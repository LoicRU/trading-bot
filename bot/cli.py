"""Interface en ligne de commande.

    python -m bot.cli check              verifie la config et l'acces aux donnees
    python -m bot.cli fetch              telecharge / rafraichit le cache d'historique
    python -m bot.cli tick               passage horaire : decide et journalise
    python -m bot.cli forward-status     etat du journal de forward testing
    python -m bot.cli forward-status --since 2026-09-03   idem, restreint a une fenetre
    python -m bot.cli forward-report [--since ...]   page HTML : P&L reellement realise en direct
    python -m bot.cli edge --segment train   le signal d'entree a-t-il un avantage ?
    python -m bot.cli explore                depistage de 40 signaux exploratoires (train uniquement)
    python -m bot.cli edge --signal <cle> --explo --segment validation   confirmation d'un candidat
    python -m bot.cli backtest           entrainement + validation
    python -m bot.cli backtest --use-holdout   consomme le bloc de reserve (une seule fois)
    python -m bot.cli daily              execution du jour + rapport du soir
    python -m bot.cli report --day ...   regenere un rapport pour une journee
    python -m bot.cli holdout-status     etat du verrou de la reserve
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Sequence

from .adapters import MarketDataError, build_adapter
from .backtest import holdout_used, reset_holdout_lock, run_backtest, split
from .config import Config, load_config
from .database import Database
from .engine import Engine, RunResult
from .forward import ForwardLog
from .models import Action, Candle
from .report import build_report, write_report


# ----------------------------------------------------------------------
def _load_candles(cfg: Config, adapter) -> List[Candle]:
    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start_ms = end_ms - cfg.market.history_days * 86_400_000
    candles = adapter.get_candles(start_ms, end_ms)
    if not candles:
        raise MarketDataError("Aucune bougie recuperee : verifie le symbole et l'unite de temps.")
    return candles


def _kill_switch_active(cfg: Config) -> bool:
    return cfg.path(cfg.runtime.kill_switch_file).exists()


def _persist(cfg: Config, result: RunResult, mode: str, segment: Optional[str] = None) -> int:
    with Database(cfg.path(cfg.runtime.db_path)) as db:
        run_id = db.create_run(
            started_at=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            mode=mode,
            adapter=cfg.market.adapter,
            symbol=cfg.market.symbol,
            timeframe=cfg.market.timeframe,
            config=cfg.to_dict(),
            segment=segment,
        )
        db.insert_decisions(run_id, result.decisions)
        db.insert_trades(run_id, result.trades)
        db.insert_scores(run_id, result.score_entries)
        db.insert_snapshots(run_id, result.snapshots)
    return run_id


def _summary(name: str, result: RunResult) -> str:
    m = result.metrics
    assert m is not None
    bd = result.score_breakdown
    pf = "inf" if m.profit_factor == float("inf") else f"{m.profit_factor:.2f}"
    return "\n".join(
        [
            f"--- {name} ---",
            f"  capital        : {m.initial_equity:,.2f} -> {m.final_equity:,.2f} "
            f"({m.total_return:+.2%})",
            f"  marche         : {m.buy_hold_return:+.2%}  |  ecart : {m.excess_return:+.2%}",
            f"  drawdown max   : -{m.max_drawdown:.2%}  (du {m.max_drawdown_start} au {m.max_drawdown_end})",
            f"  trades         : {m.trade_count}  |  reussite {m.win_rate:.1%}  |  "
            f"facteur de profit {pf}",
            f"  esperance      : {m.expectancy_r:+.2f} R par trade  |  frais {m.total_fees:,.2f}",
            f"  exposition     : {m.exposure:.1%}  |  jours sans trade : {m.days_without_trade}",
            f"  score          : {bd.get('total', 0):+.2f} "
            f"(trades {bd.get('trade_score', 0):+.2f} / abstentions {bd.get('abstention_score', 0):+.2f})",
            f"  abstentions    : {bd.get('abstention_count', 0)} notees "
            f"({bd.get('abstentions_penalisees', 0)} penalisees, "
            f"{bd.get('abstentions_recompensees', 0)} recompensees)",
            f"  score moyen    : {bd.get('moyenne_par_note', 0):+.3f} par decision notee",
        ]
    )


# ----------------------------------------------------------------------
# Commandes
# ----------------------------------------------------------------------
def _record_forward(cfg: Config, result: RunResult) -> None:
    """Alimente le journal de forward testing (append seul)."""
    log = ForwardLog(cfg.path(cfg.runtime.forward_log))
    rep = log.append_new(result, cfg.market.symbol, cfg.market.timeframe)
    print(f"  journal : {rep.summary()}")
    if rep.divergences:
        print(
            "  ATTENTION : des decisions deja journalisees ont change en etant\n"
            "  recalculees. Ton forward test n'est plus continu a partir de la.\n"
            f"  Details dans {cfg.runtime.forward_log} (lignes de type divergence).",
            file=sys.stderr,
        )


def cmd_tick(cfg: Config, args: argparse.Namespace) -> int:
    """Passage horaire : decide et journalise, sans generer de rapport.

    C'est cette commande qui fait la difference entre un backtest deguise
    et un vrai forward test : elle n'enregistre que les decisions nouvelles,
    horodatees a l'heure reelle du passage, et ne reecrit jamais le passe.
    """
    if _kill_switch_active(cfg):
        print(
            f"COUPE-CIRCUIT ACTIF : le fichier {cfg.runtime.kill_switch_file} existe.",
            file=sys.stderr,
        )
        return 4

    adapter = build_adapter(cfg)
    candles = _load_candles(cfg, adapter)
    result = Engine(cfg, adapter.interval_ms).run(candles, cfg.market.symbol)

    last = result.decisions[-1] if result.decisions else None
    if last is not None:
        print(f"Derniere decision : {last.dt:%Y-%m-%d %H:%M} UTC -> {last.action.value}")
        print(f"  motif : {last.reason}")
    print(f"  capital simule : {result.final_equity:,.2f}")
    _record_forward(cfg, result)
    _persist(cfg, result, mode="tick")
    return 0


def _parse_since(valeur: str, offset_hours: float) -> int:
    """Convertit 'AAAA-MM-JJ' ou 'AAAA-MM-JJTHH:MM' (heure locale du rapport,
    cfg.report.timezone_offset_hours) en millisecondes depuis l'epoque.

    Leve ValueError si le format ne correspond a aucun des deux.
    """
    fmt = "%Y-%m-%dT%H:%M" if "T" in valeur else "%Y-%m-%d"
    dt = datetime.strptime(valeur, fmt).replace(
        tzinfo=timezone(timedelta(hours=offset_hours))
    )
    return int(dt.timestamp() * 1000)


def cmd_forward_status(cfg: Config, args: argparse.Namespace) -> int:
    from .report import quote_currency

    log = ForwardLog(cfg.path(cfg.runtime.forward_log))
    if not log.exists():
        print("Aucun journal de forward testing. Il sera cree au premier 'tick'.")
        return 0

    since_ts = None
    if getattr(args, "since", None):
        try:
            since_ts = _parse_since(args.since, cfg.report.timezone_offset_hours)
        except ValueError:
            print(
                f"Date invalide : {args.since!r} (format attendu AAAA-MM-JJ ou "
                "AAAA-MM-JJTHH:MM)",
                file=sys.stderr,
            )
            return 1

    s = log.stats(since_ts=since_ts)
    quote = quote_currency(cfg.market.symbol)
    titre = "Journal de forward testing"
    if since_ts is not None:
        titre += f" (depuis {args.since})"
    print(titre)
    print(f"  decisions enregistrees en direct : {s['decisions_live']}")
    if s["decisions_par_action"]:
        detail = ", ".join(f"{k} : {v}" for k, v in sorted(s["decisions_par_action"].items()))
        print(f"    dont                            : {detail}")
    print(f"  trades en direct                 : {s['trades_live']}"
          f" ({s['trades_gagnants']} gagnant(s) / {s['trades_perdants']} perdant(s))")
    signe = "+" if s["pnl_total"] >= 0 else ""
    print(f"  P&L cumule (trades clotures)     : {signe}{s['pnl_total']:.2f} {quote}")
    print(f"  duree de la fenetre               : {s['jours_de_forward']} jours")
    print(f"  debut                            : {s['debut_live'] or '-'}")
    print(f"  dernier passage                  : {s['dernier_live'] or '-'}")
    if s["divergences"]:
        print(f"  DIVERGENCES DETECTEES            : {s['divergences']} (forward test rompu)")
    if since_ts is None and s["jours_de_forward"] < 90:
        print(
            "\n  Rappel : il faut au minimum 3 a 6 mois de journal continu, et\n"
            "  plusieurs centaines de decisions, avant que ces resultats\n"
            "  commencent a vouloir dire quelque chose."
        )
    return 0


def cmd_forward_report(cfg: Config, args: argparse.Namespace) -> int:
    """Genere une page HTML autonome : courbe du P&L reellement realise en
    direct (pas le backtest rejoue) plus le detail des trades du journal.
    """
    from .report import build_forward_report, write_report

    log = ForwardLog(cfg.path(cfg.runtime.forward_log))
    if not log.exists():
        print("Aucun journal de forward testing. Il sera cree au premier 'tick'.", file=sys.stderr)
        return 0

    since_ts = None
    if getattr(args, "since", None):
        try:
            since_ts = _parse_since(args.since, cfg.report.timezone_offset_hours)
        except ValueError:
            print(
                f"Date invalide : {args.since!r} (format attendu AAAA-MM-JJ ou "
                "AAAA-MM-JJTHH:MM)",
                file=sys.stderr,
            )
            return 1

    html_text = build_forward_report(cfg, log, since_ts=since_ts, since_label=args.since)
    path = write_report(cfg, html_text, "forward_pnl.html")
    print(f"Rapport ecrit : {path}")
    return 0


MS_PAR_AN = 365.25 * 24 * 3600 * 1000


def cout_total(cfg: Config, horizon_bougies: int, interval_ms: int) -> tuple[float, str]:
    """Cout complet d'un trade tenu `horizon_bougies` bougies.

    Deux composantes de nature differente, et c'est tout l'interet de les
    separer :
      - le SPREAD et les frais, payes une fois a l'aller-retour ;
      - le SWAP, paye chaque nuit, donc proportionnel a la duree.

    En crypto au comptant le second est nul et seul le premier compte : on a
    donc interet a tenir longtemps pour amortir. Sur CFD forex c'est
    l'inverse — tenir plus longtemps coute plus cher. Un horizon optimal en
    crypto peut etre ruineux en forex, et cette fonction est ce qui rend la
    difference visible.
    """
    aller_retour = 2 * (cfg.portfolio.fee_rate + cfg.portfolio.slippage_bps / 10_000.0)
    duree_an = (horizon_bougies * interval_ms) / MS_PAR_AN
    portage = cfg.portfolio.swap_annual_pct * duree_an
    total = aller_retour + portage
    jours = duree_an * 365.25
    detail = (
        f"{aller_retour:.4%} d'aller-retour"
        + (f" + {portage:.4%} de portage sur {jours:.1f} jours" if portage else "")
    )
    return total, detail


def _load_extras(cfg: Config, candles: List[Candle], args: argparse.Namespace) -> dict:
    """Charge les donnees hors-prix (taux de financement) si possible.

    Un echec n'est pas bloquant : les signaux qui en dependent sont
    simplement ecartes de la comparaison, et le nombre de tests — donc la
    correction statistique — s'ajuste tout seul.
    """
    if getattr(args, "sans_financement", False) or not candles:
        return {}
    from .funding import FundingError, fetch

    try:
        funding = fetch(
            cfg.market.symbol.replace("-", ""),
            candles[0].ts - 90 * 8 * 3600 * 1000,  # de quoi amorcer la fenetre glissante
            candles[-1].ts,
            str(cfg.path(cfg.market.cache_dir)),
        )
    except FundingError as exc:
        print(f"Taux de financement indisponibles ({exc})", file=sys.stderr)
        print("  -> les signaux de financement sont ecartes de cette comparaison.\n",
              file=sys.stderr)
        return {}

    if len(funding) < 200:
        print(f"Historique de financement trop court ({len(funding)} paiements) : "
              "signaux de financement ecartes.\n", file=sys.stderr)
        return {}
    print(f"Taux de financement : {len(funding)} paiements charges.")
    return {"funding": funding}


def cmd_edge(cfg: Config, args: argparse.Namespace) -> int:
    """Le signal d'entree predit-il quelque chose, oui ou non ?

    A lancer AVANT de passer du temps a regler des sorties : si l'entree
    n'a aucun avantage, aucun reglage de stop ne la sauvera.
    """
    from . import edge as edge_mod
    from .backtest import split

    adapter = build_adapter(cfg)
    candles = _load_candles(cfg, adapter)

    if args.segment:
        parts = split(candles, cfg)
        if args.segment not in parts:
            print(f"Bloc inconnu : {args.segment}", file=sys.stderr)
            return 1
        if args.segment == "holdout":
            print(
                "Refus : le bloc de reserve ne sert pas a explorer. "
                "Utilise 'train'.",
                file=sys.stderr,
            )
            return 3
        candles = parts[args.segment].candles
        print(f"Bloc analyse : {args.segment}")

    horizons = (
        tuple(int(h) for h in args.horizons.split(","))
        if args.horizons
        else edge_mod.DEFAULT_HORIZONS
    )
    extras = _load_extras(cfg, candles, args)

    if args.all_signals:
        data = edge_mod.compare_signals(
            cfg, candles, horizons=horizons, permutations=args.permutations, extras=extras
        )
        print()
        print(edge_mod.format_comparison(data))
        return 0

    if args.signal:
        from .signals import by_key as by_key_fige

        if getattr(args, "explo", False):
            from .signals_explo import by_key as lookup
        else:
            lookup = by_key_fige

        try:
            sig = lookup(args.signal)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if args.periods:
            if len(horizons) != 1:
                print(
                    "Avec --periods, precise un seul horizon : --horizons 72",
                    file=sys.stderr,
                )
                return 1
            rows = edge_mod.analyse_by_period(
                cfg, candles, sig, horizons[0], periods=args.periods,
                permutations=args.permutations, extras=extras,
            )
            cout, detail = cout_total(cfg, horizons[0], adapter.interval_ms)
            print()
            print(f"Cout retenu : {detail}")
            print(edge_mod.format_periods(rows, horizons[0], sig.spec.label, cout))
            return 0

        data = edge_mod.analyse_signal(
            cfg, candles, sig, horizons=horizons,
            permutations=args.permutations, extras=extras,
        )
        print(f"\nSignal : {sig.spec.label}")
        print(f"Hypothese : {sig.spec.hypothese}\n")
        print(edge_mod.format_report(data))
        return 0

    data = edge_mod.analyse(cfg, candles, horizons=horizons, permutations=args.permutations)
    print()
    print(edge_mod.format_report(data))
    return 0


def cmd_explore(cfg: Config, args: argparse.Namespace) -> int:
    """Etape 1 du protocole en deux temps : depistage de la bibliotheque
    exploratoire (40 candidats issus de TA-Lib/pandas-ta/WorldQuant, voir
    bot/signals_explo.py et catalogue_signaux.md) sur le bloc TRAIN
    uniquement. Le bloc VALIDATION n'est jamais touche ici — c'est lui qui
    sert a reconfirmer le top de ce classement, une seule fois.
    """
    from . import edge as edge_mod
    from .backtest import split

    adapter = build_adapter(cfg)
    candles = _load_candles(cfg, adapter)
    parts = split(candles, cfg)
    train = parts["train"].candles
    print(f"Bloc analyse : train ({parts['train'].eval_bars} bougies evaluees, "
          f"validation et reserve non touches)\n")

    horizons = (
        tuple(int(h) for h in args.horizons.split(","))
        if args.horizons
        else edge_mod.DEFAULT_HORIZONS
    )
    data = edge_mod.explore_signals(cfg, train, horizons=horizons, permutations=args.permutations)
    print(edge_mod.format_explore(data, top_k=args.top))
    return 0


def cmd_pooled(cfg: Config, args: argparse.Namespace) -> int:
    """Test d'avantage transversal sur tout l'univers de paires."""
    from .adapters import TIMEFRAME_MS, BinanceAdapter, CoinbaseAdapter, MarketDataError
    from .pooled import analyse_pooled, format_pooled
    from .signals import by_key
    from .universe import UNIVERS, align_on_common_grid

    try:
        sig = by_key(args.signal)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    start_ms = end_ms - cfg.market.history_days * 86_400_000
    cache = str(cfg.path(cfg.market.cache_dir))

    series: dict = {}
    ecartees: List[str] = []
    print(f"Telechargement de {len(UNIVERS)} paires en {cfg.market.timeframe}...")
    for k, sym in enumerate(UNIVERS, 1):
        try:
            if cfg.market.adapter == "coinbase":
                ad = CoinbaseAdapter(sym.replace("USDT", "-USD"), cfg.market.timeframe, cache)
            else:
                ad = BinanceAdapter(sym, cfg.market.timeframe, cache)
            candles = ad.get_candles(start_ms, end_ms)
            if len(candles) < 400:
                ecartees.append(sym)
            else:
                series[sym] = candles
        except MarketDataError:
            ecartees.append(sym)
        if k % 10 == 0:
            print(f"  {k}/{len(UNIVERS)} paires traitees")

    if len(series) < 5:
        print(
            f"Seulement {len(series)} paire(s) exploitable(s) : pas de quoi faire "
            "un test transversal.",
            file=sys.stderr,
        )
        return 2

    grille, alignees = align_on_common_grid(series)
    print(f"{len(alignees)} paires alignees sur {len(grille)} dates communes.")
    if ecartees:
        print(f"Ecartees (historique trop court ou indisponible) : {', '.join(ecartees)}")
    print(
        "\nRappel : cet univers ne contient que des paires encore cotees aujourd'hui.\n"
        "Les projets morts en sont absents, et ce sont ceux dont les cassures ont\n"
        "le plus mal fini. Tout resultat obtenu ici est donc OPTIMISTE.\n"
    )

    res = analyse_pooled(
        cfg, grille, alignees, type(sig), args.horizon,
        permutations=args.permutations,
    )
    interval = TIMEFRAME_MS[cfg.market.timeframe]
    cout, detail = cout_total(cfg, args.horizon, interval)
    print(f"Cout retenu : {detail}\n")
    print(format_pooled(res, sig.spec.label, cout, ecartees))
    return 0


def cmd_check(cfg: Config, args: argparse.Namespace) -> int:
    print(f"Configuration valide. Adaptateur : {cfg.market.adapter}")
    adapter = build_adapter(cfg)
    print(f"Marche : {adapter.describe()}")
    try:
        candles = _load_candles(cfg, adapter)
    except MarketDataError as exc:
        print(f"ECHEC de la recuperation des donnees : {exc}", file=sys.stderr)
        return 2
    first, last = candles[0], candles[-1]
    print(f"Bougies : {len(candles)}  du {first.dt:%Y-%m-%d %H:%M} au {last.dt:%Y-%m-%d %H:%M} UTC")
    try:
        parts = split(candles, cfg)
        for name, seg in parts.items():
            print(f"  bloc {name:<11} {seg.eval_bars:>6} bougies evaluees")
    except ValueError as exc:
        print(f"Decoupage impossible : {exc}", file=sys.stderr)
        return 2
    lock = holdout_used()
    print(f"Reserve : {'DEJA CONSOMMEE le ' + lock.get('used_at', '?') if lock else 'intacte'}")
    print(f"Coupe-circuit : {'ACTIF (le bot refusera de tourner)' if _kill_switch_active(cfg) else 'inactif'}")
    return 0


def cmd_fetch(cfg: Config, args: argparse.Namespace) -> int:
    adapter = build_adapter(cfg)
    candles = _load_candles(cfg, adapter)
    print(f"{len(candles)} bougies disponibles pour {adapter.describe()}.")
    print(f"Derniere : {candles[-1].dt:%Y-%m-%d %H:%M} UTC, cloture {candles[-1].close}")
    return 0


def cmd_backtest(cfg: Config, args: argparse.Namespace) -> int:
    adapter = build_adapter(cfg)
    candles = _load_candles(cfg, adapter)

    segments: Sequence[str] = args.segments.split(",") if args.segments else ("train", "validation")
    if args.use_holdout and "holdout" not in segments:
        segments = list(segments) + ["holdout"]

    try:
        results = run_backtest(
            cfg, candles, adapter.interval_ms,
            segments=segments, allow_holdout=args.use_holdout,
        )
    except PermissionError as exc:
        print(f"REFUS : {exc}", file=sys.stderr)
        return 3

    for name, result in results.items():
        print(_summary(name, result))
        run_id = _persist(cfg, result, mode="backtest", segment=name)
        if args.report:
            seg_candles = split(candles, cfg)[name].candles
            html_text = build_report(cfg, result, seg_candles, day=None,
                                     mode_label=f"backtest / bloc {name}")
            path = write_report(cfg, html_text, f"backtest_{name}.html")
            print(f"  rapport : {path}")
        print(f"  enregistre en base sous run_id={run_id}")

    if "holdout" in results:
        print(
            "\nLe bloc de reserve vient d'etre consomme et verrouille.\n"
            "Si tu modifies la strategie maintenant, son resultat ne compte plus\n"
            "comme une validation independante."
        )
    return 0


def cmd_daily(cfg: Config, args: argparse.Namespace) -> int:
    """Execution quotidienne.

    Le bot rejoue tout l'historique depuis le debut a chaque execution.
    C'est volontaire : la simulation est deterministe, donc rejouer donne
    exactement le meme etat qu'un suivi continu, sans aucun risque
    d'etat corrompu ou desynchronise apres une interruption.
    """
    if _kill_switch_active(cfg):
        print(
            f"COUPE-CIRCUIT ACTIF : le fichier {cfg.runtime.kill_switch_file} existe. "
            "Le bot ne tourne pas. Supprime-le pour reprendre.",
            file=sys.stderr,
        )
        return 4

    adapter = build_adapter(cfg)
    candles = _load_candles(cfg, adapter)
    engine = Engine(cfg, adapter.interval_ms)
    result = engine.run(candles, cfg.market.symbol)

    off = cfg.report.timezone_offset_hours
    tz = timezone(timedelta(hours=off))
    if args.day:
        day = args.day
    else:
        last_ts = result.equity_curve[-1][0]
        day = datetime.fromtimestamp(last_ts / 1000, tz=tz).strftime("%Y-%m-%d")

    run_id = _persist(cfg, result, mode="daily")
    print(_summary(f"suivi quotidien - {day}", result))
    _record_forward(cfg, result)

    day_trades = [t for t in result.trades if datetime.fromtimestamp(t.exit_ts / 1000, tz=tz).strftime("%Y-%m-%d") == day]
    day_abst = [
        d for d in result.decisions
        if d.action == Action.ABSTAIN
        and datetime.fromtimestamp(d.ts / 1000, tz=tz).strftime("%Y-%m-%d") == day
    ]
    print(f"  journee du {day} : {len(day_trades)} trade(s) cloture(s), {len(day_abst)} abstention(s)")

    fwd = ForwardLog(cfg.path(cfg.runtime.forward_log))
    html_text = build_report(
        cfg, result, candles, day=day, mode_label="suivi quotidien",
        forward_stats=fwd.stats() if fwd.exists() else None,
    )
    path = write_report(cfg, html_text, f"rapport_{day}.html")
    print(f"  rapport : {path}")
    print(f"  enregistre en base sous run_id={run_id}")
    return 0


def cmd_report(cfg: Config, args: argparse.Namespace) -> int:
    adapter = build_adapter(cfg)
    candles = _load_candles(cfg, adapter)
    engine = Engine(cfg, adapter.interval_ms)
    result = engine.run(candles, cfg.market.symbol)
    html_text = build_report(cfg, result, candles, day=args.day, mode_label="rapport a la demande")
    name = f"rapport_{args.day}.html" if args.day else "rapport_periode_complete.html"
    path = write_report(cfg, html_text, name)
    print(f"Rapport ecrit : {path}")
    return 0


def cmd_holdout_status(cfg: Config, args: argparse.Namespace) -> int:
    lock = holdout_used()
    if not lock:
        print("Bloc de reserve : INTACT. Il n'a jamais ete teste.")
        return 0
    print("Bloc de reserve : DEJA CONSOMME.")
    for key in ("used_at", "symbol", "timeframe"):
        print(f"  {key} : {lock.get(key)}")
    print(f"  resultat : {lock.get('result')}")
    print(f"  {lock.get('note', '')}")
    return 0


def cmd_holdout_reset(cfg: Config, args: argparse.Namespace) -> int:
    if not args.i_understand:
        print(
            "Cette commande efface la trace d'utilisation du bloc de reserve.\n"
            "Apres l'avoir vu une fois, le retester ne prouve plus rien : le bloc\n"
            "n'est plus independant de tes choix. Relance avec --i-understand si\n"
            "tu veux quand meme le faire.",
            file=sys.stderr,
        )
        return 5
    print("Verrou supprime." if reset_holdout_lock() else "Aucun verrou a supprimer.")
    return 0


# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bot", description="Bot de trading sur portefeuille fictif (simulation uniquement)."
    )
    p.add_argument("--config", help="chemin d'un config.json alternatif")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("check", help="verifie la config et l'acces aux donnees").set_defaults(func=cmd_check)
    sub.add_parser("fetch", help="telecharge / rafraichit le cache").set_defaults(func=cmd_fetch)

    bt = sub.add_parser("backtest", help="backtest sur les blocs d'entrainement et de validation")
    bt.add_argument("--segments", help="blocs a executer, separes par des virgules")
    bt.add_argument("--use-holdout", action="store_true", help="consomme le bloc de reserve (une seule fois)")
    bt.add_argument("--report", action="store_true", help="genere aussi un rapport HTML par bloc")
    bt.set_defaults(func=cmd_backtest)

    ed = sub.add_parser(
        "edge", help="le signal d'entree a-t-il un pouvoir predictif ?"
    )
    ed.add_argument("--segment", help="restreindre a un bloc (train, validation)")
    ed.add_argument("--horizons", help="horizons en bougies, separes par des virgules")
    ed.add_argument("--permutations", type=int, default=2000)
    ed.add_argument(
        "--all-signals", action="store_true",
        help="tester toute la liste figee de candidats, avec correction de Bonferroni",
    )
    ed.add_argument("--signal", help="tester un seul candidat de la bibliotheque")
    ed.add_argument(
        "--sans-financement", dest="sans_financement", action="store_true",
        help="ne pas telecharger les taux de financement",
    )
    ed.add_argument(
        "--periods", type=int,
        help="decouper en N tranches chronologiques pour tester la stabilite "
             "de l'avantage dans le temps (exige --signal et un seul --horizons)",
    )
    ed.add_argument(
        "--explo", action="store_true",
        help="chercher --signal dans la bibliotheque exploratoire (bot/signals_explo.py, "
             "40 candidats) plutot que dans la liste figee de bot/signals.py",
    )
    ed.set_defaults(func=cmd_edge)

    ex = sub.add_parser(
        "explore",
        help="depistage de 40 signaux exploratoires (TA-Lib/pandas-ta/WorldQuant) sur le bloc train",
    )
    ex.add_argument("--horizons", help="horizons en bougies, separes par des virgules")
    ex.add_argument("--permutations", type=int, default=2000)
    ex.add_argument("--top", type=int, default=5, help="taille du classement a reconfirmer sur validation")
    ex.set_defaults(func=cmd_explore)

    pl = sub.add_parser(
        "pooled", help="test d'avantage transversal sur tout l'univers de paires"
    )
    pl.add_argument("--signal", required=True, help="candidat a tester")
    pl.add_argument("--horizon", type=int, required=True, help="horizon en bougies")
    pl.add_argument("--permutations", type=int, default=2000)
    pl.set_defaults(func=cmd_pooled)

    sub.add_parser(
        "tick", help="passage horaire : decide et journalise, sans rapport"
    ).set_defaults(func=cmd_tick)

    fs = sub.add_parser("forward-status", help="etat du journal de forward testing")
    fs.add_argument(
        "--since",
        help="ne compter que depuis cette date/heure locale (AAAA-MM-JJ ou "
             "AAAA-MM-JJTHH:MM), ex: --since 2026-09-03 pour 'depuis hier matin'",
    )
    fs.set_defaults(func=cmd_forward_status)

    fr = sub.add_parser(
        "forward-report",
        help="page HTML : courbe du P&L reellement realise en direct (pas le backtest rejoue)",
    )
    fr.add_argument("--since", help="meme format que forward-status --since")
    fr.set_defaults(func=cmd_forward_report)

    dl = sub.add_parser("daily", help="execution du jour et rapport du soir")
    dl.add_argument("--day", help="journee a rapporter (AAAA-MM-JJ), par defaut la derniere")
    dl.set_defaults(func=cmd_daily)

    rp = sub.add_parser("report", help="regenere un rapport")
    rp.add_argument("--day", help="journee (AAAA-MM-JJ) ; sans cette option, periode complete")
    rp.set_defaults(func=cmd_report)

    sub.add_parser("holdout-status", help="etat du verrou de la reserve").set_defaults(
        func=cmd_holdout_status
    )
    hr = sub.add_parser("holdout-reset", help="efface le verrou de la reserve")
    hr.add_argument("--i-understand", action="store_true")
    hr.set_defaults(func=cmd_holdout_reset)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    try:
        return int(args.func(cfg, args))
    except MarketDataError as exc:
        print(f"Erreur de donnees de marche : {exc}", file=sys.stderr)
        return 2
    except (ValueError, RuntimeError) as exc:
        print(f"Erreur : {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
