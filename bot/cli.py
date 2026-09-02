"""Interface en ligne de commande.

    python -m bot.cli check              verifie la config et l'acces aux donnees
    python -m bot.cli fetch              telecharge / rafraichit le cache d'historique
    python -m bot.cli tick               passage horaire : decide et journalise
    python -m bot.cli forward-status     etat du journal de forward testing
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


def cmd_forward_status(cfg: Config, args: argparse.Namespace) -> int:
    log = ForwardLog(cfg.path(cfg.runtime.forward_log))
    if not log.exists():
        print("Aucun journal de forward testing. Il sera cree au premier 'tick'.")
        return 0
    s = log.stats()
    print("Journal de forward testing")
    print(f"  decisions enregistrees en direct : {s['decisions_live']}")
    print(f"  trades en direct                 : {s['trades_live']}")
    print(f"  duree du forward test            : {s['jours_de_forward']} jours")
    print(f"  debut                            : {s['debut_live'] or '-'}")
    print(f"  dernier passage                  : {s['dernier_live'] or '-'}")
    if s["divergences"]:
        print(f"  DIVERGENCES DETECTEES            : {s['divergences']} (forward test rompu)")
    if s["jours_de_forward"] < 90:
        print(
            "\n  Rappel : il faut au minimum 3 a 6 mois de journal continu, et\n"
            "  plusieurs centaines de decisions, avant que ces resultats\n"
            "  commencent a vouloir dire quelque chose."
        )
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

    sub.add_parser(
        "tick", help="passage horaire : decide et journalise, sans rapport"
    ).set_defaults(func=cmd_tick)
    sub.add_parser(
        "forward-status", help="etat du journal de forward testing"
    ).set_defaults(func=cmd_forward_status)

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
