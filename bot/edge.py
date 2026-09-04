"""Test d'avantage : le signal d'entree predit-il quoi que ce soit ?

Pourquoi cet outil existe
--------------------------
On peut passer des mois a regler des stops, des objectifs et des regles de
sortie. Mais tout ce travail ne sert a rien si le signal d'ENTREE n'a aucun
pouvoir predictif : on ne fait alors que reorganiser la facon dont on perd.

Ce module repond a une seule question, en amont de toute strategie :
apres un signal d'achat, le marche monte-t-il davantage qu'apres une
bougie prise au hasard ?

Methode
-------
Pour chaque horizon (6, 12, 24... bougies apres le signal), on compare :
  - le rendement moyen apres les N signaux reels,
  - au rendement moyen apres N bougies tirees au hasard dans la meme
    periode.

La comparaison se fait par test de permutation : on retire au hasard
2000 echantillons de meme taille et on regarde ou tombe le resultat reel
dans cette distribution. C'est plus honnete qu'un test de Student, qui
suppose des rendements independants — ce que des prix ne sont jamais.

Lecture du resultat
-------------------
p >= 0.05  : le signal ne se distingue pas du hasard. Aucun reglage de
             sortie ne sauvera cette entree. Il faut changer de signal.
p <  0.05  : le signal contient quelque chose. Attention quand meme : sur
             cinq horizons testes, un "significatif" par pur hasard est
             attendu environ une fois sur quatre. Ne te precipite pas.

Les rendements sont aussi exprimes en multiples d'ATR, parce qu'un gain
de 1 % ne vaut pas la meme chose dans un marche calme et dans un marche
agite.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .config import Config
from .models import Action, Candle
from .strategy import EmaAtrStrategy

DEFAULT_HORIZONS = (6, 12, 24, 48, 72)
PERMUTATIONS = 2000


@dataclass
class HorizonResult:
    horizon: int
    n_signals: int
    signal_mean: float          # rendement moyen apres signal
    baseline_mean: float        # rendement moyen apres une bougie quelconque
    edge: float                 # difference
    signal_mean_atr: float      # le meme, en multiples d'ATR
    baseline_mean_atr: float
    win_rate: float             # part des signaux suivis d'une hausse
    baseline_win_rate: float
    p_value: float

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def _iso(ts: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _forward_return(candles: Sequence[Candle], i: int, h: int) -> Optional[float]:
    """Rendement entre l'ouverture de la bougie suivante (le prix auquel on
    entrerait reellement) et la cloture h bougies plus loin."""
    if i + 1 >= len(candles) or i + h >= len(candles):
        return None
    entry = candles[i + 1].open
    if entry <= 0:
        return None
    return (candles[i + h].close - entry) / entry


def _run(
    candles: Sequence[Candle],
    atr_at,
    signal_idx: Sequence[int],
    warmup: int,
    horizons: Sequence[int],
    permutations: int,
    seed: int,
) -> List["HorizonResult"]:
    """Coeur de la mesure, commun a tous les signaux."""
    results: List[HorizonResult] = []

    rng = random.Random(seed)
    for h in horizons:
        eligible = [i for i in range(warmup, len(candles) - 1) if i + h < len(candles)]
        sig = [i for i in signal_idx if i + h < len(candles)]
        if len(sig) < 10 or len(eligible) < 100:
            continue

        def stats(indices: Sequence[int]):
            rets, rets_atr, ups = [], [], 0
            for i in indices:
                r = _forward_return(candles, i, h)
                if r is None:
                    continue
                rets.append(r)
                a = atr_at(i)
                if a and candles[i].close:
                    rets_atr.append(r / (a / candles[i].close))
                if r > 0:
                    ups += 1
            n = len(rets) or 1
            return (
                sum(rets) / n,
                (sum(rets_atr) / len(rets_atr)) if rets_atr else 0.0,
                ups / n,
            )

        s_mean, s_mean_atr, s_wr = stats(sig)
        b_mean, b_mean_atr, b_wr = stats(eligible)

        # Test de permutation : combien de tirages aleatoires font aussi
        # bien ou mieux que les vrais signaux ?
        k = len(sig)
        better = 0
        for _ in range(permutations):
            sample = rng.sample(eligible, k)
            total = 0.0
            count = 0
            for i in sample:
                r = _forward_return(candles, i, h)
                if r is not None:
                    total += r
                    count += 1
            if count and (total / count) >= s_mean:
                better += 1
        p = (better + 1) / (permutations + 1)  # correction de Laplace

        results.append(
            HorizonResult(
                horizon=h,
                n_signals=k,
                signal_mean=s_mean,
                baseline_mean=b_mean,
                edge=s_mean - b_mean,
                signal_mean_atr=s_mean_atr,
                baseline_mean_atr=b_mean_atr,
                win_rate=s_wr,
                baseline_win_rate=b_wr,
                p_value=p,
            )
        )
    return results


def analyse(
    cfg: Config,
    candles: Sequence[Candle],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    permutations: int = PERMUTATIONS,
    seed: int = 12345,
) -> Dict[str, object]:
    """Teste le signal d'entree de la strategie en place, filtres compris."""
    strategy = EmaAtrStrategy(cfg.strategy)
    strategy.prepare(candles)

    # Signaux bruts, avant tout filtre de risque : on mesure le pouvoir
    # predictif du signal lui-meme, pas celui du money management.
    signal_idx = [
        i
        for i in range(strategy.warmup, len(candles) - 1)
        if strategy.decide(i, candles, has_position=False).action == Action.BUY
    ]
    results = _run(
        candles, strategy.atr_at, signal_idx, strategy.warmup,
        horizons, permutations, seed,
    )
    return {
        "total_signals": len(signal_idx),
        "bars": len(candles),
        "results": results,
    }


def analyse_signal(
    cfg: Config,
    candles: Sequence[Candle],
    signal,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    permutations: int = PERMUTATIONS,
    seed: int = 12345,
    extras: Optional[Dict] = None,
) -> Dict[str, object]:
    """Teste un signal nu de la bibliotheque, sans aucun filtre."""
    from .indicators import atr as atr_series

    signal.prepare(candles, cfg.strategy, extras)
    atr_vals = atr_series(candles, cfg.strategy.atr_period)
    warmup = max(signal.warmup, cfg.strategy.atr_period + 2)

    signal_idx = [i for i in range(warmup, len(candles) - 1) if signal.fires(i)]
    results = _run(
        candles, lambda i: atr_vals[i] if i < len(atr_vals) else None,
        signal_idx, warmup, horizons, permutations, seed,
    )
    return {
        "signal": signal.spec,
        "total_signals": len(signal_idx),
        "bars": len(candles),
        "results": results,
    }


def compare_signals(
    cfg: Config,
    candles: Sequence[Candle],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    permutations: int = PERMUTATIONS,
    seed: int = 12345,
    extras: Optional[Dict] = None,
) -> Dict[str, object]:
    """Teste toute la liste figee de candidats, avec correction du nombre
    de tests effectues."""
    from .signals import build_all

    dispo_financement = bool((extras or {}).get("funding"))
    signaux = build_all(avec_financement=dispo_financement)
    rapports = [
        analyse_signal(cfg, candles, s, horizons, permutations, seed + k, extras)
        for k, s in enumerate(signaux)
    ]
    n_tests = sum(len(r["results"]) for r in rapports)  # type: ignore[arg-type]
    seuil = 0.05 / n_tests if n_tests else 0.05
    return {"rapports": rapports, "n_tests": n_tests, "seuil_corrige": seuil}


def explore_signals(
    cfg: Config,
    candles: Sequence[Candle],
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    permutations: int = PERMUTATIONS,
    seed: int = 424242,
) -> Dict[str, object]:
    """Etape 1 du protocole en deux temps : depistage de la bibliotheque
    EXPLORATOIRE (bot/signals_explo.py), sur le bloc fourni en argument
    UNIQUEMENT (l'appelant doit passer le bloc train, jamais validation ni
    holdout — cette fonction ne le verifie pas elle-meme, cmd_explore le
    garantit cote CLI).

    Ne rend AUCUN verdict de type RETENU/rien : ce n'est pas
    compare_signals. Le role de cette fonction est de classer, pas de
    trancher — le tri sert a choisir 3 a 5 candidats a reconfirmer sur le
    bloc validation, pas a conclure quoi que ce soit ici.
    """
    from .signals_explo import build_all_explo

    signaux = build_all_explo()
    rapports = [
        analyse_signal(cfg, candles, s, horizons, permutations, seed + k)
        for k, s in enumerate(signaux)
    ]
    n_tests = sum(len(r["results"]) for r in rapports)  # type: ignore[arg-type]
    seuil = 0.05 / n_tests if n_tests else 0.05
    return {"rapports": rapports, "n_tests": n_tests, "seuil_corrige": seuil}


def format_explore(data: Dict[str, object], top_k: int = 5) -> str:
    rapports = data["rapports"]  # type: ignore[assignment]
    seuil = data["seuil_corrige"]  # type: ignore[assignment]
    n_tests = data["n_tests"]

    lines = [
        "=" * 78,
        "DEPISTAGE EXPLORATOIRE — bloc TRAIN uniquement, bloc VALIDATION intact",
        "=" * 78,
        "",
        f"{len(rapports)} candidats x jusqu'a {len(DEFAULT_HORIZONS)} horizons = "
        f"{n_tests} tests. Seuil corrige de Bonferroni sur ce lot : p < {seuil:.6f}.",
        "",
        "Ce tableau CLASSE, il ne CONCLUT pas. Avec ce nombre de tests, meme",
        "un p brut < 0.001 peut etre du hasard. Ne construis rien sur ce",
        "resultat seul : l'etape suivante, obligatoire, est de reconfirmer",
        "les meilleurs candidats sur le bloc VALIDATION (jamais vu ici) avec :",
        "    python -m bot.cli edge --signal <cle> --explo --segment validation",
        "",
        f"{'signal':>28} {'categorie':>18} {'n':>5} {'meilleur horizon':>18} "
        f"{'ecart':>10} {'p':>8}",
        "-" * 96,
    ]

    scored = []
    for rap in rapports:
        spec = rap["signal"]
        res = rap["results"]
        if not res:
            lines.append(f"{spec.key:>28} {getattr(spec, 'categorie', ''):>18} "
                          f"{rap['total_signals']:>5}  pas assez de signaux")
            continue
        best = min(res, key=lambda r: r.p_value)
        scored.append((spec, best, rap["total_signals"]))
        lines.append(
            f"{spec.key:>28} {getattr(spec, 'categorie', ''):>18} {rap['total_signals']:>5} "
            f"{str(best.horizon) + 'h':>18} {best.edge:>+10.3%} {best.p_value:>8.4f}"
        )

    scored.sort(key=lambda t: t[1].p_value)
    lines.append("")
    lines.append(f"TOP {top_k} — candidats a reconfirmer sur validation (dans cet ordre) :")
    for rang, (spec, best, n) in enumerate(scored[:top_k], 1):
        marque = "significatif brut (p<0.05)" if best.p_value < 0.05 else "pas meme significatif au seuil brut"
        lines.append(
            f"  {rang}. {spec.key:<28} p={best.p_value:.4f} ({marque}), "
            f"ecart {best.edge:+.3%} a {best.horizon}h, n={n}"
        )
        if getattr(spec, "correle_avec", ""):
            lines.append(f"     correle avec : {spec.correle_avec}")

    lines += [
        "",
        "Rappel : sur un lot de cette taille, il est ATTENDU qu'au moins un",
        "candidat sorte 'significatif' au seuil brut par pur hasard, meme si",
        "rien ici ne vaut quoi que ce soit. Seule la confirmation sur",
        "validation, avec un seuil corrige de la taille du top retenu (pas de",
        "40), permet de trancher.",
    ]
    return "\n".join(lines)


def format_report(data: Dict[str, object]) -> str:
    results: List[HorizonResult] = data["results"]  # type: ignore[assignment]
    lines = [
        f"Signaux d'achat detectes : {data['total_signals']} sur {data['bars']} bougies",
        "",
        f"{'horizon':>8} {'n':>5} {'apres signal':>14} {'au hasard':>12} "
        f"{'ecart':>10} {'en ATR':>9} {'p':>7}",
        "-" * 72,
    ]
    for r in results:
        flag = "  <-- significatif" if r.significant else ""
        lines.append(
            f"{r.horizon:>6}h  {r.n_signals:>5} {r.signal_mean:>13.3%} "
            f"{r.baseline_mean:>12.3%} {r.edge:>+10.3%} "
            f"{r.signal_mean_atr - r.baseline_mean_atr:>+9.2f} {r.p_value:>7.3f}{flag}"
        )

    lines.append("")
    if not results:
        lines.append("Pas assez de signaux pour conclure quoi que ce soit.")
        return "\n".join(lines)

    signif = [r for r in results if r.significant]
    best = min(results, key=lambda r: r.p_value)

    if not signif:
        lines += [
            "VERDICT : aucun horizon ne se distingue du hasard.",
            "",
            "Le signal d'entree n'a pas de pouvoir predictif mesurable sur cette",
            "paire et cette unite de temps. Regler les stops, les objectifs ou les",
            "regles de sortie ne changera rien au fond : ca ne fera que reorganiser",
            "la facon de perdre. Il faut changer de signal d'entree, de paire, ou",
            "d'unite de temps.",
            "",
            f"(Le moins mauvais horizon est {best.horizon}h avec p={best.p_value:.3f} ;",
            " un p inferieur a 0.05 aurait ete necessaire.)",
        ]
    else:
        noms = ", ".join(f"{r.horizon}h" for r in signif)
        lines += [
            f"VERDICT : {len(signif)} horizon(s) se distinguent du hasard : {noms}.",
            "",
            "Prudence avant de s'emballer : sur cinq horizons testes, obtenir un",
            "resultat 'significatif' par pur hasard arrive environ une fois sur",
            "quatre. Un seul horizon significatif, entoure de non-significatifs,",
            "est probablement du bruit. Plusieurs horizons voisins qui pointent",
            "dans le meme sens sont un signal plus serieux.",
            "",
            "Si le resultat tient, l'etape suivante est de construire les sorties",
            "AUTOUR de cet horizon : c'est lui qui indique combien de temps",
            "l'avantage dure.",
        ]
    return "\n".join(lines)


def analyse_by_period(
    cfg: Config,
    candles: Sequence[Candle],
    signal,
    horizon: int,
    periods: int = 4,
    permutations: int = PERMUTATIONS,
    seed: int = 777,
    extras: Optional[Dict] = None,
) -> List[Dict[str, object]]:
    """Le meme signal, mesure separement sur des tranches chronologiques.

    A quoi ca sert : un avantage qui n'existe que dans les premieres
    annees d'un marche jeune, et qui a disparu depuis, ne vaut RIEN pour
    la suite. Les marches deviennent efficients avec le temps, et la
    crypto des debuts n'a plus grand-chose a voir avec celle
    d'aujourd'hui. Une mesure sur huit ans peut donc etre portee
    entierement par les deux premieres.

    Ce decoupage n'est PAS un test de significativite supplementaire :
    c'est un diagnostic de STABILITE. On regarde si l'effet est present
    partout ou concentre dans une epoque revolue.
    """
    n = len(candles)
    taille = n // periods
    out: List[Dict[str, object]] = []
    for k in range(periods):
        debut = k * taille
        fin = n if k == periods - 1 else (k + 1) * taille
        tranche = candles[debut:fin]
        if len(tranche) < horizon * 5:
            continue
        rapport = analyse_signal(
            cfg, tranche, type(signal)(), [horizon],
            permutations=permutations, seed=seed + k, extras=extras,
        )
        res = rapport["results"]
        out.append(
            {
                "periode": k + 1,
                "debut": _iso(tranche[0].ts)[:10],
                "fin": _iso(tranche[-1].ts)[:10],
                "n": rapport["total_signals"],
                "resultat": res[0] if res else None,
            }
        )
    return out


def format_periods(
    rows: List[Dict[str, object]],
    horizon: int,
    label: str,
    cout_aller_retour: float = 0.0,
) -> str:
    """Tableau de stabilite, avec la seule comparaison qui decide vraiment :
    l'avantage depasse-t-il ce qu'il en coute de le capturer ?

    Un avantage statistiquement reel mais inferieur aux frais n'est pas un
    avantage : c'est une facon elegante de payer son courtier. C'est la
    raison principale pour laquelle les strategies de suivi de tendance
    grand public ne rapportent rien sur les marches liquides.
    """
    lines = [
        f"Stabilite de « {label} » dans le temps, horizon {horizon} bougies",
        "",
        f"{'periode':>8} {'debut':>12} {'fin':>12} {'n':>6} {'ecart':>10} "
        f"{'p':>8} {'vs frais':>10}",
        "-" * 74,
    ]
    positifs = 0
    rentables = 0
    mesures = 0
    for r in rows:
        res = r["resultat"]
        if res is None:
            continue
        mesures += 1
        if res.edge > 0:
            positifs += 1
        net = res.edge - cout_aller_retour
        if net > 0:
            rentables += 1
        marque = f"{net:+.3%}" if cout_aller_retour else "-"
        lines.append(
            f"{r['periode']:>8} {r['debut']:>12} {r['fin']:>12} "
            f"{r['n']:>6} {res.edge:>+10.3%} {res.p_value:>8.4f} {marque:>10}"
        )

    lines.append("")
    if mesures == 0:
        lines.append("Pas assez de donnees par tranche.")
        return "\n".join(lines)

    derniere = rows[-1]["resultat"]
    if cout_aller_retour:
        lines.append(
            f"Cout d'un aller-retour : {cout_aller_retour:.3%} "
            "(frais des deux cotes + slippage). La colonne « vs frais » est "
            "ce qu'il reste une fois ce cout paye."
        )
        lines.append("")

    # --- forme de l'effet dans le temps -------------------------------
    if derniere is not None and derniere.edge <= 0:
        lines += [
            "FORME : l'ecart est nul ou negatif sur la tranche la plus RECENTE.",
            "Signature d'un avantage qui a existe puis s'est eteint avec la montee",
            "en efficience du marche. Un tel signal produit un backtest magnifique",
            "sur l'ensemble de la periode et perd de l'argent des demain.",
        ]
    elif positifs == mesures:
        lines.append("FORME : ecart positif sur toutes les tranches, la plus recente comprise.")
    else:
        lines.append(
            f"FORME : ecart positif sur {positifs} tranche(s) sur {mesures} — effet instable."
        )

    # --- le test qui tranche vraiment ---------------------------------
    if cout_aller_retour:
        lines.append("")
        if rentables == 0:
            lines += [
                "VERDICT ECONOMIQUE : AUCUNE tranche ne laisse de quoi payer les frais.",
                "",
                "Meme si l'avantage est statistiquement reel, il est plus petit que ce",
                "qu'il en coute de le capturer. Ce n'est pas exploitable, et aucun",
                "reglage de la strategie n'y changera quoi que ce soit : c'est une",
                "question d'arithmetique, pas de parametrage.",
                "",
                "Deux choses seulement pourraient changer ce constat : des frais",
                "nettement plus bas, ou un avantage nettement plus gros.",
            ]
        elif derniere is not None and derniere.edge - cout_aller_retour <= 0:
            lines += [
                f"VERDICT ECONOMIQUE : {rentables} tranche(s) couvrent les frais, mais PAS",
                "la plus recente. L'avantage exploitable appartient au passe.",
            ]
        else:
            lines += [
                f"VERDICT ECONOMIQUE : {rentables} tranche(s) sur {mesures} couvrent les frais,",
                "dont la plus recente. C'est le seul profil qui justifie de continuer —",
                "prochaine etape : le bloc de validation, une fois, sur ce seul candidat.",
            ]
    return "\n".join(lines)


def format_comparison(data: Dict[str, object]) -> str:
    """Tableau recapitulatif de tous les candidats, avec seuil corrige."""
    rapports = data["rapports"]  # type: ignore[assignment]
    seuil = data["seuil_corrige"]  # type: ignore[assignment]
    n_tests = data["n_tests"]

    lines = [
        f"{n_tests} tests effectues ({len(rapports)} signaux x horizons).",
        f"Seuil corrige de Bonferroni : p < {seuil:.4f} (et non 0.05).",
        "",
        "Sans cette correction, tester autant de combinaisons garantit de",
        "trouver du 'significatif' meme dans du bruit pur. Le seuil brut de",
        "0.05 n'a de sens que pour UN test decide a l'avance.",
        "",
        f"{'signal':>14} {'n':>5} {'meilleur horizon':>18} {'ecart':>10} {'p':>8}   verdict",
        "-" * 78,
    ]

    gagnants = []
    for rap in rapports:
        spec = rap["signal"]
        res = rap["results"]
        if not res:
            lines.append(f"{spec.key:>14} {rap['total_signals']:>5}  pas assez de signaux")
            continue
        best = min(res, key=lambda r: r.p_value)
        passe = best.p_value < seuil
        if passe:
            gagnants.append((spec, best))
        verdict = "RETENU" if passe else ("brut<0.05" if best.p_value < 0.05 else "rien")
        lines.append(
            f"{spec.key:>14} {rap['total_signals']:>5} {str(best.horizon) + 'h':>18} "
            f"{best.edge:>+10.3%} {best.p_value:>8.4f}   {verdict}"
        )

    lines.append("")
    if gagnants:
        for spec, best in gagnants:
            lines += [
                f"RETENU : {spec.label} ({spec.key}), horizon {best.horizon}h, "
                f"ecart {best.edge:+.3%}, p={best.p_value:.4f}",
                f"  hypothese : {spec.hypothese}",
            ]
        lines += [
            "",
            "Prochaine etape obligatoire : verifier ce signal sur le bloc de",
            "VALIDATION avant d'y croire. Un candidat qui survit a la correction",
            "sur train et s'effondre sur validation etait du bruit.",
        ]
    else:
        bruts = [
            r for rap in rapports for r in rap["results"]
            if r.p_value < 0.05
        ]
        lines.append("VERDICT : aucun candidat ne passe le seuil corrige.")
        if bruts:
            lines += [
                "",
                f"{len(bruts)} resultat(s) passent le seuil BRUT de 0.05, pour environ",
                f"{n_tests * 0.05:.1f} attendu(s) par pur hasard sur {n_tests} tests. Les horizons",
                "se chevauchant, ces tests ne sont pas independants : ce decompte ne",
                "prouve rien, ni dans un sens ni dans l'autre. Ne construis rien dessus.",
            ]
        lines += [
            "",
            "Ce n'est pas un echec de ta part : les marches liquides sont",
            "difficiles, et ne rien trouver avec des signaux classiques est le",
            "resultat le plus frequent. Les pistes serieuses restantes sont de",
            "changer de terrain (autre paire, autre unite de temps) plutot que",
            "de continuer a tester des variantes sur celui-ci.",
        ]
    return "\n".join(lines)
