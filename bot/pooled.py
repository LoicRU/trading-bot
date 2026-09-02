"""Test d'avantage transversal : un signal, plusieurs dizaines de paires.

Le probleme resolu
------------------
En journalier, une paire donne ~80 signaux en huit ans. Aucune conclusion
possible. Quarante paires en donnent des milliers, sans changer
l'hypothese : le momentum n'est pas suppose etre une propriete du
Bitcoin, mais du comportement des marches.

Le piege statistique, et comment il est traite
-----------------------------------------------
On ne peut PAS mettre bout a bout les observations de 40 paires crypto en
les traitant comme independantes. Elles ne le sont pas : quand le marche
monte, tout monte ensemble. Empiler des observations correlees en faisant
comme si elles ne l'etaient pas gonfle artificiellement la significativite
— c'est la faute qui produit des p-values spectaculaires et fausses.

La parade employee ici est le DECALAGE CIRCULAIRE COMMUN : a chaque
permutation, on decale les dates de tous les signaux du MEME nombre de
jours, pour toutes les paires a la fois. Ainsi :

  - la structure de correlation entre paires est preservee intacte,
  - le regroupement temporel des signaux aussi (les cassures arrivent en
    grappes, pas uniformement),
  - seul le lien entre le signal et ce qui le suit est detruit.

C'est ce lien, et lui seul, qu'on veut tester.

Ce qu'il faut regarder en priorite
-----------------------------------
Pas la p-value : la PART DE PAIRES ou l'ecart est positif. Un vrai
phenomene transversal se retrouve sur la majorite des paires. Un artefact
se concentre sur trois ou quatre qui portent tout le resultat.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import Config
from .models import Candle


@dataclass
class PooledResult:
    horizon: int
    n_signals: int
    n_paires: int
    signal_mean: float
    baseline_mean: float
    edge: float
    p_value: float
    paires_positives: int
    paires_mesurees: int
    par_paire: Dict[str, Tuple[int, float]]  # symbole -> (n signaux, ecart)

    @property
    def part_positive(self) -> float:
        return self.paires_positives / self.paires_mesurees if self.paires_mesurees else 0.0


def _forward(serie: Sequence[Optional[Candle]], i: int, h: int) -> Optional[float]:
    """Rendement entre l'ouverture de la bougie suivante et la cloture h plus loin."""
    if i + 1 >= len(serie) or i + h >= len(serie):
        return None
    entree, sortie = serie[i + 1], serie[i + h]
    if entree is None or sortie is None or entree.open <= 0:
        return None
    return (sortie.close - entree.open) / entree.open


def analyse_pooled(
    cfg: Config,
    grille: Sequence[int],
    alignees: Dict[str, List[Optional[Candle]]],
    signal_class,
    horizon: int,
    permutations: int = 2000,
    seed: int = 4242,
) -> PooledResult:
    taille = len(grille)

    # --- signaux et rendements, paire par paire ----------------------
    signaux: List[Tuple[str, int]] = []
    rendements: Dict[str, List[Optional[float]]] = {}
    par_paire: Dict[str, Tuple[int, float]] = {}

    for sym, serie in alignees.items():
        compactes = [c for c in serie if c is not None]
        if len(compactes) < horizon * 5:
            continue

        sig = signal_class()
        sig.prepare(compactes, cfg.strategy)
        # Correspondance entre l'indice compacte et l'indice sur la grille
        positions = [k for k, c in enumerate(serie) if c is not None]
        declenchements = {
            positions[j] for j in range(sig.warmup, len(compactes) - 1) if sig.fires(j)
        }

        rendements[sym] = [_forward(serie, i, horizon) for i in range(taille)]

        locaux = [(sym, i) for i in sorted(declenchements) if rendements[sym][i] is not None]
        signaux.extend(locaux)

        if locaux:
            vals = [rendements[sym][i] for _, i in locaux]
            eligibles = [r for r in rendements[sym] if r is not None]
            base = sum(eligibles) / len(eligibles) if eligibles else 0.0
            par_paire[sym] = (len(vals), sum(vals) / len(vals) - base)

    if not signaux:
        raise ValueError("Aucun signal sur l'univers : rien a mesurer.")

    def moyenne(paires: Sequence[Tuple[str, int]]) -> float:
        vals = [
            rendements[s][i] for s, i in paires
            if 0 <= i < taille and rendements[s][i] is not None
        ]
        return sum(vals) / len(vals) if vals else 0.0

    observe = moyenne(signaux)
    tous = [r for serie in rendements.values() for r in serie if r is not None]
    baseline = sum(tous) / len(tous) if tous else 0.0

    # --- permutation par decalage circulaire commun -------------------
    rng = random.Random(seed)
    mieux = 0
    for _ in range(permutations):
        lag = rng.randrange(horizon + 1, taille - horizon - 1)
        decales = [(s, (i + lag) % taille) for s, i in signaux]
        if moyenne(decales) >= observe:
            mieux += 1
    p = (mieux + 1) / (permutations + 1)

    positives = sum(1 for _, (_, e) in par_paire.items() if e > 0)

    return PooledResult(
        horizon=horizon,
        n_signals=len(signaux),
        n_paires=len(rendements),
        signal_mean=observe,
        baseline_mean=baseline,
        edge=observe - baseline,
        p_value=p,
        paires_positives=positives,
        paires_mesurees=len(par_paire),
        par_paire=par_paire,
    )


def format_pooled(res: PooledResult, label: str, cout: float, ecartees: Sequence[str]) -> str:
    net = res.edge - cout
    lines = [
        f"Test transversal — « {label} », horizon {res.horizon} bougies",
        "",
        f"  paires retenues        : {res.paires_mesurees}"
        + (f"  ({len(ecartees)} ecartees faute d'historique)" if ecartees else ""),
        f"  signaux au total       : {res.n_signals}",
        f"  rendement apres signal : {res.signal_mean:+.3%}",
        f"  rendement au hasard    : {res.baseline_mean:+.3%}",
        f"  ecart                  : {res.edge:+.3%}",
        f"  cout aller-retour      : {cout:.3%}",
        f"  NET                    : {net:+.3%}",
        f"  p (decalage circulaire): {res.p_value:.4f}",
        "",
        f"  paires a ecart positif : {res.paires_positives}/{res.paires_mesurees} "
        f"({res.part_positive:.0%})",
        "",
    ]

    # Les paires qui portent le resultat
    tri = sorted(res.par_paire.items(), key=lambda kv: kv[1][1], reverse=True)
    lines.append("  meilleures / pires paires :")
    for sym, (n, e) in tri[:3]:
        lines.append(f"    {sym:>12} {n:>4} signaux {e:>+9.3%}")
    lines.append("    ...")
    for sym, (n, e) in tri[-3:]:
        lines.append(f"    {sym:>12} {n:>4} signaux {e:>+9.3%}")
    lines.append("")

    # --- lecture ------------------------------------------------------
    if res.part_positive < 0.6:
        lines += [
            "VERDICT : l'ecart n'est positif que sur une minorite de paires.",
            "Un vrai phenomene transversal se retrouverait sur la plupart d'entre",
            "elles. Ici, quelques paires portent le resultat : c'est la signature",
            "d'un artefact, pas d'un comportement de marche.",
        ]
    elif net <= 0:
        lines += [
            "VERDICT : l'ecart ne couvre pas les frais.",
            "Meme largement partage entre les paires, il n'est pas exploitable.",
        ]
    elif res.p_value >= 0.05:
        lines += [
            "VERDICT : marge economique presente, mais l'ecart ne se distingue pas",
            "du hasard une fois la correlation entre paires prise en compte.",
            "C'est exactement ce que le decalage circulaire sert a detecter : sans",
            "lui, ce meme resultat aurait paru tres significatif.",
        ]
    else:
        lines += [
            f"VERDICT : ecart positif sur {res.part_positive:.0%} des paires, superieur",
            f"aux frais de {net:+.3%}, et p={res.p_value:.4f} en tenant compte de la",
            "correlation entre paires.",
            "",
            "C'est le premier profil de toute cette recherche qui justifie d'aller",
            "plus loin. Deux reserves a garder en tete avant de s'emballer :",
            "  - le biais du survivant gonfle ce chiffre d'un montant inconnu",
            "    (les paires mortes ne sont pas dans l'univers),",
            "  - rien n'a encore ete verifie hors echantillon.",
        ]
    return "\n".join(lines)
