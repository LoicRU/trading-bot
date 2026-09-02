"""Univers de paires pour les tests transversaux.

Pourquoi plusieurs paires
--------------------------
En journalier, une seule paire donne une centaine de signaux en huit ans.
Aucun test ne peut conclure avec si peu. Mais l'hypothese testee — le
momentum apres cassure — n'a rien de propre au Bitcoin : c'est un
phenomene documente sur toutes les classes d'actifs. La mesurer sur
quarante paires simultanement multiplie les observations par quarante
sans changer l'hypothese d'un iota.

BIAIS DU SURVIVANT : L'AVERTISSEMENT LE PLUS IMPORTANT DE CE FICHIER
---------------------------------------------------------------------
Cette liste est composee de paires qui existent ENCORE aujourd'hui. Les
projets morts, delistes, effondres a zero en ont disparu — et ils etaient
nombreux en crypto. Or ce sont precisement ceux dont les cassures a la
hausse ont le plus mal fini.

Consequence : tout resultat obtenu ici est OPTIMISTE, et personne ne sait
exactement de combien. On ne peut pas corriger ce biais sans la liste
historique complete des paires delistees, que Binance ne publie pas
commodement. Il faut donc le garder en tete a chaque lecture : un
avantage mesure ici qui serait juste au-dessus des frais est
probablement, en realite, en dessous.

La liste est FIGEE. Elle a ete etablie avant de voir le moindre resultat,
et ne doit pas etre modifiee ensuite : retirer les paires qui deplaisent
apres coup fabriquerait exactement le resultat qu'on cherche a eviter.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .models import Candle

# 45 paires USDT parmi les plus anciennes et les plus liquides de Binance.
# Celles qui n'ont pas assez d'historique seront simplement ecartees, et
# le rapport dira lesquelles.
UNIVERS: List[str] = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT",
    "SOLUSDT", "DOGEUSDT", "DOTUSDT", "LTCUSDT", "LINKUSDT",
    "TRXUSDT", "AVAXUSDT", "ATOMUSDT", "ETCUSDT", "XLMUSDT",
    "BCHUSDT", "NEARUSDT", "FILUSDT", "ALGOUSDT", "VETUSDT",
    "ICPUSDT", "HBARUSDT", "EOSUSDT", "AAVEUSDT", "XTZUSDT",
    "SANDUSDT", "MANAUSDT", "THETAUSDT", "AXSUSDT", "CHZUSDT",
    "ZECUSDT", "IOTAUSDT", "NEOUSDT", "DASHUSDT", "COMPUSDT",
    "SNXUSDT", "YFIUSDT", "SUSHIUSDT", "CRVUSDT", "ENJUSDT",
    "ZILUSDT", "QTUMUSDT", "KSMUSDT", "RUNEUSDT", "ONEUSDT",
]


def align_on_common_grid(
    series: Dict[str, List[Candle]], min_bars: int = 400
) -> Tuple[List[int], Dict[str, List[Candle]]]:
    """Ramene toutes les paires sur une grille de dates commune.

    Indispensable pour le test de permutation : pour respecter la
    correlation entre paires — en crypto, tout monte et tout descend
    ensemble — il faut pouvoir decaler simultanement toutes les series
    du meme nombre de jours. Ca exige un index de dates partage.

    Les paires trop courtes sont ecartees plutot que completees : inventer
    des bougies manquantes fabriquerait des signaux qui n'ont jamais existe.
    """
    utilisables = {s: c for s, c in series.items() if len(c) >= min_bars}
    if not utilisables:
        return [], {}

    # Grille = dates presentes dans la paire de reference la plus longue
    reference = max(utilisables.values(), key=len)
    grille = [c.ts for c in reference]
    index = {ts: k for k, ts in enumerate(grille)}

    alignees: Dict[str, List[Candle]] = {}
    for sym, candles in utilisables.items():
        rangee: List[Candle] = [None] * len(grille)  # type: ignore[list-item]
        for c in candles:
            k = index.get(c.ts)
            if k is not None:
                rangee[k] = c
        # On ne garde que les paires reellement presentes sur la grille
        presentes = sum(1 for c in rangee if c is not None)
        if presentes >= min_bars:
            alignees[sym] = rangee
    return grille, alignees


def couverture(alignees: Dict[str, List[Candle]]) -> Dict[str, int]:
    return {s: sum(1 for c in serie if c is not None) for s, serie in alignees.items()}
