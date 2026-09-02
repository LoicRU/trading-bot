"""Interface generique d'acces au marche.

Tout le reste du bot ne connait QUE cette interface. Il ignore
completement s'il parle a Binance, a un fichier CSV ou a un generateur
de donnees. C'est ce qui permettra plus tard de brancher OANDA (forex)
ou Alpaca (actions) en ecrivant un seul fichier, sans toucher au moteur.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..models import Candle

# Duree d'une bougie en millisecondes, par libelle d'unite de temps.
TIMEFRAME_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


class MarketDataError(RuntimeError):
    """Erreur de recuperation des donnees de marche."""


class MarketAdapter(ABC):
    """Contrat minimal qu'un adaptateur doit remplir."""

    name: str = "base"

    def __init__(self, symbol: str, timeframe: str) -> None:
        if timeframe not in TIMEFRAME_MS:
            raise MarketDataError(
                f"Unite de temps non supportee : {timeframe}. "
                f"Choix possibles : {', '.join(TIMEFRAME_MS)}"
            )
        self.symbol = symbol
        self.timeframe = timeframe
        self.interval_ms = TIMEFRAME_MS[timeframe]

    @abstractmethod
    def get_candles(self, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[Candle]:
        """Renvoie les bougies closes, triees par temps croissant.

        La derniere bougie renvoyee doit toujours etre CLOSE. Une bougie
        en cours de formation ne doit jamais sortir d'un adaptateur :
        elle changerait de valeur et fausserait toute la chaine.
        """

    @abstractmethod
    def get_price(self) -> float:
        """Dernier prix connu (utilise seulement pour l'affichage)."""

    def describe(self) -> str:
        return f"{self.name}:{self.symbol}@{self.timeframe}"


def read_cache(path) -> List[Candle]:
    """Relit un cache de bougies au format CSV. Cache absent = liste vide."""
    import csv
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    out: List[Candle] = []
    with p.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out.append(
                Candle(
                    ts=int(row["ts"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return out


def write_cache(path, candles: List[Candle]) -> None:
    """Ecriture atomique : un plantage en cours d'ecriture ne laisse jamais
    un cache tronque, qui produirait des backtests faux et silencieux."""
    import csv
    from pathlib import Path

    p = Path(path)
    tmp = p.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ts", "open", "high", "low", "close", "volume"])
        for c in candles:
            writer.writerow([c.ts, c.open, c.high, c.low, c.close, c.volume])
    tmp.replace(p)


def merge_candles(cached: List[Candle], fresh: List[Candle]) -> List[Candle]:
    merged = {c.ts: c for c in cached}
    merged.update({c.ts: c for c in fresh})
    return sanity_check([merged[k] for k in sorted(merged)])


def sanity_check(candles: List[Candle]) -> List[Candle]:
    """Filtre defensif : ecarte les bougies aberrantes ou hors ordre."""
    clean: List[Candle] = []
    for c in candles:
        if c.high < c.low:
            continue
        if min(c.open, c.high, c.low, c.close) <= 0:
            continue
        if not (c.low <= c.open <= c.high and c.low <= c.close <= c.high):
            continue
        if clean and c.ts <= clean[-1].ts:
            continue
        clean.append(c)
    return clean
