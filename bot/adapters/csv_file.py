"""Adaptateur CSV : rejoue un historique telecharge a la main.

Utile quand l'API de l'exchange est inaccessible (pare-feu, restriction
geographique) ou pour travailler sur des donnees d'une autre source.

Format attendu, avec en-tete : ts,open,high,low,close,volume
ts en millisecondes UTC, ou en secondes (detecte automatiquement),
ou en date ISO (2024-01-31T14:00:00Z).
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from ..models import Candle
from .base import MarketAdapter, MarketDataError, sanity_check


def _parse_ts(raw: str) -> int:
    raw = raw.strip()
    try:
        value = float(raw)
        # < 10^11 => secondes ; sinon millisecondes
        return int(value * 1000) if value < 1e11 else int(value)
    except ValueError:
        pass
    text = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise MarketDataError(f"Horodatage illisible dans le CSV : {raw!r}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


class CsvAdapter(MarketAdapter):
    name = "csv"

    def __init__(self, symbol: str, timeframe: str, csv_path: str) -> None:
        super().__init__(symbol, timeframe)
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise MarketDataError(f"Fichier CSV introuvable : {self.csv_path}")
        self._cache: Optional[List[Candle]] = None

    def _load(self) -> List[Candle]:
        if self._cache is not None:
            return self._cache
        out: List[Candle] = []
        with self.csv_path.open(newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if not reader.fieldnames:
                raise MarketDataError("CSV vide ou sans en-tete.")
            cols = {c.lower().strip(): c for c in reader.fieldnames}
            required = ["ts", "open", "high", "low", "close"]
            missing = [c for c in required if c not in cols]
            if missing:
                raise MarketDataError(
                    f"Colonnes manquantes dans {self.csv_path.name} : {missing}. "
                    "Attendu : ts,open,high,low,close,volume"
                )
            for row in reader:
                out.append(
                    Candle(
                        ts=_parse_ts(row[cols["ts"]]),
                        open=float(row[cols["open"]]),
                        high=float(row[cols["high"]]),
                        low=float(row[cols["low"]]),
                        close=float(row[cols["close"]]),
                        volume=float(row[cols["volume"]]) if "volume" in cols and row[cols["volume"]] else 0.0,
                    )
                )
        out.sort(key=lambda c: c.ts)
        self._cache = sanity_check(out)
        return self._cache

    def get_candles(self, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[Candle]:
        candles = self._load()
        lo = start_ms if start_ms is not None else -1
        hi = end_ms if end_ms is not None else 1 << 62
        return [c for c in candles if lo <= c.ts <= hi]

    def get_price(self) -> float:
        candles = self._load()
        if not candles:
            raise MarketDataError("Aucune bougie dans le CSV.")
        return candles[-1].close
