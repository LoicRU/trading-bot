"""Adaptateur Binance : donnees de marche publiques.

Aucune cle API, aucun compte, aucune authentification : l'endpoint
/api/v3/klines est public. Ce module est en LECTURE SEULE, il n'a
aucun moyen technique de passer un ordre.

Les donnees telechargees sont mises en cache dans data/ au format CSV
pour ne pas re-telecharger deux ans d'historique a chaque execution.
"""

from __future__ import annotations

import csv
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

from ..models import Candle
from .base import MarketAdapter, MarketDataError, sanity_check

BASE_URL = "https://api.binance.com"
MAX_LIMIT = 1000  # plafond impose par Binance sur /klines


class BinanceAdapter(MarketAdapter):
    name = "binance"

    def __init__(self, symbol: str, timeframe: str, cache_dir: str = "data") -> None:
        super().__init__(symbol, timeframe)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------
    @property
    def cache_file(self) -> Path:
        return self.cache_dir / f"{self.symbol}_{self.timeframe}.csv"

    def _read_cache(self) -> List[Candle]:
        if not self.cache_file.exists():
            return []
        out: List[Candle] = []
        with self.cache_file.open(newline="", encoding="utf-8") as fh:
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

    def _write_cache(self, candles: List[Candle]) -> None:
        tmp = self.cache_file.with_suffix(".csv.tmp")
        with tmp.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["ts", "open", "high", "low", "close", "volume"])
            for c in candles:
                writer.writerow([c.ts, c.open, c.high, c.low, c.close, c.volume])
        tmp.replace(self.cache_file)  # ecriture atomique : pas de cache a moitie ecrit

    # ------------------------------------------------------------------
    # Reseau
    # ------------------------------------------------------------------
    def _request(self, path: str, params: dict) -> list:
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{BASE_URL}{path}?{query}"
        req = urllib.request.Request(url, headers={"User-Agent": "paper-trading-bot/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise MarketDataError(
                    "Binance : trop de requetes (429). Attends une minute avant de relancer."
                ) from exc
            if exc.code == 451:
                raise MarketDataError(
                    "Binance : acces refuse depuis ce pays (451). Utilise l'adaptateur "
                    "'csv' avec un historique telecharge, ou 'synthetic' pour tester."
                ) from exc
            raise MarketDataError(f"Binance a repondu HTTP {exc.code} : {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise MarketDataError(
                f"Impossible de joindre Binance ({exc.reason}). Verifie ta connexion, "
                "ou bascule sur l'adaptateur 'synthetic' / 'csv' dans config.json."
            ) from exc

    def _fetch_klines(self, start_ms: int, end_ms: int) -> List[Candle]:
        """Telecharge par pages de 1000 bougies, en avancant dans le temps."""
        out: List[Candle] = []
        cursor = start_ms
        while cursor < end_ms:
            rows = self._request(
                "/api/v3/klines",
                {
                    "symbol": self.symbol,
                    "interval": self.timeframe,
                    "startTime": cursor,
                    "endTime": end_ms,
                    "limit": MAX_LIMIT,
                },
            )
            if not rows:
                break
            for r in rows:
                out.append(
                    Candle(
                        ts=int(r[0]),
                        open=float(r[1]),
                        high=float(r[2]),
                        low=float(r[3]),
                        close=float(r[4]),
                        volume=float(r[5]),
                    )
                )
            last_ts = int(rows[-1][0])
            if last_ts + self.interval_ms <= cursor:
                break  # securite anti-boucle infinie
            cursor = last_ts + self.interval_ms
            if len(rows) < MAX_LIMIT:
                break
            time.sleep(0.25)  # on reste poli avec l'API publique
        return out

    # ------------------------------------------------------------------
    # Interface publique
    # ------------------------------------------------------------------
    def get_candles(self, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[Candle]:
        now_ms = int(time.time() * 1000)
        # On coupe la bougie en cours : elle n'est pas encore close.
        last_closed = (now_ms // self.interval_ms) * self.interval_ms - self.interval_ms
        end_ms = min(end_ms or last_closed, last_closed)
        start_ms = start_ms or (end_ms - 730 * 86_400_000)

        cached = self._read_cache()
        if cached and cached[-1].ts >= end_ms and cached[0].ts <= start_ms:
            return sanity_check([c for c in cached if start_ms <= c.ts <= end_ms])

        fetch_from = start_ms
        if cached and cached[0].ts <= start_ms <= cached[-1].ts:
            fetch_from = cached[-1].ts + self.interval_ms  # on complete le cache

        fresh = self._fetch_klines(fetch_from, end_ms) if fetch_from <= end_ms else []

        merged = {c.ts: c for c in cached}
        merged.update({c.ts: c for c in fresh})
        allc = sanity_check([merged[k] for k in sorted(merged)])
        if allc:
            self._write_cache(allc)
        return [c for c in allc if start_ms <= c.ts <= end_ms]

    def get_price(self) -> float:
        data = self._request("/api/v3/ticker/price", {"symbol": self.symbol})
        return float(data["price"])  # type: ignore[index]
