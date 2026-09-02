"""Adaptateur Coinbase Exchange : donnees de marche publiques.

Pourquoi cet adaptateur existe alors qu'il y a deja Binance
------------------------------------------------------------
Binance refuse les requetes venant d'adresses IP americaines (erreur
HTTP 451). Or les runners GitHub Actions sont heberges aux Etats-Unis.
Autrement dit : l'adaptateur Binance marche tres bien sur ta machine et
echouera sur GitHub Actions.

Coinbase, entreprise americaine, n'a pas cette restriction. C'est donc
l'adaptateur a utiliser pour l'execution automatisee.

Comme Binance : aucune cle, aucun compte, lecture seule.

Attention aux symboles : ici c'est "BTC-USD" et non "BTCUSDT".
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

from ..models import Candle
from .base import (
    MarketAdapter,
    MarketDataError,
    merge_candles,
    read_cache,
    sanity_check,
    write_cache,
)

BASE_URL = "https://api.exchange.coinbase.com"
MAX_CANDLES = 300  # plafond impose par Coinbase sur /candles

# Coinbase n'accepte que ces granularites, en secondes.
GRANULARITY = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
    "6h": 21600,
    "1d": 86400,
}


class CoinbaseAdapter(MarketAdapter):
    name = "coinbase"

    def __init__(self, symbol: str, timeframe: str, cache_dir: str = "data") -> None:
        super().__init__(symbol, timeframe)
        if timeframe not in GRANULARITY:
            raise MarketDataError(
                f"Coinbase n'accepte que ces unites de temps : {', '.join(GRANULARITY)}. "
                f"Recu : {timeframe}"
            )
        self.granularity = GRANULARITY[timeframe]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def cache_file(self) -> Path:
        return self.cache_dir / f"coinbase_{self.symbol}_{self.timeframe}.csv"

    # ------------------------------------------------------------------
    def _request(self, path: str, params: Optional[dict] = None):
        url = f"{BASE_URL}{path}"
        if params:
            url += "?" + "&".join(f"{k}={v}" for k, v in params.items())
        req = urllib.request.Request(url, headers={"User-Agent": "paper-trading-bot/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                raise MarketDataError(
                    "Coinbase : trop de requetes (429). Relance dans une minute."
                ) from exc
            if exc.code == 404:
                raise MarketDataError(
                    f"Coinbase ne connait pas la paire '{self.symbol}'. "
                    "Le format attendu est 'BTC-USD', pas 'BTCUSDT'."
                ) from exc
            raise MarketDataError(f"Coinbase a repondu HTTP {exc.code} : {exc.reason}") from exc
        except urllib.error.URLError as exc:
            raise MarketDataError(
                f"Impossible de joindre Coinbase ({exc.reason})."
            ) from exc

    def _fetch(self, start_ms: int, end_ms: int) -> List[Candle]:
        """Coinbase plafonne a 300 bougies par requete et renvoie du plus
        recent au plus ancien. On avance donc par fenetres successives."""
        out: List[Candle] = []
        window = MAX_CANDLES * self.granularity * 1000
        cursor = start_ms
        while cursor < end_ms:
            stop = min(cursor + window, end_ms)
            rows = self._request(
                f"/products/{self.symbol}/candles",
                {
                    "granularity": self.granularity,
                    "start": _iso(cursor),
                    "end": _iso(stop),
                },
            )
            if isinstance(rows, dict):  # Coinbase renvoie {"message": "..."} en cas d'erreur
                raise MarketDataError(f"Coinbase : {rows.get('message', rows)}")
            for r in rows:
                # format : [temps_en_secondes, plus_bas, plus_haut, ouverture, cloture, volume]
                out.append(
                    Candle(
                        ts=int(r[0]) * 1000,
                        open=float(r[3]),
                        high=float(r[2]),
                        low=float(r[1]),
                        close=float(r[4]),
                        volume=float(r[5]),
                    )
                )
            cursor = stop
            time.sleep(0.15)  # l'API publique tolere ~10 requetes/seconde
        out.sort(key=lambda c: c.ts)
        return out

    # ------------------------------------------------------------------
    def get_candles(self, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[Candle]:
        now_ms = int(time.time() * 1000)
        last_closed = (now_ms // self.interval_ms) * self.interval_ms - self.interval_ms
        end_ms = min(end_ms or last_closed, last_closed)
        start_ms = start_ms or (end_ms - 730 * 86_400_000)

        cached = read_cache(self.cache_file)
        if cached and cached[-1].ts >= end_ms and cached[0].ts <= start_ms:
            return sanity_check([c for c in cached if start_ms <= c.ts <= end_ms])

        fetch_from = start_ms
        if cached and cached[0].ts <= start_ms <= cached[-1].ts:
            fetch_from = cached[-1].ts + self.interval_ms

        fresh = self._fetch(fetch_from, end_ms) if fetch_from <= end_ms else []
        allc = merge_candles(cached, fresh)
        if allc:
            write_cache(self.cache_file, allc)
        return [c for c in allc if start_ms <= c.ts <= end_ms]

    def get_price(self) -> float:
        data = self._request(f"/products/{self.symbol}/ticker")
        return float(data["price"])  # type: ignore[index]


def _iso(ms: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")
