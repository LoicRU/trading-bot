"""Adaptateur synthetique : marche fabrique, deterministe, hors-ligne.

A quoi ca sert :
  1. Tester toute la chaine sans connexion reseau.
  2. Verifier le comportement du bot sur des regimes de marche connus
     d'avance (hausse, baisse, marche plat), ce qu'un historique reel
     ne permet pas de commander.

ATTENTION : ces prix sont INVENTES. Un bon resultat ici ne prouve
strictement rien sur le marche reel. C'est un banc de test technique,
pas une validation de strategie.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional

from ..models import Candle
from .base import MarketAdapter

# Regimes traverses en boucle : (nom, derive par barre, volatilite par barre)
REGIMES = [
    ("hausse", 0.0012, 0.012),
    ("plat", 0.0000, 0.004),
    ("baisse", -0.0010, 0.016),
    ("plat", 0.0001, 0.006),
    ("hausse forte", 0.0022, 0.020),
    ("krach", -0.0035, 0.035),
]


class SyntheticAdapter(MarketAdapter):
    name = "synthetic"

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        bars: int = 8000,
        seed: int = 42,
        start_price: float = 30000.0,
        regime_length: int = 400,
    ) -> None:
        super().__init__(symbol, timeframe)
        self.bars = bars
        self.seed = seed
        self.start_price = start_price
        self.regime_length = regime_length
        self._cache: Optional[List[Candle]] = None

    def _generate(self) -> List[Candle]:
        if self._cache is not None:
            return self._cache

        rng = random.Random(self.seed)
        candles: List[Candle] = []
        price = self.start_price
        # On termine sur la derniere bougie close avant maintenant.
        end_ts = (int(__import__("time").time() * 1000) // self.interval_ms) * self.interval_ms
        start_ts = end_ts - self.bars * self.interval_ms

        for i in range(self.bars):
            _, drift, vol = REGIMES[(i // self.regime_length) % len(REGIMES)]
            shock = rng.gauss(0.0, 1.0)
            ret = drift + vol * shock
            open_ = price
            close = max(open_ * math.exp(ret), 1e-6)
            # Meches proportionnelles a la volatilite du regime
            wick = abs(rng.gauss(0.0, vol * 0.6))
            high = max(open_, close) * (1.0 + wick)
            low = min(open_, close) * (1.0 - abs(rng.gauss(0.0, vol * 0.6)))
            low = max(low, 1e-7)
            candles.append(
                Candle(
                    ts=start_ts + i * self.interval_ms,
                    open=round(open_, 2),
                    high=round(high, 2),
                    low=round(low, 2),
                    close=round(close, 2),
                    volume=round(abs(rng.gauss(100, 30)), 4),
                )
            )
            price = close

        self._cache = candles
        return candles

    def get_candles(self, start_ms: Optional[int] = None, end_ms: Optional[int] = None) -> List[Candle]:
        candles = self._generate()
        lo = start_ms if start_ms is not None else -1
        hi = end_ms if end_ms is not None else 1 << 62
        return [c for c in candles if lo <= c.ts <= hi]

    def get_price(self) -> float:
        return self._generate()[-1].close
