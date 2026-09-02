"""Strategie de decision : croisement d'EMA filtre par la volatilite.

Volontairement simple et lisible. L'objectif de la V1 n'est pas d'etre
rentable, c'est d'avoir une infrastructure honnete dans laquelle on
pourra ensuite tester des strategies serieuses.

Point important : la strategie renvoie TOUJOURS une decision motivee,
y compris quand elle choisit de ne pas trader. L'abstention est un choix
enregistre et evalue, pas un silence.

Garantie anti-triche : decide(i, ...) ne lit jamais un indice > i.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .config import StrategyConfig
from .indicators import atr as atr_series
from .indicators import crossed_down, crossed_up, ema
from .models import Action, Candle, Decision


class EmaAtrStrategy:
    name = "ema_cross_atr_filter"

    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self._ready = False
        self.ema_fast: List[Optional[float]] = []
        self.ema_slow: List[Optional[float]] = []
        self.atr: List[Optional[float]] = []

    # ------------------------------------------------------------------
    def prepare(self, candles: Sequence[Candle]) -> None:
        """Precalcule les indicateurs sur toute la serie.

        Ce precalcul n'introduit aucune fuite du futur : chaque valeur a
        l'indice i n'est construite qu'a partir des bougies <= i (voir
        indicators.py), et decide() ne lit jamais au-dela de i.
        """
        closes = [c.close for c in candles]
        self.ema_fast = ema(closes, self.cfg.ema_fast)
        self.ema_slow = ema(closes, self.cfg.ema_slow)
        self.atr = atr_series(candles, self.cfg.atr_period)
        self._ready = True

    @property
    def warmup(self) -> int:
        """Nombre de bougies necessaires avant que la strategie puisse decider."""
        return max(self.cfg.ema_slow, self.cfg.atr_period) + 2

    def atr_at(self, i: int) -> Optional[float]:
        return self.atr[i] if 0 <= i < len(self.atr) else None

    # ------------------------------------------------------------------
    def decide(self, i: int, candles: Sequence[Candle], has_position: bool) -> Decision:
        if not self._ready:
            raise RuntimeError("prepare() doit etre appele avant decide().")

        candle = candles[i]
        ef, es, a = self.ema_fast[i], self.ema_slow[i], self.atr[i]

        context = {
            "ema_fast": round(ef, 6) if ef is not None else None,
            "ema_slow": round(es, 6) if es is not None else None,
            "atr": round(a, 6) if a is not None else None,
            "atr_pct": round(a / candle.close, 6) if a and candle.close else None,
            "close": candle.close,
        }

        def mk(action: Action, reason: str) -> Decision:
            return Decision(
                ts=candle.ts,
                symbol="",
                action=action,
                reason=reason,
                price=candle.close,
                context=context,
            )

        if ef is None or es is None or a is None:
            return mk(Action.ABSTAIN, "periode de chauffe : indicateurs pas encore disponibles")

        atr_pct = a / candle.close if candle.close else 0.0

        # --- Sortie -----------------------------------------------------
        if has_position:
            if self.cfg.exit_on_cross and crossed_down(self.ema_fast, self.ema_slow, i):
                return mk(Action.SELL, "croisement baissier des EMA : sortie du trade")
            return mk(
                Action.ABSTAIN,
                "position deja ouverte, aucun signal de sortie (le stop et l'objectif restent actifs)",
            )

        # --- Entree -----------------------------------------------------
        if not crossed_up(self.ema_fast, self.ema_slow, i):
            trend = "haussiere" if ef > es else "baissiere"
            return mk(
                Action.ABSTAIN,
                f"aucun croisement haussier sur cette bougie (tendance {trend} deja en place)",
            )

        if atr_pct < self.cfg.min_atr_pct:
            return mk(
                Action.ABSTAIN,
                f"signal d'achat ignore : marche trop plat (ATR {atr_pct:.2%} < "
                f"{self.cfg.min_atr_pct:.2%}), le mouvement ne paierait pas les frais",
            )

        if atr_pct > self.cfg.max_atr_pct:
            return mk(
                Action.ABSTAIN,
                f"signal d'achat ignore : volatilite excessive (ATR {atr_pct:.2%} > "
                f"{self.cfg.max_atr_pct:.2%}), risque non maitrise",
            )

        return mk(
            Action.BUY,
            f"croisement haussier des EMA {self.cfg.ema_fast}/{self.cfg.ema_slow} "
            f"avec volatilite exploitable (ATR {atr_pct:.2%})",
        )
