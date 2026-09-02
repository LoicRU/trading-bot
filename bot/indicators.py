"""Indicateurs techniques, en Python pur (aucune dependance).

Regle absolue respectee ici : chaque valeur d'indicateur a l'indice i
n'utilise QUE les bougies d'indice <= i. Aucune fuite du futur.
Les periodes de chauffe renvoient None, jamais une valeur approximative.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .models import Candle


def sma(values: Sequence[float], period: int) -> List[Optional[float]]:
    """Moyenne mobile simple."""
    out: List[Optional[float]] = [None] * len(values)
    if period <= 0:
        return out
    running = 0.0
    for i, v in enumerate(values):
        running += v
        if i >= period:
            running -= values[i - period]
        if i >= period - 1:
            out[i] = running / period
    return out


def ema(values: Sequence[float], period: int) -> List[Optional[float]]:
    """Moyenne mobile exponentielle, amorcee par une SMA sur la premiere fenetre."""
    out: List[Optional[float]] = [None] * len(values)
    if period <= 0 or len(values) < period:
        return out
    k = 2.0 / (period + 1.0)
    seed = sum(values[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(values)):
        prev = values[i] * k + prev * (1.0 - k)
        out[i] = prev
    return out


def true_range(current: Candle, previous: Optional[Candle]) -> float:
    if previous is None:
        return current.high - current.low
    return max(
        current.high - current.low,
        abs(current.high - previous.close),
        abs(current.low - previous.close),
    )


def atr(candles: Sequence[Candle], period: int) -> List[Optional[float]]:
    """Average True Range, lissage de Wilder (le standard)."""
    out: List[Optional[float]] = [None] * len(candles)
    if period <= 0 or len(candles) < period:
        return out

    trs = [true_range(candles[i], candles[i - 1] if i > 0 else None) for i in range(len(candles))]
    seed = sum(trs[:period]) / period
    out[period - 1] = seed
    prev = seed
    for i in range(period, len(candles)):
        prev = (prev * (period - 1) + trs[i]) / period
        out[i] = prev
    return out


def crossed_up(fast: Sequence[Optional[float]], slow: Sequence[Optional[float]], i: int) -> bool:
    """La rapide vient-elle de passer au-dessus de la lente sur la bougie i ?"""
    if i < 1:
        return False
    a0, b0, a1, b1 = fast[i - 1], slow[i - 1], fast[i], slow[i]
    if None in (a0, b0, a1, b1):
        return False
    return a0 <= b0 and a1 > b1


def crossed_down(fast: Sequence[Optional[float]], slow: Sequence[Optional[float]], i: int) -> bool:
    """La rapide vient-elle de passer sous la lente sur la bougie i ?"""
    if i < 1:
        return False
    a0, b0, a1, b1 = fast[i - 1], slow[i - 1], fast[i], slow[i]
    if None in (a0, b0, a1, b1):
        return False
    return a0 >= b0 and a1 < b1
