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


def rsi(values: Sequence[float], period: int = 14) -> List[Optional[float]]:
    """Relative Strength Index, lissage de Wilder.

    Mesure entre 0 et 100 la force relative des hausses face aux baisses.
    Sous 30, on parle traditionnellement de survente ; au-dessus de 70,
    de surachat. Que ces seuils aient une valeur predictive reste
    precisement ce qu'il faut verifier plutot que croire.
    """
    out: List[Optional[float]] = [None] * len(values)
    if period < 1 or len(values) <= period:
        return out

    gains = losses = 0.0
    for i in range(1, period + 1):
        delta = values[i] - values[i - 1]
        gains += max(delta, 0.0)
        losses += max(-delta, 0.0)
    avg_gain, avg_loss = gains / period, losses / period
    out[period] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1 + avg_gain / avg_loss)

    for i in range(period + 1, len(values)):
        delta = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(delta, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-delta, 0.0)) / period
        out[i] = 100.0 if avg_loss == 0 else 100.0 - 100.0 / (1 + avg_gain / avg_loss)
    return out


def rolling_max(values: Sequence[float], window: int) -> List[Optional[float]]:
    """Plus haut des `window` valeurs PRECEDENTES (l'indice i exclu).

    L'exclusion de i est volontaire : inclure la bougie courante ferait
    fuir le present dans le signal et rendrait tout cassage de plus haut
    trivialement vrai.
    """
    out: List[Optional[float]] = [None] * len(values)
    for i in range(window, len(values)):
        out[i] = max(values[i - window:i])
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
