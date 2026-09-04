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


def crossed_above(serie: Sequence[Optional[float]], level: float, i: int) -> bool:
    """La serie vient-elle de passer au-dessus d'un niveau fixe sur la bougie i ?

    Generalisation de crossed_up a un seuil constant plutot qu'une seconde
    serie (RSI qui repasse au-dessus de 30, MACD qui repasse au-dessus de 0...).
    """
    if i < 1:
        return False
    a0, a1 = serie[i - 1], serie[i]
    if a0 is None or a1 is None:
        return False
    return a0 <= level < a1


def crossed_below(serie: Sequence[Optional[float]], level: float, i: int) -> bool:
    """La serie vient-elle de passer sous un niveau fixe sur la bougie i ?"""
    if i < 1:
        return False
    a0, a1 = serie[i - 1], serie[i]
    if a0 is None or a1 is None:
        return False
    return a0 >= level > a1


# ----------------------------------------------------------------------
# Indicateurs supplementaires pour le balayage exploratoire (bot/signals_explo.py)
#
# Meme regle que plus haut : la valeur a l'indice i n'utilise que les
# bougies <= i. Toutes les fenetres glissantes incluent la bougie
# courante (ce n'est PAS une fuite du futur — seul rolling_max/rolling_min
# l'excluent volontairement, pour la definition specifique d'une cassure
# "au-dessus du plus haut PRECEDENT").
# ----------------------------------------------------------------------

def rolling_min(values: Sequence[float], window: int) -> List[Optional[float]]:
    """Plus bas des `window` valeurs PRECEDENTES (l'indice i exclu). Voir rolling_max."""
    out: List[Optional[float]] = [None] * len(values)
    for i in range(window, len(values)):
        out[i] = min(values[i - window:i])
    return out


def stddev(values: Sequence[float], period: int) -> List[Optional[float]]:
    """Ecart-type (population) sur une fenetre glissante incluant la bougie courante."""
    out: List[Optional[float]] = [None] * len(values)
    if period <= 1:
        return out
    for i in range(period - 1, len(values)):
        window = values[i - period + 1:i + 1]
        m = sum(window) / period
        out[i] = (sum((v - m) ** 2 for v in window) / period) ** 0.5
    return out


def _ema_skip_none(series: Sequence[Optional[float]], period: int) -> List[Optional[float]]:
    """EMA d'une serie qui commence par des None (periode de chauffe amont).

    Utilise par macd()/trix(), qui enchainent des EMA sur des series
    elles-memes derivees d'une EMA. Une fois demarree (apres les None de
    tete), la serie ne doit plus avoir de trou — sinon ema() amorcerait
    silencieusement sur une mauvaise fenetre, l'exact type d'erreur qu'on
    ne doit jamais laisser passer sans bruit ici.
    """
    n = len(series)
    out: List[Optional[float]] = [None] * n
    start = next((i for i, v in enumerate(series) if v is not None), n)
    tail = series[start:]
    if any(v is None for v in tail):
        raise ValueError("serie avec un trou apres son demarrage (indicateur mal chaine)")
    computed = ema([float(v) for v in tail], period)  # type: ignore[list-item]
    for i, v in enumerate(computed):
        out[start + i] = v
    return out


def macd(
    values: Sequence[float], fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Ligne MACD, ligne de signal (EMA de la ligne MACD) et histogramme."""
    ema_fast, ema_slow = ema(values, fast), ema(values, slow)
    line: List[Optional[float]] = [
        (a - b) if a is not None and b is not None else None
        for a, b in zip(ema_fast, ema_slow)
    ]
    sig_line = _ema_skip_none(line, signal)
    hist: List[Optional[float]] = [
        (a - b) if a is not None and b is not None else None
        for a, b in zip(line, sig_line)
    ]
    return line, sig_line, hist


def bollinger(
    values: Sequence[float], period: int = 20, k: float = 2.0
) -> tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Bande mediane (SMA), superieure et inferieure (+/- k ecarts-types)."""
    mid = sma(values, period)
    dev = stddev(values, period)
    upper = [m + k * d if m is not None and d is not None else None for m, d in zip(mid, dev)]
    lower = [m - k * d if m is not None and d is not None else None for m, d in zip(mid, dev)]
    return mid, upper, lower


def keltner(
    candles: Sequence[Candle], ema_period: int = 20, atr_period: int = 10, k: float = 2.0
) -> tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """Canal de Keltner : EMA du prix +/- k fois l'ATR."""
    mid = ema([c.close for c in candles], ema_period)
    a = atr(candles, atr_period)
    upper = [m + k * v if m is not None and v is not None else None for m, v in zip(mid, a)]
    lower = [m - k * v if m is not None and v is not None else None for m, v in zip(mid, a)]
    return mid, upper, lower


def _sma_skip_none(series: Sequence[Optional[float]], period: int) -> List[Optional[float]]:
    """SMA d'une serie qui commence par des None. Voir _ema_skip_none."""
    n = len(series)
    out: List[Optional[float]] = [None] * n
    start = next((i for i, v in enumerate(series) if v is not None), n)
    tail = series[start:]
    if any(v is None for v in tail):
        raise ValueError("serie avec un trou apres son demarrage (indicateur mal chaine)")
    computed = sma([float(v) for v in tail], period)  # type: ignore[list-item]
    for i, v in enumerate(computed):
        out[start + i] = v
    return out


def stochastic(
    candles: Sequence[Candle], period: int = 14, smooth_k: int = 3, smooth_d: int = 3
) -> tuple[List[Optional[float]], List[Optional[float]]]:
    """Oscillateur stochastique lent : %K lisse et %D (SMA de %K)."""
    n = len(candles)
    raw: List[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        window = candles[i - period + 1:i + 1]
        hh = max(c.high for c in window)
        ll = min(c.low for c in window)
        raw[i] = 50.0 if hh == ll else (candles[i].close - ll) / (hh - ll) * 100.0
    k_line = _sma_skip_none(raw, smooth_k)
    d_line = _sma_skip_none(k_line, smooth_d)
    return k_line, d_line


def williams_r(candles: Sequence[Candle], period: int = 14) -> List[Optional[float]]:
    """Williams %R : entre -100 (survente) et 0 (surachat)."""
    n = len(candles)
    out: List[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        window = candles[i - period + 1:i + 1]
        hh = max(c.high for c in window)
        ll = min(c.low for c in window)
        out[i] = -50.0 if hh == ll else (hh - candles[i].close) / (hh - ll) * -100.0
    return out


def cci(candles: Sequence[Candle], period: int = 20) -> List[Optional[float]]:
    """Commodity Channel Index."""
    n = len(candles)
    tp = [(c.high + c.low + c.close) / 3.0 for c in candles]
    tp_sma = sma(tp, period)
    out: List[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        m = tp_sma[i]
        if m is None:
            continue
        window = tp[i - period + 1:i + 1]
        mean_dev = sum(abs(v - m) for v in window) / period
        out[i] = 0.0 if mean_dev == 0 else (tp[i] - m) / (0.015 * mean_dev)
    return out


def roc(values: Sequence[float], period: int) -> List[Optional[float]]:
    """Rate of Change : variation relative sur `period` bougies."""
    out: List[Optional[float]] = [None] * len(values)
    for i in range(period, len(values)):
        base = values[i - period]
        if base:
            out[i] = (values[i] - base) / base * 100.0
    return out


def obv(candles: Sequence[Candle]) -> List[Optional[float]]:
    """On Balance Volume : volume cumule, signe par le sens de cloture a cloture."""
    out: List[Optional[float]] = [0.0] * len(candles)
    for i in range(1, len(candles)):
        prev = out[i - 1] or 0.0
        if candles[i].close > candles[i - 1].close:
            out[i] = prev + candles[i].volume
        elif candles[i].close < candles[i - 1].close:
            out[i] = prev - candles[i].volume
        else:
            out[i] = prev
    return out


def mfi(candles: Sequence[Candle], period: int = 14) -> List[Optional[float]]:
    """Money Flow Index : RSI pondere par le volume sur le prix typique."""
    n = len(candles)
    tp = [(c.high + c.low + c.close) / 3.0 for c in candles]
    out: List[Optional[float]] = [None] * n
    for i in range(period, n):
        pos = neg = 0.0
        for j in range(i - period + 1, i + 1):
            flow = tp[j] * candles[j].volume
            if tp[j] > tp[j - 1]:
                pos += flow
            elif tp[j] < tp[j - 1]:
                neg += flow
        out[i] = 100.0 if neg == 0 else 100.0 - 100.0 / (1 + pos / neg)
    return out


def cmf(candles: Sequence[Candle], period: int = 20) -> List[Optional[float]]:
    """Chaikin Money Flow : pression acheteuse/vendeuse ponderee par le volume."""
    n = len(candles)
    mfv = []
    for c in candles:
        span = c.high - c.low
        mfv.append(0.0 if span == 0 else ((c.close - c.low) - (c.high - c.close)) / span * c.volume)
    out: List[Optional[float]] = [None] * n
    for i in range(period - 1, n):
        vol_sum = sum(candles[j].volume for j in range(i - period + 1, i + 1))
        out[i] = 0.0 if vol_sum == 0 else sum(mfv[i - period + 1:i + 1]) / vol_sum
    return out


def aroon(candles: Sequence[Candle], period: int = 14) -> tuple[List[Optional[float]], List[Optional[float]]]:
    """Aroon up / Aroon down : nombre de bougies depuis le plus haut / plus bas recent."""
    n = len(candles)
    up: List[Optional[float]] = [None] * n
    down: List[Optional[float]] = [None] * n
    for i in range(period, n):
        window = candles[i - period:i + 1]
        idx_high = max(range(len(window)), key=lambda k: window[k].high)
        idx_low = min(range(len(window)), key=lambda k: window[k].low)
        up[i] = (idx_high / period) * 100.0
        down[i] = (idx_low / period) * 100.0
    return up, down


def trix(values: Sequence[float], period: int = 15) -> List[Optional[float]]:
    """TRIX : taux de variation d'une triple EMA (filtre le bruit court terme)."""
    e1 = ema(values, period)
    e2 = _ema_skip_none(e1, period)
    e3 = _ema_skip_none(e2, period)
    out: List[Optional[float]] = [None] * len(values)
    for i in range(1, len(values)):
        a, b = e3[i], e3[i - 1]
        if a is not None and b is not None and b != 0:
            out[i] = (a - b) / b * 100.0
    return out


def vortex(candles: Sequence[Candle], period: int = 14) -> tuple[List[Optional[float]], List[Optional[float]]]:
    """Vortex Indicator : VI+ (mouvement haussier) et VI- (mouvement baissier)."""
    n = len(candles)
    vm_plus = [0.0] + [abs(candles[i].high - candles[i - 1].low) for i in range(1, n)]
    vm_minus = [0.0] + [abs(candles[i].low - candles[i - 1].high) for i in range(1, n)]
    trs = [true_range(candles[i], candles[i - 1] if i > 0 else None) for i in range(n)]
    plus: List[Optional[float]] = [None] * n
    minus: List[Optional[float]] = [None] * n
    for i in range(period, n):
        tr_sum = sum(trs[i - period + 1:i + 1])
        if tr_sum:
            plus[i] = sum(vm_plus[i - period + 1:i + 1]) / tr_sum
            minus[i] = sum(vm_minus[i - period + 1:i + 1]) / tr_sum
    return plus, minus


def awesome_oscillator(candles: Sequence[Candle], fast: int = 5, slow: int = 34) -> List[Optional[float]]:
    """Awesome Oscillator : SMA(5) - SMA(34) du point milieu (haut+bas)/2."""
    mid = [(c.high + c.low) / 2.0 for c in candles]
    s_fast, s_slow = sma(mid, fast), sma(mid, slow)
    return [
        (a - b) if a is not None and b is not None else None
        for a, b in zip(s_fast, s_slow)
    ]


def volume_sma(candles: Sequence[Candle], period: int = 20) -> List[Optional[float]]:
    return sma([c.volume for c in candles], period)
