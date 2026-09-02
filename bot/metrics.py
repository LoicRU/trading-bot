"""Metriques de performance.

Le resultat net est la metrique la moins utile de la liste. Celles qui
comptent vraiment sont plus bas : le drawdown maximum (la pire perte
cumulee traversee, celle qui te ferait couper le bot en panique), et la
comparaison avec "acheter et ne rien faire" - parce qu'un bot qui gagne
20 % quand le marche en a pris 60 n'est pas un bon bot, c'est un mauvais
bot dans un bon marche.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Sequence, Tuple

from .models import Candle, Trade

MS_PER_YEAR = 365.25 * 24 * 3600 * 1000


@dataclass
class Metrics:
    initial_equity: float
    final_equity: float
    total_return: float
    buy_hold_return: float
    excess_return: float
    max_drawdown: float
    max_drawdown_start: str
    max_drawdown_end: str
    longest_drawdown_bars: int
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    trade_count: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    expectancy_r: float
    best_trade_r: float
    worst_trade_r: float
    total_fees: float
    exposure: float
    bars: int
    days_covered: float
    days_without_trade: int

    def to_dict(self) -> Dict:
        return asdict(self)


def _iso(ts: int) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def max_drawdown(equity_curve: Sequence[Tuple[int, float]]) -> Tuple[float, str, str, int]:
    """Renvoie (drawdown max, date du sommet, date du creux, duree la plus longue en barres)."""
    if not equity_curve:
        return 0.0, "", "", 0

    peak = equity_curve[0][1]
    peak_ts = equity_curve[0][0]
    worst = 0.0
    worst_peak_ts = peak_ts
    worst_trough_ts = peak_ts

    longest = 0
    current_len = 0

    for ts, eq in equity_curve:
        if eq >= peak:
            peak = eq
            peak_ts = ts
            longest = max(longest, current_len)
            current_len = 0
        else:
            current_len += 1
            dd = (peak - eq) / peak if peak > 0 else 0.0
            if dd > worst:
                worst = dd
                worst_peak_ts = peak_ts
                worst_trough_ts = ts
    longest = max(longest, current_len)

    return worst, _iso(worst_peak_ts), _iso(worst_trough_ts), longest


def compute(
    equity_curve: Sequence[Tuple[int, float]],
    trades: Sequence[Trade],
    candles: Sequence[Candle],
    total_fees: float,
    bars_in_position: int,
    interval_ms: int,
) -> Metrics:
    if not equity_curve:
        raise ValueError("Courbe d'equite vide : rien a mesurer.")

    initial = equity_curve[0][1]
    final = equity_curve[-1][1]
    total_return = (final - initial) / initial if initial else 0.0

    bh_return = 0.0
    if candles:
        first_close, last_close = candles[0].close, candles[-1].close
        bh_return = (last_close - first_close) / first_close if first_close else 0.0

    span_ms = max(equity_curve[-1][0] - equity_curve[0][0], 1)
    years = span_ms / MS_PER_YEAR
    days = span_ms / 86_400_000

    annualized = ((final / initial) ** (1 / years) - 1) if years > 0 and initial > 0 and final > 0 else 0.0

    # Rendements barre a barre pour la volatilite
    rets: List[float] = []
    for k in range(1, len(equity_curve)):
        prev = equity_curve[k - 1][1]
        if prev > 0:
            rets.append(equity_curve[k][1] / prev - 1.0)

    vol_ann, sharpe = 0.0, 0.0
    if len(rets) > 2:
        mean = sum(rets) / len(rets)
        var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
        sd = math.sqrt(var)
        bars_per_year = MS_PER_YEAR / interval_ms
        vol_ann = sd * math.sqrt(bars_per_year)
        if sd > 0:
            sharpe = (mean / sd) * math.sqrt(bars_per_year)

    dd, dd_start, dd_end, longest_dd = max_drawdown(equity_curve)

    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]
    gross_win = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))

    win_rate = len(wins) / len(trades) if trades else 0.0
    avg_win = gross_win / len(wins) if wins else 0.0
    avg_loss = -gross_loss / len(losses) if losses else 0.0
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0)
    r_values = [t.r_multiple for t in trades]
    expectancy_r = sum(r_values) / len(r_values) if r_values else 0.0

    traded_days = {t.entry_ts // 86_400_000 for t in trades}
    total_days = {c.ts // 86_400_000 for c in candles}
    days_without_trade = len(total_days - traded_days)

    return Metrics(
        initial_equity=round(initial, 2),
        final_equity=round(final, 2),
        total_return=round(total_return, 6),
        buy_hold_return=round(bh_return, 6),
        excess_return=round(total_return - bh_return, 6),
        max_drawdown=round(dd, 6),
        max_drawdown_start=dd_start,
        max_drawdown_end=dd_end,
        longest_drawdown_bars=longest_dd,
        annualized_return=round(annualized, 6),
        annualized_volatility=round(vol_ann, 6),
        sharpe=round(sharpe, 4),
        trade_count=len(trades),
        win_rate=round(win_rate, 4),
        avg_win=round(avg_win, 2),
        avg_loss=round(avg_loss, 2),
        profit_factor=round(profit_factor, 4) if profit_factor != float("inf") else float("inf"),
        expectancy_r=round(expectancy_r, 4),
        best_trade_r=round(max(r_values), 4) if r_values else 0.0,
        worst_trade_r=round(min(r_values), 4) if r_values else 0.0,
        total_fees=round(total_fees, 2),
        exposure=round(bars_in_position / len(equity_curve), 4) if equity_curve else 0.0,
        bars=len(equity_curve),
        days_covered=round(days, 1),
        days_without_trade=days_without_trade,
    )
