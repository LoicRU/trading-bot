"""Moteur d'execution.

Un seul et meme moteur sert au backtest et au suivi quotidien : c'est
volontaire. Deux moteurs differents, c'est la garantie que le comportement
en reel divergera de ce qui a ete teste.

Regle de synchronisation, la plus importante du fichier :
    la decision est prise a la CLOTURE de la bougie i,
    elle est executee a l'OUVERTURE de la bougie i+1.
On ne peut donc jamais acheter au prix qui a justement declenche le
signal. C'est un detail qui parait mineur et qui suffit, a lui seul, a
transformer une strategie perdante en strategie "gagnante" sur le papier.

Ordre d'evaluation dans une bougie, du plus defavorable au plus favorable :
le stop est teste AVANT l'objectif. Si les deux sont touches dans la meme
bougie, on suppose que c'est le stop qui est parti en premier. On ne peut
pas savoir, donc on choisit l'hypothese pessimiste.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import metrics as metrics_mod
from .config import Config
from .models import Action, Candle, Decision, Trade
from .portfolio import Portfolio
from .risk import RiskManager
from .scoring import ScoreEntry, Scorer
from .strategy import EmaAtrStrategy


@dataclass
class RunResult:
    symbol: str
    equity_curve: List[Tuple[int, float]] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    decisions: List[Decision] = field(default_factory=list)
    score_entries: List[ScoreEntry] = field(default_factory=list)
    snapshots: List[Dict] = field(default_factory=list)
    halted_days: List[str] = field(default_factory=list)
    bars_in_position: int = 0
    metrics: Optional[metrics_mod.Metrics] = None
    score_breakdown: Dict = field(default_factory=dict)
    warmup_bars: int = 0

    @property
    def final_equity(self) -> float:
        return self.equity_curve[-1][1] if self.equity_curve else 0.0


class Engine:
    def __init__(self, cfg: Config, interval_ms: int) -> None:
        self.cfg = cfg
        self.interval_ms = interval_ms
        self.strategy = EmaAtrStrategy(cfg.strategy)
        self.portfolio = Portfolio(
            cfg.portfolio.initial_cash, cfg.portfolio.fee_rate, cfg.portfolio.slippage_bps
        )
        self.risk = RiskManager(cfg.risk)
        self.scorer = Scorer(cfg.scoring, risk_stop_mult=cfg.risk.atr_stop_mult)

    # ------------------------------------------------------------------
    def run(self, candles: Sequence[Candle], symbol: str) -> RunResult:
        if len(candles) < self.strategy.warmup + 5:
            raise ValueError(
                f"Pas assez de bougies ({len(candles)}) : il en faut au moins "
                f"{self.strategy.warmup + 5} pour cette configuration."
            )

        self.strategy.prepare(candles)
        result = RunResult(symbol=symbol, warmup_bars=self.strategy.warmup)
        pf, risk, scorer = self.portfolio, self.risk, self.scorer
        halted_seen: set[str] = set()

        for i in range(self.strategy.warmup, len(candles) - 1):
            decision_bar = candles[i]
            exec_bar = candles[i + 1]

            equity_now = pf.equity({symbol: decision_bar.close})
            risk.start_day_if_needed(exec_bar.day, equity_now)
            risk.update_day(equity_now)
            if risk.halted and exec_bar.day not in halted_seen:
                halted_seen.add(exec_bar.day)
                result.halted_days.append(exec_bar.day)

            raw = self.strategy.decide(i, candles, has_position=pf.position(symbol) is not None)
            raw.symbol = symbol

            action = raw.action
            reason = raw.reason
            was_flat = pf.position(symbol) is None

            # ---------- Sorties -----------------------------------------
            pos = pf.position(symbol)
            if pos is not None:
                trade: Optional[Trade] = None
                if exec_bar.low <= pos.stop:
                    trade = pf.sell(symbol, pos.stop, exec_bar.ts, "stop touche")
                elif exec_bar.high >= pos.take_profit:
                    trade = pf.sell(symbol, pos.take_profit, exec_bar.ts, "objectif atteint")
                elif action == Action.SELL:
                    trade = pf.sell(symbol, exec_bar.open, exec_bar.ts, "signal de sortie de la strategie")

                if trade is not None:
                    scorer.score_trade(trade)
                    result.trades.append(trade)
                    if action != Action.SELL:
                        action = Action.SELL
                        reason = f"sortie automatique : {trade.exit_reason}"

            # ---------- Entrees -----------------------------------------
            if pf.position(symbol) is None and raw.action == Action.BUY:
                allowed, why = risk.can_open(len(pf.positions))
                atr_value = self.strategy.atr_at(i)
                if not allowed:
                    action, reason = Action.ABSTAIN, f"signal d'achat bloque : {why}"
                elif not atr_value:
                    action, reason = Action.ABSTAIN, "signal d'achat bloque : ATR indisponible"
                else:
                    entry_price = exec_bar.open
                    stop, take = risk.stops(entry_price, atr_value)
                    equity_for_size = pf.equity({symbol: exec_bar.open})
                    qty, risk_amount, note = risk.position_size(
                        equity_for_size, pf.cash, entry_price, stop
                    )
                    if qty <= 0:
                        action, reason = Action.ABSTAIN, f"signal d'achat bloque : {note}"
                    else:
                        pf.buy(
                            symbol, entry_price, qty, exec_bar.ts,
                            stop=stop, take_profit=take, risk_amount=risk_amount,
                            reason=raw.reason,
                        )
                        action = Action.BUY
                        reason = f"{raw.reason} | {note} | stop {stop:.2f} / objectif {take:.2f}"

            # ---------- Enregistrement de la decision effective ----------
            effective = Decision(
                ts=decision_bar.ts,
                symbol=symbol,
                action=action,
                reason=reason,
                price=decision_bar.close,
                context=raw.context,
            )
            result.decisions.append(effective)

            # Seules les abstentions prises SANS position ouverte sont notees :
            # ce sont les vraies decisions de "ne pas entrer". S'abstenir alors
            # qu'on detient deja une position n'est pas un choix d'entree, le
            # penaliser n'aurait aucun sens.
            if action == Action.ABSTAIN and was_flat:
                scorer.register_abstention(i, effective, self.strategy.atr_at(i))

            newly_scored = scorer.evaluate_pending(i, candles)
            result.score_entries.extend(newly_scored)

            # ---------- Etat de fin de bougie ----------------------------
            prices = {symbol: exec_bar.close}
            pf.check_invariants(prices)
            equity = pf.equity(prices)
            open_pos = pf.position(symbol)
            if open_pos is not None:
                result.bars_in_position += 1

            result.equity_curve.append((exec_bar.ts, equity))
            result.snapshots.append(
                {
                    "ts": exec_bar.ts,
                    "day": exec_bar.day,
                    "equity": equity,
                    "cash": pf.cash,
                    "position_qty": open_pos.qty if open_pos else 0.0,
                    "price": exec_bar.close,
                    "cumulative_score": scorer.total,
                }
            )

        # Les scores de trades ont ete ajoutes au scorer au fil de l'eau ;
        # on complete la liste exportee avec ceux-la.
        result.score_entries = list(scorer.entries)
        result.score_breakdown = scorer.breakdown()

        used_candles = list(candles[self.strategy.warmup + 1:])
        result.metrics = metrics_mod.compute(
            equity_curve=result.equity_curve,
            trades=result.trades,
            candles=used_candles,
            total_fees=pf.fees_paid,
            bars_in_position=result.bars_in_position,
            interval_ms=self.interval_ms,
        )
        return result
