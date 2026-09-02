"""Portefeuille fictif.

Trois principes non negociables :
  1. Les frais sont preleves a l'achat ET a la vente, toujours.
  2. Le prix d'execution est degrade par le slippage, toujours dans le
     sens defavorable. Un simulateur qui te sert au prix affiche te
     ment, et c'est exactement comme ca qu'on croit avoir une strategie
     gagnante qui perd de l'argent en reel.
  3. On ne peut pas depenser plus que le cash disponible. Pas de levier
     en V1 : c'est volontaire.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .models import Position, Trade


class InsufficientFunds(RuntimeError):
    pass


class PositionError(RuntimeError):
    pass


class Portfolio:
    def __init__(self, initial_cash: float, fee_rate: float = 0.001, slippage_bps: float = 5.0) -> None:
        if initial_cash <= 0:
            raise ValueError("Le capital initial doit etre strictement positif.")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_rate = float(fee_rate)
        self.slippage = float(slippage_bps) / 10_000.0
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.fees_paid = 0.0
        self.realized_pnl = 0.0

    # ------------------------------------------------------------------
    # Prix d'execution
    # ------------------------------------------------------------------
    def fill_price(self, price: float, side: str) -> float:
        """Le slippage joue toujours contre nous : plus cher a l'achat,
        moins cher a la vente."""
        if side == "buy":
            return price * (1.0 + self.slippage)
        if side == "sell":
            return price * (1.0 - self.slippage)
        raise ValueError(f"Sens inconnu : {side}")

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    def buy(
        self,
        symbol: str,
        price: float,
        qty: float,
        ts: int,
        stop: float,
        take_profit: float,
        risk_amount: float,
        reason: str = "",
    ) -> Position:
        if symbol in self.positions:
            raise PositionError(f"Position deja ouverte sur {symbol} (V1 : une seule a la fois).")
        if qty <= 0:
            raise ValueError("La quantite doit etre strictement positive.")

        fill = self.fill_price(price, "buy")
        notional = fill * qty
        fee = notional * self.fee_rate
        total = notional + fee
        if total > self.cash + 1e-9:
            raise InsufficientFunds(
                f"Achat de {qty:.8f} {symbol} a {fill:.2f} = {total:.2f} "
                f"mais seulement {self.cash:.2f} disponibles."
            )

        self.cash -= total
        self.fees_paid += fee
        pos = Position(
            symbol=symbol,
            qty=qty,
            entry_price=fill,
            entry_ts=ts,
            stop=stop,
            take_profit=take_profit,
            risk_amount=risk_amount,
            entry_fee=fee,
            entry_reason=reason,
        )
        self.positions[symbol] = pos
        return pos

    def sell(self, symbol: str, price: float, ts: int, reason: str) -> Trade:
        pos = self.positions.get(symbol)
        if pos is None:
            raise PositionError(f"Aucune position ouverte sur {symbol}.")

        fill = self.fill_price(price, "sell")
        notional = fill * pos.qty
        fee = notional * self.fee_rate
        self.cash += notional - fee
        self.fees_paid += fee

        gross = (fill - pos.entry_price) * pos.qty
        pnl = gross - fee - pos.entry_fee
        cost_basis = pos.entry_price * pos.qty
        trade = Trade(
            symbol=symbol,
            qty=pos.qty,
            entry_ts=pos.entry_ts,
            entry_price=pos.entry_price,
            exit_ts=ts,
            exit_price=fill,
            fees=fee + pos.entry_fee,
            pnl=pnl,
            pnl_pct=pnl / cost_basis if cost_basis else 0.0,
            risk_amount=pos.risk_amount,
            exit_reason=reason,
            entry_reason=pos.entry_reason,
        )
        self.realized_pnl += pnl
        self.trades.append(trade)
        del self.positions[symbol]
        return trade

    # ------------------------------------------------------------------
    # Etat
    # ------------------------------------------------------------------
    def position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def equity(self, prices: Dict[str, float]) -> float:
        """Cash + valeur de marche des positions (au prix affiche, sans slippage :
        le slippage sera compte au moment reel de la sortie)."""
        total = self.cash
        for sym, pos in self.positions.items():
            total += pos.qty * prices.get(sym, pos.entry_price)
        return total

    def exposure(self, prices: Dict[str, float]) -> float:
        eq = self.equity(prices)
        if eq <= 0:
            return 0.0
        invested = sum(p.qty * prices.get(s, p.entry_price) for s, p in self.positions.items())
        return invested / eq

    def check_invariants(self, prices: Dict[str, float]) -> None:
        """Verifie que la comptabilite tient. Appele par les tests et par
        le moteur a chaque barre : mieux vaut planter tot que produire un
        rapport faux."""
        if self.cash < -1e-6:
            raise AssertionError(f"Cash negatif : {self.cash}")
        for sym, pos in self.positions.items():
            if pos.qty <= 0:
                raise AssertionError(f"Position a quantite nulle ou negative sur {sym}")
        expected = self.initial_cash + self.realized_pnl
        actual_closed_part = self.cash + sum(
            p.entry_price * p.qty + p.entry_fee for p in self.positions.values()
        )
        if abs(expected - actual_closed_part) > 1e-4 * max(1.0, abs(expected)):
            raise AssertionError(
                "Incoherence comptable : "
                f"attendu {expected:.6f}, obtenu {actual_closed_part:.6f}"
            )
