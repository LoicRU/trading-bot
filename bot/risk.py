"""Gestion du risque.

C'est le module qui a le droit de dire NON a la strategie. La strategie
propose, le risque dispose. Separer les deux est ce qui evite qu'une
strategie enthousiaste ne vide le compte.

La taille de position est calculee a partir du RISQUE, pas d'un montant
fixe : on decide d'abord combien on accepte de perdre si le stop est
touche, et la taille en decoule. C'est l'inverse de ce que font la
plupart des debutants, et c'est ce qui compte le plus sur la duree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .config import RiskConfig


@dataclass
class DayState:
    """Suivi de la journee en cours pour la limite de perte journaliere."""

    day: str
    start_equity: float
    halted: bool = False
    halt_reason: str = ""


class RiskManager:
    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg
        self.day: Optional[DayState] = None

    # ------------------------------------------------------------------
    # Journee
    # ------------------------------------------------------------------
    def start_day_if_needed(self, day: str, equity: float) -> None:
        if self.day is None or self.day.day != day:
            self.day = DayState(day=day, start_equity=equity)

    def update_day(self, equity: float) -> None:
        """Coupe les nouveaux trades si la perte du jour depasse le plafond."""
        if self.day is None or self.day.halted:
            return
        loss = (self.day.start_equity - equity) / self.day.start_equity if self.day.start_equity else 0.0
        if loss >= self.cfg.max_daily_loss_pct:
            self.day.halted = True
            self.day.halt_reason = (
                f"perte journaliere de {loss:.2%} >= plafond {self.cfg.max_daily_loss_pct:.2%}"
            )

    @property
    def halted(self) -> bool:
        return bool(self.day and self.day.halted)

    @property
    def halt_reason(self) -> str:
        return self.day.halt_reason if self.day else ""

    # ------------------------------------------------------------------
    # Stops et taille
    # ------------------------------------------------------------------
    def stops(self, entry_price: float, atr_value: float) -> Tuple[float, float]:
        """Stop et objectif calcules en multiples d'ATR : ils s'adaptent
        automatiquement a la volatilite du moment."""
        stop = entry_price - self.cfg.atr_stop_mult * atr_value
        take = entry_price + self.cfg.atr_tp_mult * atr_value
        return max(stop, 1e-9), take

    def position_size(
        self, equity: float, cash: float, entry_price: float, stop_price: float
    ) -> Tuple[float, float, str]:
        """Renvoie (quantite, montant risque, explication).

        Quantite = 0 signifie qu'aucune position ne doit etre ouverte.
        """
        distance = entry_price - stop_price
        if distance <= 0:
            return 0.0, 0.0, "stop incoherent (au-dessus du prix d'entree)"

        risk_budget = equity * self.cfg.risk_per_trade_pct
        qty = risk_budget / distance

        # Plafond de taille : on ne met jamais plus de X % du capital sur une ligne.
        max_notional = equity * self.cfg.max_position_pct
        if qty * entry_price > max_notional:
            qty = max_notional / entry_price

        # Plafond de cash reel disponible (frais compris, avec une marge).
        affordable = (cash * 0.995) / entry_price
        if qty > affordable:
            qty = affordable

        if qty <= 0:
            return 0.0, 0.0, "cash insuffisant"

        risk_amount = qty * distance
        note = (
            f"taille {qty:.8f} (risque {risk_amount:.2f} = "
            f"{risk_amount / equity:.2%} du capital)"
        )
        return qty, risk_amount, note

    # ------------------------------------------------------------------
    # Autorisation
    # ------------------------------------------------------------------
    def can_open(self, open_positions: int) -> Tuple[bool, str]:
        if self.halted:
            return False, f"trading suspendu pour la journee ({self.halt_reason})"
        if open_positions >= self.cfg.max_open_positions:
            return False, f"deja {open_positions} position(s) ouverte(s), plafond atteint"
        return True, ""
