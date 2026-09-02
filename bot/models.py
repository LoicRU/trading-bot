"""Structures de donnees de base partagees par tout le bot.

Aucune logique metier ici : uniquement les objets qui circulent entre
les modules (bougies, decisions, positions, trades).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional


class Action(str, Enum):
    """Les trois decisions possibles du bot.

    ABSTAIN est une decision a part entiere, pas une absence de decision :
    elle est enregistree, motivee, et evaluee par le systeme de score.
    """

    BUY = "BUY"
    SELL = "SELL"
    ABSTAIN = "ABSTAIN"


@dataclass(frozen=True)
class Candle:
    """Une bougie OHLCV. ts est en millisecondes UTC (ouverture de la bougie)."""

    ts: int
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.ts / 1000, tz=timezone.utc)

    @property
    def day(self) -> str:
        """Cle de journee UTC, utilisee pour la limite de perte journaliere."""
        return self.dt.strftime("%Y-%m-%d")


@dataclass
class Decision:
    """Ce que la strategie a decide sur une bougie donnee, et pourquoi."""

    ts: int
    symbol: str
    action: Action
    reason: str
    price: float
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def dt(self) -> datetime:
        return datetime.fromtimestamp(self.ts / 1000, tz=timezone.utc)


@dataclass
class Position:
    """Une position ouverte (V1 : uniquement du long, pas de vente a decouvert)."""

    symbol: str
    qty: float
    entry_price: float
    entry_ts: int
    stop: float
    take_profit: float
    risk_amount: float  # montant en devise risque entre l'entree et le stop (= 1 R)
    entry_fee: float = 0.0
    entry_reason: str = ""

    def unrealized(self, price: float) -> float:
        return (price - self.entry_price) * self.qty


@dataclass
class Trade:
    """Un aller-retour complet, une fois la position fermee."""

    symbol: str
    qty: float
    entry_ts: int
    entry_price: float
    exit_ts: int
    exit_price: float
    fees: float
    pnl: float
    pnl_pct: float
    risk_amount: float
    exit_reason: str
    entry_reason: str = ""
    score: Optional[float] = None

    @property
    def r_multiple(self) -> float:
        """Resultat exprime en multiple du risque initial.

        -1 R = le stop a ete touche exactement comme prevu.
        +2 R = le trade a rapporte deux fois ce qu'il risquait.
        C'est la seule mesure qui permette de comparer des trades de
        tailles differentes de facon honnete.
        """
        if self.risk_amount <= 0:
            return 0.0
        return self.pnl / self.risk_amount

    @property
    def is_win(self) -> bool:
        return self.pnl > 0
