"""Bibliotheque de signaux d'entree candidats.

A quoi ca sert
--------------
Le test d'avantage a montre que le croisement d'EMA ne predit rien. La
question devient : est-ce que QUELQUE CHOSE predit quoi que ce soit sur
cette paire ? Plutot que de bricoler un signal, d'ecrire une strategie
autour, de la backtester et de recommencer — des heures pour chaque
candidat — on teste ici les signaux nus, en quelques secondes chacun.

LE PIEGE A EVITER ABSOLUMENT
----------------------------
Tester beaucoup de signaux jusqu'a en trouver un "significatif" est la
facon la plus efficace de se mentir a soi-meme. Avec un seuil de 5 %, un
signal sur vingt ressort significatif PAR PUR HASARD, meme si aucun ne
vaut rien.

D'ou deux regles appliquees par le code, pas laissees a la bonne volonte :

1. La liste des candidats est FIGEE ici, decidee a l'avance. On ne
   l'allonge pas apres avoir vu les resultats.
2. Le seuil est corrige du nombre de tests effectues (correction de
   Bonferroni). Tester 5 signaux sur 5 horizons, c'est 25 tests : le
   seuil devient 0.05/25 = 0.002.

Un signal qui passe cette barre merite qu'on s'y interesse. Un signal a
p=0.04 dans ces conditions ne prouve rien du tout.

Tous les signaux sont des EVENEMENTS : ils se declenchent sur une
transition, pas sur un etat permanent. Un signal qui serait vrai 40 % du
temps ne se compare pas a un tirage ponctuel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .config import StrategyConfig
from .indicators import atr as atr_series
from .indicators import crossed_up, ema, rolling_max, rsi
from .models import Candle


@dataclass
class SignalSpec:
    key: str
    label: str
    hypothese: str


class Signal:
    """Un signal candidat : on le prepare une fois, puis on l'interroge
    bougie par bougie. fires(i) ne doit JAMAIS lire au-dela de i."""

    spec: SignalSpec

    besoin_financement = False  # ce signal exige-t-il les taux de financement ?

    def prepare(self, candles: Sequence[Candle], cfg: StrategyConfig, extras=None) -> None:
        raise NotImplementedError

    def fires(self, i: int) -> bool:
        raise NotImplementedError

    @property
    def warmup(self) -> int:
        return 60


class EmaCross(Signal):
    spec = SignalSpec(
        "ema_cross",
        "Croisement haussier d'EMA",
        "suivre la tendance quand la moyenne courte repasse au-dessus de la longue",
    )

    def prepare(self, candles, cfg, extras=None):
        closes = [c.close for c in candles]
        self.fast = ema(closes, cfg.ema_fast)
        self.slow = ema(closes, cfg.ema_slow)
        self._warmup = cfg.ema_slow + 2

    def fires(self, i):
        return crossed_up(self.fast, self.slow, i)

    @property
    def warmup(self):
        return self._warmup


class RsiRebound(Signal):
    spec = SignalSpec(
        "rsi_rebond",
        "Sortie de survente (RSI repasse au-dessus de 30)",
        "retour a la moyenne : acheter apres un exces de baisse",
    )
    period = 14
    seuil = 30.0

    def prepare(self, candles, cfg, extras=None):
        self.rsi = rsi([c.close for c in candles], self.period)

    def fires(self, i):
        if i < 1 or self.rsi[i] is None or self.rsi[i - 1] is None:
            return False
        return self.rsi[i - 1] <= self.seuil < self.rsi[i]

    @property
    def warmup(self):
        return self.period + 2


class DonchianBreakout(Signal):
    spec = SignalSpec(
        "cassure",
        "Cassure du plus haut des 48 dernieres bougies",
        "momentum reel : acheter la force, pas un indicateur retarde",
    )
    window = 48

    def prepare(self, candles, cfg, extras=None):
        self.high_max = rolling_max([c.high for c in candles], self.window)
        self.closes = [c.close for c in candles]

    def fires(self, i):
        if i < 1 or self.high_max[i] is None or self.high_max[i - 1] is None:
            return False
        # premiere bougie qui cloture au-dessus du plus haut precedent
        return self.closes[i] > self.high_max[i] and self.closes[i - 1] <= self.high_max[i - 1]

    @property
    def warmup(self):
        return self.window + 2


class DipBelowTrend(Signal):
    spec = SignalSpec(
        "creux",
        "Prix a plus de 1,5 ATR sous la moyenne longue",
        "retour a la moyenne : acheter l'ecartement excessif",
    )
    k = 1.5

    def prepare(self, candles, cfg, extras=None):
        closes = [c.close for c in candles]
        self.slow = ema(closes, cfg.ema_slow)
        self.atr = atr_series(candles, cfg.atr_period)
        self.closes = closes
        self._warmup = max(cfg.ema_slow, cfg.atr_period) + 2

    def _below(self, i: int) -> Optional[bool]:
        if self.slow[i] is None or self.atr[i] is None:
            return None
        return self.closes[i] < self.slow[i] - self.k * self.atr[i]

    def fires(self, i):
        if i < 1:
            return False
        now, before = self._below(i), self._below(i - 1)
        return now is True and before is False

    @property
    def warmup(self):
        return self._warmup


class MomentumFlip(Signal):
    spec = SignalSpec(
        "momentum",
        "Rendement sur 24 bougies qui redevient positif",
        "momentum brut, sans lissage par une moyenne mobile",
    )
    window = 24

    def prepare(self, candles, cfg, extras=None):
        self.closes = [c.close for c in candles]

    def _mom(self, i: int) -> Optional[float]:
        j = i - self.window
        if j < 0 or self.closes[j] <= 0:
            return None
        return self.closes[i] / self.closes[j] - 1.0

    def fires(self, i):
        a, b = self._mom(i), self._mom(i - 1)
        if a is None or b is None:
            return False
        return b <= 0 < a

    @property
    def warmup(self):
        return self.window + 2


class FundingCapitulation(Signal):
    """Taux de financement au plus bas de son mois : positionnement
    vendeur sature."""

    spec = SignalSpec(
        "financement_bas",
        "Taux de financement dans son dernier decile mensuel",
        "positionnement vendeur sature : plus personne pour vendre, "
        "les rachats forces poussent a la hausse",
    )
    besoin_financement = True

    def prepare(self, candles, cfg, extras=None):
        from .funding import events_to_indices, percentile_events

        self._indices = set()
        funding = (extras or {}).get("funding")
        if funding:
            lows, _ = percentile_events(funding)
            self._indices = events_to_indices(candles, lows)

    def fires(self, i):
        return i in self._indices

    @property
    def warmup(self):
        return 60


class FundingEuphoria(Signal):
    """Le controle du precedent : positionnement acheteur sature.

    On s'attend a un avantage NUL ou NEGATIF ici. Ce candidat est la pour
    verifier que la mesure ne trouve pas systematiquement du positif —
    un test sans controle est un test qui se croit sur parole."""

    spec = SignalSpec(
        "financement_haut",
        "Taux de financement dans son premier decile mensuel (controle)",
        "positionnement acheteur sature : on attend un avantage nul ou negatif",
    )
    besoin_financement = True

    def prepare(self, candles, cfg, extras=None):
        from .funding import events_to_indices, percentile_events

        self._indices = set()
        funding = (extras or {}).get("funding")
        if funding:
            _, highs = percentile_events(funding)
            self._indices = events_to_indices(candles, highs)

    def fires(self, i):
        return i in self._indices

    @property
    def warmup(self):
        return 60


# Liste FIGEE des candidats. Ne pas l'allonger apres avoir vu des resultats :
# chaque ajout apres coup invalide la correction statistique.
#
# Serie 2 (financement) ajoutee APRES l'echec complet de la serie 1 sur les
# signaux de prix. C'est une nouvelle serie pre-enregistree, avec sa propre
# correction — pas un elargissement opportuniste de la precedente. La reserve
# reste intacte et servira d'arbitre final, une seule fois.
CANDIDATS: List[Callable[[], Signal]] = [
    EmaCross,
    RsiRebound,
    DonchianBreakout,
    DipBelowTrend,
    MomentumFlip,
    FundingCapitulation,
    FundingEuphoria,
]


def build_all(avec_financement: bool = True) -> List[Signal]:
    return [
        klass() for klass in CANDIDATS
        if avec_financement or not klass.besoin_financement
    ]


def by_key(key: str) -> Signal:
    for klass in CANDIDATS:
        if klass.spec.key == key:
            return klass()
    dispo = ", ".join(k.spec.key for k in CANDIDATS)
    raise ValueError(f"Signal inconnu : {key}. Disponibles : {dispo}")
