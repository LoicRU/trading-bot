"""Systeme de recompense / malus.

Deux sources de score :

1. LES TRADES FERMES -> score = R-multiple (resultat / risque initial).
   Perdre 50 EUR en ayant risque 50 EUR vaut -1. Gagner 150 en ayant
   risque 50 vaut +3. Cette normalisation est indispensable : sans elle,
   le bot apprendrait juste que "gros trade = gros score" et
   augmenterait la taille jusqu'a exploser.

2. LES ABSTENTIONS -> evaluees APRES COUP, avec un delai.
   C'est le coeur du probleme que tu as identifie. Si ne rien faire
   rapporte 0 et ne coute rien, ne jamais trader devient la strategie
   optimale et le bot se fige. Donc :
     - s'abstenir alors que le marche offrait un beau mouvement
       haussier  -> MALUS (cout d'opportunite),
     - s'abstenir alors que le marche a chute
       -> BONUS (perte evitee),
     - s'abstenir dans un marche qui n'a rien fait
       -> petit BONUS de patience.

AVERTISSEMENT IMPORTANT SUR LE FUTUR :
L'evaluation d'une abstention regarde les bougies POSTERIEURES a la
decision. C'est legitime pour NOTER une decision deja prise, ca ne
l'est absolument pas pour EN PRENDRE une. Ce module n'est donc jamais
appele par la strategie, uniquement par le moteur, et uniquement une
fois que horizon_bars bougies se sont ecoulees. Ne branche jamais
ces scores dans une decision d'entree sans un decalage temporel
equivalent, sinon tu te fabriques un bot qui triche.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .config import ScoringConfig
from .models import Action, Candle, Decision, Trade


@dataclass
class ScoreEntry:
    ts: int                 # horodatage de la decision notee
    kind: str               # "trade" ou "abstention"
    value: float
    detail: str
    evaluated_ts: Optional[int] = None  # quand la note a pu etre attribuee


class Scorer:
    def __init__(self, cfg: ScoringConfig, risk_stop_mult: float = 2.0) -> None:
        self.cfg = cfg
        self.risk_stop_mult = risk_stop_mult
        self.entries: List[ScoreEntry] = []
        self._pending: List[tuple[int, Decision, Optional[float]]] = []
        self._last_registered: Optional[int] = None
        self.skipped_overlapping = 0

    # ------------------------------------------------------------------
    # Trades
    # ------------------------------------------------------------------
    def score_trade(self, trade: Trade) -> float:
        r = trade.r_multiple
        capped = max(-self.cfg.cap_r, min(self.cfg.cap_r, r))
        detail = (
            f"trade ferme ({trade.exit_reason}) : {trade.pnl:+.2f} "
            f"pour {trade.risk_amount:.2f} risques = {r:+.2f} R"
        )
        trade.score = capped
        self.entries.append(ScoreEntry(ts=trade.exit_ts, kind="trade", value=capped, detail=detail))
        return capped

    # ------------------------------------------------------------------
    # Abstentions
    # ------------------------------------------------------------------
    def register_abstention(self, index: int, decision: Decision, atr_value: Optional[float]) -> None:
        """Met l'abstention en file d'attente ; elle sera notee plus tard.

        Les fenetres d'evaluation ne se chevauchent JAMAIS. Sans cette
        regle, une meme hausse ratee serait comptee autant de fois qu'il
        y a de bougies dans l'horizon (24 fois avec les valeurs par
        defaut), et le cout d'opportunite ecraserait completement le
        score des trades. On ne note donc qu'une abstention tous les
        horizon_bars : chaque mouvement rate est compte une fois, une
        seule.
        """
        if self._last_registered is not None and index - self._last_registered < self.cfg.horizon_bars:
            self.skipped_overlapping += 1
            return
        self._last_registered = index
        self._pending.append((index, decision, atr_value))

    def evaluate_pending(self, current_index: int, candles: Sequence[Candle]) -> List[ScoreEntry]:
        """Note toutes les abstentions dont l'horizon est ecoule."""
        produced: List[ScoreEntry] = []
        still_pending = []
        for index, decision, atr_value in self._pending:
            target = index + self.cfg.horizon_bars
            if target > current_index or target >= len(candles):
                still_pending.append((index, decision, atr_value))
                continue
            entry = self._evaluate_one(index, target, decision, atr_value, candles)
            if entry is not None:
                self.entries.append(entry)
                produced.append(entry)
        self._pending = still_pending
        return produced

    def _evaluate_one(
        self,
        index: int,
        target: int,
        decision: Decision,
        atr_value: Optional[float],
        candles: Sequence[Candle],
    ) -> Optional[ScoreEntry]:
        start = candles[index].close
        end = candles[target].close
        if start <= 0:
            return None

        move = end - start

        # On convertit le mouvement rate en "R" : combien de fois le risque
        # qu'un trade aurait pris (stop = risk_stop_mult x ATR) ce mouvement
        # represente-t-il ? Sans ATR exploitable, on ne note pas.
        if not atr_value or atr_value <= 0:
            return None
        one_r = self.risk_stop_mult * atr_value
        move_r = move / one_r

        threshold = self.cfg.opportunity_threshold_r
        cap = self.cfg.cap_r

        if move_r >= threshold:
            # Le marche est monte franchement et le bot n'etait pas la.
            value = -self.cfg.opportunity_weight * min(move_r, cap)
            detail = (
                f"abstention penalisee : le marche a pris {move / start:+.2%} "
                f"({move_r:+.2f} R) en {self.cfg.horizon_bars} bougies - occasion ratee"
            )
        elif move_r <= -threshold:
            # Le marche a chute : ne pas etre entre etait le bon choix.
            value = self.cfg.avoided_weight * min(abs(move_r), cap)
            detail = (
                f"abstention recompensee : le marche a perdu {move / start:.2%} "
                f"({move_r:+.2f} R) - perte evitee"
            )
        else:
            # Marche sans direction : la patience vaut un petit bonus.
            value = self.cfg.patience_reward
            detail = (
                f"abstention neutre recompensee : marche sans direction "
                f"({move / start:+.2%}, {move_r:+.2f} R) - patience justifiee"
            )

        return ScoreEntry(
            ts=decision.ts,
            kind="abstention",
            value=round(value, 4),
            detail=detail,
            evaluated_ts=candles[target].ts,
        )

    def flush_unevaluated(self) -> int:
        """Nombre d'abstentions restees sans note (horizon non ecoule)."""
        return len(self._pending)

    # ------------------------------------------------------------------
    # Agregats
    # ------------------------------------------------------------------
    @property
    def total(self) -> float:
        return round(sum(e.value for e in self.entries), 4)

    def total_between(self, start_ts: int, end_ts: int) -> float:
        return round(
            sum(e.value for e in self.entries if start_ts <= e.ts <= end_ts), 4
        )

    def breakdown(self) -> dict:
        trades = [e for e in self.entries if e.kind == "trade"]
        absts = [e for e in self.entries if e.kind == "abstention"]
        count = len(self.entries)
        return {
            "total": self.total,
            "trade_score": round(sum(e.value for e in trades), 4),
            "abstention_score": round(sum(e.value for e in absts), 4),
            "trade_count": len(trades),
            "abstention_count": len(absts),
            "abstentions_penalisees": sum(1 for e in absts if e.value < 0),
            "abstentions_recompensees": sum(1 for e in absts if e.value > 0),
            "moyenne_par_note": round(self.total / count, 4) if count else 0.0,
            "fenetres_ignorees": self.skipped_overlapping,
            "non_evaluees": self.flush_unevaluated(),
        }
