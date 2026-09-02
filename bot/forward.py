"""Journal de forward testing : la preuve que les decisions ont ete prises
AVANT de connaitre la suite.

Pourquoi ce fichier existe
--------------------------
Le bot rejoue tout son historique a chaque execution. C'est excellent pour
la fiabilite, mais ca cree un probleme de credibilite : si tu lances le bot
une fois par soir, il recalcule les decisions de la journee ecoulee, et
RIEN ne prouve qu'il les aurait prises au bon moment. Un backtest deguise
en suivi temps reel, autrement dit.

Ce journal resout ca. A chaque passage horaire, seules les decisions
NOUVELLES sont ajoutees, horodatees a l'heure reelle de leur enregistrement,
et jamais modifiees ensuite. Le fichier est en append seul, versionne par
git : l'historique des commits date chaque ligne de facon independante de
ce que le bot pourrait raconter apres coup.

C'est ce qui transforme "j'ai lance mon bot pendant six mois" en un vrai
forward test, celui de l'etape 3 de la sequence avant l'argent reel.

Controle d'integrite
--------------------
A chaque passage, les decisions deja journalisees sont recalculees et
comparees a ce qui avait ete ecrit. Un desaccord signifie que quelque
chose a bouge : parametres modifies, donnees de marche revisees par
l'exchange, ou bug. La divergence est enregistree telle quelle, jamais
masquee. Un forward test dont les parametres ont change en cours de route
n'est plus un forward test, et il vaut mieux le savoir.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from .engine import RunResult
from .models import Decision, Trade


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc).isoformat(timespec="minutes")


@dataclass
class AppendReport:
    decisions_added: int = 0
    trades_added: int = 0
    divergences: int = 0
    first_run: bool = False
    last_ts: Optional[int] = None

    def summary(self) -> str:
        if self.first_run:
            return (
                "journal initialise : point de depart pose sur l'historique existant. "
                "Le forward test commence au prochain passage."
            )
        bits = [f"{self.decisions_added} decision(s) enregistree(s) en direct"]
        if self.trades_added:
            bits.append(f"{self.trades_added} trade(s)")
        if self.divergences:
            bits.append(f"ATTENTION : {self.divergences} divergence(s) detectee(s)")
        return ", ".join(bits)


class ForwardLog:
    """Journal append seul, au format JSON Lines (une ligne = un evenement)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Lecture
    # ------------------------------------------------------------------
    def exists(self) -> bool:
        return self.path.exists()

    def read(self) -> Iterator[Dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Une ligne corrompue ne doit pas faire perdre tout le journal.
                    continue

    def _logged_decisions(self) -> Dict[int, Dict[str, Any]]:
        return {
            rec["ts"]: rec for rec in self.read() if rec.get("type") == "decision"
        }

    def last_decision_ts(self) -> Optional[int]:
        """Horodatage au-dela duquel une decision est consideree comme nouvelle.

        C'est le maximum entre le point de depart pose a la creation du
        journal et la derniere decision reellement enregistree.
        """
        best: Optional[int] = None
        for rec in self.read():
            ts = None
            if rec.get("type") == "decision":
                ts = int(rec["ts"])
            elif rec.get("type") == "meta" and rec.get("watermark_ts") is not None:
                ts = int(rec["watermark_ts"])
            if ts is not None and (best is None or ts > best):
                best = ts
        return best

    def last_trade_ts(self) -> Optional[int]:
        best: Optional[int] = None
        for rec in self.read():
            ts = None
            if rec.get("type") == "trade":
                ts = int(rec["exit_ts"])
            elif rec.get("type") == "meta" and rec.get("trade_watermark_ts") is not None:
                ts = int(rec["trade_watermark_ts"])
            if ts is not None and (best is None or ts > best):
                best = ts
        return best

    # ------------------------------------------------------------------
    # Ecriture (append seul, jamais de reecriture)
    # ------------------------------------------------------------------
    def _append(self, records: Sequence[Dict[str, Any]]) -> None:
        if not records:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    def append_new(self, result: RunResult, symbol: str, timeframe: str) -> AppendReport:
        """Ajoute uniquement ce qui n'a jamais ete journalise."""
        first_run = not self.path.exists()
        known = self._logged_decisions()
        last_dec = self.last_decision_ts()
        last_trd = self.last_trade_ts()
        recorded_at = _now_iso()
        report = AppendReport(first_run=first_run)
        out: List[Dict[str, Any]] = []

        if first_run:
            # On ne recopie PAS l'historique rejoue dans le journal : il est
            # deja reproductible a partir des donnees de marche, et le
            # commiter chaque heure n'aurait aucun sens. On pose seulement
            # un point de depart. Tout ce qui suivra sera du temps reel.
            wm = max((d.ts for d in result.decisions), default=None)
            twm = max((t.exit_ts for t in result.trades), default=None)
            self._append([
                {
                    "type": "meta",
                    "event": "journal_cree",
                    "recorded_at": recorded_at,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "watermark_ts": wm,
                    "watermark_iso": _iso(wm) if wm else None,
                    "trade_watermark_ts": twm,
                    "note": (
                        "Point de depart du forward test. Tout ce qui figure "
                        "au-dessus de cette ligne n'existe pas : l'historique "
                        "anterieur est rejoue depuis les donnees de marche et ne "
                        "constitue pas une preuve. Seules les lignes qui suivent, "
                        "ajoutees passage apres passage, comptent."
                    ),
                }
            ])
            report.last_ts = wm
            return report

        # --- controle d'integrite sur ce qui existe deja ----------------
        for d in result.decisions:
            prev = known.get(d.ts)
            if prev is None:
                continue
            if prev.get("action") != d.action.value:
                report.divergences += 1
                out.append(
                    {
                        "type": "divergence",
                        "ts": d.ts,
                        "iso": _iso(d.ts),
                        "recorded_at": recorded_at,
                        "journalise": prev.get("action"),
                        "recalcule": d.action.value,
                        "motif_journalise": prev.get("reason"),
                        "motif_recalcule": d.reason,
                        "note": (
                            "La decision recalculee ne correspond plus a celle "
                            "enregistree. Parametres modifies, donnees revisees "
                            "ou bug : le forward test n'est plus continu."
                        ),
                    }
                )

        # --- nouvelles decisions ----------------------------------------
        for d in result.decisions:
            if last_dec is not None and d.ts <= last_dec:
                continue
            out.append(self._decision_record(d, recorded_at, live=True))
            report.decisions_added += 1
            report.last_ts = d.ts

        # --- nouveaux trades --------------------------------------------
        for t in result.trades:
            if last_trd is not None and t.exit_ts <= last_trd:
                continue
            out.append(self._trade_record(t, recorded_at, live=True))
            report.trades_added += 1

        self._append(out)
        return report

    @staticmethod
    def _decision_record(d: Decision, recorded_at: str, live: bool) -> Dict[str, Any]:
        return {
            "type": "decision",
            "ts": d.ts,
            "iso": _iso(d.ts),
            "recorded_at": recorded_at,
            "live": live,
            "symbol": d.symbol,
            "action": d.action.value,
            "reason": d.reason,
            "price": round(d.price, 8),
        }

    @staticmethod
    def _trade_record(t: Trade, recorded_at: str, live: bool) -> Dict[str, Any]:
        return {
            "type": "trade",
            "exit_ts": t.exit_ts,
            "iso": _iso(t.exit_ts),
            "recorded_at": recorded_at,
            "live": live,
            "symbol": t.symbol,
            "entry_ts": t.entry_ts,
            "entry_price": round(t.entry_price, 8),
            "exit_price": round(t.exit_price, 8),
            "qty": round(t.qty, 10),
            "pnl": round(t.pnl, 4),
            "r_multiple": round(t.r_multiple, 4),
            "exit_reason": t.exit_reason,
        }

    # ------------------------------------------------------------------
    # Statistiques
    # ------------------------------------------------------------------
    def stats(self) -> Dict[str, Any]:
        live_decisions = 0
        live_trades = 0
        divergences = 0
        first_live: Optional[str] = None
        last_live: Optional[str] = None
        started_at: Optional[str] = None

        for rec in self.read():
            kind = rec.get("type")
            if kind == "meta" and rec.get("event") == "journal_cree":
                started_at = rec.get("recorded_at")
            elif kind == "divergence":
                divergences += 1
            elif kind == "decision":
                live_decisions += 1
                stamp = rec.get("recorded_at")
                first_live = first_live or stamp
                last_live = stamp
            elif kind == "trade":
                live_trades += 1

        # La duree se compte depuis la creation du journal, pas depuis la
        # premiere decision : un bot qui n'a rien decide pendant trois
        # semaines a quand meme trois semaines de forward test.
        jours = 0.0
        debut = started_at or first_live
        if debut and last_live:
            try:
                d0 = datetime.fromisoformat(debut)
                d1 = datetime.fromisoformat(last_live)
                jours = round((d1 - d0).total_seconds() / 86400, 1)
            except ValueError:
                jours = 0.0

        return {
            "decisions_live": live_decisions,
            "trades_live": live_trades,
            "divergences": divergences,
            "debut_live": debut,
            "dernier_live": last_live,
            "jours_de_forward": jours,
        }
