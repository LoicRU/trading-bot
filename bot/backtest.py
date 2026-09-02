"""Backtest et decoupage entrainement / validation / reserve.

Le decoupage est CHRONOLOGIQUE, jamais aleatoire. Melanger les dates
sur des donnees financieres revient a laisser le bot voir le futur.

Le troisieme bloc, la RESERVE, est protege par un verrou sur disque.
La regle : tu ne le testes qu'une seule fois, tout a la fin, quand la
strategie est figee. Si tu le testes, que tu ajustes un parametre, puis
que tu le retestes, il ne vaut plus rien : tu viens de l'optimiser lui
aussi, sans t'en rendre compte. Le verrou est la pour t'obliger a un
geste conscient, pas pour t'empecher de travailler.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .config import ROOT, Config
from .engine import Engine, RunResult
from .models import Candle
from .strategy import EmaAtrStrategy

SEGMENTS = ("train", "validation", "holdout")
LOCK_FILE = ROOT / ".holdout_used.json"


@dataclass
class Segment:
    name: str
    candles: List[Candle]      # bougies evaluees, precedees de leur chauffe
    start_ts: int
    end_ts: int
    eval_bars: int


def split(candles: Sequence[Candle], cfg: Config) -> Dict[str, Segment]:
    """Coupe la serie en trois blocs successifs dans le temps.

    Chaque bloc recoit en prefixe les bougies necessaires au calcul des
    indicateurs, pour que l'evaluation commence bien au premier jour du
    bloc et pas quelques dizaines de bougies plus tard.
    """
    n = len(candles)
    if n < 200:
        raise ValueError(f"Historique trop court pour un decoupage serieux : {n} bougies.")

    pad = EmaAtrStrategy(cfg.strategy).warmup + 1
    b = cfg.backtest
    i_train_end = int(n * b.train_ratio)
    i_val_end = int(n * (b.train_ratio + b.validation_ratio))

    bounds = {
        "train": (0, i_train_end),
        "validation": (i_train_end, i_val_end),
        "holdout": (i_val_end, n),
    }

    out: Dict[str, Segment] = {}
    for name, (start, end) in bounds.items():
        if end - start < pad + 20:
            raise ValueError(
                f"Bloc '{name}' trop court ({end - start} bougies) pour la periode de "
                f"chauffe requise ({pad}). Augmente history_days ou reduis les periodes d'EMA."
            )
        pad_start = max(0, start - pad)
        chunk = list(candles[pad_start:end])
        out[name] = Segment(
            name=name,
            candles=chunk,
            start_ts=candles[start].ts,
            end_ts=candles[end - 1].ts,
            eval_bars=end - start,
        )
    return out


# ----------------------------------------------------------------------
# Verrou de la reserve
# ----------------------------------------------------------------------
def holdout_used() -> Optional[dict]:
    if LOCK_FILE.exists():
        try:
            return json.loads(LOCK_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {"used_at": "inconnu"}
    return None


def mark_holdout_used(cfg: Config, result: RunResult) -> None:
    LOCK_FILE.write_text(
        json.dumps(
            {
                "used_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "symbol": cfg.market.symbol,
                "timeframe": cfg.market.timeframe,
                "strategy_params": cfg.strategy.__dict__,
                "risk_params": cfg.risk.__dict__,
                "result": {
                    "final_equity": result.final_equity,
                    "total_return": result.metrics.total_return if result.metrics else None,
                    "max_drawdown": result.metrics.max_drawdown if result.metrics else None,
                    "trades": result.metrics.trade_count if result.metrics else None,
                },
                "note": (
                    "Ce bloc a ete consomme. Si tu modifies la strategie apres avoir vu "
                    "ce resultat, il n'est plus une validation independante."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def reset_holdout_lock() -> bool:
    if LOCK_FILE.exists():
        LOCK_FILE.unlink()
        return True
    return False


# ----------------------------------------------------------------------
def run_segment(cfg: Config, segment: Segment, interval_ms: int) -> RunResult:
    engine = Engine(cfg, interval_ms)
    return engine.run(segment.candles, cfg.market.symbol)


def run_backtest(
    cfg: Config,
    candles: Sequence[Candle],
    interval_ms: int,
    segments: Sequence[str] = ("train", "validation"),
    allow_holdout: bool = False,
) -> Dict[str, RunResult]:
    """Lance le backtest sur les blocs demandes.

    'holdout' n'est jamais inclus par defaut, et exige allow_holdout=True
    ainsi qu'un verrou disque non pose.
    """
    parts = split(candles, cfg)
    results: Dict[str, RunResult] = {}

    for name in segments:
        if name not in parts:
            raise ValueError(f"Bloc inconnu : {name}. Choix : {', '.join(SEGMENTS)}")
        if name == "holdout":
            if not allow_holdout:
                raise PermissionError(
                    "Le bloc de reserve est protege. Relance avec --use-holdout, "
                    "et seulement quand la strategie est definitivement figee."
                )
            used = holdout_used()
            if used:
                raise PermissionError(
                    "Le bloc de reserve a DEJA ete consomme le "
                    f"{used.get('used_at')}. Le retester apres avoir vu son resultat "
                    "ne prouve plus rien. Si tu veux vraiment repartir de zero, "
                    "supprime .holdout_used.json en connaissance de cause."
                )
        results[name] = run_segment(cfg, parts[name], interval_ms)
        if name == "holdout":
            mark_holdout_used(cfg, results[name])

    return results
