"""Persistance SQLite.

Tout est trace : les trades, mais aussi TOUTES les decisions, y compris
les abstentions et leur motif. C'est ce qui permettra de repondre a la
question qui compte vraiment quand une serie se passe mal : "pourquoi
le bot a-t-il fait ca ?".

Le format est volontairement plat et lisible : si tu passes un jour a
de l'argent reel, cette base est deja ce qu'il faut pour justifier
chaque operation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .models import Decision, Trade
from .scoring import ScoreEntry

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at   TEXT NOT NULL,
    mode         TEXT NOT NULL,
    adapter      TEXT NOT NULL,
    symbol       TEXT NOT NULL,
    timeframe    TEXT NOT NULL,
    segment      TEXT,
    config_json  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id),
    ts           INTEGER NOT NULL,
    symbol       TEXT NOT NULL,
    action       TEXT NOT NULL,
    reason       TEXT NOT NULL,
    price        REAL NOT NULL,
    context_json TEXT
);

CREATE TABLE IF NOT EXISTS trades (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id),
    symbol       TEXT NOT NULL,
    qty          REAL NOT NULL,
    entry_ts     INTEGER NOT NULL,
    entry_price  REAL NOT NULL,
    exit_ts      INTEGER NOT NULL,
    exit_price   REAL NOT NULL,
    fees         REAL NOT NULL,
    pnl          REAL NOT NULL,
    pnl_pct      REAL NOT NULL,
    risk_amount  REAL NOT NULL,
    r_multiple   REAL NOT NULL,
    entry_reason TEXT,
    exit_reason  TEXT,
    score        REAL
);

CREATE TABLE IF NOT EXISTS scores (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       INTEGER NOT NULL REFERENCES runs(id),
    ts           INTEGER NOT NULL,
    kind         TEXT NOT NULL,
    value        REAL NOT NULL,
    detail       TEXT,
    evaluated_ts INTEGER
);

CREATE TABLE IF NOT EXISTS snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        INTEGER NOT NULL REFERENCES runs(id),
    ts            INTEGER NOT NULL,
    day           TEXT NOT NULL,
    equity        REAL NOT NULL,
    cash          REAL NOT NULL,
    position_qty  REAL NOT NULL,
    price         REAL NOT NULL,
    cumulative_score REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_run_ts ON decisions(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_trades_run_exit  ON trades(run_id, exit_ts);
CREATE INDEX IF NOT EXISTS idx_scores_run_ts    ON scores(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_snapshots_run_ts ON snapshots(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_snapshots_day    ON snapshots(run_id, day);
"""


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # ------------------------------------------------------------------
    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.conn.commit()
        self.close()

    # ------------------------------------------------------------------
    def create_run(
        self,
        started_at: str,
        mode: str,
        adapter: str,
        symbol: str,
        timeframe: str,
        config: Dict[str, Any],
        segment: Optional[str] = None,
    ) -> int:
        cur = self.conn.execute(
            "INSERT INTO runs (started_at, mode, adapter, symbol, timeframe, segment, config_json)"
            " VALUES (?,?,?,?,?,?,?)",
            (started_at, mode, adapter, symbol, timeframe, segment, json.dumps(config)),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def insert_decisions(self, run_id: int, decisions: Iterable[Decision]) -> None:
        self.conn.executemany(
            "INSERT INTO decisions (run_id, ts, symbol, action, reason, price, context_json)"
            " VALUES (?,?,?,?,?,?,?)",
            [
                (run_id, d.ts, d.symbol, d.action.value, d.reason, d.price, json.dumps(d.context))
                for d in decisions
            ],
        )
        self.conn.commit()

    def insert_trades(self, run_id: int, trades: Iterable[Trade]) -> None:
        self.conn.executemany(
            "INSERT INTO trades (run_id, symbol, qty, entry_ts, entry_price, exit_ts, exit_price,"
            " fees, pnl, pnl_pct, risk_amount, r_multiple, entry_reason, exit_reason, score)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    run_id, t.symbol, t.qty, t.entry_ts, t.entry_price, t.exit_ts, t.exit_price,
                    t.fees, t.pnl, t.pnl_pct, t.risk_amount, t.r_multiple,
                    t.entry_reason, t.exit_reason, t.score,
                )
                for t in trades
            ],
        )
        self.conn.commit()

    def insert_scores(self, run_id: int, entries: Iterable[ScoreEntry]) -> None:
        self.conn.executemany(
            "INSERT INTO scores (run_id, ts, kind, value, detail, evaluated_ts) VALUES (?,?,?,?,?,?)",
            [(run_id, e.ts, e.kind, e.value, e.detail, e.evaluated_ts) for e in entries],
        )
        self.conn.commit()

    def insert_snapshots(self, run_id: int, rows: Sequence[Dict[str, Any]]) -> None:
        self.conn.executemany(
            "INSERT INTO snapshots (run_id, ts, day, equity, cash, position_qty, price,"
            " cumulative_score) VALUES (?,?,?,?,?,?,?,?)",
            [
                (
                    run_id, r["ts"], r["day"], r["equity"], r["cash"],
                    r["position_qty"], r["price"], r["cumulative_score"],
                )
                for r in rows
            ],
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Lectures
    # ------------------------------------------------------------------
    def latest_run_id(self) -> Optional[int]:
        row = self.conn.execute("SELECT id FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        return int(row["id"]) if row else None

    def run(self, run_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()

    def trades_between(self, run_id: int, start_ts: int, end_ts: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM trades WHERE run_id=? AND exit_ts BETWEEN ? AND ? ORDER BY exit_ts",
            (run_id, start_ts, end_ts),
        ).fetchall()

    def decisions_between(
        self, run_id: int, start_ts: int, end_ts: int, action: Optional[str] = None
    ) -> List[sqlite3.Row]:
        sql = "SELECT * FROM decisions WHERE run_id=? AND ts BETWEEN ? AND ?"
        args: List[Any] = [run_id, start_ts, end_ts]
        if action:
            sql += " AND action=?"
            args.append(action)
        sql += " ORDER BY ts"
        return self.conn.execute(sql, args).fetchall()

    def scores_between(self, run_id: int, start_ts: int, end_ts: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM scores WHERE run_id=? AND ts BETWEEN ? AND ? ORDER BY ts",
            (run_id, start_ts, end_ts),
        ).fetchall()

    def snapshots(self, run_id: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM snapshots WHERE run_id=? ORDER BY ts", (run_id,)
        ).fetchall()

    def all_trades(self, run_id: int) -> List[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM trades WHERE run_id=? ORDER BY exit_ts", (run_id,)
        ).fetchall()
