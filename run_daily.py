#!/usr/bin/env python3
"""Point d'entree pour l'execution automatique quotidienne.

A brancher sur cron (Linux/macOS) ou sur le planificateur de taches
(Windows). Exemple cron, tous les soirs a 22h05 heure locale :

    5 22 * * *  cd /chemin/vers/trading_bot && /usr/bin/python3 run_daily.py

La sortie est ecrite dans logs/daily.log en plus de la sortie standard,
pour qu'une execution ratee laisse une trace consultable.
"""

from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bot.cli import main  # noqa: E402

LOG_DIR = ROOT / "logs"


class Tee:
    """Ecrit a la fois sur la sortie standard et dans le fichier de log."""

    def __init__(self, stream, handle):
        self.stream = stream
        self.handle = handle

    def write(self, text: str) -> int:
        self.stream.write(text)
        self.handle.write(text)
        return len(text)

    def flush(self) -> None:
        self.stream.flush()
        self.handle.flush()


if __name__ == "__main__":
    LOG_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with (LOG_DIR / "daily.log").open("a", encoding="utf-8") as fh:
        fh.write(f"\n========== execution du {stamp} ==========\n")
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = Tee(out, fh), Tee(err, fh)
        try:
            code = main(["daily"] + sys.argv[1:])
        except Exception:  # noqa: BLE001 - on veut la trace complete dans le log
            traceback.print_exc()
            code = 99
        finally:
            sys.stdout, sys.stderr = out, err
    raise SystemExit(code)
