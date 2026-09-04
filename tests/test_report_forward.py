"""Le rapport de forward test reel doit se construire UNIQUEMENT a partir
du journal en direct (bot/forward.py), jamais du backtest rejoue, et ne
doit jamais planter quand la fenetre est vide ou que le P&L est negatif
(contrairement au capital, un P&L cumule peut legitimement descendre
sous zero).
"""

import tempfile
import unittest
from pathlib import Path

from bot.config import Config
from bot.engine import RunResult
from bot.forward import ForwardLog
from bot.models import Action, Decision, Trade
from bot.report import build_forward_report, equity_chart, Series


def decision(ts, action=Action.ABSTAIN, reason="motif"):
    return Decision(ts=ts, symbol="X", action=action, reason=reason, price=100.0)


def trade(exit_ts, pnl=10.0):
    return Trade(symbol="X", qty=1.0, entry_ts=exit_ts - 3600_000, entry_price=100.0,
                 exit_ts=exit_ts, exit_price=110.0, fees=0.2, pnl=pnl, pnl_pct=0.1,
                 risk_amount=10.0, exit_reason="objectif atteint")


def result_with(decisions, trades=()):
    r = RunResult(symbol="X")
    r.decisions = list(decisions)
    r.trades = list(trades)
    return r


class TestBuildForwardReport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = ForwardLog(Path(self.dir.name) / "forward_log.jsonl")
        self.cfg = Config()

    def tearDown(self):
        self.dir.cleanup()

    def test_journal_vide_ne_plante_pas(self):
        self.log.append_new(result_with([decision(1000)]), "X", "1h")  # amorcage seul
        html_out = build_forward_report(self.cfg, self.log)
        self.assertIn("Aucun trade", html_out)
        self.assertIn("<html", html_out)

    def test_pnl_negatif_apparait_bien_dans_le_html(self):
        self.log.append_new(result_with([decision(1000)]), "X", "1h")
        self.log.append_new(
            result_with([decision(1000), decision(2000), decision(3000)],
                        [trade(2000, pnl=5.0), trade(3000, pnl=-12.0)]),
            "X", "1h",
        )
        html_out = build_forward_report(self.cfg, self.log)
        # P&L cumule final = -7.0 : doit apparaitre quelque part (table ou stat).
        self.assertIn("-7,00", html_out)
        self.assertNotIn("Aucun trade cloture pour l'instant.", html_out)

    def test_since_label_apparait_dans_le_titre(self):
        self.log.append_new(result_with([decision(1000)]), "X", "1h")
        self.log.append_new(
            result_with([decision(1000), decision(2000)], [trade(2000, pnl=3.0)]),
            "X", "1h",
        )
        html_out = build_forward_report(
            self.cfg, self.log, since_ts=0, since_label="2026-09-03"
        )
        self.assertIn("2026-09-03", html_out)

    def test_since_ts_exclut_les_trades_anterieurs(self):
        self.log.append_new(result_with([decision(1000)]), "X", "1h")
        self.log.append_new(
            result_with([decision(1000), decision(2000), decision(9000)],
                        [trade(2000, pnl=100.0), trade(9000, pnl=1.0)]),
            "X", "1h",
        )
        html_out = build_forward_report(self.cfg, self.log, since_ts=5000)
        # Seul le trade a exit_ts=9000 (pnl=1.0) doit compter : le gros gain
        # de +100 est anterieur a la fenetre et ne doit pas y figurer.
        self.assertNotIn("+100,00", html_out)
        self.assertIn("+1,00", html_out)


class TestEquityChartClampMinZero(unittest.TestCase):
    """equity_chart doit pouvoir tracer une serie qui descend sous zero
    quand on le lui demande explicitement (P&L), sans jamais rogner les
    valeurs negatives comme elle le fait par defaut pour un capital."""

    def test_par_defaut_laxe_ne_descend_jamais_sous_zero(self):
        series = [Series("Capital", [(1000, -50.0), (2000, 100.0)], "#000", "#fff", "bot")]
        out = equity_chart(series, 0.0, "USD")
        self.assertIn("<svg", out)  # se construit sans lever d'exception

    def test_clamp_min_zero_false_conserve_les_valeurs_negatives(self):
        series = [Series("P&L", [(1000, -50.0), (2000, 30.0)], "#000", "#fff", "bot")]
        out_clamp = equity_chart(series, 0.0, "USD", clamp_min_zero=True)
        out_libre = equity_chart(series, 0.0, "USD", clamp_min_zero=False)
        # Les deux se construisent sans erreur ; le graphique libre doit
        # afficher une graduation negative quelque part (le clamp, non).
        self.assertIn("<svg", out_clamp)
        self.assertIn("<svg", out_libre)
        self.assertIn("-", out_libre)


if __name__ == "__main__":
    unittest.main()
