"""Le journal de forward testing doit etre append seul et incorruptible.

Si ces tests passent, on sait que le journal ne peut pas etre reecrit
apres coup — c'est toute sa raison d'etre.
"""

import json
import tempfile
import unittest
from pathlib import Path

from bot.engine import RunResult
from bot.forward import ForwardLog
from bot.models import Action, Decision, Trade


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


class TestForwardLog(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.log = ForwardLog(Path(self.dir.name) / "forward_log.jsonl")

    def tearDown(self):
        self.dir.cleanup()

    def test_amorcage_pose_un_point_de_depart_sans_recopier_lhistorique(self):
        rep = self.log.append_new(result_with([decision(1000), decision(2000)]), "X", "1h")
        self.assertTrue(rep.first_run)
        self.assertEqual(rep.decisions_added, 0)
        lignes = [l for l in self.log.path.read_text().splitlines() if l.strip()]
        self.assertEqual(len(lignes), 1, "seule la ligne meta doit exister")
        self.assertEqual(self.log.stats()["decisions_live"], 0)
        self.assertEqual(self.log.last_decision_ts(), 2000)

    def test_seules_les_nouvelles_decisions_sont_ajoutees(self):
        self.log.append_new(result_with([decision(1000), decision(2000)]), "X", "1h")
        rep = self.log.append_new(
            result_with([decision(1000), decision(2000), decision(3000)]), "X", "1h"
        )
        self.assertEqual(rep.decisions_added, 1)
        self.assertEqual(self.log.stats()["decisions_live"], 1)

    def test_rien_de_nouveau_najoute_rien(self):
        self.log.append_new(result_with([decision(1000)]), "X", "1h")
        before = self.log.path.read_text()
        rep = self.log.append_new(result_with([decision(1000)]), "X", "1h")
        self.assertEqual(rep.decisions_added, 0)
        self.assertEqual(self.log.path.read_text(), before)

    def test_le_passe_nest_jamais_reecrit(self):
        """Meme si le bot recalcule une decision differemment, la ligne
        d'origine reste intacte dans le fichier."""
        self.log.append_new(result_with([decision(1000)]), "X", "1h")
        self.log.append_new(
            result_with([decision(1000), decision(2000, Action.BUY, "achat")]), "X", "1h"
        )
        self.log.append_new(
            result_with([decision(1000), decision(2000, Action.SELL, "vente")]), "X", "1h"
        )
        lignes = [json.loads(l) for l in self.log.path.read_text().splitlines() if l.strip()]
        originales = [r for r in lignes if r.get("type") == "decision" and r["ts"] == 2000]
        self.assertEqual(len(originales), 1)
        self.assertEqual(originales[0]["action"], "BUY")

    def test_une_divergence_est_detectee_et_journalisee(self):
        self.log.append_new(result_with([decision(1000)]), "X", "1h")
        self.log.append_new(
            result_with([decision(1000), decision(2000, Action.BUY, "achat")]), "X", "1h"
        )
        rep = self.log.append_new(
            result_with([decision(1000), decision(2000, Action.SELL, "vente"),
                         decision(3000)]), "X", "1h"
        )
        self.assertEqual(rep.divergences, 1)
        self.assertEqual(self.log.stats()["divergences"], 1)

    def test_les_trades_sont_journalises(self):
        self.log.append_new(result_with([decision(1000)]), "X", "1h")
        rep = self.log.append_new(
            result_with([decision(1000), decision(2000)], [trade(2000)]), "X", "1h"
        )
        self.assertEqual(rep.trades_added, 1)
        self.assertEqual(self.log.stats()["trades_live"], 1)

    def test_une_ligne_corrompue_ne_detruit_pas_le_journal(self):
        self.log.append_new(result_with([decision(1000)]), "X", "1h")
        self.log.append_new(result_with([decision(1000), decision(2000)]), "X", "1h")
        with self.log.path.open("a", encoding="utf-8") as fh:
            fh.write("{ceci n'est pas du json\n")
        self.assertEqual(self.log.stats()["decisions_live"], 1)

    def test_journal_absent(self):
        self.assertFalse(self.log.exists())
        self.assertIsNone(self.log.last_decision_ts())

    def test_pnl_total_cumule_les_trades_en_direct(self):
        self.log.append_new(result_with([decision(1000)]), "X", "1h")
        self.log.append_new(
            result_with([decision(1000), decision(2000), decision(3000)],
                        [trade(2000, pnl=10.0), trade(3000, pnl=-4.0)]),
            "X", "1h",
        )
        s = self.log.stats()
        self.assertAlmostEqual(s["pnl_total"], 6.0)
        self.assertEqual(s["trades_gagnants"], 1)
        self.assertEqual(s["trades_perdants"], 1)

    def test_since_ts_restreint_aux_decisions_et_trades_apres_ce_moment(self):
        self.log.append_new(result_with([decision(1000)]), "X", "1h")
        self.log.append_new(
            result_with(
                [decision(1000), decision(2000), decision(3000), decision(4000)],
                [trade(2000, pnl=10.0), trade(4000, pnl=-4.0)],
            ),
            "X", "1h",
        )
        # Sans filtre : tout ce qui est en direct.
        self.assertEqual(self.log.stats()["decisions_live"], 3)
        self.assertEqual(self.log.stats()["trades_live"], 2)
        self.assertAlmostEqual(self.log.stats()["pnl_total"], 6.0)

        # Avec since_ts=3000 : seules les lignes a partir de 3000 comptent.
        s = self.log.stats(since_ts=3000)
        self.assertEqual(s["decisions_live"], 2)  # ts 3000 et 4000
        self.assertEqual(s["trades_live"], 1)      # exit_ts 4000 seulement
        self.assertAlmostEqual(s["pnl_total"], -4.0)

        # since_ts au-dela de tout : fenetre vide, pas d'erreur.
        s_vide = self.log.stats(since_ts=10_000)
        self.assertEqual(s_vide["decisions_live"], 0)
        self.assertEqual(s_vide["trades_live"], 0)
        self.assertAlmostEqual(s_vide["pnl_total"], 0.0)

    def test_live_trades_et_live_decisions_sont_tries_et_filtres(self):
        self.log.append_new(result_with([decision(1000)]), "X", "1h")
        self.log.append_new(
            result_with(
                [decision(1000), decision(3000), decision(2000)],
                [trade(4000, pnl=1.0), trade(3000, pnl=2.0)],
            ),
            "X", "1h",
        )
        decs = self.log.live_decisions()
        self.assertEqual([d["ts"] for d in decs], [2000, 3000])
        trades = self.log.live_trades()
        self.assertEqual([t["exit_ts"] for t in trades], [3000, 4000])

        decs_filtres = self.log.live_decisions(since_ts=3000)
        self.assertEqual([d["ts"] for d in decs_filtres], [3000])
        trades_filtres = self.log.live_trades(since_ts=3500)
        self.assertEqual([t["exit_ts"] for t in trades_filtres], [4000])


if __name__ == "__main__":
    unittest.main()
