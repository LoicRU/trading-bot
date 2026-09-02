"""Strategie et scoring testes sur des marches fabriques dont on connait
la bonne reponse d'avance. C'est la seule facon de verifier une regle de
decision : sur des vraies donnees, on ne sait jamais ce qu'on devrait
attendre."""

import unittest

from bot.config import ScoringConfig, StrategyConfig
from bot.models import Action, Candle, Decision, Trade
from bot.scoring import Scorer
from bot.strategy import EmaAtrStrategy


def candles_from(prices, vol=0.01):
    out = []
    for i, p in enumerate(prices):
        out.append(
            Candle(
                ts=i * 3_600_000,
                open=p,
                high=p * (1 + vol),
                low=p * (1 - vol),
                close=p,
                volume=1.0,
            )
        )
    return out


class TestStrategy(unittest.TestCase):
    def setUp(self):
        self.cfg = StrategyConfig(ema_fast=5, ema_slow=12, atr_period=7,
                                  min_atr_pct=0.004, max_atr_pct=0.08)

    def test_marche_totalement_plat_ne_declenche_aucun_achat(self):
        strat = EmaAtrStrategy(self.cfg)
        candles = candles_from([100.0] * 80, vol=0.0005)
        strat.prepare(candles)
        actions = {
            strat.decide(i, candles, has_position=False).action
            for i in range(strat.warmup, len(candles))
        }
        self.assertEqual(actions, {Action.ABSTAIN})

    def test_signal_dachat_sur_retournement_haussier(self):
        strat = EmaAtrStrategy(self.cfg)
        prices = [100 - i * 0.8 for i in range(40)] + [68 + i * 1.6 for i in range(40)]
        candles = candles_from(prices, vol=0.012)
        strat.prepare(candles)
        actions = [
            strat.decide(i, candles, has_position=False).action
            for i in range(strat.warmup, len(candles))
        ]
        self.assertIn(Action.BUY, actions)

    def test_filtre_de_volatilite_bloque_le_marche_trop_plat(self):
        cfg = StrategyConfig(ema_fast=5, ema_slow=12, atr_period=7,
                             min_atr_pct=0.50, max_atr_pct=0.90)  # seuil absurde exprès
        strat = EmaAtrStrategy(cfg)
        prices = [100 - i * 0.8 for i in range(40)] + [68 + i * 1.6 for i in range(40)]
        candles = candles_from(prices, vol=0.012)
        strat.prepare(candles)
        decisions = [
            strat.decide(i, candles, has_position=False)
            for i in range(strat.warmup, len(candles))
        ]
        self.assertNotIn(Action.BUY, [d.action for d in decisions])
        self.assertTrue(any("trop plat" in d.reason for d in decisions))

    def test_chaque_decision_est_motivee(self):
        strat = EmaAtrStrategy(self.cfg)
        candles = candles_from([100 + (i % 7) for i in range(80)])
        strat.prepare(candles)
        for i in range(strat.warmup, len(candles)):
            d = strat.decide(i, candles, has_position=False)
            self.assertTrue(d.reason.strip(), "une decision sans motif est inexploitable")


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.cfg = ScoringConfig(horizon_bars=5, opportunity_threshold_r=1.0,
                                 opportunity_weight=0.5, avoided_weight=0.3,
                                 patience_reward=0.05, cap_r=3.0)

    def _abstention(self, prices, atr_value=1.0):
        scorer = Scorer(self.cfg, risk_stop_mult=2.0)
        candles = candles_from(prices)
        decision = Decision(ts=candles[0].ts, symbol="X", action=Action.ABSTAIN,
                            reason="test", price=candles[0].close)
        scorer.register_abstention(0, decision, atr_value)
        produced = scorer.evaluate_pending(len(candles) - 1, candles)
        return produced[0] if produced else None

    def test_sabstenir_pendant_une_hausse_est_penalise(self):
        entry = self._abstention([100, 102, 104, 106, 108, 112])
        self.assertIsNotNone(entry)
        self.assertLess(entry.value, 0)
        self.assertIn("ratee", entry.detail)

    def test_sabstenir_pendant_une_baisse_est_recompense(self):
        entry = self._abstention([100, 98, 96, 94, 92, 88])
        self.assertIsNotNone(entry)
        self.assertGreater(entry.value, 0)
        self.assertIn("evitee", entry.detail)

    def test_sabstenir_dans_un_marche_plat_donne_un_petit_bonus(self):
        entry = self._abstention([100, 100.1, 99.9, 100.05, 100, 100.02])
        self.assertIsNotNone(entry)
        self.assertAlmostEqual(entry.value, self.cfg.patience_reward)

    def test_le_piege_du_ne_rien_faire_est_bien_evite(self):
        """Point central : si ne jamais trader rapportait toujours autant que
        trader, le bot convergerait vers l'inaction. Une abstention pendant
        une forte hausse doit couter PLUS que ce que rapporte la patience."""
        rate = self._abstention([100, 103, 106, 109, 112, 116])
        plat = self._abstention([100, 100.1, 99.9, 100, 100, 100])
        self.assertLess(rate.value, 0)
        self.assertGreater(plat.value, 0)
        self.assertGreater(abs(rate.value), plat.value)

    def test_abstention_non_evaluee_tant_que_lhorizon_nest_pas_ecoule(self):
        scorer = Scorer(self.cfg, risk_stop_mult=2.0)
        candles = candles_from([100, 101, 102])
        d = Decision(ts=0, symbol="X", action=Action.ABSTAIN, reason="t", price=100)
        scorer.register_abstention(0, d, 1.0)
        self.assertEqual(scorer.evaluate_pending(2, candles), [])
        self.assertEqual(scorer.flush_unevaluated(), 1)

    def test_score_de_trade_en_r_multiple(self):
        scorer = Scorer(self.cfg, risk_stop_mult=2.0)
        gagnant = Trade(symbol="X", qty=1, entry_ts=0, entry_price=100, exit_ts=1,
                        exit_price=120, fees=0, pnl=100.0, pnl_pct=0.2,
                        risk_amount=50.0, exit_reason="objectif")
        perdant = Trade(symbol="X", qty=1, entry_ts=0, entry_price=100, exit_ts=1,
                        exit_price=95, fees=0, pnl=-50.0, pnl_pct=-0.05,
                        risk_amount=50.0, exit_reason="stop touche")
        self.assertAlmostEqual(scorer.score_trade(gagnant), 2.0)
        self.assertAlmostEqual(scorer.score_trade(perdant), -1.0)
        self.assertAlmostEqual(scorer.total, 1.0)

    def test_score_plafonne(self):
        scorer = Scorer(self.cfg, risk_stop_mult=2.0)
        enorme = Trade(symbol="X", qty=1, entry_ts=0, entry_price=100, exit_ts=1,
                       exit_price=1000, fees=0, pnl=5000.0, pnl_pct=9.0,
                       risk_amount=50.0, exit_reason="objectif")
        self.assertAlmostEqual(scorer.score_trade(enorme), self.cfg.cap_r)


if __name__ == "__main__":
    unittest.main()
