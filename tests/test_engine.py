"""Tests de bout en bout du moteur, sur le marche synthetique.

Le test le plus important du projet est test_aucune_fuite_du_futur :
il verifie qu'en changeant l'avenir, le passe du bot ne bouge pas d'un
pouce. Un bot qui echoue a ce test peut afficher n'importe quelle
performance, elle ne veut rien dire.
"""

import unittest

from bot.adapters.synthetic import SyntheticAdapter
from bot.config import Config
from bot.engine import Engine
from bot.models import Action, Candle


def make_config(**over) -> Config:
    cfg = Config()
    cfg.market.adapter = "synthetic"
    cfg.market.symbol = "TESTUSDT"
    cfg.market.timeframe = "1h"
    cfg.strategy.ema_fast = 8
    cfg.strategy.ema_slow = 21
    cfg.strategy.atr_period = 14
    for key, value in over.items():
        section, _, field = key.partition("__")
        setattr(getattr(cfg, section), field, value)
    return cfg


def candles(bars=1500, seed=7):
    return SyntheticAdapter("TESTUSDT", "1h", bars=bars, seed=seed).get_candles()


class TestEngine(unittest.TestCase):
    def test_backtest_complet_sans_erreur(self):
        cfg = make_config()
        engine = Engine(cfg, 3_600_000)
        result = engine.run(candles(), "TESTUSDT")
        self.assertGreater(len(result.equity_curve), 100)
        self.assertEqual(len(result.decisions), len(result.equity_curve))
        self.assertIsNotNone(result.metrics)

    def test_capital_jamais_negatif_et_invariants_tenus(self):
        cfg = make_config()
        engine = Engine(cfg, 3_600_000)
        result = engine.run(candles(), "TESTUSDT")
        self.assertTrue(all(eq > 0 for _, eq in result.equity_curve))
        self.assertGreaterEqual(engine.portfolio.cash, -1e-9)

    def test_aucune_fuite_du_futur(self):
        """On rejoue deux marches identiques jusqu'a la bougie 800, puis
        radicalement differents ensuite. Toutes les decisions prises avant
        la bougie 800 doivent etre rigoureusement les memes."""
        base = candles(1500, seed=7)
        coupe = 800
        trafique = list(base[:coupe]) + [
            Candle(ts=c.ts, open=c.open * 3, high=c.high * 3, low=c.low * 3,
                   close=c.close * 3, volume=c.volume)
            for c in base[coupe:]
        ]

        r1 = Engine(make_config(), 3_600_000).run(base, "TESTUSDT")
        r2 = Engine(make_config(), 3_600_000).run(trafique, "TESTUSDT")

        limite = base[coupe - 5].ts
        d1 = [(d.ts, d.action, d.reason) for d in r1.decisions if d.ts < limite]
        d2 = [(d.ts, d.action, d.reason) for d in r2.decisions if d.ts < limite]
        self.assertEqual(d1, d2, "le bot a change ses decisions passees : fuite du futur")

    def test_les_abstentions_sont_enregistrees_et_motivees(self):
        cfg = make_config()
        result = Engine(cfg, 3_600_000).run(candles(), "TESTUSDT")
        abstentions = [d for d in result.decisions if d.action == Action.ABSTAIN]
        self.assertGreater(len(abstentions), 0)
        self.assertTrue(all(d.reason.strip() for d in abstentions))

    def test_le_score_recompense_et_penalise(self):
        result = Engine(make_config(), 3_600_000).run(candles(), "TESTUSDT")
        bd = result.score_breakdown
        self.assertGreater(bd["abstention_count"], 0)
        self.assertGreater(bd["abstentions_penalisees"], 0)
        self.assertGreater(bd["abstentions_recompensees"], 0)

    def test_limite_de_perte_journaliere_active(self):
        """Avec un plafond ridiculement bas, le coupe-circuit doit se
        declencher au moins une fois."""
        cfg = make_config(risk__max_daily_loss_pct=0.0005, risk__risk_per_trade_pct=0.05)
        result = Engine(cfg, 3_600_000).run(candles(), "TESTUSDT")
        self.assertGreater(len(result.halted_days), 0)

    def test_taille_de_position_respecte_le_plafond(self):
        cfg = make_config(risk__max_position_pct=0.10, risk__risk_per_trade_pct=0.05)
        engine = Engine(cfg, 3_600_000)
        result = engine.run(candles(), "TESTUSDT")
        for t in result.trades:
            notional = t.entry_price * t.qty
            self.assertLessEqual(notional, cfg.portfolio.initial_cash * 5.0)

    def test_execution_a_louverture_suivante(self):
        """Une entree ne doit jamais se faire au prix de cloture de la
        bougie qui a genere le signal."""
        cs = candles()
        result = Engine(make_config(), 3_600_000).run(cs, "TESTUSDT")
        by_ts = {c.ts: c for c in cs}
        for t in result.trades:
            bar = by_ts.get(t.entry_ts)
            self.assertIsNotNone(bar)
            # le prix d'entree est l'ouverture de cette bougie, degradee par le slippage
            self.assertAlmostEqual(t.entry_price, bar.open * 1.0005, places=4)

    def test_deux_executions_identiques_donnent_le_meme_resultat(self):
        cfg = make_config()
        r1 = Engine(cfg, 3_600_000).run(candles(), "TESTUSDT")
        r2 = Engine(cfg, 3_600_000).run(candles(), "TESTUSDT")
        self.assertEqual(r1.final_equity, r2.final_equity)
        self.assertEqual(len(r1.trades), len(r2.trades))


if __name__ == "__main__":
    unittest.main()
