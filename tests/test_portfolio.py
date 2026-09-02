"""Le portefeuille doit etre comptablement irreprochable.

Si ces tests passent, on sait au moins que les chiffres du rapport ne
sont pas inventes par un bug d'arrondi.
"""

import unittest

from bot.portfolio import InsufficientFunds, Portfolio, PositionError


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self.pf = Portfolio(initial_cash=10_000.0, fee_rate=0.001, slippage_bps=10.0)

    def test_slippage_joue_toujours_contre_nous(self):
        self.assertGreater(self.pf.fill_price(100.0, "buy"), 100.0)
        self.assertLess(self.pf.fill_price(100.0, "sell"), 100.0)

    def test_frais_preleves_a_l_achat_et_a_la_vente(self):
        self.pf.buy("BTC", 100.0, 10.0, ts=0, stop=90.0, take_profit=130.0, risk_amount=100.0)
        frais_achat = self.pf.fees_paid
        self.assertGreater(frais_achat, 0)
        self.pf.sell("BTC", 100.0, ts=1, reason="test")
        self.assertGreater(self.pf.fees_paid, frais_achat)

    def test_un_aller_retour_a_prix_constant_perd_de_l_argent(self):
        """Frais + slippage : acheter et revendre au meme prix DOIT couter.
        Un simulateur ou cette operation est neutre est un simulateur qui ment."""
        self.pf.buy("BTC", 100.0, 10.0, ts=0, stop=90.0, take_profit=130.0, risk_amount=100.0)
        trade = self.pf.sell("BTC", 100.0, ts=1, reason="test")
        self.assertLess(trade.pnl, 0)
        self.assertLess(self.pf.equity({"BTC": 100.0}), 10_000.0)

    def test_impossible_de_depenser_plus_que_le_cash(self):
        with self.assertRaises(InsufficientFunds):
            self.pf.buy("BTC", 100.0, 1000.0, ts=0, stop=90.0, take_profit=130.0, risk_amount=1.0)

    def test_pas_de_double_position(self):
        self.pf.buy("BTC", 100.0, 1.0, ts=0, stop=90.0, take_profit=130.0, risk_amount=10.0)
        with self.assertRaises(PositionError):
            self.pf.buy("BTC", 100.0, 1.0, ts=1, stop=90.0, take_profit=130.0, risk_amount=10.0)

    def test_vente_sans_position(self):
        with self.assertRaises(PositionError):
            self.pf.sell("BTC", 100.0, ts=0, reason="test")

    def test_invariants_apres_operations(self):
        self.pf.check_invariants({"BTC": 100.0})
        self.pf.buy("BTC", 100.0, 5.0, ts=0, stop=90.0, take_profit=130.0, risk_amount=50.0)
        self.pf.check_invariants({"BTC": 105.0})
        self.pf.sell("BTC", 120.0, ts=1, reason="objectif")
        self.pf.check_invariants({"BTC": 120.0})
        self.assertAlmostEqual(
            self.pf.cash, 10_000.0 + self.pf.realized_pnl, places=6
        )

    def test_r_multiple(self):
        self.pf.buy("BTC", 100.0, 10.0, ts=0, stop=95.0, take_profit=115.0, risk_amount=50.0)
        trade = self.pf.sell("BTC", 115.0, ts=1, reason="objectif")
        # ~150 de gain brut pour 50 risques => proche de +3 R, un peu moins avec les frais
        self.assertGreater(trade.r_multiple, 2.0)
        self.assertLess(trade.r_multiple, 3.0)


if __name__ == "__main__":
    unittest.main()
