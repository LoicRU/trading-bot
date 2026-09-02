"""Les indicateurs doivent etre justes ET aveugles au futur."""

import unittest

from bot.indicators import atr, crossed_down, crossed_up, ema, sma
from bot.models import Candle


def mk(prices):
    return [
        Candle(ts=i * 60_000, open=p, high=p * 1.01, low=p * 0.99, close=p, volume=1.0)
        for i, p in enumerate(prices)
    ]


class TestIndicators(unittest.TestCase):
    def test_sma_valeur_connue(self):
        out = sma([1, 2, 3, 4, 5], 3)
        self.assertIsNone(out[1])
        self.assertAlmostEqual(out[2], 2.0)
        self.assertAlmostEqual(out[4], 4.0)

    def test_ema_serie_constante_reste_constante(self):
        out = ema([10.0] * 20, 5)
        self.assertAlmostEqual(out[-1], 10.0)

    def test_ema_chauffe_renvoie_none(self):
        out = ema([1, 2, 3], 5)
        self.assertTrue(all(v is None for v in out))

    def test_aucune_fuite_du_futur(self):
        """Modifier la fin de la serie ne doit RIEN changer au debut.
        C'est le test qui attrape les bugs de lookahead."""
        base = [100 + i for i in range(60)]
        modifie = base[:40] + [999] * 20

        e1, e2 = ema(base, 10), ema(modifie, 10)
        self.assertEqual(e1[:40], e2[:40])

        a1, a2 = atr(mk(base), 14), atr(mk(modifie), 14)
        self.assertEqual(a1[:40], a2[:40])

    def test_atr_positif_et_chauffe(self):
        out = atr(mk([100 + i for i in range(40)]), 14)
        self.assertIsNone(out[12])
        self.assertIsNotNone(out[13])
        self.assertTrue(all(v > 0 for v in out[13:]))

    def test_croisements(self):
        fast = [1.0, 1.0, 3.0, 3.0, 1.0]
        slow = [2.0, 2.0, 2.0, 2.0, 2.0]
        self.assertTrue(crossed_up(fast, slow, 2))
        self.assertFalse(crossed_up(fast, slow, 3))
        self.assertTrue(crossed_down(fast, slow, 4))

    def test_croisement_ignore_les_none(self):
        fast = [None, None, 3.0]
        slow = [None, 2.0, 2.0]
        self.assertFalse(crossed_up(fast, slow, 1))


if __name__ == "__main__":
    unittest.main()
