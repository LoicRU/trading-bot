"""Les 17 indicateurs ajoutes pour le balayage exploratoire (bot/signals_explo.py).

Meme exigence que pour les indicateurs d'origine : une valeur juste ET
aveugle au futur. Le test le plus important ici est
test_aucune_fuite_du_futur_tous_indicateurs, qui rejoue TOUS les nouveaux
indicateurs sur deux series identiques jusqu'a un point puis divergentes,
et verifie qu'aucun n'a change son passe.
"""

import unittest

from bot.indicators import (
    aroon,
    awesome_oscillator,
    bollinger,
    cci,
    cmf,
    crossed_above,
    crossed_below,
    keltner,
    macd,
    mfi,
    obv,
    roc,
    rolling_min,
    stddev,
    stochastic,
    trix,
    volume_sma,
    vortex,
    williams_r,
)
from bot.models import Candle


def mk(prices, volumes=None, high=None, low=None):
    n = len(prices)
    volumes = volumes or [100.0] * n
    return [
        Candle(
            ts=i * 60_000,
            open=prices[i],
            high=(high[i] if high else prices[i] * 1.01),
            low=(low[i] if low else prices[i] * 0.99),
            close=prices[i],
            volume=volumes[i],
        )
        for i in range(n)
    ]


class TestValeursConnues(unittest.TestCase):
    def test_rolling_min_exclut_la_bougie_courante(self):
        out = rolling_min([5, 4, 3, 10, 1], 3)
        # a l'indice 3 : min(5,4,3) = 3, la valeur 10 en indice 3 elle-meme est exclue
        self.assertAlmostEqual(out[3], 3.0)

    def test_stddev_serie_constante_est_nulle(self):
        out = stddev([10.0] * 10, 5)
        self.assertAlmostEqual(out[-1], 0.0)

    def test_stddev_valeur_connue(self):
        # [1,2,3,4,5] -> ecart-type population = sqrt(2) ~ 1.4142
        out = stddev([1, 2, 3, 4, 5], 5)
        self.assertAlmostEqual(out[4], 2.0 ** 0.5, places=6)

    def test_bollinger_bande_mediane_est_la_sma(self):
        mid, upper, lower = bollinger([1, 2, 3, 4, 5], period=5, k=2.0)
        self.assertAlmostEqual(mid[4], 3.0)
        self.assertGreater(upper[4], mid[4])
        self.assertLess(lower[4], mid[4])

    def test_macd_serie_constante_est_nulle(self):
        line, sig, hist = macd([50.0] * 60, fast=12, slow=26, signal=9)
        self.assertAlmostEqual(line[-1], 0.0, places=6)
        self.assertAlmostEqual(sig[-1], 0.0, places=6)
        self.assertAlmostEqual(hist[-1], 0.0, places=6)

    def test_trix_serie_constante_est_nulle(self):
        out = trix([50.0] * 80, period=10)
        self.assertAlmostEqual(out[-1], 0.0, places=6)

    def test_roc_valeur_connue(self):
        out = roc([100, 100, 100, 110], period=3)
        self.assertAlmostEqual(out[3], 10.0)

    def test_obv_monte_avec_les_clotures_en_hausse(self):
        out = obv(mk([10, 11, 10.5, 12]))
        self.assertEqual(out[0], 0.0)
        self.assertGreater(out[1], out[0])
        self.assertLess(out[2], out[1])
        self.assertGreater(out[3], out[2])

    def test_williams_r_borne_entre_moins_100_et_0(self):
        out = williams_r(mk([100 + (i % 5) for i in range(30)]), period=14)
        for v in out:
            if v is not None:
                self.assertLessEqual(v, 0.0)
                self.assertGreaterEqual(v, -100.0)

    def test_stochastic_borne_entre_0_et_100(self):
        k, d = stochastic(mk([100 + (i % 7) for i in range(40)]), period=14)
        for v in k:
            if v is not None:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 100.0)
        for v in d:
            if v is not None:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 100.0)

    def test_mfi_borne_entre_0_et_100(self):
        out = mfi(mk([100 + (i % 5) for i in range(40)], volumes=[100 + i for i in range(40)]), period=14)
        for v in out:
            if v is not None:
                self.assertGreaterEqual(v, 0.0)
                self.assertLessEqual(v, 100.0)

    def test_cmf_serie_plate_est_nulle(self):
        # high == low == close partout : la formule division-par-zero doit
        # etre geree (0.0), pas planter.
        candles = [Candle(ts=i, open=10, high=10, low=10, close=10, volume=5) for i in range(30)]
        out = cmf(candles, period=20)
        self.assertAlmostEqual(out[-1], 0.0)

    def test_aroon_plus_haut_au_dernier_indice_donne_100(self):
        prix = list(range(1, 20))  # strictement croissant : le plus haut est toujours le dernier
        up, down = aroon(mk(prix), period=14)
        self.assertAlmostEqual(up[-1], 100.0)

    def test_vortex_composantes_positives(self):
        plus, minus = vortex(mk([100 + (i % 5) for i in range(40)]), period=14)
        for v in plus:
            if v is not None:
                self.assertGreaterEqual(v, 0.0)
        for v in minus:
            if v is not None:
                self.assertGreaterEqual(v, 0.0)

    def test_awesome_oscillator_serie_constante_est_nulle(self):
        out = awesome_oscillator(mk([50.0] * 40))
        self.assertAlmostEqual(out[-1], 0.0)

    def test_volume_sma_valeur_connue(self):
        out = volume_sma(mk([1] * 5, volumes=[10, 20, 30, 40, 50]), period=5)
        self.assertAlmostEqual(out[4], 30.0)

    def test_keltner_bande_mediane_est_lema(self):
        from bot.indicators import ema

        candles = mk([100 + i for i in range(40)])
        mid, upper, lower = keltner(candles, ema_period=20, atr_period=10, k=2.0)
        e = ema([c.close for c in candles], 20)
        self.assertAlmostEqual(mid[-1], e[-1])
        self.assertGreater(upper[-1], mid[-1])
        self.assertLess(lower[-1], mid[-1])

    def test_cci_serie_plate_est_nulle(self):
        candles = [Candle(ts=i, open=10, high=10, low=10, close=10, volume=1) for i in range(30)]
        out = cci(candles, period=20)
        self.assertAlmostEqual(out[-1], 0.0)


class TestCroisementsGeneriques(unittest.TestCase):
    def test_crossed_above(self):
        serie = [10.0, 20.0, 40.0, 20.0]
        self.assertTrue(crossed_above(serie, 30.0, 2))
        self.assertFalse(crossed_above(serie, 30.0, 1))

    def test_crossed_below(self):
        serie = [40.0, 20.0, 10.0]
        self.assertTrue(crossed_below(serie, 30.0, 1))

    def test_crossed_above_ignore_les_none(self):
        self.assertFalse(crossed_above([None, 40.0], 30.0, 1))


class TestAucuneFuiteDuFutur(unittest.TestCase):
    """Deux series identiques jusqu'a l'indice 40, puis divergentes : tout
    indicateur qui changerait sa valeur AVANT 40 fuit le futur."""

    def setUp(self):
        base_prix = [100 + (i % 13) + 0.1 * i for i in range(80)]
        base_vol = [100 + (i % 17) * 3 for i in range(80)]
        self.a = mk(base_prix, volumes=base_vol)
        modifie_prix = base_prix[:40] + [999 + i for i in range(40)]
        modifie_vol = base_vol[:40] + [1 + i for i in range(40)]
        self.b = mk(modifie_prix, volumes=modifie_vol)
        self.COUPURE = 40

    def _assert_stable(self, out_a, out_b, nom):
        self.assertEqual(
            out_a[: self.COUPURE], out_b[: self.COUPURE],
            f"{nom} a change une valeur passee suite a une modification future",
        )

    def test_tous_les_nouveaux_indicateurs(self):
        closes_a = [c.close for c in self.a]
        closes_b = [c.close for c in self.b]

        self._assert_stable(rolling_min(closes_a, 10), rolling_min(closes_b, 10), "rolling_min")
        self._assert_stable(stddev(closes_a, 10), stddev(closes_b, 10), "stddev")

        for idx, nom in enumerate(("macd_line", "macd_signal", "macd_hist")):
            self._assert_stable(macd(closes_a)[idx], macd(closes_b)[idx], nom)
        for idx, nom in enumerate(("boll_mid", "boll_upper", "boll_lower")):
            self._assert_stable(bollinger(closes_a, 20)[idx], bollinger(closes_b, 20)[idx], nom)
        for idx, nom in enumerate(("kelt_mid", "kelt_upper", "kelt_lower")):
            self._assert_stable(keltner(self.a, 20, 10)[idx], keltner(self.b, 20, 10)[idx], nom)
        for idx, nom in enumerate(("stoch_k", "stoch_d")):
            self._assert_stable(stochastic(self.a, 14)[idx], stochastic(self.b, 14)[idx], nom)

        self._assert_stable(williams_r(self.a, 14), williams_r(self.b, 14), "williams_r")
        self._assert_stable(cci(self.a, 20), cci(self.b, 20), "cci")
        self._assert_stable(roc(closes_a, 12), roc(closes_b, 12), "roc")
        self._assert_stable(obv(self.a), obv(self.b), "obv")
        self._assert_stable(mfi(self.a, 14), mfi(self.b, 14), "mfi")
        self._assert_stable(cmf(self.a, 20), cmf(self.b, 20), "cmf")
        for idx, nom in enumerate(("aroon_up", "aroon_down")):
            self._assert_stable(aroon(self.a, 14)[idx], aroon(self.b, 14)[idx], nom)
        self._assert_stable(trix(closes_a, 15), trix(closes_b, 15), "trix")
        for idx, nom in enumerate(("vortex_plus", "vortex_minus")):
            self._assert_stable(vortex(self.a, 14)[idx], vortex(self.b, 14)[idx], nom)
        self._assert_stable(awesome_oscillator(self.a), awesome_oscillator(self.b), "awesome_oscillator")
        self._assert_stable(volume_sma(self.a, 20), volume_sma(self.b, 20), "volume_sma")


if __name__ == "__main__":
    unittest.main()
