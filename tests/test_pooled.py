"""Le test transversal doit resister a la correlation entre paires.

C'est LE point critique : quarante paires crypto montent et descendent
ensemble. Un test qui les traiterait comme independantes produirait des
p-values spectaculaires et fausses. Ces tests verifient que ce n'est pas
le cas.
"""

import random
import unittest

from bot.config import Config
from bot.models import Candle
from bot.pooled import analyse_pooled, format_pooled
from bot.signals import DonchianBreakout
from bot.universe import align_on_common_grid


def cfg_test():
    cfg = Config()
    cfg.strategy.ema_fast = 8
    cfg.strategy.ema_slow = 21
    cfg.strategy.atr_period = 14
    return cfg


def marche_correle(n_paires=10, n=800, seed=5, part_commune=0.8):
    """Plusieurs paires partageant un facteur commun dominant — comme la
    crypto, ou tout suit le Bitcoin. Aucun avantage n'y est injecte."""
    rng = random.Random(seed)
    commun = [rng.gauss(0, 0.02) for _ in range(n)]
    series = {}
    for p in range(n_paires):
        prix = 100.0
        bougies = []
        for i in range(n):
            ret = part_commune * commun[i] + (1 - part_commune) * rng.gauss(0, 0.02)
            o = prix
            c = o * (1 + ret)
            bougies.append(
                Candle(ts=i * 86_400_000, open=round(o, 4),
                       high=round(max(o, c) * 1.01, 4), low=round(min(o, c) * 0.99, 4),
                       close=round(c, 4), volume=1.0)
            )
            prix = c
        series[f"P{p}USDT"] = bougies
    return series


class TestPooled(unittest.TestCase):
    def test_pas_davantage_sur_des_paires_correlees_sans_signal(self):
        """Le cas qui compte : beaucoup de paires, fortement correlees,
        aucun avantage reel. Le test ne doit pas crier victoire."""
        series = marche_correle()
        grille, alignees = align_on_common_grid(series, min_bars=300)
        res = analyse_pooled(
            cfg_test(), grille, alignees, DonchianBreakout, horizon=20,
            permutations=300, seed=1,
        )
        self.assertGreater(res.n_signals, 50)
        self.assertGreater(
            res.p_value, 0.01,
            "le test transversal trouve un avantage dans des paires correlees sans signal",
        )

    def test_compte_les_paires_positives(self):
        series = marche_correle()
        grille, alignees = align_on_common_grid(series, min_bars=300)
        res = analyse_pooled(
            cfg_test(), grille, alignees, DonchianBreakout, horizon=20,
            permutations=200, seed=2,
        )
        self.assertLessEqual(res.paires_positives, res.paires_mesurees)
        self.assertGreater(res.paires_mesurees, 0)

    def test_detecte_un_avantage_partage_par_toutes_les_paires(self):
        """Avantage injecte apres chaque cassure, sur TOUTES les paires.
        Le test doit le voir malgre la correlation."""
        from bot.indicators import rolling_max

        series = {}
        base = marche_correle(n_paires=8, n=900, seed=11)
        for sym, bougies in base.items():
            highs = rolling_max([c.high for c in bougies], 48)
            out, boost, prix = [], 0, bougies[0].close
            for i, c in enumerate(bougies):
                if highs[i] is not None and c.close > highs[i]:
                    boost = 20
                drift = 0.004 if boost > 0 else 0.0
                boost = max(0, boost - 1)
                o = prix
                cl = o * (1 + (c.close / c.open - 1) + drift)
                out.append(Candle(ts=c.ts, open=round(o, 4),
                                  high=round(max(o, cl) * 1.005, 4),
                                  low=round(min(o, cl) * 0.995, 4),
                                  close=round(cl, 4), volume=1.0))
                prix = cl
            series[sym] = out

        grille, alignees = align_on_common_grid(series, min_bars=300)
        res = analyse_pooled(
            cfg_test(), grille, alignees, DonchianBreakout, horizon=20,
            permutations=300, seed=3,
        )
        self.assertGreater(res.edge, 0)
        self.assertGreater(
            res.part_positive, 0.7,
            "un avantage present sur toutes les paires doit se voir sur la plupart",
        )

    def test_le_rapport_signale_un_resultat_porte_par_peu_de_paires(self):
        from bot.pooled import PooledResult
        res = PooledResult(
            horizon=20, n_signals=500, n_paires=10, signal_mean=0.02,
            baseline_mean=0.0, edge=0.02, p_value=0.001,
            paires_positives=3, paires_mesurees=10,
            par_paire={f"P{i}USDT": (50, 0.02 if i < 3 else -0.01) for i in range(10)},
        )
        texte = format_pooled(res, "test", 0.003, [])
        self.assertIn("minorite de paires", texte)

    def test_le_rapport_mentionne_toujours_le_biais_du_survivant_si_positif(self):
        from bot.pooled import PooledResult
        res = PooledResult(
            horizon=20, n_signals=2000, n_paires=40, signal_mean=0.03,
            baseline_mean=0.005, edge=0.025, p_value=0.001,
            paires_positives=34, paires_mesurees=40,
            par_paire={f"P{i}USDT": (50, 0.02) for i in range(40)},
        )
        texte = format_pooled(res, "test", 0.003, [])
        self.assertIn("survivant", texte)


class TestUniverse(unittest.TestCase):
    def test_la_grille_est_chronologique_et_partagee(self):
        series = marche_correle(n_paires=5, n=600, seed=8)
        grille, alignees = align_on_common_grid(series, min_bars=300)
        self.assertEqual(grille, sorted(grille))
        for serie in alignees.values():
            self.assertEqual(len(serie), len(grille))

    def test_les_paires_trop_courtes_sont_ecartees(self):
        series = marche_correle(n_paires=3, n=600, seed=9)
        series["COURTUSDT"] = series["P0USDT"][:50]
        _, alignees = align_on_common_grid(series, min_bars=300)
        self.assertNotIn("COURTUSDT", alignees)

    def test_univers_sans_doublon(self):
        from bot.universe import UNIVERS
        self.assertEqual(len(UNIVERS), len(set(UNIVERS)))


if __name__ == "__main__":
    unittest.main()
