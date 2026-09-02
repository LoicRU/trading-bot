"""Le test d'avantage doit dire 'rien' quand il n'y a rien.

Un outil de diagnostic qui trouve des signaux dans du bruit est pire
qu'inutile : il donne confiance dans une strategie qui n'existe pas.
"""

import random
import unittest

from bot.config import Config
from bot.edge import analyse, format_report
from bot.models import Candle


def marche_aleatoire(n=3000, seed=3, vol=0.01):
    """Marche parfaitement aleatoire : par construction, AUCUN signal
    ne peut y avoir d'avantage."""
    rng = random.Random(seed)
    price = 100.0
    out = []
    for i in range(n):
        ret = rng.gauss(0, vol)
        o = price
        c = o * (1 + ret)
        out.append(
            Candle(
                ts=i * 3_600_000,
                open=round(o, 4),
                high=round(max(o, c) * (1 + abs(rng.gauss(0, vol / 2))), 4),
                low=round(min(o, c) * (1 - abs(rng.gauss(0, vol / 2))), 4),
                close=round(c, 4),
                volume=1.0,
            )
        )
        price = c
    return out


def cfg_test():
    cfg = Config()
    cfg.strategy.ema_fast = 8
    cfg.strategy.ema_slow = 21
    cfg.strategy.atr_period = 14
    cfg.strategy.min_atr_pct = 0.0001
    cfg.strategy.max_atr_pct = 0.5
    return cfg


class TestEdge(unittest.TestCase):
    def test_aucun_avantage_sur_un_marche_aleatoire(self):
        """Sur du bruit pur, aucun horizon ne doit ressortir significatif.
        Un ou deux faux positifs sur cinq horizons restent possibles, donc
        on verifie qu'ils ne sont pas majoritaires."""
        data = analyse(cfg_test(), marche_aleatoire(), permutations=400, seed=7)
        results = data["results"]
        self.assertGreater(len(results), 0, "aucun horizon exploitable")
        significatifs = [r for r in results if r.significant]
        self.assertLess(
            len(significatifs), len(results) / 2,
            "le test trouve un avantage dans du bruit pur",
        )

    def test_les_p_values_sont_bien_des_probabilites(self):
        data = analyse(cfg_test(), marche_aleatoire(), permutations=200, seed=11)
        for r in data["results"]:
            self.assertGreater(r.p_value, 0.0)
            self.assertLessEqual(r.p_value, 1.0)

    def test_le_rapport_conclut_explicitement(self):
        data = analyse(cfg_test(), marche_aleatoire(), permutations=200, seed=13)
        texte = format_report(data)
        self.assertIn("VERDICT", texte)

    def test_deterministe(self):
        c = marche_aleatoire()
        a = analyse(cfg_test(), c, permutations=200, seed=5)
        b = analyse(cfg_test(), c, permutations=200, seed=5)
        self.assertEqual(
            [r.p_value for r in a["results"]], [r.p_value for r in b["results"]]
        )

    def test_detecte_un_avantage_reel_quand_il_existe(self):
        """Marche truque : apres chaque croisement haussier, on injecte une
        derive positive. Le test DOIT la voir, sinon il est aveugle."""
        base = marche_aleatoire(3000, seed=21, vol=0.006)
        from bot.indicators import crossed_up, ema

        closes = [c.close for c in base]
        ef, es = ema(closes, 8), ema(closes, 21)

        truque = []
        boost = 0
        price = base[0].close
        for i, c in enumerate(base):
            if crossed_up(ef, es, i):
                boost = 24  # 24 bougies de derive favorable apres le signal
            drift = 0.004 if boost > 0 else 0.0
            boost = max(0, boost - 1)
            o = price
            cl = o * (1 + (c.close / c.open - 1) + drift)
            truque.append(
                Candle(ts=c.ts, open=round(o, 4), high=round(max(o, cl) * 1.002, 4),
                       low=round(min(o, cl) * 0.998, 4), close=round(cl, 4), volume=1.0)
            )
            price = cl

        data = analyse(cfg_test(), truque, permutations=400, seed=3)
        significatifs = [r for r in data["results"] if r.significant]
        self.assertGreater(
            len(significatifs), 0,
            "le test ne detecte pas un avantage pourtant injecte volontairement",
        )


class TestSignalLibrary(unittest.TestCase):
    """Chaque candidat doit se declencher raisonnablement et ne jamais
    lire le futur."""

    def setUp(self):
        from bot.signals import build_all
        self.candles = marche_aleatoire(2500, seed=17)
        self.cfg = cfg_test()
        # Les signaux de financement ont leurs propres tests : sans donnees
        # de financement ils ne peuvent pas se declencher, par construction.
        self.signaux = build_all(avec_financement=False)

    def test_tous_les_signaux_se_declenchent_parfois(self):
        for s in self.signaux:
            s.prepare(self.candles, self.cfg.strategy)
            n = sum(1 for i in range(s.warmup, len(self.candles) - 1) if s.fires(i))
            self.assertGreater(n, 5, f"{s.spec.key} ne se declenche presque jamais")
            self.assertLess(
                n, len(self.candles) * 0.25,
                f"{s.spec.key} se declenche trop souvent pour etre un evenement",
            )

    def test_aucun_signal_ne_lit_le_futur(self):
        """On modifie la fin de la serie : les declenchements du debut ne
        doivent pas bouger d'un pouce."""
        from bot.models import Candle
        coupe = 1500
        trafique = list(self.candles[:coupe]) + [
            Candle(ts=c.ts, open=c.open * 4, high=c.high * 4, low=c.low * 4,
                   close=c.close * 4, volume=c.volume)
            for c in self.candles[coupe:]
        ]
        from bot.signals import build_all
        for a, b in zip(self.signaux, build_all(avec_financement=False)):
            a.prepare(self.candles, self.cfg.strategy)
            b.prepare(trafique, self.cfg.strategy)
            limite = coupe - 60
            fa = [i for i in range(a.warmup, limite) if a.fires(i)]
            fb = [i for i in range(b.warmup, limite) if b.fires(i)]
            self.assertEqual(fa, fb, f"{a.spec.key} lit le futur")

    def test_comparaison_applique_bonferroni(self):
        from bot.edge import compare_signals, format_comparison
        data = compare_signals(self.cfg, self.candles, permutations=120, seed=4)
        self.assertLess(data["seuil_corrige"], 0.05)
        self.assertGreater(data["n_tests"], 5)
        self.assertIn("Bonferroni", format_comparison(data))

    def test_les_cles_sont_uniques(self):
        cles = [s.spec.key for s in self.signaux]
        self.assertEqual(len(cles), len(set(cles)))



class TestFundingSignals(unittest.TestCase):
    """Les signaux de financement doivent se comporter proprement meme
    sans donnees, et surtout ne jamais lire le futur."""

    def test_sans_donnees_ils_ne_se_declenchent_jamais(self):
        from bot.signals import FundingCapitulation
        s = FundingCapitulation()
        candles = marche_aleatoire(500)
        s.prepare(candles, cfg_test().strategy, extras=None)
        self.assertFalse(any(s.fires(i) for i in range(len(candles))))

    def test_ecartes_de_la_comparaison_sans_donnees(self):
        from bot.edge import compare_signals
        data = compare_signals(
            cfg_test(), marche_aleatoire(1500), permutations=60, seed=2, extras=None
        )
        cles = {r["signal"].key for r in data["rapports"]}
        self.assertNotIn("financement_bas", cles)
        self.assertNotIn("financement_haut", cles)

    def test_percentiles_calcules_sur_fenetre_glissante_anterieure(self):
        """Modifier la fin de la serie de financement ne doit rien changer
        aux evenements du debut."""
        import random as _r
        from bot.funding import percentile_events

        rng = _r.Random(9)
        base = [(i * 8 * 3600 * 1000, rng.gauss(0.0001, 0.0002)) for i in range(600)]
        trafique = base[:400] + [(ts, r * 50) for ts, r in base[400:]]

        lows_a, highs_a = percentile_events(base)
        lows_b, highs_b = percentile_events(trafique)
        limite = base[390][0]
        self.assertEqual([t for t in lows_a if t < limite], [t for t in lows_b if t < limite])
        self.assertEqual([t for t in highs_a if t < limite], [t for t in highs_b if t < limite])

    def test_un_evenement_agit_sur_une_bougie_posterieure(self):
        from bot.funding import events_to_indices
        candles = marche_aleatoire(10)
        # evenement place juste apres la cloture de la bougie 3
        event = candles[3].ts + 1
        idx = events_to_indices(candles, [event])
        self.assertTrue(all(i >= 4 for i in idx), "le signal agirait avant de connaitre la donnee")



class TestStabilite(unittest.TestCase):
    """Le diagnostic de stabilite doit distinguer un effet permanent d'un
    effet eteint."""

    def _marche_avec_avantage(self, n, boost_jusqua):
        """Avantage apres cassure, mais UNIQUEMENT avant l'indice donne :
        au-dela, le marche redevient aleatoire. C'est le profil d'un
        avantage qui s'eteint."""
        from bot.indicators import rolling_max
        base = marche_aleatoire(n, seed=31, vol=0.006)
        highs = rolling_max([c.high for c in base], 48)
        out, boost, price = [], 0, base[0].close
        for i, c in enumerate(base):
            if i < boost_jusqua and highs[i] is not None and c.close > highs[i]:
                boost = 72
            drift = 0.003 if boost > 0 else 0.0
            boost = max(0, boost - 1)
            o = price
            cl = o * (1 + (c.close / c.open - 1) + drift)
            out.append(Candle(ts=c.ts, open=round(o, 4), high=round(max(o, cl) * 1.002, 4),
                              low=round(min(o, cl) * 0.998, 4), close=round(cl, 4), volume=1.0))
            price = cl
        return out

    def test_detecte_un_avantage_eteint(self):
        from bot.edge import analyse_by_period, format_periods
        from bot.signals import DonchianBreakout
        candles = self._marche_avec_avantage(4000, boost_jusqua=2000)
        rows = analyse_by_period(
            cfg_test(), candles, DonchianBreakout(), horizon=72,
            periods=4, permutations=200,
        )
        self.assertGreaterEqual(len(rows), 3)
        premiere = rows[0]["resultat"]
        derniere = rows[-1]["resultat"]
        self.assertIsNotNone(premiere)
        self.assertIsNotNone(derniere)
        self.assertGreater(
            premiere.edge, derniere.edge,
            "la stabilite ne distingue pas une epoque avantageuse d'une epoque neutre",
        )
        texte = format_periods(rows, 72, "test")
        self.assertIn("periode", texte)

    def test_tranches_chronologiques_non_chevauchantes(self):
        from bot.edge import analyse_by_period
        from bot.signals import DonchianBreakout
        candles = marche_aleatoire(4000, seed=41)
        rows = analyse_by_period(
            cfg_test(), candles, DonchianBreakout(), horizon=24,
            periods=4, permutations=60,
        )
        debuts = [r["debut"] for r in rows]
        self.assertEqual(debuts, sorted(debuts))



class TestVerdictEconomique(unittest.TestCase):
    """Un avantage plus petit que les frais n'est pas un avantage."""

    def _rows(self, ecarts):
        from bot.edge import HorizonResult
        return [
            {
                "periode": k + 1, "debut": "2020-01-01", "fin": "2021-01-01", "n": 100,
                "resultat": HorizonResult(
                    horizon=72, n_signals=100, signal_mean=e, baseline_mean=0.0,
                    edge=e, signal_mean_atr=0.0, baseline_mean_atr=0.0,
                    win_rate=0.5, baseline_win_rate=0.5, p_value=0.01,
                ),
            }
            for k, e in enumerate(ecarts)
        ]

    def test_avantage_positif_mais_sous_les_frais_est_rejete(self):
        from bot.edge import format_periods
        # tous positifs, tous inferieurs au cout de 0,3 %
        texte = format_periods(self._rows([0.002, 0.0018, 0.0012, 0.0017]), 72, "x", 0.003)
        self.assertIn("AUCUNE tranche ne laisse de quoi payer les frais", texte)

    def test_avantage_superieur_aux_frais_est_retenu(self):
        from bot.edge import format_periods
        texte = format_periods(self._rows([0.01, 0.009, 0.008, 0.011]), 72, "x", 0.003)
        self.assertIn("dont la plus recente", texte)

    def test_avantage_eteint_est_signale(self):
        from bot.edge import format_periods
        texte = format_periods(self._rows([0.02, 0.015, 0.004, 0.0005]), 72, "x", 0.003)
        self.assertIn("appartient au passe", texte)

    def test_sans_cout_pas_de_verdict_economique(self):
        from bot.edge import format_periods
        texte = format_periods(self._rows([0.01, 0.01, 0.01, 0.01]), 72, "x", 0.0)
        self.assertNotIn("VERDICT ECONOMIQUE", texte)


if __name__ == "__main__":
    unittest.main()
