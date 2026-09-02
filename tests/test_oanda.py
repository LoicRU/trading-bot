"""Adaptateur OANDA : tout ce qui se teste sans reseau.

Le parsing est isole dans une fonction pure exactement pour ca — pouvoir
verifier les regles qui comptent (bougies incompletes ecartees,
horodatage a la nanoseconde) sans dependre d'un jeton ni d'une connexion.
"""

import unittest

from bot.adapters.base import MarketDataError
from bot.adapters.oanda import GRANULARITE, OandaAdapter, parse_candles
from bot.config import Config, validate


def payload(*bougies):
    return {"instrument": "EUR_USD", "granularity": "H1", "candles": list(bougies)}


def bougie(temps, o, h, l, c, complete=True, volume=100):
    return {
        "complete": complete,
        "volume": volume,
        "time": temps,
        "mid": {"o": str(o), "h": str(h), "l": str(l), "c": str(c)},
    }


class TestParsing(unittest.TestCase):
    def test_bougie_incomplete_ecartee(self):
        """Une bougie en cours de formation change encore de valeur.
        L'inclure fausserait indicateurs, signaux et backtest."""
        data = payload(
            bougie("2024-01-01T00:00:00.000000000Z", 1.10, 1.11, 1.09, 1.105),
            bougie("2024-01-01T01:00:00.000000000Z", 1.105, 1.12, 1.10, 1.115, complete=False),
        )
        out = parse_candles(data)
        self.assertEqual(len(out), 1)

    def test_horodatage_nanoseconde(self):
        out = parse_candles(payload(bougie("2024-01-01T00:00:00.000000000Z", 1, 2, 0.5, 1.5)))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].dt.strftime("%Y-%m-%d %H:%M"), "2024-01-01 00:00")

    def test_horodatage_sans_fraction(self):
        out = parse_candles(payload(bougie("2024-06-15T13:00:00Z", 1, 2, 0.5, 1.5)))
        self.assertEqual(out[0].dt.strftime("%Y-%m-%d %H:%M"), "2024-06-15 13:00")

    def test_ordre_ohlc_respecte(self):
        """OANDA nomme les champs o/h/l/c ; une inversion silencieuse
        produirait des bougies aberrantes que rien ne rattraperait."""
        out = parse_candles(payload(bougie("2024-01-01T00:00:00.000000000Z",
                                           1.1000, 1.1050, 1.0950, 1.1020)))
        c = out[0]
        self.assertAlmostEqual(c.open, 1.1000)
        self.assertAlmostEqual(c.high, 1.1050)
        self.assertAlmostEqual(c.low, 1.0950)
        self.assertAlmostEqual(c.close, 1.1020)

    def test_payload_vide(self):
        self.assertEqual(parse_candles({"candles": []}), [])
        self.assertEqual(parse_candles({}), [])

    def test_horodatage_illisible_refuse(self):
        with self.assertRaises(MarketDataError):
            parse_candles(payload(bougie("pas une date", 1, 2, 0.5, 1.5)))


class TestAdaptateur(unittest.TestCase):
    def test_instrument_mal_forme_refuse(self):
        with self.assertRaises(MarketDataError):
            OandaAdapter("EURUSD", "1h", token="x")

    def test_unite_de_temps_non_supportee(self):
        with self.assertRaises(MarketDataError):
            OandaAdapter("EUR_USD", "3m", token="x")

    def test_granularites_couvrent_nos_unites(self):
        for tf in ("1h", "4h", "1d"):
            self.assertIn(tf, GRANULARITE)

    def test_sans_jeton_message_explicite(self):
        import os
        ancien = os.environ.pop("OANDA_TOKEN", None)
        try:
            ad = OandaAdapter("EUR_USD", "1h")
            with self.assertRaises(MarketDataError) as ctx:
                ad._request({"granularity": "H1", "count": 1})
            self.assertIn("OANDA_TOKEN", str(ctx.exception))
        finally:
            if ancien is not None:
                os.environ["OANDA_TOKEN"] = ancien

    def test_environnement_par_defaut_est_practice(self):
        ad = OandaAdapter("EUR_USD", "1h", token="x")
        self.assertEqual(ad.environnement, "practice")
        self.assertIn("fxpractice", ad.host)


class TestConfigOanda(unittest.TestCase):
    def test_symbole_crypto_refuse_sur_oanda(self):
        cfg = Config()
        cfg.market.adapter = "oanda"
        cfg.market.symbol = "BTCUSDT"
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_compte_reel_refuse_par_defaut(self):
        """Ce projet est un simulateur : viser l'environnement reel doit
        exiger une modification consciente du code, pas une ligne de config."""
        cfg = Config()
        cfg.market.adapter = "oanda"
        cfg.market.symbol = "EUR_USD"
        cfg.market.oanda_env = "live"
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_config_oanda_livree_est_valide(self):
        from pathlib import Path
        from bot.config import load_config
        chemin = Path(__file__).resolve().parent.parent / "config.oanda.json"
        if chemin.exists():
            cfg = load_config(str(chemin))
            self.assertEqual(cfg.market.adapter, "oanda")
            self.assertEqual(cfg.portfolio.fee_rate, 0.0)

    def test_cout_forex_bien_inferieur_au_cout_crypto(self):
        """Le fondement meme du changement de terrain."""
        from pathlib import Path
        from bot.config import load_config
        racine = Path(__file__).resolve().parent.parent
        fx = load_config(str(racine / "config.oanda.json"))
        crypto = load_config(str(racine / "config.json"))

        def cout(c):
            return 2 * (c.portfolio.fee_rate + c.portfolio.slippage_bps / 10_000.0)

        self.assertLess(cout(fx) * 5, cout(crypto))



class TestCoutAvecPortage(unittest.TestCase):
    """Le swap est un cout par jour : l'ignorer fausserait la comparaison
    entre horizons, exactement comme ignorer les frais l'a fait en crypto."""

    def _cfg(self, swap):
        c = Config()
        c.portfolio.fee_rate = 0.0
        c.portfolio.slippage_bps = 1.0
        c.portfolio.swap_annual_pct = swap
        return c

    def test_sans_swap_le_cout_ne_depend_pas_de_lhorizon(self):
        from bot.cli import cout_total
        c = self._cfg(0.0)
        court, _ = cout_total(c, 24, 3_600_000)
        long, _ = cout_total(c, 240, 3_600_000)
        self.assertAlmostEqual(court, long)

    def test_avec_swap_tenir_plus_longtemps_coute_plus_cher(self):
        from bot.cli import cout_total
        c = self._cfg(0.0241)
        court, _ = cout_total(c, 24, 3_600_000)
        long, _ = cout_total(c, 240, 3_600_000)
        self.assertGreater(long, court)

    def test_le_portage_forex_sur_trois_jours_egale_le_spread(self):
        """Constat central de la calibration OANDA : sur 3 jours, le swap
        pese autant que l'aller-retour."""
        from bot.cli import cout_total
        c = self._cfg(0.0241)
        total, _ = cout_total(c, 72, 3_600_000)
        aller_retour = 2 * (c.portfolio.fee_rate + c.portfolio.slippage_bps / 10_000.0)
        portage = total - aller_retour
        self.assertGreater(portage, aller_retour * 0.8)
        self.assertLess(portage, aller_retour * 1.2)

    def test_forex_reste_bien_moins_cher_que_crypto_meme_avec_portage(self):
        from pathlib import Path
        from bot.cli import cout_total
        from bot.config import load_config
        racine = Path(__file__).resolve().parent.parent
        fx = load_config(str(racine / "config.oanda.json"))
        crypto = load_config(str(racine / "config.json"))
        c_fx, _ = cout_total(fx, 72, 3_600_000)
        c_cr, _ = cout_total(crypto, 72, 3_600_000)
        self.assertLess(c_fx * 5, c_cr)

    def test_swap_en_pourcentage_refuse(self):
        """2.41 au lieu de 0.0241 doit planter, pas produire un cout 100 fois trop grand."""
        from bot.config import validate
        c = self._cfg(2.41)
        with self.assertRaises(ValueError):
            validate(c)


if __name__ == "__main__":
    unittest.main()
