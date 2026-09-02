"""La configuration doit refuser tot ce qui produirait un backtest faux.

Une config incoherente qui passe silencieusement, c'est des semaines de
resultats a jeter sans savoir pourquoi.
"""

import json
import tempfile
import unittest
from pathlib import Path

from bot.config import Config, load_config, validate


def write_cfg(data: dict) -> str:
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump(data, tmp)
    tmp.close()
    return tmp.name


class TestConfig(unittest.TestCase):
    def test_les_deux_configs_livrees_sont_valides(self):
        root = Path(__file__).resolve().parent.parent
        for name in ("config.json", "config.ci.json"):
            cfg = load_config(str(root / name))
            self.assertIn(cfg.market.adapter, ("binance", "coinbase", "csv", "synthetic"))

    def test_config_ci_utilise_coinbase(self):
        """Binance renvoie 451 depuis les IP americaines : la config
        d'integration continue ne doit jamais l'utiliser."""
        root = Path(__file__).resolve().parent.parent
        cfg = load_config(str(root / "config.ci.json"))
        self.assertEqual(cfg.market.adapter, "coinbase")

    def test_les_deux_configs_ont_la_meme_strategie(self):
        """Si la strategie differe entre local et CI, le forward test ne
        teste pas ce que tu as backteste."""
        root = Path(__file__).resolve().parent.parent
        a = load_config(str(root / "config.json"))
        b = load_config(str(root / "config.ci.json"))
        self.assertEqual(a.strategy, b.strategy)
        self.assertEqual(a.risk, b.risk)
        self.assertEqual(a.scoring, b.scoring)
        self.assertEqual(a.portfolio, b.portfolio)

    def test_symbole_coinbase_mal_forme_refuse(self):
        cfg = Config()
        cfg.market.adapter = "coinbase"
        cfg.market.symbol = "BTCUSDT"
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_symbole_binance_mal_forme_refuse(self):
        cfg = Config()
        cfg.market.adapter = "binance"
        cfg.market.symbol = "BTC-USD"
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_ema_rapide_plus_lente_que_la_lente_refusee(self):
        cfg = Config()
        cfg.strategy.ema_fast = 50
        cfg.strategy.ema_slow = 20
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_risque_par_trade_absurde_refuse(self):
        cfg = Config()
        cfg.risk.risk_per_trade_pct = 0.9
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_ratios_de_decoupage_incoherents_refuses(self):
        cfg = Config()
        cfg.backtest.train_ratio = 0.8
        with self.assertRaises(ValueError):
            validate(cfg)

    def test_cle_inconnue_refusee(self):
        """Une faute de frappe dans config.json doit faire echouer le
        lancement, pas etre ignoree en silence."""
        path = write_cfg({"strategy": {"ema_fastt": 10}})
        with self.assertRaises(ValueError):
            load_config(path)

    def test_valeurs_par_defaut_si_cle_absente(self):
        path = write_cfg({"strategy": {"ema_fast": 5}})
        cfg = load_config(path)
        self.assertEqual(cfg.strategy.ema_fast, 5)
        self.assertEqual(cfg.strategy.ema_slow, 26)


if __name__ == "__main__":
    unittest.main()
