"""Fabrique d'adaptateurs de marche.

Pour brancher un nouveau marche (OANDA forex, Alpaca actions...),
il suffit d'ecrire une classe qui herite de MarketAdapter et de
l'ajouter au dictionnaire ci-dessous. Rien d'autre ne bouge.
"""

from __future__ import annotations

from ..config import Config
from .base import MarketAdapter, MarketDataError, TIMEFRAME_MS, sanity_check
from .binance import BinanceAdapter
from .coinbase import CoinbaseAdapter
from .csv_file import CsvAdapter
from .oanda import OandaAdapter
from .synthetic import SyntheticAdapter

__all__ = [
    "MarketAdapter",
    "MarketDataError",
    "TIMEFRAME_MS",
    "sanity_check",
    "BinanceAdapter",
    "CoinbaseAdapter",
    "OandaAdapter",
    "CsvAdapter",
    "SyntheticAdapter",
    "build_adapter",
]


def build_adapter(cfg: Config) -> MarketAdapter:
    m = cfg.market
    if m.adapter == "binance":
        return BinanceAdapter(m.symbol, m.timeframe, str(cfg.path(m.cache_dir)))
    if m.adapter == "coinbase":
        return CoinbaseAdapter(m.symbol, m.timeframe, str(cfg.path(m.cache_dir)))
    if m.adapter == "oanda":
        return OandaAdapter(
            m.symbol, m.timeframe, str(cfg.path(m.cache_dir)),
            environnement=m.oanda_env,
        )
    if m.adapter == "csv":
        if not m.csv_path:
            raise MarketDataError("market.csv_path est obligatoire avec l'adaptateur csv.")
        return CsvAdapter(
            m.symbol, m.timeframe, str(cfg.path(m.csv_path)),
            tz_offset_hours=m.csv_tz_offset_hours,
        )
    if m.adapter == "synthetic":
        return SyntheticAdapter(m.symbol, m.timeframe, bars=m.synthetic_bars, seed=m.synthetic_seed)
    raise MarketDataError(f"Adaptateur inconnu : {m.adapter}")
