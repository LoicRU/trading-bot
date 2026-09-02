"""Chargement et validation de la configuration.

Un seul fichier config.json pilote tout le bot. Aucun parametre n'est
ecrit en dur ailleurs dans le code : si tu veux changer le comportement,
c'est ici et nulle part ailleurs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class MarketConfig:
    adapter: str = "binance"          # binance | coinbase | csv | synthetic
    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    history_days: int = 730
    cache_dir: str = "data"
    csv_path: Optional[str] = None
    synthetic_seed: int = 42
    synthetic_bars: int = 8000


@dataclass
class PortfolioConfig:
    initial_cash: float = 10000.0
    fee_rate: float = 0.001           # 0.1 % par cote, applique a l'achat ET a la vente
    slippage_bps: float = 5.0         # points de base de degradation du prix d'execution


@dataclass
class StrategyConfig:
    ema_fast: int = 12
    ema_slow: int = 26
    atr_period: int = 14
    min_atr_pct: float = 0.004        # sous ce seuil le marche est trop plat : on s'abstient
    max_atr_pct: float = 0.08         # au-dessus il est trop chaotique : on s'abstient
    exit_on_cross: bool = True


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.01  # fraction du capital risquee entre l'entree et le stop
    max_position_pct: float = 0.25    # plafond de taille de position en % du capital
    atr_stop_mult: float = 2.0
    atr_tp_mult: float = 3.0
    max_daily_loss_pct: float = 0.03  # au-dela, arret des trades pour la journee
    max_open_positions: int = 1


@dataclass
class ScoringConfig:
    horizon_bars: int = 24            # delai avant d'evaluer une abstention
    opportunity_threshold_r: float = 1.0
    opportunity_weight: float = 0.5
    avoided_weight: float = 0.3
    patience_reward: float = 0.05
    cap_r: float = 3.0


@dataclass
class BacktestConfig:
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    holdout_ratio: float = 0.2


@dataclass
class ReportConfig:
    output_dir: str = "reports"
    timezone_offset_hours: float = 2.0


@dataclass
class RuntimeConfig:
    db_path: str = "data/bot.sqlite"
    kill_switch_file: str = "KILL_SWITCH"
    forward_log: str = "forward_log.jsonl"


@dataclass
class Config:
    market: MarketConfig = field(default_factory=MarketConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    # --- chemins resolus par rapport a la racine du projet -------------
    def path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else ROOT / p

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


_SECTIONS = {
    "market": MarketConfig,
    "portfolio": PortfolioConfig,
    "strategy": StrategyConfig,
    "risk": RiskConfig,
    "scoring": ScoringConfig,
    "backtest": BacktestConfig,
    "report": ReportConfig,
    "runtime": RuntimeConfig,
}


def load_config(path: Optional[str] = None) -> Config:
    """Charge config.json ; toute cle absente reprend sa valeur par defaut."""
    cfg_path = Path(path) if path else ROOT / "config.json"
    raw: Dict[str, Any] = {}
    if cfg_path.exists():
        raw = json.loads(cfg_path.read_text(encoding="utf-8"))

    sections = {}
    for name, klass in _SECTIONS.items():
        data = raw.get(name, {}) or {}
        known = {k: v for k, v in data.items() if k in klass.__annotations__}
        unknown = set(data) - set(known)
        if unknown:
            raise ValueError(
                f"config.json : cles inconnues dans la section '{name}' : {sorted(unknown)}"
            )
        sections[name] = klass(**known)

    cfg = Config(**sections)
    validate(cfg)
    return cfg


def validate(cfg: Config) -> None:
    """Garde-fous : mieux vaut planter au demarrage qu'apres 2 ans de backtest."""
    errors = []

    if cfg.market.adapter not in ("binance", "coinbase", "csv", "synthetic"):
        errors.append(f"market.adapter inconnu : {cfg.market.adapter}")
    if cfg.market.adapter == "coinbase" and "-" not in cfg.market.symbol:
        errors.append(
            f"Coinbase attend un symbole de la forme 'BTC-USD' (recu : {cfg.market.symbol})"
        )
    if cfg.market.adapter == "binance" and "-" in cfg.market.symbol:
        errors.append(
            f"Binance attend un symbole de la forme 'BTCUSDT' (recu : {cfg.market.symbol})"
        )
    if cfg.market.adapter == "csv" and not cfg.market.csv_path:
        errors.append("market.adapter=csv exige market.csv_path")

    if cfg.portfolio.initial_cash <= 0:
        errors.append("portfolio.initial_cash doit etre > 0")
    if not 0 <= cfg.portfolio.fee_rate < 0.1:
        errors.append("portfolio.fee_rate doit etre entre 0 et 0.1")
    if cfg.portfolio.slippage_bps < 0:
        errors.append("portfolio.slippage_bps doit etre >= 0")

    if cfg.strategy.ema_fast >= cfg.strategy.ema_slow:
        errors.append("strategy.ema_fast doit etre strictement < ema_slow")
    if cfg.strategy.atr_period < 2:
        errors.append("strategy.atr_period doit etre >= 2")
    if cfg.strategy.min_atr_pct >= cfg.strategy.max_atr_pct:
        errors.append("strategy.min_atr_pct doit etre < max_atr_pct")

    if not 0 < cfg.risk.risk_per_trade_pct <= 0.1:
        errors.append("risk.risk_per_trade_pct doit etre entre 0 et 0.1 (10 %)")
    if not 0 < cfg.risk.max_position_pct <= 1.0:
        errors.append("risk.max_position_pct doit etre entre 0 et 1")
    if cfg.risk.atr_stop_mult <= 0:
        errors.append("risk.atr_stop_mult doit etre > 0")
    if cfg.risk.max_open_positions < 1:
        errors.append("risk.max_open_positions doit etre >= 1")

    total = cfg.backtest.train_ratio + cfg.backtest.validation_ratio + cfg.backtest.holdout_ratio
    if abs(total - 1.0) > 1e-6:
        errors.append(f"backtest : les trois ratios doivent totaliser 1.0 (actuel : {total})")

    if cfg.scoring.horizon_bars < 1:
        errors.append("scoring.horizon_bars doit etre >= 1")

    if errors:
        raise ValueError("Configuration invalide :\n  - " + "\n  - ".join(errors))
