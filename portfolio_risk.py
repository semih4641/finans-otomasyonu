"""Portfolio-level risk management: correlation, sector limits, drawdown controls, total exposure."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Optional

import numpy as np
import pandas as pd

from risk import RiskConfig, RiskGate, PositionRisk, load_risk_config

logger = logging.getLogger(__name__)

DEFAULT_MAX_SECTOR_EXPOSURE_PCT = 30.0
DEFAULT_MAX_CORRELATION_EXPOSURE_PCT = 40.0
DEFAULT_MAX_PORTFOLIO_RISK_PCT = 6.0
DEFAULT_MAX_DRAWDOWN_PCT = 10.0
DEFAULT_CORRELATION_LOOKBACK_DAYS = 60
DEFAULT_MIN_CORRELATION_THRESHOLD = 0.7


@dataclass(frozen=True)
class PortfolioConfig:
    """Portfolio-level risk settings."""
    max_sector_exposure_pct: float = DEFAULT_MAX_SECTOR_EXPOSURE_PCT
    max_correlation_exposure_pct: float = DEFAULT_MAX_CORRELATION_EXPOSURE_PCT
    max_portfolio_risk_pct: float = DEFAULT_MAX_PORTFOLIO_RISK_PCT
    max_drawdown_pct: float = DEFAULT_MAX_DRAWDOWN_PCT
    correlation_lookback_days: int = DEFAULT_CORRELATION_LOOKBACK_DAYS
    min_correlation_threshold: float = DEFAULT_MIN_CORRELATION_THRESHOLD
    sector_mapping: dict[str, str] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return all([
            0 < self.max_sector_exposure_pct <= 100,
            0 < self.max_correlation_exposure_pct <= 100,
            0 < self.max_portfolio_risk_pct <= 100,
            0 < self.max_drawdown_pct <= 100,
            self.correlation_lookback_days > 0,
            0 <= self.min_correlation_threshold <= 1,
        ])


def load_portfolio_config(environ: Optional[Mapping[str, str]] = None) -> PortfolioConfig:
    import os
    from dotenv import load_dotenv
    if environ is None:
        load_dotenv()
    env = os.environ if environ is None else environ

    def _parse_float(name: str, default: float, maximum: float = 100.0) -> float:
        raw = env.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            logger.error("❌ Portfolio config: %s geçersiz sayı: %r", name, raw)
            return default
        if not math.isfinite(value) or value <= 0 or value > maximum:
            logger.error("❌ Portfolio config: %s 0-%s aralığında olmalı: %r", name, maximum, raw)
            return default
        return value

    def _parse_int(name: str, default: int) -> int:
        raw = env.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.error("❌ Portfolio config: %s tam sayı olmalı: %r", name, raw)
            return default
        if value <= 0:
            logger.error("❌ Portfolio config: %s pozitif olmalı: %r", name, raw)
            return default
        return value

    sector_mapping: dict[str, str] = {}
    raw_sectors = env.get("SECTOR_MAPPING", "")
    if raw_sectors:
        for pair in raw_sectors.split(","):
            if ":" in pair:
                sym, sec = pair.split(":", 1)
                sector_mapping[sym.strip().upper()] = sec.strip()

    return PortfolioConfig(
        max_sector_exposure_pct=_parse_float("MAX_SECTOR_EXPOSURE_PCT", DEFAULT_MAX_SECTOR_EXPOSURE_PCT),
        max_correlation_exposure_pct=_parse_float("MAX_CORRELATION_EXPOSURE_PCT", DEFAULT_MAX_CORRELATION_EXPOSURE_PCT),
        max_portfolio_risk_pct=_parse_float("MAX_PORTFOLIO_RISK_PCT", DEFAULT_MAX_PORTFOLIO_RISK_PCT),
        max_drawdown_pct=_parse_float("MAX_DRAWDOWN_PCT", DEFAULT_MAX_DRAWDOWN_PCT),
        correlation_lookback_days=_parse_int("CORRELATION_LOOKBACK_DAYS", DEFAULT_CORRELATION_LOOKBACK_DAYS),
        min_correlation_threshold=_parse_float("MIN_CORRELATION_THRESHOLD", DEFAULT_MIN_CORRELATION_THRESHOLD, maximum=1.0),
        sector_mapping=sector_mapping,
    )


PORTFOLIO_CONFIG = load_portfolio_config()


def get_portfolio_config() -> PortfolioConfig:
    return PORTFOLIO_CONFIG


@dataclass(frozen=True)
class PortfolioGate:
    allowed: bool
    code: str
    message: str
    kill_switch: bool = False
    details: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PortfolioState:
    """Current portfolio snapshot for risk checks."""
    positions: list[Mapping[str, Any]]
    equity: float
    peak_equity: float
    daily_pnl: float
    account_size: float
    price_data: dict[str, pd.DataFrame] = field(default_factory=dict)


def _get_sector(symbol: str, config: PortfolioConfig) -> str:
    return config.sector_mapping.get(symbol.upper(), "UNKNOWN")


def _calculate_correlation_matrix(price_data: dict[str, pd.DataFrame], lookback: int) -> pd.DataFrame:
    """Calculate correlation matrix from close prices."""
    if not price_data:
        return pd.DataFrame()

    closes = {}
    for symbol, df in price_data.items():
        if df is not None and len(df) >= lookback:
            close_col = "close" if "close" in df.columns else "Close"
            if close_col in df.columns:
                closes[symbol] = df[close_col].tail(lookback).pct_change().dropna()

    if len(closes) < 2:
        return pd.DataFrame()

    returns_df = pd.DataFrame(closes)
    returns_df = returns_df.dropna(axis=1, how="all")
    if returns_df.shape[1] < 2:
        return pd.DataFrame()

    return returns_df.corr()


def _portfolio_var(positions: list[Mapping], price_data: dict, config: PortfolioConfig) -> float:
    """Estimate portfolio VaR (95%) using parametric method with correlation."""
    if not positions:
        return 0.0

    risk_config = load_risk_config()
    if not risk_config.is_valid or risk_config.account_size is None:
        return 0.0

    weights = {}
    for pos in positions:
        symbol = pos.get("symbol", "").split(":")[-1]
        qty = pos.get("quantity") or pos.get("size")
        entry = pos.get("entry") or pos.get("entry_price")
        if qty and entry and qty > 0 and entry > 0:
            notional = qty * entry
            weights[symbol] = notional

    if not weights:
        return 0.0

    total_notional = sum(weights.values())
    if total_notional <= 0:
        return 0.0

    w = np.array([weights.get(s, 0) / total_notional for s in weights])
    symbols = list(weights.keys())

    corr = _calculate_correlation_matrix(price_data, config.correlation_lookback_days)
    if corr.empty or not set(symbols).issubset(set(corr.index)):
        return sum(weights.values()) * 0.02

    corr_matrix = corr.loc[symbols, symbols].values
    vols = np.array([0.02] * len(symbols))

    cov = np.outer(vols, vols) * corr_matrix
    portfol_vol = math.sqrt(w @ cov @ w)
    var_95 = 1.645 * portfol_vol * total_notional

    return var_95


def sector_exposure_gate(
    new_position: Mapping,
    portfolio: PortfolioState,
    config: Optional[PortfolioConfig] = None,
) -> PortfolioGate:
    """Check if adding position would exceed sector concentration limit."""
    cfg = config or PORTFOLIO_CONFIG
    if not cfg.is_valid:
        return PortfolioGate(False, "CONFIG_INVALID", "portfolio config geçersiz.", True)

    symbol = new_position.get("symbol", "").split(":")[-1]
    sector = _get_sector(symbol, cfg)
    if sector == "UNKNOWN":
        return PortfolioGate(True, "SECTOR_UNKNOWN", "sektör bilinmiyor, kontrol atlandı.", details={"sector": sector})

    entry = new_position.get("entry_num") or new_position.get("entry")
    qty = new_position.get("quantity")
    if not entry or not qty or entry <= 0 or qty <= 0:
        return PortfolioGate(False, "INVALID_POSITION", "yeni pozisyon entry/qty geçersiz.", details={"entry": entry, "qty": qty})

    new_notional = entry * qty

    sector_notional = 0.0
    for pos in portfolio.positions:
        sym = pos.get("symbol", "").split(":")[-1]
        if _get_sector(sym, cfg) == sector:
            q = pos.get("quantity") or pos.get("size")
            e = pos.get("entry") or pos.get("entry_price")
            if q and e and q > 0 and e > 0:
                sector_notional += q * e

    total_notional = 0.0
    for pos in portfolio.positions:
        q = pos.get("quantity") or pos.get("size")
        e = pos.get("entry") or pos.get("entry_price")
        if q and e and q > 0 and e > 0:
            total_notional += q * e
    total_notional += new_notional

    if total_notional <= 0:
        return PortfolioGate(True, "NO_EXPOSURE", "toplam pozisyon yok.")

    sector_pct = (sector_notional + new_notional) / total_notional * 100

    if sector_pct > cfg.max_sector_exposure_pct:
        return PortfolioGate(
            False,
            "SECTOR_LIMIT_EXCEEDED",
            f"sektör konsantrasyonu aşıldı: {sector} %{sector_pct:.1f} > %{cfg.max_sector_exposure_pct:.1f}",
            True,
            details={"sector": sector, "current_pct": sector_pct, "limit_pct": cfg.max_sector_exposure_pct},
        )

    return PortfolioGate(
        True,
        "SECTOR_LIMIT_OK",
        f"sektör limiti uygun: {sector} %{sector_pct:.1f} / %{cfg.max_sector_exposure_pct:.1f}",
        details={"sector": sector, "current_pct": sector_pct, "limit_pct": cfg.max_sector_exposure_pct},
    )


def correlation_exposure_gate(
    new_position: Mapping,
    portfolio: PortfolioState,
    config: Optional[PortfolioConfig] = None,
) -> PortfolioGate:
    """Check if adding position would create excessive correlated exposure."""
    cfg = config or PORTFOLIO_CONFIG
    if not cfg.is_valid:
        return PortfolioGate(False, "CONFIG_INVALID", "portfolio config geçersiz.", True)

    symbol = new_position.get("symbol", "").split(":")[-1]
    entry = new_position.get("entry_num") or new_position.get("entry")
    qty = new_position.get("quantity")
    if not entry or not qty or entry <= 0 or qty <= 0:
        return PortfolioGate(False, "INVALID_POSITION", "yeni pozisyon entry/qty geçersiz.")

    new_notional = entry * qty

    corr = _calculate_correlation_matrix(portfolio.price_data, cfg.correlation_lookback_days)
    if corr.empty or symbol not in corr.index:
        return PortfolioGate(True, "CORR_DATA_MISSING", "korelasyon verisi yok, kontrol atlandı.")

    correlated_notional = 0.0
    total_notional = new_notional
    correlated_symbols = []

    for pos in portfolio.positions:
        sym = pos.get("symbol", "").split(":")[-1]
        q = pos.get("quantity") or pos.get("size")
        e = pos.get("entry") or pos.get("entry_price")
        if q and e and q > 0 and e > 0:
            notional = q * e
            total_notional += notional
            if sym in corr.index and corr.loc[symbol, sym] >= cfg.min_correlation_threshold:
                correlated_notional += notional
                correlated_symbols.append(sym)

    if total_notional <= 0:
        return PortfolioGate(True, "NO_EXPOSURE", "toplam pozisyon yok.")

    corr_pct = correlated_notional / total_notional * 100

    if corr_pct > cfg.max_correlation_exposure_pct:
        return PortfolioGate(
            False,
            "CORRELATION_LIMIT_EXCEEDED",
            f"korelasyon konsantrasyonu aşıldı: %{corr_pct:.1f} > %{cfg.max_correlation_exposure_pct:.1f} "
            f"(eşik >={cfg.min_correlation_threshold:.2f})",
            True,
            details={"correlated_pct": corr_pct, "limit_pct": cfg.max_correlation_exposure_pct, "symbols": correlated_symbols},
        )

    return PortfolioGate(
        True,
        "CORRELATION_LIMIT_OK",
        f"korelasyon limiti uygun: %{corr_pct:.1f} / %{cfg.max_correlation_exposure_pct:.1f}",
        details={"correlated_pct": corr_pct, "limit_pct": cfg.max_correlation_exposure_pct, "symbols": correlated_symbols},
    )


def portfolio_risk_gate(
    new_position: Mapping,
    portfolio: PortfolioState,
    risk_config: Optional[RiskConfig] = None,
    portfolio_config: Optional[PortfolioConfig] = None,
) -> PortfolioGate:
    """Check if total portfolio risk (sum of position risks) would exceed limit."""
    rcfg = risk_config or load_risk_config()
    pcfg = portfolio_config or PORTFOLIO_CONFIG

    if not rcfg.is_valid or rcfg.account_size is None:
        return PortfolioGate(False, "RISK_CONFIG_INVALID", "risk config geçersiz veya ACCOUNT_SIZE yok.", True)
    if not pcfg.is_valid:
        return PortfolioGate(False, "PORTFOLIO_CONFIG_INVALID", "portfolio config geçersiz.", True)

    current_risk = 0.0
    for pos in portfolio.positions:
        entry = pos.get("entry") or pos.get("entry_price")
        sl = pos.get("sl") or pos.get("stop_loss")
        qty = pos.get("quantity") or pos.get("size")
        if entry and sl and qty and entry > sl > 0 and qty > 0:
            risk_per_unit = entry - sl
            current_risk += risk_per_unit * qty

    new_entry = new_position.get("entry_num") or new_position.get("entry")
    new_sl = new_position.get("sl") or new_position.get("stop_loss")
    new_qty = new_position.get("quantity")
    if new_entry and new_sl and new_qty and new_entry > new_sl > 0 and new_qty > 0:
        new_risk = (new_entry - new_sl) * new_qty
    else:
        new_risk = 0.0

    total_risk = current_risk + new_risk
    risk_pct = total_risk / rcfg.account_size * 100 if rcfg.account_size > 0 else float("inf")

    if risk_pct > pcfg.max_portfolio_risk_pct:
        return PortfolioGate(
            False,
            "PORTFOLIO_RISK_EXCEEDED",
            f"toplam portföy riski aşıldı: %{risk_pct:.2f} > %{pcfg.max_portfolio_risk_pct:.2f}",
            True,
            details={"total_risk_pct": risk_pct, "limit_pct": pcfg.max_portfolio_risk_pct, "current_risk": current_risk, "new_risk": new_risk},
        )

    return PortfolioGate(
        True,
        "PORTFOLIO_RISK_OK",
        f"portföy riski uygun: %{risk_pct:.2f} / %{pcfg.max_portfolio_risk_pct:.2f}",
        details={"total_risk_pct": risk_pct, "limit_pct": pcfg.max_portfolio_risk_pct},
    )


def drawdown_gate(
    portfolio: PortfolioState,
    config: Optional[PortfolioConfig] = None,
) -> PortfolioGate:
    """Kill switch if portfolio drawdown exceeds limit."""
    cfg = config or PORTFOLIO_CONFIG
    if not cfg.is_valid:
        return PortfolioGate(False, "CONFIG_INVALID", "portfolio config geçersiz.", True)

    if portfolio.peak_equity <= 0:
        return PortfolioGate(True, "NO_PEAK", "peak equity hesaplanamadı.")

    drawdown_pct = (portfolio.peak_equity - portfolio.equity) / portfolio.peak_equity * 100

    if drawdown_pct >= cfg.max_drawdown_pct:
        return PortfolioGate(
            False,
            "MAX_DRAWDOWN_EXCEEDED",
            f"KILL SWITCH: max drawdown aşıldı: %{drawdown_pct:.2f} >= %{cfg.max_drawdown_pct:.2f}",
            True,
            details={"drawdown_pct": drawdown_pct, "limit_pct": cfg.max_drawdown_pct, "equity": portfolio.equity, "peak": portfolio.peak_equity},
        )

    return PortfolioGate(
        True,
        "DRAWDOWN_OK",
        f"drawdown kontrolü uygun: %{drawdown_pct:.2f} / %{cfg.max_drawdown_pct:.2f}",
        details={"drawdown_pct": drawdown_pct, "limit_pct": cfg.max_drawdown_pct},
    )


def portfolio_gate(
    new_position: Mapping,
    portfolio: PortfolioState,
    risk_config: Optional[RiskConfig] = None,
    portfolio_config: Optional[PortfolioConfig] = None,
) -> PortfolioGate:
    """Apply all portfolio-level gates in sequence."""
    gates = [
        drawdown_gate(portfolio, portfolio_config),
        portfolio_risk_gate(new_position, portfolio, risk_config, portfolio_config),
        sector_exposure_gate(new_position, portfolio, portfolio_config),
        correlation_exposure_gate(new_position, portfolio, portfolio_config),
    ]

    for gate in gates:
        if not gate.allowed:
            return gate

    return PortfolioGate(
        True,
        "ALL_GATES_PASSED",
        "tüm portföy risk kapıları geçildi.",
        details={g.code: g.details for g in gates},
    )


def calculate_portfolio_metrics(portfolio: PortfolioState) -> dict:
    """Calculate current portfolio risk metrics for reporting."""
    metrics = {
        "total_positions": len(portfolio.positions),
        "total_notional": 0.0,
        "total_risk": 0.0,
        "sector_exposure": {},
        "correlation_clusters": [],
        "var_95": 0.0,
        "drawdown_pct": 0.0,
    }

    if portfolio.peak_equity > 0:
        metrics["drawdown_pct"] = round((portfolio.peak_equity - portfolio.equity) / portfolio.peak_equity * 100, 2)

    positions_by_sector: dict[str, float] = {}
    all_notional = 0.0
    all_risk = 0.0

    for pos in portfolio.positions:
        symbol = pos.get("symbol", "").split(":")[-1]
        entry = pos.get("entry") or pos.get("entry_price")
        sl = pos.get("sl") or pos.get("stop_loss")
        qty = pos.get("quantity") or pos.get("size")
        if entry and qty and entry > 0 and qty > 0:
            notional = entry * qty
            all_notional += notional
            sector = _get_sector(symbol, PORTFOLIO_CONFIG)
            positions_by_sector[sector] = positions_by_sector.get(sector, 0) + notional

            if sl and entry > sl > 0:
                all_risk += (entry - sl) * qty

    metrics["total_notional"] = round(all_notional, 2)
    metrics["total_risk"] = round(all_risk, 2)
    metrics["sector_exposure"] = {k: round(v, 2) for k, v in positions_by_sector.items()}

    if PORTFOLIO_CONFIG.is_valid and portfolio.price_data:
        metrics["var_95"] = round(_portfolio_var(portfolio.positions, portfolio.price_data, PORTFOLIO_CONFIG), 2)

    return metrics