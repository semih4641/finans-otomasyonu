"""Portfolio-level risk management: correlation, sector limits, drawdown controls, total exposure."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from risk import RiskConfig, load_risk_config

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
    configuration_errors: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        limits = (self.max_sector_exposure_pct, self.max_correlation_exposure_pct,
                  self.max_portfolio_risk_pct, self.max_drawdown_pct)
        threshold = self.min_correlation_threshold
        return (
            not self.configuration_errors
            and all(isinstance(value, Real) and not isinstance(value, bool)
                    and math.isfinite(value) and 0 < value <= 100 for value in limits)
            and isinstance(self.correlation_lookback_days, Integral)
            and not isinstance(self.correlation_lookback_days, bool)
            and self.correlation_lookback_days >= 3
            and isinstance(threshold, Real) and not isinstance(threshold, bool)
            and math.isfinite(threshold) and 0 <= threshold <= 1
        )


def load_portfolio_config(environ: Optional[Mapping[str, str]] = None) -> PortfolioConfig:
    import os
    from dotenv import load_dotenv
    if environ is None:
        load_dotenv()
    env = os.environ if environ is None else environ
    errors: list[str] = []

    def _parse_float(name: str, default: float, maximum: float = 100.0, allow_zero: bool = False) -> float:
        raw = env.get(name)
        if raw is None or not raw.strip():
            return default
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{name} geçersiz sayı")
            logger.error("❌ Portfolio config: %s geçersiz sayı: %r", name, raw)
            return default
        if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero) or value > maximum:
            errors.append(f"{name} geçersiz aralık")
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
            errors.append(f"{name} tam sayı olmalı")
            logger.error("❌ Portfolio config: %s tam sayı olmalı: %r", name, raw)
            return default
        if value < 3:
            errors.append(f"{name} en az 3 olmalı")
            logger.error("❌ Portfolio config: %s en az 3 olmalı: %r", name, raw)
            return default
        return value

    sector_mapping: dict[str, str] = {}
    raw_sectors = env.get("SECTOR_MAPPING", "")
    if raw_sectors:
        for pair in raw_sectors.split(","):
            if ":" in pair:
                sym, sec = pair.split(":", 1)
                if sym.strip() and sec.strip():
                    sector_mapping[sym.strip().upper()] = sec.strip()

    return PortfolioConfig(
        max_sector_exposure_pct=_parse_float("MAX_SECTOR_EXPOSURE_PCT", DEFAULT_MAX_SECTOR_EXPOSURE_PCT),
        max_correlation_exposure_pct=_parse_float("MAX_CORRELATION_EXPOSURE_PCT", DEFAULT_MAX_CORRELATION_EXPOSURE_PCT),
        max_portfolio_risk_pct=_parse_float("MAX_PORTFOLIO_RISK_PCT", DEFAULT_MAX_PORTFOLIO_RISK_PCT),
        max_drawdown_pct=_parse_float("MAX_DRAWDOWN_PCT", DEFAULT_MAX_DRAWDOWN_PCT),
        correlation_lookback_days=_parse_int("CORRELATION_LOOKBACK_DAYS", DEFAULT_CORRELATION_LOOKBACK_DAYS),
        min_correlation_threshold=_parse_float("MIN_CORRELATION_THRESHOLD", DEFAULT_MIN_CORRELATION_THRESHOLD, maximum=1.0, allow_zero=True),
        sector_mapping=sector_mapping,
        configuration_errors=tuple(errors),
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


def _positive_number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) and number > 0 and not isinstance(value, bool) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _position_notional(position: Mapping) -> float | None:
    entry = _positive_number(position.get("entry_num", position.get("entry", position.get("entry_price"))))
    quantity = _positive_number(position.get("quantity", position.get("size")))
    if entry is None or quantity is None:
        return None
    notional = entry * quantity
    return notional if math.isfinite(notional) else None


def _symbol(position: Mapping) -> str:
    return str(position.get("symbol") or "").split(":")[-1].upper()


def _calculate_correlation_matrix(price_data: dict[str, pd.DataFrame], lookback: int) -> pd.DataFrame:
    """Calculate correlation matrix from close prices."""
    if not price_data:
        return pd.DataFrame()

    closes = {}
    for symbol, df in price_data.items():
        if df is not None and len(df) >= lookback:
            close_col = "close" if "close" in df.columns else "Close"
            if close_col in df.columns:
                prices = pd.to_numeric(df[close_col], errors="coerce").tail(lookback)
                closes[symbol.upper()] = prices.where(prices > 0).pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)

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

    weights = {}
    for pos in positions:
        symbol = _symbol(pos)
        notional = _position_notional(pos)
        if notional is not None:
            weights[symbol] = weights.get(symbol, 0.0) + notional

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
    if not np.isfinite(corr_matrix).all():
        return total_notional * 0.02
    vols = np.array([0.02] * len(symbols))

    cov = np.outer(vols, vols) * corr_matrix
    variance = float(w @ cov @ w)
    if variance < 0 or not math.isfinite(variance):
        return total_notional * 0.02
    portfol_vol = math.sqrt(variance)
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

    symbol = _symbol(new_position)
    new_notional = _position_notional(new_position)
    equity = _positive_number(portfolio.equity)
    if not symbol or new_notional is None:
        return PortfolioGate(False, "INVALID_POSITION", "yeni pozisyon entry/qty geçersiz.")
    if equity is None:
        return PortfolioGate(False, "INVALID_EQUITY", "portföy özkaynağı sonlu ve pozitif olmalı.", True)
    sector = _get_sector(symbol, cfg)
    if sector == "UNKNOWN":
        return PortfolioGate(True, "SECTOR_UNKNOWN", "sektör bilinmiyor, kontrol atlandı.", details={"sector": sector})

    sector_notional = 0.0
    for pos in portfolio.positions:
        sym = _symbol(pos)
        if _get_sector(sym, cfg) == sector:
            notional = _position_notional(pos)
            if notional is None:
                return PortfolioGate(False, "INVALID_EXISTING_POSITION", "mevcut sektör pozisyonunun tutarı hesaplanamadı.", True)
            sector_notional += notional

    sector_pct = (sector_notional + new_notional) / equity * 100

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

    symbol = _symbol(new_position)
    new_notional = _position_notional(new_position)
    equity = _positive_number(portfolio.equity)
    if not symbol or new_notional is None:
        return PortfolioGate(False, "INVALID_POSITION", "yeni pozisyon entry/qty geçersiz.")
    if equity is None:
        return PortfolioGate(False, "INVALID_EQUITY", "portföy özkaynağı sonlu ve pozitif olmalı.", True)

    existing_notionals = []
    for pos in portfolio.positions:
        notional = _position_notional(pos)
        if notional is None:
            return PortfolioGate(False, "INVALID_EXISTING_POSITION", "mevcut pozisyonun tutarı hesaplanamadı.", True)
        existing_notionals.append((_symbol(pos), notional))

    corr = _calculate_correlation_matrix(portfolio.price_data, cfg.correlation_lookback_days)
    if (corr.empty or symbol not in corr.index) and not any(_symbol(pos) == symbol for pos in portfolio.positions):
        return PortfolioGate(True, "CORR_DATA_MISSING", "korelasyon verisi yok, kontrol atlandı.")

    correlated_notional = 0.0
    correlated_symbols = []

    for sym, notional in existing_notionals:
        if sym == symbol or (symbol in corr.index and sym in corr.columns and corr.loc[symbol, sym] >= cfg.min_correlation_threshold):
            correlated_notional += notional
            correlated_symbols.append(sym)

    # A cluster includes the proposed position itself, measured against equity.
    corr_pct = (correlated_notional + new_notional) / equity * 100 if correlated_symbols else 0.0

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
    account_size = _positive_number(rcfg.account_size)
    if account_size is None:
        return PortfolioGate(False, "RISK_CONFIG_INVALID", "ACCOUNT_SIZE sonlu ve pozitif olmalı.", True)
    equity = _positive_number(portfolio.equity)
    if equity is None:
        return PortfolioGate(False, "INVALID_EQUITY", "portföy özkaynağı sonlu ve pozitif olmalı.", True)

    current_risk = 0.0
    for pos in portfolio.positions:
        entry = _positive_number(pos.get("entry_num", pos.get("entry", pos.get("entry_price"))))
        sl = _positive_number(pos.get("sl", pos.get("stop_loss")))
        qty = _positive_number(pos.get("quantity", pos.get("size")))
        if entry is None or sl is None or qty is None:
            return PortfolioGate(False, "INVALID_EXISTING_POSITION", "mevcut pozisyonun stop riski hesaplanamadı.", True)
        current_risk += max(entry - sl, 0.0) * qty

    new_entry = _positive_number(new_position.get("entry_num", new_position.get("entry")))
    new_sl = _positive_number(new_position.get("sl", new_position.get("stop_loss")))
    new_qty = _positive_number(new_position.get("quantity"))
    if new_entry is not None and new_sl is not None and new_qty is not None and new_entry > new_sl:
        new_risk = (new_entry - new_sl) * new_qty
    else:
        return PortfolioGate(False, "INVALID_POSITION", "yeni pozisyon entry/stop/qty geçersiz.")

    total_risk = current_risk + new_risk
    risk_pct = total_risk / equity * 100

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

    peak = _positive_number(portfolio.peak_equity)
    try:
        equity = float(portfolio.equity)
    except (TypeError, ValueError):
        equity = float("nan")
    if peak is None or not math.isfinite(equity):
        return PortfolioGate(False, "INVALID_EQUITY", "özkaynak veya zirve değeri geçersiz.", True)

    drawdown_pct = max(0.0, (peak - equity) / peak * 100)

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
    gates = []
    for check in (
        lambda: drawdown_gate(portfolio, portfolio_config),
        lambda: portfolio_risk_gate(new_position, portfolio, risk_config, portfolio_config),
        lambda: sector_exposure_gate(new_position, portfolio, portfolio_config),
        lambda: correlation_exposure_gate(new_position, portfolio, portfolio_config),
    ):
        gate = check()
        if not gate.allowed:
            return gate
        gates.append(gate)

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
