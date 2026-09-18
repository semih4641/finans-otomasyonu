"""Safety-first position sizing and risk gates.

This module only calculates and reports risk.  It never connects to an
exchange, sends an order, or sends a Telegram message.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict, dataclass
from datetime import date, datetime
from numbers import Integral, Real
from typing import Any, Iterable, Mapping, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEFAULT_POSITION_RISK_PCT = 1.0
DEFAULT_MAX_OPEN_POSITIONS = 5
DEFAULT_DAILY_LOSS_LIMIT_PCT = 2.0


def _positive_finite(value: Any, maximum: Optional[float] = None) -> bool:
    """Validate direct configuration values as well as environment values."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return False
    try:
        return math.isfinite(value) and value > 0 and (
            maximum is None or value <= maximum
        )
    except (TypeError, ValueError, OverflowError):
        return False


@dataclass(frozen=True)
class RiskConfig:
    """Validated risk settings loaded from the environment."""

    account_size: Optional[float] = None
    position_risk_pct: float = DEFAULT_POSITION_RISK_PCT
    max_open_positions: int = DEFAULT_MAX_OPEN_POSITIONS
    daily_loss_limit_pct: float = DEFAULT_DAILY_LOSS_LIMIT_PCT
    configuration_errors: tuple[str, ...] = ()

    @property
    def is_valid(self) -> bool:
        return (
            not self.configuration_errors
            and (self.account_size is None or _positive_finite(self.account_size))
            and _positive_finite(self.position_risk_pct, 100.0)
            and _positive_finite(self.daily_loss_limit_pct, 100.0)
            and isinstance(self.max_open_positions, Integral)
            and not isinstance(self.max_open_positions, bool)
            and self.max_open_positions >= 1
        )


@dataclass(frozen=True)
class PositionRisk:
    """A calculation result; ``quantity`` is never an order instruction."""

    entry: Optional[float]
    stop_loss: Optional[float]
    risk_per_unit: Optional[float]
    risk_amount: Optional[float]
    quantity: Optional[float]
    notional_value: Optional[float]
    status: str
    message: str

    @property
    def is_calculable(self) -> bool:
        return self.status == "OK"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RiskGate:
    """Explicit result of a risk limit check."""

    allowed: bool
    code: str
    message: str
    kill_switch: bool = False


def _parse_positive_float(
    name: str,
    raw: Optional[str],
    default: float,
    errors: list[str],
    *,
    maximum: Optional[float] = None,
) -> float:
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        message = f"{name} geçersiz sayı: {raw!r}"
        errors.append(message)
        logger.error("❌ Risk yapılandırması: %s", message)
        return default
    if not math.isfinite(value) or value <= 0 or (
        maximum is not None and value > maximum
    ):
        bound = f" ve <= {maximum:g}" if maximum is not None else ""
        message = f"{name} pozitif{bound} olmalı: {raw!r}"
        errors.append(message)
        logger.error("❌ Risk yapılandırması: %s", message)
        return default
    return value


def load_risk_config(environ: Optional[Mapping[str, str]] = None) -> RiskConfig:
    """Load risk settings without ever assuming a real account exists."""
    if environ is None:
        load_dotenv()
    env = os.environ if environ is None else environ
    errors: list[str] = []

    account_size: Optional[float] = None
    raw_account = env.get("ACCOUNT_SIZE")
    if raw_account is not None and raw_account.strip():
        try:
            account_size = float(raw_account)
        except (TypeError, ValueError):
            message = f"ACCOUNT_SIZE geçersiz sayı: {raw_account!r}"
            errors.append(message)
            logger.error("❌ Risk yapılandırması: %s", message)
        else:
            if not math.isfinite(account_size) or account_size <= 0:
                message = f"ACCOUNT_SIZE pozitif ve sonlu olmalı: {raw_account!r}"
                errors.append(message)
                logger.error("❌ Risk yapılandırması: %s", message)
                account_size = None

    position_risk_pct = _parse_positive_float(
        "POSITION_RISK_PCT",
        env.get("POSITION_RISK_PCT"),
        DEFAULT_POSITION_RISK_PCT,
        errors,
        maximum=100.0,
    )
    daily_loss_limit_pct = _parse_positive_float(
        "DAILY_LOSS_LIMIT_PCT",
        env.get("DAILY_LOSS_LIMIT_PCT"),
        DEFAULT_DAILY_LOSS_LIMIT_PCT,
        errors,
        maximum=100.0,
    )

    max_open_positions = DEFAULT_MAX_OPEN_POSITIONS
    raw_max = env.get("MAX_OPEN_POSITIONS")
    if raw_max is not None and raw_max.strip():
        try:
            max_open_positions = int(raw_max)
        except (TypeError, ValueError):
            message = f"MAX_OPEN_POSITIONS tam sayı olmalı: {raw_max!r}"
            errors.append(message)
            logger.error("❌ Risk yapılandırması: %s", message)
        else:
            if max_open_positions < 1:
                message = f"MAX_OPEN_POSITIONS en az 1 olmalı: {raw_max!r}"
                errors.append(message)
                logger.error("❌ Risk yapılandırması: %s", message)
                max_open_positions = DEFAULT_MAX_OPEN_POSITIONS

    return RiskConfig(
        account_size=account_size,
        position_risk_pct=position_risk_pct,
        max_open_positions=max_open_positions,
        daily_loss_limit_pct=daily_loss_limit_pct,
        configuration_errors=tuple(errors),
    )


RISK_CONFIG = load_risk_config()


def get_risk_config() -> RiskConfig:
    """Return the environment configuration captured at module load time."""
    return RISK_CONFIG


def calculate_position_risk(
    entry: Any,
    stop_loss: Any,
    config: Optional[RiskConfig] = None,
) -> PositionRisk:
    """Calculate risk budget and quantity from entry and stop-loss.

    Missing ``ACCOUNT_SIZE`` deliberately produces no quantity.  This avoids
    silently treating an unknown account as a live or zero-sized account.
    """
    cfg = config or RISK_CONFIG
    try:
        entry_value = float(entry)
        stop_value = float(stop_loss)
    except (TypeError, ValueError, OverflowError):
        logger.error("❌ Risk hesabı: entry ve stop_loss sayısal olmalı.")
        return PositionRisk(
            None,
            None,
            None,
            None,
            None,
            None,
            "INVALID_INPUT",
            "entry ve stop_loss sonlu sayılar olmalı; pozisyon boyutu hesaplanmadı.",
        )

    if not math.isfinite(entry_value) or not math.isfinite(stop_value):
        logger.error("❌ Risk hesabı: entry ve stop_loss sonlu sayılar olmalı.")
        return PositionRisk(
            entry_value,
            stop_value,
            None,
            None,
            None,
            None,
            "INVALID_INPUT",
            "entry ve stop_loss sonlu sayılar olmalı; pozisyon boyutu hesaplanmadı.",
        )

    risk_per_unit = entry_value - stop_value
    if entry_value <= 0 or stop_value <= 0 or risk_per_unit <= 0:
        logger.error(
            "❌ Risk hesabı: geçersiz entry/stop mesafesi (entry=%s, stop=%s).",
            entry_value,
            stop_value,
        )
        return PositionRisk(
            entry_value,
            stop_value,
            risk_per_unit,
            None,
            None,
            None,
            "INVALID_INPUT",
            "entry stop_loss değerinden büyük ve pozitif olmalı; pozisyon boyutu hesaplanmadı.",
        )
    if not cfg.is_valid:
        logger.error("❌ Risk hesabı: risk yapılandırması geçersiz.")
        return PositionRisk(
            entry_value,
            stop_value,
            risk_per_unit,
            None,
            None,
            None,
            "CONFIG_INVALID",
            "risk yapılandırması geçersiz; pozisyon boyutu hesaplanmadı.",
        )
    if cfg.account_size is None:
        return PositionRisk(
            entry_value,
            stop_value,
            risk_per_unit,
            None,
            None,
            None,
            "ACCOUNT_SIZE_UNSET",
            "ACCOUNT_SIZE tanımlı değil; pozisyon miktarı hesaplanmadı.",
        )

    risk_amount = cfg.account_size * (cfg.position_risk_pct / 100.0)
    quantity = risk_amount / risk_per_unit
    notional = quantity * entry_value
    if not all(_positive_finite(value) for value in (risk_amount, quantity, notional)):
        logger.error("❌ Risk hesabı: pozisyon boyutu sonlu ve pozitif hesaplanamadı.")
        return PositionRisk(
            entry_value,
            stop_value,
            risk_per_unit,
            None,
            None,
            None,
            "INVALID_INPUT",
            "pozisyon boyutu sonlu ve pozitif hesaplanamadı; giriş değerlerini kontrol edin.",
        )
    return PositionRisk(
        entry_value,
        stop_value,
        risk_per_unit,
        risk_amount,
        quantity,
        notional,
        "OK",
        "pozisyon boyutu yalnızca risk bilgisi olarak hesaplandı; canlı emir gönderilmedi.",
    )


def format_position_risk(result: PositionRisk) -> str:
    """Format a safe, explicit risk line for an existing bot message."""
    if result.status == "OK":
        return (
            f"  ⚖️ Boyut önerisi: ~{result.quantity:,.4g} birim "
            f"(≈{result.notional_value:,.2f} değerinde; "
            f"risk tutarı ≈{result.risk_amount:,.2f})\n"
        )
    return f"  ⚠️ Risk bilgisi: {result.message}\n"


def count_open_positions(positions: Iterable[Any]) -> int:
    """Purely count active positions in mappings or opaque position objects."""
    count = 0
    closed_states = {"closed", "settled", "cancelled", "canceled", "done"}
    for position in positions or ():
        if isinstance(position, Mapping):
            if position.get("is_open") is False:
                continue
            state = str(position.get("status", position.get("state", ""))).lower()
            if state in closed_states:
                continue
        count += 1
    return count


def daily_realized_net_pnl(
    records: Iterable[Mapping[str, Any]],
    target_date: date,
) -> float:
    """Purely sum realized net P/L amounts closed on ``target_date``.

    Records must expose ``closed_at`` (ISO/date-compatible) and one of
    ``realized_net_pnl``, ``net_pnl_amount`` or ``net_pnl``.  Percentage
    fields such as ``net_pnl_pct`` are not interpreted as currency.
    """
    total = 0.0
    for record in records:
        closed_at = record.get("closed_at")
        if isinstance(closed_at, datetime):
            record_date = closed_at.date()
        elif isinstance(closed_at, date):
            record_date = closed_at
        elif isinstance(closed_at, str):
            try:
                record_date = date.fromisoformat(closed_at[:10])
            except ValueError:
                continue
        else:
            continue
        if record_date != target_date:
            continue
        amount = record.get(
            "realized_net_pnl",
            record.get("net_pnl_amount", record.get("net_pnl")),
        )
        try:
            value = float(amount)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            total += value
    return total


def open_position_gate(
    open_position_count: int,
    config: Optional[RiskConfig] = None,
) -> RiskGate:
    """Gate a new position based on the configured open-position limit."""
    cfg = config or RISK_CONFIG
    if not cfg.is_valid:
        return RiskGate(False, "CONFIG_INVALID", "risk yapılandırması geçersiz.", True)
    if (
        isinstance(open_position_count, bool)
        or not isinstance(open_position_count, Integral)
        or open_position_count < 0
    ):
        logger.error("❌ Risk kapısı: açık pozisyon sayısı negatif olmayan bir tam sayı olmalı.")
        return RiskGate(
            False, "INVALID_OPEN_COUNT",
            "açık pozisyon sayısı negatif olmayan bir tam sayı olmalı.", True,
        )
    if open_position_count >= cfg.max_open_positions:
        return RiskGate(
            False,
            "MAX_OPEN_POSITIONS",
            f"limit: {open_position_count}/{cfg.max_open_positions} açık pozisyon; yeni pozisyon kilitlendi.",
            True,
        )
    return RiskGate(
        True,
        "OPEN_POSITION_LIMIT_OK",
        f"açık pozisyon limiti uygun: {open_position_count}/{cfg.max_open_positions}.",
    )


def daily_loss_gate(
    realized_net_pnl: Any,
    config: Optional[RiskConfig] = None,
) -> RiskGate:
    """Gate a new position based on realized daily net P/L."""
    cfg = config or RISK_CONFIG
    if not cfg.is_valid:
        return RiskGate(False, "CONFIG_INVALID", "risk yapılandırması geçersiz.", True)
    if cfg.account_size is None:
        return RiskGate(
            False,
            "ACCOUNT_SIZE_UNSET",
            "ACCOUNT_SIZE tanımlı değil; günlük zarar limiti doğrulanamadı ve işlem kilitlendi.",
            True,
        )
    try:
        pnl = float(realized_net_pnl)
    except (TypeError, ValueError, OverflowError):
        logger.error("❌ Risk kapısı: günlük net P/L sayısal olmalı.")
        return RiskGate(False, "INVALID_DAILY_PNL", "günlük net P/L geçersiz.", True)
    if not math.isfinite(pnl):
        logger.error("❌ Risk kapısı: günlük net P/L sonlu olmalı.")
        return RiskGate(False, "INVALID_DAILY_PNL", "günlük net P/L sonlu olmalı.", True)

    limit_amount = cfg.account_size * (cfg.daily_loss_limit_pct / 100.0)
    if pnl <= -limit_amount:
        return RiskGate(
            False,
            "DAILY_LOSS_LIMIT",
            f"kill-switch: günlük net P/L {pnl:,.2f}; "
            f"limit -{limit_amount:,.2f} ({cfg.daily_loss_limit_pct:g}%).",
            True,
        )
    return RiskGate(
        True,
        "DAILY_LOSS_LIMIT_OK",
        f"günlük zarar limiti uygun: {pnl:,.2f} / -{limit_amount:,.2f}.",
    )


def risk_gate(
    open_position_count: int,
    realized_net_pnl: Any,
    config: Optional[RiskConfig] = None,
) -> RiskGate:
    """Apply both safety gates; no function here can place an order."""
    open_gate = open_position_gate(open_position_count, config)
    if not open_gate.allowed:
        return open_gate
    return daily_loss_gate(realized_net_pnl, config)
