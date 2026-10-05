"""Explainable data and daily-stock anomaly checks; not a manipulation detector.

Only completed candles are accepted. Historical callers omit ``now`` so a
backtest never depends on the wall clock or prices after its decision candle.
"""

from dataclasses import dataclass
import math
import os

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SafetyConfig:
    volume_ratio: float = 5.0
    price_move_pct: float = 6.0
    range_pct: float = 8.0
    stock_max_age_days: float = 7.0
    crypto_max_age_hours: float = 3.0

    @classmethod
    def from_env(cls):
        defaults = cls()
        names = {
            "volume_ratio": "ANOMALY_VOLUME_RATIO",
            "price_move_pct": "ANOMALY_PRICE_MOVE_PCT",
            "range_pct": "ANOMALY_RANGE_PCT",
            "stock_max_age_days": "STOCK_MAX_DATA_AGE_DAYS",
            "crypto_max_age_hours": "CRYPTO_MAX_DATA_AGE_HOURS",
        }
        values = {}
        for field, name in names.items():
            raw = os.getenv(name)
            try:
                value = float(raw) if raw is not None else getattr(defaults, field)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{name} pozitif ve sonlu bir sayı olmalı.") from exc
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} pozitif ve sonlu bir sayı olmalı.")
            values[field] = value
        return cls(**values)


def validated_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a numeric copy, rejecting malformed bars instead of filling them."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Tamamlanmış fiyat geçmişi boş.")
    if (not isinstance(frame.index, pd.DatetimeIndex) or frame.index.hasnans
            or not frame.index.is_unique or not frame.index.is_monotonic_increasing):
        raise ValueError("Mum tarihleri geçerli, benzersiz ve artan sırada olmalı.")
    data = frame.rename(columns=lambda name: str(name).lower()).copy()
    required = ["open", "high", "low", "close", "volume"]
    if not data.columns.is_unique or not set(required).issubset(data.columns):
        raise ValueError("Açılış, yüksek, düşük, kapanış ve hacim sütunları gerekli.")
    data[required] = data[required].apply(pd.to_numeric, errors="coerce")
    values = data[required].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Fiyat veya hacimde eksik/sonlu olmayan değer var.")
    if (values[:, :4] <= 0).any() or (values[:, 4] < 0).any():
        raise ValueError("Fiyatlar pozitif, hacim sıfır veya pozitif olmalı.")
    if ((data["low"] > data[["open", "close"]].min(axis=1)).any()
            or (data["high"] < data[["open", "close"]].max(axis=1)).any()):
        raise ValueError("Açılış/kapanış mumun düşük-yüksek aralığı dışında.")
    data.attrs["completed_only"] = True
    return data


def assess_signal_safety(frame: pd.DataFrame, market="stock", *, now=None,
                         config: SafetyConfig | None = None) -> dict:
    """Assess an already validated, completed frame. Thresholds are heuristics."""
    result = {"blocked": False, "codes": [], "reasons": [], "warnings": [], "metrics": {}}

    def block(code, reason):
        result["blocked"] = True
        result["codes"].append(code)
        result["reasons"].append(reason)

    try:
        config = config or SafetyConfig.from_env()
    except ValueError as exc:
        block("INVALID_SAFETY_CONFIG", str(exc))
        return result

    if now is not None:
        zone = "UTC" if market == "crypto" else "Europe/Istanbul"
        current = pd.Timestamp(now)
        current = current.tz_localize(zone) if current.tzinfo is None else current.tz_convert(zone)
        last = frame.index[-1]
        last = last.tz_localize(zone) if last.tzinfo is None else last.tz_convert(zone)
        if pd.isna(current):
            block("INVALID_CLOCK", "Verinin güncelliği doğrulanamadı.")
            return result
        if last > current:
            block("FUTURE_DATA", "Son mumun zamanı rapor zamanından ileride.")
        elif market == "crypto":
            hours = (current - last).total_seconds() / 3600
            if hours > config.crypto_max_age_hours:
                block("STALE_DATA", f"Son tamamlanmış saatlik mum {hours:.1f} saat eski; güncel sinyal üretilmedi.")
        else:
            days = (current.date() - last.date()).days
            if days > config.stock_max_age_days:
                block("STALE_DATA", f"Son günlük mum {days} takvim günü eski; güncel sinyal üretilmedi.")

    if frame["volume"].iloc[-1] <= 0:
        block("NO_VOLUME", "Son tamamlanmış mumda işlem hacmi yok; sinyal üretilmedi.")

    # Daily-stock thresholds must not be applied to hourly crypto candles.
    if market != "stock" or len(frame) < 21:
        return result
    previous = frame.iloc[-21:-1]
    baseline = float(previous["volume"].median())
    if baseline <= 0:
        block("INSUFFICIENT_VOLUME", "Önceki 20 mumda yeterli işlem hacmi geçmişi yok.")
        return result
    last = frame.iloc[-1]
    previous_close = float(frame["close"].iloc[-2])
    ratio = float(last["volume"] / baseline)
    change = float((last["close"] / previous_close - 1) * 100)
    candle_range = float((last["high"] - last["low"]) / previous_close * 100)
    if not all(math.isfinite(value) for value in (ratio, change, candle_range)):
        block("INVALID_METRICS", "Fiyat/hacim oranları güvenilir biçimde hesaplanamadı.")
        return result
    result["metrics"] = {"volume_ratio": ratio, "change_pct": change, "range_pct": candle_range}
    if ratio >= config.volume_ratio:
        explanation = (f"Hacim önceki 20 mumun medyanının {ratio:.1f} katı; "
                       f"kapanış değişimi %{change:.1f}, mum aralığı %{candle_range:.1f}.")
        if abs(change) >= config.price_move_pct or candle_range >= config.range_pct:
            block("PRICE_VOLUME_ANOMALY", explanation + " Aşırı fiyat-hacim hareketi nedeniyle sinyal durduruldu.")
        else:
            result["warnings"].append(explanation + " Tek başına hacim artışı engelleme nedeni değil.")
    return result
