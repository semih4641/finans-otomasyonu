"""Advanced signal filters: ADX, Volume Profile, Regime Detection, Order Flow, On-Chain (crypto)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegime(Enum):
    BULL_TRENDING = "BULL_TRENDING"
    BULL_RANGING = "BULL_RANGING"
    BEAR_TRENDING = "BEAR_TRENDING"
    BEAR_RANGING = "BEAR_RANGING"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class SignalFeatures:
    """Container for all computed features for a symbol."""
    # Trend
    adx: Optional[float] = None
    adx_trend: Optional[str] = None
    di_plus: Optional[float] = None
    di_minus: Optional[float] = None

    # Momentum
    rsi: Optional[float] = None
    rsi_trend: Optional[str] = None
    macd_hist: Optional[float] = None
    macd_hist_trend: Optional[str] = None

    # Volatility
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    bb_width: Optional[float] = None
    bb_position: Optional[float] = None

    # Volume
    volume_ratio: Optional[float] = None
    volume_trend: Optional[str] = None
    obv: Optional[float] = None
    obv_trend: Optional[str] = None
    vwap_distance: Optional[float] = None

    # Market Structure
    regime: MarketRegime = MarketRegime.UNKNOWN
    regime_score: float = 0.0
    higher_highs: int = 0
    lower_lows: int = 0
    trend_strength: float = 0.0

    # Crypto-specific
    funding_rate: Optional[float] = None
    open_interest_change: Optional[float] = None
    exchange_netflow: Optional[float] = None

    # Metadata
    symbol: str = ""
    market: str = "stock"
    timestamp: Optional[datetime] = None


def _completed_candles(df: pd.DataFrame, analyze_completed_only: bool = True) -> pd.DataFrame:
    if analyze_completed_only and df is not None and len(df) > 2:
        return df.iloc[:-1]
    return df


def calculate_adx(df: pd.DataFrame, period: int = 14) -> dict[str, Optional[float]]:
    """Calculate ADX, +DI, -DI using Wilder's smoothing."""
    try:
        df = _completed_candles(df)
        high = df["high"] if "high" in df.columns else df["High"]
        low = df["low"] if "low" in df.columns else df["Low"]
        close = df["close"] if "close" in df.columns else df["Close"]

        plus_dm = high.diff()
        minus_dm = low.diff().abs()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, min_periods=period, adjust=False).mean() / atr)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        return {
            "adx": round(float(adx.iloc[-1]), 2) if not pd.isna(adx.iloc[-1]) else None,
            "di_plus": round(float(plus_di.iloc[-1]), 2) if not pd.isna(plus_di.iloc[-1]) else None,
            "di_minus": round(float(minus_di.iloc[-1]), 2) if not pd.isna(minus_di.iloc[-1]) else None,
        }
    except Exception as e:
        logger.error(f"❌ ADX hesaplanamadı: {e}")
        return {"adx": None, "di_plus": None, "di_minus": None}


def calculate_bollinger_bands(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> dict[str, Optional[float]]:
    """Calculate Bollinger Bands width and position."""
    try:
        df = _completed_candles(df)
        close = df["close"] if "close" in df.columns else df["Close"]

        sma = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std

        bb_width = (upper - lower) / sma
        bb_position = (close - lower) / (upper - lower)

        return {
            "bb_width": round(float(bb_width.iloc[-1]), 4) if not pd.isna(bb_width.iloc[-1]) else None,
            "bb_position": round(float(bb_position.iloc[-1]), 4) if not pd.isna(bb_position.iloc[-1]) else None,
            "upper": round(float(upper.iloc[-1]), 2) if not pd.isna(upper.iloc[-1]) else None,
            "lower": round(float(lower.iloc[-1]), 2) if not pd.isna(lower.iloc[-1]) else None,
            "middle": round(float(sma.iloc[-1]), 2) if not pd.isna(sma.iloc[-1]) else None,
        }
    except Exception as e:
        logger.error(f"❌ Bollinger Bands hesaplanamadı: {e}")
        return {"bb_width": None, "bb_position": None, "upper": None, "lower": None, "middle": None}


def calculate_obv(df: pd.DataFrame) -> dict[str, Optional[float]]:
    """Calculate On-Balance Volume and trend."""
    try:
        df = _completed_candles(df)
        close = df["close"] if "close" in df.columns else df["Close"]
        volume = df["volume"] if "volume" in df.columns else df.get("Volume")

        if volume is None:
            return {"obv": None, "obv_trend": None}

        price_change = close.diff()
        obv = (volume * np.sign(price_change)).cumsum()
        obv_sma = obv.rolling(window=20).mean()

        obv_trend = "UP" if obv.iloc[-1] > obv_sma.iloc[-1] else "DOWN"

        return {
            "obv": round(float(obv.iloc[-1]), 2) if not pd.isna(obv.iloc[-1]) else None,
            "obv_trend": obv_trend,
        }
    except Exception as e:
        logger.error(f"❌ OBV hesaplanamadı: {e}")
        return {"obv": None, "obv_trend": None}


def calculate_vwap_distance(df: pd.DataFrame, period: int = 20) -> Optional[float]:
    """Calculate distance from VWAP as percentage."""
    try:
        df = _completed_candles(df)
        close = df["close"] if "close" in df.columns else df["Close"]
        high = df["high"] if "high" in df.columns else df["High"]
        low = df["low"] if "low" in df.columns else df["Low"]
        volume = df["volume"] if "volume" in df.columns else df.get("Volume")

        if volume is None:
            return None

        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(window=period).sum() / volume.rolling(window=period).sum()
        distance = (close.iloc[-1] - vwap.iloc[-1]) / vwap.iloc[-1] * 100

        return round(float(distance), 2) if not pd.isna(distance) else None
    except Exception as e:
        logger.error(f"❌ VWAP distance hesaplanamadı: {e}")
        return None


def detect_market_regime(df: pd.DataFrame, lookback: int = 50) -> tuple[MarketRegime, float]:
    """Detect market regime using ADX, volatility, and trend structure."""
    try:
        df = _completed_candles(df)
        close = df["close"] if "close" in df.columns else df["Close"]
        high = df["high"] if "high" in df.columns else df["High"]
        low = df["low"] if "low" in df.columns else df["Low"]

        if len(close) < lookback:
            return MarketRegime.UNKNOWN, 0.0

        recent = close.tail(lookback)

        adx_data = calculate_adx(df)
        adx = adx_data.get("adx") or 0
        di_plus = adx_data.get("di_plus") or 0
        di_minus = adx_data.get("di_minus") or 0

        returns = recent.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252 if len(returns) > 100 else len(returns))
        vol_percentile = min(volatility * 100, 100)

        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma_20
        current_price = close.iloc[-1]

        higher_highs = 0
        lower_lows = 0
        for i in range(-10, -1):
            if high.iloc[i] > high.iloc[i-1]:
                higher_highs += 1
            if low.iloc[i] < low.iloc[i-1]:
                lower_lows += 1

        trend_up = current_price > sma_20 > sma_50
        trend_down = current_price < sma_20 < sma_50
        strong_trend = adx > 25
        ranging = adx < 20

        regime = MarketRegime.UNKNOWN
        score = 0.0

        if strong_trend and trend_up:
            regime = MarketRegime.BULL_TRENDING
            score = min(adx / 50.0, 1.0)
        elif strong_trend and trend_down:
            regime = MarketRegime.BEAR_TRENDING
            score = min(adx / 50.0, 1.0)
        elif ranging and trend_up:
            regime = MarketRegime.BULL_RANGING
            score = 1.0 - (adx / 20.0)
        elif ranging and trend_down:
            regime = MarketRegime.BEAR_RANGING
            score = 1.0 - (adx / 20.0)
        elif vol_percentile > 70:
            regime = MarketRegime.HIGH_VOLATILITY
            score = vol_percentile / 100.0
        elif vol_percentile < 30:
            regime = MarketRegime.LOW_VOLATILITY
            score = 1.0 - vol_percentile / 30.0

        return regime, round(score, 3)
    except Exception as e:
        logger.error(f"❌ Regime detection hatası: {e}")
        return MarketRegime.UNKNOWN, 0.0


def detect_volume_anomaly(df: pd.DataFrame, lookback: int = 20) -> dict[str, Any]:
    """Detect volume anomalies: spikes, dry-ups, accumulation/distribution."""
    try:
        df = _completed_candles(df)
        volume = df["volume"] if "volume" in df.columns else df.get("Volume")
        close = df["close"] if "close" in df.columns else df["Close"]

        if volume is None or len(volume) < lookback + 1:
            return {"ratio": None, "trend": None, "anomaly": False, "type": None}

        avg_volume = volume.rolling(window=lookback).mean().iloc[-1]
        curr_volume = volume.iloc[-1]
        ratio = curr_volume / avg_volume if avg_volume > 0 else 1.0

        vol_sma_short = volume.rolling(5).mean().iloc[-1]
        vol_sma_long = volume.rolling(20).mean().iloc[-1]
        vol_trend = "INCREASING" if vol_sma_short > vol_sma_long else "DECREASING"

        price_change = close.pct_change().iloc[-1]
        anomaly = False
        anomaly_type = None

        if ratio > 2.0 and price_change > 0:
            anomaly = True
            anomaly_type = "ACCUMULATION"
        elif ratio > 2.0 and price_change < 0:
            anomaly = True
            anomaly_type = "DISTRIBUTION"
        elif ratio < 0.5:
            anomaly = True
            anomaly_type = "VOLUME_DRY_UP"

        return {
            "ratio": round(ratio, 2),
            "trend": vol_trend,
            "anomaly": anomaly,
            "type": anomaly_type,
        }
    except Exception as e:
        logger.error(f"❌ Volume anomaly detection hatası: {e}")
        return {"ratio": None, "trend": None, "anomaly": False, "type": None}


def extract_features(df: pd.DataFrame, symbol: str, market: str = "stock") -> SignalFeatures:
    """Extract all features for ML model and advanced filtering."""
    df = _completed_candles(df)
    close = df["close"] if "close" in df.columns else df["Close"]
    high = df["high"] if "high" in df.columns else df["High"]
    low = df["low"] if "low" in df.columns else df["Low"]

    adx_data = calculate_adx(df)
    bb_data = calculate_bollinger_bands(df)
    obv_data = calculate_obv(df)
    vwap_dist = calculate_vwap_distance(df)
    vol_data = detect_volume_anomaly(df)
    regime, regime_score = detect_market_regime(df)

    rsi = None
    try:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = gain / loss
        rsi = round(float((100 - 100/(1+rs)).iloc[-1]), 2)
    except Exception:
        pass

    macd_hist = None
    try:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = round(float((macd - signal).iloc[-1]), 4)
    except Exception:
        pass

    atr = None
    atr_pct = None
    try:
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_val = tr.rolling(14).mean().iloc[-1]
        atr = round(float(atr_val), 4)
        atr_pct = round(atr_val / close.iloc[-1] * 100, 2)
    except Exception:
        pass

    higher_highs = sum(1 for i in range(-10, -1) if high.iloc[i] > high.iloc[i-1])
    lower_lows = sum(1 for i in range(-10, -1) if low.iloc[i] < low.iloc[i-1])

    return SignalFeatures(
        symbol=symbol,
        market=market,
        timestamp=datetime.now(),
        adx=adx_data["adx"],
        di_plus=adx_data["di_plus"],
        di_minus=adx_data["di_minus"],
        rsi=rsi,
        macd_hist=macd_hist,
        atr=atr,
        atr_pct=atr_pct,
        bb_width=bb_data["bb_width"],
        bb_position=bb_data["bb_position"],
        volume_ratio=vol_data["ratio"],
        volume_trend=vol_data["trend"],
        obv=obv_data["obv"],
        obv_trend=obv_data["obv_trend"],
        vwap_distance=vwap_dist,
        regime=regime,
        regime_score=regime_score,
        higher_highs=higher_highs,
        lower_lows=lower_lows,
        trend_strength=regime_score,
    )


def score_signal_advanced(features: SignalFeatures, base_score: int, market: str) -> tuple[int, list[str]]:
    """Advanced scoring using extracted features. Returns (adjusted_score, reasons)."""
    score = base_score
    reasons = []

    if features.regime == MarketRegime.BULL_TRENDING:
        score += 1
        reasons.append(f"📈 Rejim: BOĞA Trending (skor +1)")
    elif features.regime == MarketRegime.BEAR_TRENDING:
        score -= 1
        reasons.append(f"📉 Rejim: AYI Trending (skor -1)")
    elif features.regime == MarketRegime.HIGH_VOLATILITY:
        score -= 1
        reasons.append(f"⚡ Rejim: Yüksek Volatilite (skor -1)")

    if features.adx is not None:
        if features.adx > 30 and features.di_plus and features.di_minus:
            if features.di_plus > features.di_minus:
                score += 1
                reasons.append(f"📊 ADX: {features.adx:.0f} Güçlü Trend (+DI > -DI) (skor +1)")
            else:
                score -= 1
                reasons.append(f"📊 ADX: {features.adx:.0f} Güçlü Trend (-DI > +DI) (skor -1)")
        elif features.adx < 20:
            reasons.append(f"📊 ADX: {features.adx:.0f} Zayıf Trend (Range)")

    if features.volume_ratio is not None:
        if features.volume_ratio > 1.5 and features.volume_trend == "INCREASING":
            score += 1
            reasons.append(f"🔊 Hacim: {features.volume_ratio:.1f}x Artan (skor +1)")
        elif features.volume_ratio < 0.5:
            score -= 1
            reasons.append(f"🔇 Hacim: {features.volume_ratio:.1f}x Kuruma (skor -1)")

    if features.bb_position is not None:
        if features.bb_position < 0.1:
            score += 1
            reasons.append(f"📉 BB Alt Bandına Yakın ({features.bb_position:.2f}) (skor +1)")
        elif features.bb_position > 0.9:
            score -= 1
            reasons.append(f"📈 BB Üst Bandına Yakın ({features.bb_position:.2f}) (skor -1)")

    if features.vwap_distance is not None:
        if features.vwap_distance < -1.0:
            score += 1
            reasons.append(f"📊 VWAP Altında %{abs(features.vwap_distance):.1f} (skor +1)")
        elif features.vwap_distance > 1.0:
            score -= 1
            reasons.append(f"📊 VWAP Üstünde %{features.vwap_distance:.1f} (skor -1)")

    if features.obv_trend == "UP" and features.rsi and features.rsi < 50:
        score += 1
        reasons.append(f"📈 OBV Yukarı + RSI < 50 (skor +1)")
    elif features.obv_trend == "DOWN" and features.rsi and features.rsi > 50:
        score -= 1
        reasons.append(f"📉 OBV Aşağı + RSI > 50 (skor -1)")

    if market == "crypto":
        if features.funding_rate is not None:
            if features.funding_rate > 0.01:
                score -= 1
                reasons.append(f"💸 Funding Rate Yüksek: {features.funding_rate:.4f} (skor -1)")
            elif features.funding_rate < -0.01:
                score += 1
                reasons.append(f"💰 Funding Rate Negatif: {features.funding_rate:.4f} (skor +1)")

        if features.open_interest_change is not None:
            if features.open_interest_change > 0.05:
                score += 1
                reasons.append(f"📈 OI Artışı: %{features.open_interest_change*100:.1f} (skor +1)")
            elif features.open_interest_change < -0.05:
                score -= 1
                reasons.append(f"📉 OI Düşüşü: %{features.open_interest_change*100:.1f} (skor -1)")

    return max(0, score), reasons


def generate_advanced_signals(df: pd.DataFrame, symbol: str, market: str = "stock") -> dict:
    """Generate advanced signal output with features and adjusted scoring."""
    features = extract_features(df, symbol, market)

    base_signals = []
    if features.rsi is not None:
        if features.rsi < 30:
            base_signals.append("🔴 RSI Aşırı Satım")
        elif features.rsi < 40:
            base_signals.append("📈 RSI Toparlanma")

    if features.macd_hist is not None and features.macd_hist > 0:
        base_signals.append("🔀 MACD Boğa Kesişimi")

    if features.bb_width is not None and features.bb_width < 0.05:
        base_signals.append("🎯 Bollinger Sıkışması")

    if features.volume_ratio is not None and features.volume_ratio > 1.5:
        base_signals.append("🔊 Hacim Patlaması")

    base_score = len(base_signals)
    adjusted_score, reasons = score_signal_advanced(features, base_score, market)

    return {
        "features": features,
        "base_signals": base_signals,
        "base_score": base_score,
        "adjusted_score": adjusted_score,
        "score_reasons": reasons,
        "regime": features.regime.value,
        "regime_score": features.regime_score,
    }