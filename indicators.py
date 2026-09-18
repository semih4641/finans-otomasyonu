"""
Teknik İndikatör Hesaplama Motoru
=================================
RSI, MACD ve yardımcı formatlama fonksiyonları.
Saf hesaplama fonksiyonları — Telegram veya bot state'e bağımlılık yok.
"""

import logging

import pandas as pd

from market_data import completed_candles

logger = logging.getLogger(__name__)

# Analizde yalnızca KAPANMIŞ mumlar kullanılsın (oluşmakta olan son mum atılır)
ANALYZE_COMPLETED_CANDLES_ONLY = True


def _completed_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Oluşmakta olan son mumu atar; analiz yalnızca kapanmış mumlar üzerinden yapılır."""
    return completed_candles(df, ANALYZE_COMPLETED_CANDLES_ONLY)


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Wilder yumuşatmalı RSI serisi — vektörel hesap.
    Wilder'in özyinelemeli ortalaması, alpha=1/period olan EMA'ya eşdeğerdir
    (ewm adjust=False); Python döngüsüne kıyasla çok daha hızlıdır.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.where(avg_loss != 0)
    rsi = (100 - (100 / (1 + rs))).mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    return rsi.mask((avg_loss == 0) & (avg_gain == 0), 50.0)


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> float | None:
    """
    14 günlük RSI (Relative Strength Index) hesaplar.

    Args:
        df: OHLCV DataFrame ('close' sütunu gerekli)
        period: RSI periyodu (varsayılan 14)

    Returns:
        float: Son RSI değeri veya None
    """
    try:
        close = df["close"] if "close" in df.columns else df["Close"]
        close = _completed_candles(close.to_frame()).iloc[:, 0]

        rsi = _rsi_series(close, period)
        last = rsi.iloc[-1]
        if pd.isna(last):
            return None
        return round(float(last), 2)

    except Exception as e:
        logger.error(f"❌ RSI hesaplanamadı: {e}")
        return None


def calculate_macd(df: pd.DataFrame) -> dict | None:
    """
    Standart MACD hesaplar (12, 26, 9).

    Args:
        df: OHLCV DataFrame ('close' sütunu gerekli)

    Returns:
        dict: {"macd", "signal", "histogram"} veya None
    """
    try:
        close = df["close"] if "close" in df.columns else df["Close"]
        close = _completed_candles(close.to_frame()).iloc[:, 0]

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()

        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            "macd": round(macd_line.iloc[-1], 4),
            "signal": round(signal_line.iloc[-1], 4),
            "histogram": round(histogram.iloc[-1], 4),
        }

    except Exception as e:
        logger.error(f"❌ MACD hesaplanamadı: {e}")
        return None


def get_rsi_status(rsi_value: float) -> str:
    """RSI değerine göre durum metni döndürür."""
    if rsi_value is None:
        return "❓ Hesaplanamadı"
    if rsi_value < 30:
        return "🔴 AŞIRI SATIM"
    elif rsi_value > 70:
        return "🟢 AŞIRI ALIM"
    else:
        return "⚪ NÖTR"


def format_crypto_price(p: float) -> str:
    """Küçük küsuratlı kriptolar için (SHIB vb.) dinamik fiyat formatlayıcı."""
    formatted = f"{p:.8f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted
