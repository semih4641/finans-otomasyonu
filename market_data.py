"""Explicit completed-bar handling for BIST daily analysis."""

from datetime import time

import pandas as pd

BIST_TIMEZONE = "Europe/Istanbul"
# Full-day close is 18:10; allow 10 minutes for the daily provider bar to settle.
# Half-days conservatively become eligible at the same time or on the next day.
# https://www.borsaistanbul.com/piyasalar/pay-piyasasi/islem-saatleri
BIST_DAILY_READY = time(18, 20)


def completed_candles(df, enabled=True):
    if df is None or not enabled or df.attrs.get("completed_only", False):
        return df
    return df.iloc[:-1]


def completed_bist_daily(df: pd.DataFrame, now=None) -> pd.DataFrame:
    """Keep past sessions; exclude today's incomplete bar and future dates."""
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans or not df.index.is_unique:
        raise ValueError("BIST günlük verisi benzersiz zaman damgaları içermeli")
    current = pd.Timestamp.now(tz=BIST_TIMEZONE) if now is None else pd.Timestamp(now)
    current = current.tz_localize(BIST_TIMEZONE) if current.tzinfo is None else current.tz_convert(BIST_TIMEZONE)
    index = df.index.tz_localize(BIST_TIMEZONE) if df.index.tz is None else df.index.tz_convert(BIST_TIMEZONE)
    dates = index.date
    allowed = (dates < current.date()) | ((dates == current.date()) & (current.time() >= BIST_DAILY_READY))
    result = df.loc[allowed].sort_index().copy()
    result.attrs["completed_only"] = True
    return result


def fetch_bist_history(symbol: str, period="5y") -> pd.DataFrame:
    """Download adjusted daily BIST prices without Telegram or credentials."""
    if not symbol.upper().endswith(".IS"):
        raise ValueError("BIST eğitimi yalnız .IS sembollerini kabul eder")
    import yfinance as yf
    history = yf.Ticker(symbol.upper()).history(period=period, interval="1d", auto_adjust=True, timeout=25)
    if history.empty:
        raise ValueError(f"{symbol}: fiyat geçmişi boş")
    return completed_bist_daily(history)
