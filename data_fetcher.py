"""
Piyasa Verisi Çekme Katmanı
============================
Kripto borsalarından (ccxt), Yahoo Finance'tan hisse ve temettü verisi çeker.
Asenkron TTL önbellek ve eşzamanlı istek paylaşımı içerir.
"""

import asyncio
import logging
import time
from datetime import datetime

import pandas as pd
import yfinance as yf
import ccxt.async_support as ccxt

from market_data import completed_bist_daily

logger = logging.getLogger(__name__)

# --- Konfigürasyon ---
# Kripto OHLCV mum sayısı — SMA200 / Altın Kesişim için en az ~250 gerekir
CRYPTO_OHLCV_LIMIT = 500

# Yahoo Finance verisi TTL önbellek süresi (saniye) — rate-limit koruması
STOCK_CACHE_TTL_SECONDS = 600

# Aynı anda kaç hisse verisi paralel çekilsin
STOCK_FETCH_CONCURRENCY = 4

# Türkiye'den Binance erişimi bazen engelli olabildiği için
# birden fazla borsayı sırayla dener. Bybit sıkıntı çıkardığı için çıkarıldı.
CRYPTO_EXCHANGES = ["gate", "kucoin", "kraken", "binance"]


# ============================================================================
# KRİPTO VERİSİ ÇEKME
# ============================================================================
async def fetch_crypto_data(symbol: str) -> dict | None:
    """
    Kripto borsalarından bir paritenin anlık fiyatını ve
    son N mumluk 1 saatlik OHLCV verisini çeker.
    Birden fazla borsayı sırayla dener (gate -> kucoin -> kraken -> binance).

    Args:
        symbol: Kripto paritesi (örn. "BTC/USDT", "DOT/USDT")

    Returns:
        dict: {"price", "ohlcv_df", "exchange"} veya hata durumunda None
    """
    for exchange_id in CRYPTO_EXCHANGES:
        exchange = None
        try:
            exchange_class = getattr(ccxt, exchange_id)
            exchange = exchange_class({"enableRateLimit": True})

            # Anlık fiyat
            ticker = await exchange.fetch_ticker(symbol)
            price = ticker.get("last", 0)

            # Son N mumun OHLCV verisi (1 saatlik) — Altın Kesişim (SMA200) için uzun geçmiş şart
            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe="1h", limit=CRYPTO_OHLCV_LIMIT)

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)

            logger.info(
                f"✅ Kripto verisi çekildi: {symbol} ({exchange_id}) — Fiyat: {price}"
            )
            return {"price": price, "ohlcv_df": df, "exchange": exchange_id}

        except Exception as e:
            logger.warning(f"⚠️ {exchange_id} başarısız ({symbol}): {e}")
            continue
        finally:
            if exchange:
                await exchange.close()

    logger.error(f"❌ Kripto verisi hiçbir borsadan çekilemedi: {symbol}")
    return None


# ============================================================================
# HİSSE VE TEMETTÜ VERİSİ
# ============================================================================
def fetch_stock_data(ticker_symbol: str) -> dict | None:
    """
    Yahoo Finance üzerinden bir hissenin anlık fiyatını ve
    1 yıllık geçmiş verisini çeker.
    Borsa İstanbul hisseleri için otomatik .IS son eki dener.

    Args:
        ticker_symbol: Hisse kodu (örn. "AAPL", "THYAO.IS", "ASELS")

    Returns:
        dict: {"price", "history_df", "info", "ticker"} veya hata durumunda None
    """
    # Denenecek semboller: önce olduğu gibi, sonra .IS ekleyerek
    candidates = [ticker_symbol]
    if not ticker_symbol.endswith(".IS"):
        candidates.append(f"{ticker_symbol}.IS")

    for symbol in candidates:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            # Anlık fiyat (farklı anahtarları dene)
            price = info.get("currentPrice") or info.get("regularMarketPrice", 0)

            # 1 yıllık geçmiş veri
            history_df = stock.history(period="2y", auto_adjust=True)
            if symbol.endswith(".IS") and not history_df.empty:
                history_df = completed_bist_daily(history_df)

            if history_df.empty:
                logger.warning(f"⚠️ Hisse geçmiş verisi boş: {symbol}")
                continue  # .IS ile dene

            logger.info(f"✅ Hisse verisi çekildi: {symbol} — Fiyat: {price}")
            return {"price": price, "history_df": history_df, "info": info, "ticker": symbol}

        except Exception as e:
            logger.warning(f"⚠️ Hisse verisi çekilemedi ({symbol}): {e}")
            continue

    logger.error(f"❌ Hisse verisi hiçbir sembolle çekilemedi: {ticker_symbol}")
    return None


def fetch_dividend_data(ticker_symbol: str) -> dict | None:
    """
    Yahoo Finance üzerinden bir hissenin temettü bilgilerini çeker.
    Borsa İstanbul hisseleri için otomatik .IS son eki dener.

    Args:
        ticker_symbol: Hisse kodu (örn. "AAPL", "MSFT", "ASELS")

    Returns:
        dict: {"dividend_yield", "dividends", "ex_date", "calendar"} veya None
    """
    # Denenecek semboller
    candidates = [ticker_symbol]
    if not ticker_symbol.endswith(".IS"):
        candidates.append(f"{ticker_symbol}.IS")

    for symbol in candidates:
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            # info boşsa veya geçersizse sonraki adayı dene
            if not info or info.get("trailingPegRatio") is None and not info.get("shortName"):
                continue

            dividends = stock.dividends

            dividend_yield = info.get("dividendYield", 0)
            if dividend_yield:
                dividend_yield = dividend_yield * 100  # Yüzde olarak

            # Yaklaşan temettü tarihi
            ex_date = info.get("exDividendDate", None)
            if ex_date:
                ex_date = datetime.fromtimestamp(ex_date)

            # Temettü takvimi
            try:
                calendar = stock.calendar
            except Exception:
                calendar = None

            logger.info(f"✅ Temettü verisi çekildi: {symbol}")
            return {
                "dividend_yield": dividend_yield,
                "dividends": dividends,
                "ex_date": ex_date,
                "calendar": calendar,
                "info": info,
            }

        except Exception as e:
            logger.warning(f"⚠️ Temettü verisi çekilemedi ({symbol}): {e}")
            continue

    logger.error(f"❌ Temettü verisi hiçbir sembolle çekilemedi: {ticker_symbol}")
    return None


# ============================================================================
# ASENKron ERİŞİM + TTL ÖNBELLEK
# ============================================================================
_stock_cache: dict = {}
_stock_cache_ts: dict = {}
_stock_fetch_tasks: dict = {}
_STOCK_SEMAPHORE = asyncio.Semaphore(STOCK_FETCH_CONCURRENCY)


async def _cached_fetch(fn, ticker_symbol: str):
    """Share in-flight requests and start the TTL when data arrives."""
    key = f"{fn.__name__}:{ticker_symbol.upper()}"
    now = time.monotonic()
    if key in _stock_cache and (now - _stock_cache_ts.get(key, 0)) < STOCK_CACHE_TTL_SECONDS:
        logger.info(f"📦 Önbellekten verildi: {key}")
        return _stock_cache[key]

    async def fetch_and_cache():
        try:
            async with _STOCK_SEMAPHORE:
                data = await asyncio.to_thread(fn, ticker_symbol)
            if data is not None:
                _stock_cache[key] = data
                _stock_cache_ts[key] = time.monotonic()
            return data
        finally:
            _stock_fetch_tasks.pop(key, None)

    if key not in _stock_fetch_tasks:
        task = asyncio.create_task(fetch_and_cache())
        # Retrieve errors even if every waiter was cancelled.
        task.add_done_callback(lambda done: done.exception() if not done.cancelled() else None)
        _stock_fetch_tasks[key] = task
    return await asyncio.shield(_stock_fetch_tasks[key])


async def fetch_stock_data_async(ticker_symbol: str) -> dict | None:
    """fetch_stock_data'nın asenkron + cache'lı sürümü."""
    return await _cached_fetch(fetch_stock_data, ticker_symbol)


async def fetch_dividend_data_async(ticker_symbol: str) -> dict | None:
    """fetch_dividend_data'nın asenkron + cache'lı sürümü."""
    return await _cached_fetch(fetch_dividend_data, ticker_symbol)
