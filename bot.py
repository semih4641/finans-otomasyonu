"""
Telegram Finans Botu
====================
Hisse senedi, kripto para ve temettü takibi yapan,
teknik indikatörleri (RSI, MACD) hesaplayan asenkron Telegram botu.

Kütüphaneler: python-telegram-bot (v20+), yfinance, ccxt, pandas, apscheduler
"""

# ============================================================================
# 1. KÜTÜPHANE VE ORTAM KURULUMU
# ============================================================================
import os
import logging
import asyncio
import json
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
import ccxt.async_support as ccxt
from dotenv import load_dotenv

from telegram import Update, BotCommand
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
)
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from risk import calculate_position_risk, format_position_risk
from portfolio_risk import PortfolioState, portfolio_gate, calculate_portfolio_metrics, get_portfolio_config
from signals_advanced import extract_features, generate_advanced_signals, score_signal_advanced, SignalFeatures
from ml_model import SignalMLModel, RegimeAwareModelEnsemble
import paper

# .env dosyasından konfigürasyonu yükle
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# Arka plan zamanlayıcısının takip edeceği varsayılan varlıklar
WATCHLIST_CRYPTO = ["BTC/USDT", "ETH/USDT", "DOT/USDT"]
WATCHLIST_STOCKS = ["AAPL", "THYAO.IS", "MSFT"]

# AL sinyali taraması yapılacak hisseler (BIST 30 + Özel İstekler)
SCAN_STOCKS = [
    # BIST 30
    "AKBNK.IS", "ALARK.IS", "ASELS.IS", "ASTOR.IS", "BIMAS.IS", 
    "BRISA.IS", "CCOLA.IS", "ENKAI.IS", "EREGL.IS", "FROTO.IS", 
    "GARAN.IS", "GUBRF.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS", 
    "KONTR.IS", "KRDMD.IS", "MIATK.IS", "ODAS.IS", "OYAKC.IS", 
    "PGSUS.IS", "PETKM.IS", "SAHOL.IS", "SASA.IS", "SISE.IS", 
    "TCELL.IS", "THYAO.IS", "TOASO.IS", "TUPRS.IS", "YKBNK.IS",
    # Kullanıcı Eklemeleri
    "KOTON.IS", "AAGYO.IS", "LINK.IS", "ALTNY.IS"
]

# AL sinyali taraması yapılacak kripto paralar (Top 20)
SCAN_CRYPTO = [
    "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT", 
    "ADA/USDT", "AVAX/USDT", "LINK/USDT", "DOT/USDT", "MATIC/USDT", 
    "DOGE/USDT", "LTC/USDT", "SHIB/USDT", "UNI/USDT", "ATOM/USDT",
    "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "INJ/USDT"
]

# Zamanlayıcı aralığı (dakika)
SCHEDULER_INTERVAL_MINUTES = 15

# --- Faz 1: profesyonel analiz ayarları ---
# Kripto OHLCV mum sayısı — SMA200 / Altın Kesişim için en az ~250 gerekir
CRYPTO_OHLCV_LIMIT = 500
# Analizde yalnızca KAPANMIŞ mumlar kullanılsın (oluşmakta olan son mum atılır)
ANALYZE_COMPLETED_CANDLES_ONLY = True
# Aynı sinyal tekrar spam yapmasın: sembol+sinyal türü başına sessizlik süresi (saat)
SIGNAL_COOLDOWN_HOURS = 24
# Cooldown durumunun kalıcı olarak saklandığı dosya
SIGNAL_STATE_FILE = "signal_state.json"
# Yahoo Finance verisi TTL önbellek süresi (saniye) — rate-limit koruması
STOCK_CACHE_TTL_SECONDS = 600
# Aynı anda kaç hisse verisi paralel çekilsin
STOCK_FETCH_CONCURRENCY = 4

# --- Faz 2: sinyal kalite filtreleri ---
# Piyasa bazlı sinyal ağırlıkları — 52 BIST + 20 kripto backtest'iyle kalibre edildi:
# hissede MACD(+%1.66) ve RSI Toparlanma(+%2.09) güçlü; Aşırı Satım negatif (-%0.56).
# Kripto saatlikte RSI Toparlanma geçersiz (%24 WR); Hacim Patlaması en güvenilir sinyal.
SIGNAL_SCORES_STOCK = {
    "📈 RSI Toparlanma": 2,
    "🔴 RSI Aşırı Satım": 0,
    "🔀 MACD Boğa Kesişimi": 2,
    "✨ Altın Kesişim": 3,
    "🔊 Hacim Patlaması": 1,
}
SIGNAL_SCORES_CRYPTO = {
    "📈 RSI Toparlanma": 0,
    "🔴 RSI Aşırı Satım": 1,
    "🔀 MACD Boğa Kesişimi": 1,
    "✨ Altın Kesişim": 3,
    "🔊 Hacim Patlaması": 2,
}
# BOĞA / nötr rejimde gereken minimum toplam skor (hisse)
MIN_SIGNAL_SCORE = 2
# AYI rejiminde ters tepki sinyalleri için daha güçlü teyit gerekir (hisse)
MIN_SIGNAL_SCORE_BEAR = 4
# Kriptoda eşik daha yüksek — skor 2 kümesi maliyetsiz bile sıfır beklentiliydi
MIN_SIGNAL_SCORE_CRYPTO = 3
# Stop mesafesi fiyatın bu yüzdesini geçemez (aşırı geniş stop engellenir)
MAX_STOP_LOSS_PCT = 0.04
# Take Profit hedefleri en az bu Risk/Ödül oranını verecek şekilde kurulur
MIN_RISK_REWARD_RATIO = 1.8

# --- ML Model Ayarları ---
ML_MODEL_TYPE = "rf"  # rf, gb, lr
ML_MIN_CONFIDENCE = 0.55
USE_ML_SCORING = True

# --- Portfolio Risk Ayarları ---
ENABLE_PORTFOLIO_RISK = True

# Logging yapılandırması
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("finans_bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# --- ML Model Ensemble (lazy init) ---
_ml_ensemble: RegimeAwareModelEnsemble | None = None
_ml_models_loaded = False

# --- Portfolio State Tracking ---
_portfolio_state = PortfolioState(
    positions=[],
    equity=100000.0,
    peak_equity=100000.0,
    daily_pnl=0.0,
    account_size=100000.0,
    price_data={},
)


def _get_ml_ensemble() -> RegimeAwareModelEnsemble:
    global _ml_ensemble, _ml_models_loaded
    if _ml_ensemble is None:
        _ml_ensemble = RegimeAwareModelEnsemble(ML_MODEL_TYPE)
        # Try to load existing models
        from pathlib import Path
        model_dir = Path("models")
        if model_dir.exists():
            for model_file in model_dir.glob("*.pkl"):
                try:
                    model = SignalMLModel.load(model_file)
                    if model.regime:
                        _ml_ensemble.models[model.regime] = model
                    else:
                        _ml_ensemble.global_model = model
                    _ml_models_loaded = True
                    logger.info(f"📂 ML model yüklendi: {model_file}")
                except Exception as e:
                    logger.warning(f"⚠️ Model yüklenemedi {model_file}: {e}")
    return _ml_ensemble


def _update_portfolio_state() -> None:
    """Update portfolio state from paper trading open positions."""
    global _portfolio_state
    try:
        open_positions = paper._load_open()
        if not open_positions:
            return

        _portfolio_state.positions = open_positions

        # Calculate equity from settled trades
        if paper.SETTLED_CSV.exists():
            df = pd.read_csv(paper.SETTLED_CSV, encoding="utf-8-sig")
            if not df.empty:
                returns = df["net_pnl_pct"] / 100
                equity = 100000.0 * (1 + returns).cumprod().iloc[-1]
                _portfolio_state.equity = equity
                _portfolio_state.peak_equity = max(_portfolio_state.peak_equity, equity)

        # Update price data for correlation calculation
        _portfolio_state.price_data = {}  # Will be populated async if needed

    except Exception as e:
        logger.warning(f"⚠️ Portfolio state güncellenemedi: {e}")


# ============================================================================
# 1.5 SİNYAL TEKRAR ÖNLEME (COOLDOWN)
# ============================================================================
def _signal_category(signal_text: str) -> str:
    """Sinyal metninden sabit kategori anahtarı üretir.
    Örn: '📈 RSI Toparlanma (RSI: 28.3, ...)' -> '📈 RSI Toparlanma'
    """
    return signal_text.split(" (")[0].strip()


def _load_signal_state() -> dict:
    """Cooldown durum dosyasını diskten okur; bozuksa sıfırdan başlar."""
    try:
        with open(SIGNAL_STATE_FILE, encoding="utf-8") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items() if isinstance(v, str)}
    except FileNotFoundError:
        return {}
    except Exception as e:
        logger.warning(f"⚠️ Cooldown durumu okunamadı, sıfırlanıyor: {e}")
        return {}


def _save_signal_state(state: dict) -> None:
    """Süresi dolmuş kayıtları ayıklayıp durum dosyasını diske yazar."""
    cutoff = datetime.now() - timedelta(hours=SIGNAL_COOLDOWN_HOURS)
    pruned = {}
    for key, iso_ts in state.items():
        try:
            if datetime.fromisoformat(iso_ts) >= cutoff:
                pruned[key] = iso_ts
        except ValueError:
            continue
    try:
        with open(SIGNAL_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(pruned, f, ensure_ascii=False, indent=1)
    except Exception as e:
        logger.error(f"❌ Cooldown durumu yazılamadı: {e}")


def _cooldown_active(key: str) -> bool:
    """Bu anahtar için hâlâ sessizlik süresi devam ediyor mu?"""
    state = _load_signal_state()
    iso_ts = state.get(key)
    if not iso_ts:
        return False
    try:
        return datetime.now() - datetime.fromisoformat(iso_ts) < timedelta(
            hours=SIGNAL_COOLDOWN_HOURS
        )
    except ValueError:
        return False


def _cooldown_mark(keys: list) -> None:
    """Verilen anahtarları 'bildirildi' olarak işaretler."""
    if not keys:
        return
    state = _load_signal_state()
    now_iso = datetime.now().isoformat()
    for key in keys:
        state[key] = now_iso
    _save_signal_state(state)
    logger.info(f"🔇 {len(keys)} sinyal için {SIGNAL_COOLDOWN_HOURS}h cooldown başlatıldı.")


def filter_cooldown(symbol_key: str, signals: list) -> list:
    """Cooldown'da olmayan sinyalleri döndürür — otomatik tarama için."""
    fresh = []
    for s in signals:
        if not _cooldown_active(f"{symbol_key}|{_signal_category(s)}"):
            fresh.append(s)
    return fresh


def mark_signals_sent(symbol_key: str, signals: list) -> None:
    """Gönderilen sinyalleri cooldown'a sokar."""
    _cooldown_mark([f"{symbol_key}|{_signal_category(s)}" for s in signals])


# ============================================================================
# 2. KRİPTO VERİSİ ÇEKME FONKSİYONU
# ============================================================================

# Türkiye'den Binance erişimi bazen engelli olabildiği için
# birden fazla borsayı sırayla dener. Bybit sıkıntı çıkardığı için çıkarıldı.
CRYPTO_EXCHANGES = ["gate", "kucoin", "kraken", "binance"]


async def fetch_crypto_data(symbol: str) -> dict | None:
    """
    Kripto borsalarından bir paritenin anlık fiyatını ve
    son 100 mumluk 1 saatlik OHLCV verisini çeker.
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
# 3. HİSSE VE TEMETTÜ VERİSİ FONKSİYONLARI
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
            history_df = stock.history(period="1y")

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


# --- Faz 1: async erişim + TTL önbellek (event loop'u bloklamamak için) ---
_stock_cache: dict = {}
_stock_cache_ts: dict = {}
_STOCK_SEMAPHORE = asyncio.Semaphore(STOCK_FETCH_CONCURRENCY)


async def _cached_fetch(fn, ticker_symbol: str, *args):
    """Sync yfinance çağrısını thread'e taşıyıp TTL cache ile döndürür."""
    key = f"{fn.__name__}:{ticker_symbol.upper()}"
    now = time.monotonic()
    if key in _stock_cache and (now - _stock_cache_ts.get(key, 0)) < STOCK_CACHE_TTL_SECONDS:
        logger.info(f"📦 Önbellekten verildi: {key}")
        return _stock_cache[key]

    async with _STOCK_SEMAPHORE:
        data = await asyncio.to_thread(fn, ticker_symbol)

    if data is not None:
        _stock_cache[key] = data
        _stock_cache_ts[key] = now
    return data


async def fetch_stock_data_async(ticker_symbol: str) -> dict | None:
    """fetch_stock_data'nın asenkron + cache'lı sürümü."""
    return await _cached_fetch(fetch_stock_data, ticker_symbol)


async def fetch_dividend_data_async(ticker_symbol: str) -> dict | None:
    """fetch_dividend_data'nın asenkron + cache'lı sürümü."""
    return await _cached_fetch(fetch_dividend_data, ticker_symbol)


# ============================================================================
# 4. İNDİKATÖR HESAPLAMA MOTORU
# ============================================================================
def _completed_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Oluşmakta olan son mumu atar; analiz yalnızca kapanmış mumlar üzerinden yapılır."""
    if ANALYZE_COMPLETED_CANDLES_ONLY and df is not None and len(df) > 2:
        return df.iloc[:-1]
    return df


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
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


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


def _min_score(market: str, trend: str) -> int:
    """Piyasa türü ve rejime göre gereken minimum konfluans skoru."""
    if market == "crypto":
        return MIN_SIGNAL_SCORE_CRYPTO
    return MIN_SIGNAL_SCORE_BEAR if trend == "AYI" else MIN_SIGNAL_SCORE


def sizing_hint(entry: float, sl: float) -> str:
    """Return risk-only sizing information; never place an order."""
    return format_position_risk(calculate_position_risk(entry, sl))


def detect_buy_signals(df: pd.DataFrame, market: str = "stock", symbol: str = "") -> dict:
    """
    Fiyat verisinden AL sinyallerini tespit eder (Gelişmiş: ADX, Volume Profile, Regime, ML).
    
    market: "stock" | "crypto" — kalibrasyon ağırlıkları ve skor eşiği buna göre seçilir.

    AL Sinyali Koşulları:
    1. RSI Aşırı Satım Toparlanması: RSI 30 altından yukarı çıkıyor
    2. MACD Boğa Kesişimi: MACD çizgisi sinyal çizgisini yukarı kesiyor
    3. Altın Kesişim: 50 günlük SMA, 200 günlük SMA'yı yukarı kesiyor
    4. Hacim Patlaması: Son hacim, 20 günlük ortalama hacmin 1.5 katı
    5. ADX Trend Gücü: ADX > 25 ve +DI > -DI
    6. Bollinger Band Sıkışması: Bant genişliği < %5
    7. VWAP Altında: Fiyat VWAP'ın %1+ altında
    8. OBY Diverjans: OBV yukarı, RSI < 50

    Args:
        df: OHLCV DataFrame
        market: "stock" | "crypto"
        symbol: Sembol adı (ML scoring için)

    Returns:
        dict: {"signals", "tp1", "tp2", "sl", "vade", "score", "trend", "rr", "ml_score", "regime", "features"}
        Faz 2 kalite kapıları: konfluans skoru rejim eşiğini geçmeyen
        kurulumlar boş sinyal listesiyle bastırılır.
    """
    signals = []
    base_signals = []
    try:
        close = df["close"] if "close" in df.columns else df["Close"]
        volume = df["volume"] if "volume" in df.columns else df.get("Volume")

        # Yalnızca kapanmış mumlar üzerinden analiz
        completed = _completed_candles(df)
        close = completed["close"] if "close" in completed.columns else completed["Close"]
        volume = (
            completed["volume"]
            if "volume" in completed.columns
            else completed.get("Volume")
        )

        if len(close) < 30:
            return {"signals": [], "tp1": 0.0, "tp2": 0.0, "sl": 0.0, "vade": "",
                    "score": 0, "trend": "", "rr": None, "ml_score": 0.5, "regime": "UNKNOWN", "features": None}

        # --- 1. RSI Aşırı Satım Toparlanması ---
        rsi = _rsi_series(close)

        if len(rsi) >= 2:
            prev_rsi = rsi.iloc[-2]
            curr_rsi = rsi.iloc[-1]
            # RSI 30 altından yukarı çıkıyor (toparlanma)
            if prev_rsi < 30 and curr_rsi >= 30:
                signals.append(f"📈 RSI Toparlanma (RSI: {curr_rsi:.1f}, önceki: {prev_rsi:.1f})")
                base_signals.append("📈 RSI Toparlanma")
            # RSI hâlâ aşırı satım bölgesinde (potansiyel dip)
            elif curr_rsi < 30:
                signals.append(f"🔴 RSI Aşırı Satım (RSI: {curr_rsi:.1f})")
                base_signals.append("🔴 RSI Aşırı Satım")

        # --- 2. MACD Boğa Kesişimi ---
        if len(close) >= 26:
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            if len(macd_line) >= 2:
                # MACD sinyal çizgisini yukarı kesiyor
                prev_diff = macd_line.iloc[-2] - signal_line.iloc[-2]
                curr_diff = macd_line.iloc[-1] - signal_line.iloc[-1]
                if prev_diff < 0 and curr_diff >= 0:
                    signals.append("🔀 MACD Boğa Kesişimi (MACD sinyal çizgisini yukarı kesti)")
                    base_signals.append("🔀 MACD Boğa Kesişimi")

        # --- 3. Altın Kesişim (SMA 50 > SMA 200) ---
        if len(close) >= 200:
            sma_50 = close.rolling(window=50).mean()
            sma_200 = close.rolling(window=200).mean()

            if len(sma_50) >= 2 and len(sma_200) >= 2:
                prev_above = sma_50.iloc[-2] > sma_200.iloc[-2]
                curr_above = sma_50.iloc[-1] > sma_200.iloc[-1]
                if not prev_above and curr_above:
                    signals.append("✨ Altın Kesişim (SMA50 > SMA200 — güçlü yükseliş sinyali)")
                    base_signals.append("✨ Altın Kesişim")

        # --- 4. Hacim Patlaması ---
        if volume is not None and len(volume) >= 21:
            avg_volume = volume.rolling(window=20).mean().iloc[-1]
            curr_volume = volume.iloc[-1]
            if avg_volume > 0 and curr_volume > avg_volume * 1.5:
                ratio = curr_volume / avg_volume
                signals.append(f"🔊 Hacim Patlaması (hacim ortalamanın {ratio:.1f}x üzerinde)")
                base_signals.append("🔊 Hacim Patlaması")

    except Exception as e:
        logger.error(f"❌ Temel sinyal tespiti hatası: {e}")

    # --- Gelişmiş Özellikler ve ML Scoring ---
    ml_score = 0.5
    regime = "UNKNOWN"
    features: SignalFeatures | None = None
    advanced_signals = []
    score_reasons = []

    if signals and USE_ML_SCORING:
        try:
            # Extract features for advanced analysis
            features = extract_features(df, symbol, market)
            regime = features.regime.value

            # Generate advanced signals
            advanced_result = generate_advanced_signals(df, symbol, market)
            advanced_signals = advanced_result.get("base_signals", [])
            ml_score = advanced_result.get("adjusted_score", 0) / max(advanced_result.get("base_score", 1), 1)
            score_reasons = advanced_result.get("score_reasons", [])

            # ML Model prediction
            ensemble = _get_ml_ensemble()
            if ensemble.global_model or features.regime in ensemble.models:
                ml_prob, model_used = ensemble.predict(features)
                ml_score = ml_prob
                logger.debug(f"ML prediction ({model_used}): {ml_prob:.3f} for {symbol}")

            # Apply advanced scoring
            base_score = len(base_signals)
            weights = SIGNAL_SCORES_CRYPTO if market == "crypto" else SIGNAL_SCORES_STOCK
            weighted_score = sum(weights.get(_signal_category(s), 1) for s in base_signals)

            # Combine traditional and ML scoring
            final_score = int(weighted_score * 0.6 + ml_score * 10 * 0.4)
            for reason in score_reasons:
                signals.append(f"  → {reason}")

        except Exception as e:
            logger.error(f"❌ Gelişsin analizi hatası: {e}")

    result = {
        "signals": signals, "tp1": 0.0, "tp2": 0.0, "sl": 0.0, "vade": "",
        "score": 0, "trend": "", "rr": None,
        "ml_score": ml_score,
        "regime": regime,
        "features": features,
    }

    if signals:
        try:
            close = completed["close"] if "close" in completed.columns else completed["Close"]
            high = completed["high"] if "high" in completed.columns else completed.get("High", close)
            low = completed["low"] if "low" in completed.columns else completed.get("Low", close)

            # ATR hesaplama (14 günlük)
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean().iloc[-1]

            curr_price = close.iloc[-1]

            # --- Konfluans skoru: piyasa bazlı ağırlık tablosuyla ---
            weights = SIGNAL_SCORES_CRYPTO if market == "crypto" else SIGNAL_SCORES_STOCK
            score = sum(weights.get(_signal_category(s), 1) for s in base_signals)

            # ML skorunu da dahil et (0-1 arası, 10 ile çarpıp ağırlıklı ekle)
            if USE_ML_SCORING:
                score = int(score * 0.7 + ml_score * 10 * 0.3)

            # --- Trend rejimi: SMA200 üstü BOĞA, altı AYI piyasası ---
            trend = "BİLİNMİYOR"
            if len(close) >= 200:
                sma200 = close.rolling(window=200).mean().iloc[-1]
                if pd.notna(sma200):
                    trend = "BOĞA" if curr_price > sma200 else "AYI"

            # --- SL: teknik seviye ama % risk bütçesiyle sınırlı ---
            recent_low = low.tail(10).min()
            sl_raw = min(curr_price - (atr * 1.5), recent_low)
            sl = max(sl_raw, curr_price * (1 - MAX_STOP_LOSS_PCT))

            # --- TP hedefleri minimum R/R sağlayacak şekilde kurulur ---
            risk = curr_price - sl
            tp1 = curr_price + max(atr * 1.5, risk * MIN_RISK_REWARD_RATIO)
            tp2 = curr_price + max(atr * 3.0, risk * MIN_RISK_REWARD_RATIO * 2)
            rr = round((tp1 - curr_price) / risk, 3) if risk > 0 else None

            # Vade
            if any("Altın Kesişim" in str(s) for s in base_signals):
                vade = "Orta-Uzun Vade (1-3 Ay)"
            else:
                vade = "Kısa Vade (1-3 Hafta)"

            result["score"] = score
            result["trend"] = trend
            result["rr"] = rr

            # --- Kalite kapıları ---
            threshold = _min_score(market, trend)
            gated_reason = None
            if risk <= 0:
                gated_reason = f"geçersiz stop mesafesi (risk={risk})"
            elif score < threshold:
                gated_reason = f"konfluans {score} < {threshold} [{trend}]"
            elif USE_ML_SCORING and ml_score < ML_MIN_CONFIDENCE:
                gated_reason = f"ML güven skoru düşük: {ml_score:.2f} < {ML_MIN_CONFIDENCE}"

            if gated_reason:
                logger.info(f"🚫 Sinyal bastırıldı: {gated_reason}")
                signals.clear()
                result.update({"signals": [], "tp1": 0.0, "tp2": 0.0, "sl": 0.0, "vade": ""})
            else:
                result["tp1"] = float(tp1)
                result["tp2"] = float(tp2)
                result["sl"] = float(sl)
                result["vade"] = vade

        except Exception as e:
            logger.error(f"❌ TP/SL hesaplama hatası: {e}")

    return result


# ============================================================================
# 5. BAŞLANGIÇ VE MENÜ KOMUTLARI (/start, /help)
# ============================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Botu başlatan /start komutu."""
    msg = update.effective_message
    welcome_message = (
        "🤖 <b>Finans Takip Botu</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "Merhaba! Ben senin kişisel finans asistanınım.\n"
        "Hisse senedi, kripto para ve temettü takibi yapıyorum.\n\n"
        "📊 <b>Neler yapabilirim?</b>\n"
        "• Kripto ve hisse anlık fiyat sorgulama\n"
        "• RSI ve MACD teknik analiz\n"
        "• Temettü bilgisi ve takvimi\n"
        "• Otomatik RSI ve temettü uyarıları\n\n"
        "Komutları görmek için /help yazın."
    )
    await msg.reply_text(welcome_message, parse_mode="HTML")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/help komutu — kullanılabilir komutları listeler."""
    msg = update.effective_message
    help_message = (
        "📋 <b>Kullanılabilir Komutlar</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "🪙 <b>Kripto Komutları</b>\n"
        "  /kripto BTC  →  BTC/USDT anlık fiyat + RSI + MACD\n"
        "  /kripto DOT  →  DOT/USDT anlık fiyat + RSI + MACD\n\n"
        "📈 <b>Hisse Komutları</b>\n"
        "  /hisse AAPL     →  Apple hisse fiyat + RSI + MACD\n"
        "  /hisse THYAO.IS →  Türk Hava Yolları fiyat + analiz\n\n"
        "💰 <b>Temettü Komutları</b>\n"
        "  /temettu AAPL  →  Apple temettü bilgileri\n"
        "  /temettu MSFT  →  Microsoft temettü bilgileri\n\n"
        "🚀 <b>Sinyal Tarama</b>\n"
        "  /sinyal        →  BIST hisselerini AL sinyali için tara\n"
        "  /sinyal ASELS  →  Tek hisse sinyal analizi\n"
        "  /kriptosinyal  →  Kripto paraları AL sinyali için tara\n"
        "  /tamtara       →  Hem hisse hem kripto tara\n\n"
        "🤖 <b>ML & Portföy</b>\n"
        "  /mltrain       →  ML modellerini eğit (quick opsiyonel)\n"
        "  /portfoy       →  Portföy risk durumu ve metrikler\n\n"
        "🔧 <b>Diğer</b>\n"
        "  /chatid   →  Chat ID'nizi öğrenin\n"
        "  /izle     →  İzleme listesini görüntüle\n"
        "  /start    →  Botu yeniden başlat\n"
        "  /help     →  Bu mesajı göster\n\n"
        "⏰ <i>Bot her 15 dakikada bir AL sinyali taraması yapar\n"
        "ve fırsat bulursa otomatik uyarı gönderir.</i>"
    )
    await msg.reply_text(help_message, parse_mode="HTML")


async def chatid_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/chatid komutu — kullanıcının Chat ID'sini gösterir."""
    msg = update.effective_message
    chat_id = update.effective_chat.id
    await msg.reply_text(
        f"🆔 <b>Chat ID'niz:</b> <code>{chat_id}</code>\n\n"
        f"Bu değeri <code>.env</code> dosyasındaki <code>CHAT_ID</code> alanına yazın.",
        parse_mode="HTML",
    )


# ============================================================================
# 6. KRİPTO VE HİSSE SORGU KOMUTLARI (/kripto, /hisse)
# ============================================================================
async def kripto_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /kripto <sembol> komutu.
    Örnek: /kripto BTC, /kripto DOT
    """
    msg = update.effective_message
    if not context.args:
        await msg.reply_text(
            "⚠️ Kullanım: <code>/kripto BTC</code>\n"
            "Bir kripto sembolü belirtin.",
            parse_mode="HTML",
        )
        return

    symbol = context.args[0].upper()
    pair = f"{symbol}/USDT"

    await msg.reply_text(f"⏳ <i>{pair} verisi çekiliyor...</i>", parse_mode="HTML")

    data = await fetch_crypto_data(pair)
    if data is None:
        await msg.reply_text(
            f"❌ <b>{pair}</b> verisi çekilemedi.\n"
            "Sembolü kontrol edin (örn: BTC, ETH, DOT).",
            parse_mode="HTML",
        )
        return

    price = data["price"]
    df = data["ohlcv_df"]

    rsi = calculate_rsi(df)
    macd = calculate_macd(df)
    rsi_status = get_rsi_status(rsi)

    message = (
        f"🪙 <b>{pair}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💵 <b>Fiyat:</b> ${price:,.4f}\n\n"
        f"📊 <b>Teknik İndikatörler</b>\n"
        f"  RSI (14): <b>{rsi}</b> — {rsi_status}\n"
    )

    if macd:
        message += (
            f"  MACD: <b>{macd['macd']}</b>\n"
            f"  Signal: <b>{macd['signal']}</b>\n"
            f"  Histogram: <b>{macd['histogram']}</b>\n"
        )

    message += f"\n⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
    await msg.reply_text(message, parse_mode="HTML")


async def hisse_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /hisse <ticker> komutu.
    Örnek: /hisse AAPL, /hisse THYAO.IS
    """
    msg = update.effective_message
    if not context.args:
        await msg.reply_text(
            "⚠️ Kullanım: <code>/hisse AAPL</code>\n"
            "Bir hisse kodu belirtin.",
            parse_mode="HTML",
        )
        return

    ticker = context.args[0].upper()

    await msg.reply_text(f"⏳ <i>{ticker} verisi çekiliyor...</i>", parse_mode="HTML")

    data = await fetch_stock_data_async(ticker)
    if data is None:
        await msg.reply_text(
            f"❌ <b>{ticker}</b> verisi çekilemedi.\n"
            "Hisse kodunu kontrol edin (örn: AAPL, MSFT, THYAO.IS).",
            parse_mode="HTML",
        )
        return

    price = data["price"]
    info = data["info"]
    df = data["history_df"]

    rsi = calculate_rsi(df)
    macd = calculate_macd(df)
    rsi_status = get_rsi_status(rsi)

    company_name = info.get("shortName", ticker)
    currency = info.get("currency", "USD")
    market_cap = info.get("marketCap", 0)
    day_high = info.get("dayHigh", 0)
    day_low = info.get("dayLow", 0)

    # Piyasa değerini okunabilir formata çevir
    if market_cap >= 1_000_000_000_000:
        market_cap_str = f"{market_cap / 1_000_000_000_000:.2f}T"
    elif market_cap >= 1_000_000_000:
        market_cap_str = f"{market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:
        market_cap_str = f"{market_cap / 1_000_000:.2f}M"
    else:
        market_cap_str = f"{market_cap:,.0f}"

    message = (
        f"📈 <b>{company_name} ({ticker})</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💵 <b>Fiyat:</b> {price:,.2f} {currency}\n"
        f"📉 <b>Gün Aralığı:</b> {day_low:,.2f} — {day_high:,.2f}\n"
        f"🏢 <b>Piyasa Değeri:</b> {market_cap_str} {currency}\n\n"
        f"📊 <b>Teknik İndikatörler</b>\n"
        f"  RSI (14): <b>{rsi}</b> — {rsi_status}\n"
    )

    if macd:
        message += (
            f"  MACD: <b>{macd['macd']}</b>\n"
            f"  Signal: <b>{macd['signal']}</b>\n"
            f"  Histogram: <b>{macd['histogram']}</b>\n"
        )

    message += f"\n⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
    await msg.reply_text(message, parse_mode="HTML")


# ============================================================================
# 7. TEMETTÜ SORGU KOMUTU (/temettu)
# ============================================================================
async def temettu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /temettu <ticker> komutu.
    Örnek: /temettu AAPL, /temettu MSFT
    """
    msg = update.effective_message
    if not context.args:
        await msg.reply_text(
            "⚠️ Kullanım: <code>/temettu AAPL</code>\n"
            "Bir hisse kodu belirtin.",
            parse_mode="HTML",
        )
        return

    ticker = context.args[0].upper()

    await msg.reply_text(
        f"⏳ <i>{ticker} temettü bilgileri çekiliyor...</i>", parse_mode="HTML"
    )

    data = await fetch_dividend_data_async(ticker)
    if data is None:
        await msg.reply_text(
            f"❌ <b>{ticker}</b> temettü verisi çekilemedi.",
            parse_mode="HTML",
        )
        return

    dividend_yield = data["dividend_yield"]
    dividends = data["dividends"]
    ex_date = data["ex_date"]
    info = data["info"]
    company_name = info.get("shortName", ticker)

    message = (
        f"💰 <b>{company_name} ({ticker}) — Temettü Bilgileri</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    # Temettü verimi
    if dividend_yield:
        message += f"📊 <b>Temettü Verimi:</b> %{dividend_yield:.2f}\n"
    else:
        message += "📊 <b>Temettü Verimi:</b> Bilgi yok\n"

    # Yaklaşan temettü tarihi
    if ex_date:
        days_until = (ex_date - datetime.now()).days
        if days_until > 0:
            message += (
                f"📅 <b>Yaklaşan Ex-Temettü Tarihi:</b> {ex_date.strftime('%d.%m.%Y')}\n"
                f"⏳ <b>Kalan Gün:</b> {days_until} gün\n"
            )
        else:
            message += f"📅 <b>Son Ex-Temettü Tarihi:</b> {ex_date.strftime('%d.%m.%Y')}\n"
    else:
        message += "📅 <b>Ex-Temettü Tarihi:</b> Bilgi yok\n"

    # Son 5 temettü ödemesi
    if dividends is not None and not dividends.empty:
        message += "\n📋 <b>Son 5 Temettü Ödemesi:</b>\n"
        recent = dividends.tail(5)
        for date, amount in recent.items():
            date_str = date.strftime("%d.%m.%Y")
            message += f"  • {date_str}: <b>${amount:.4f}</b>\n"
    else:
        message += "\n📋 <i>Temettü geçmişi bulunamadı.</i>\n"

    message += f"\n⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
    await msg.reply_text(message, parse_mode="HTML")


# ============================================================================
# 8. İZLEME LİSTESİ KOMUTU (/izle)
# ============================================================================
async def izle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/izle komutu — mevcut izleme listesini gösterir."""
    msg = update.effective_message
    crypto_list = "\n".join([f"  • {c}" for c in WATCHLIST_CRYPTO])
    stock_list = "\n".join([f"  • {s}" for s in WATCHLIST_STOCKS])
    scan_list = "\n".join([f"  • {s}" for s in SCAN_STOCKS])

    message = (
        "👁️ <b>İzleme & Tarama Listesi</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"🪙 <b>Kripto İzleme:</b>\n{crypto_list}\n\n"
        f"📈 <b>Hisse İzleme:</b>\n{stock_list}\n\n"
        f"🔍 <b>AL Sinyali Tarama ({len(SCAN_STOCKS)} hisse):</b>\n{scan_list}\n\n"
        f"⏰ <i>Her {SCHEDULER_INTERVAL_MINUTES} dakikada bir taranır.</i>"
    )
    await msg.reply_text(message, parse_mode="HTML")


# ============================================================================
# 8a. PERFORMANS KOMUTU (/performans)
# ============================================================================
async def performans_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/performans komutu — canlı paper-trading istatistiklerini gösterir."""
    msg = update.effective_message
    import paper
    await msg.reply_text(paper.performance_summary(), parse_mode="HTML")


# ============================================================================
# 8b. AL SİNYALİ TARAMA KOMUTU (/sinyal)
# ============================================================================
async def sinyal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /sinyal komutu — hisseleri AL sinyali için tarar.
    /sinyal         → Tüm SCAN_STOCKS listesini tarar
    /sinyal ASELS   → Tek bir hisseyi tarar
    """
    msg = update.effective_message

    # Tek hisse mi yoksa tüm liste mi?
    if context.args:
        ticker = context.args[0].upper()
        await msg.reply_text(
            f"🔍 <i>{ticker} AL sinyali analiz ediliyor...</i>", parse_mode="HTML"
        )
        stock_data = await fetch_stock_data_async(ticker)
        if stock_data is None:
            await msg.reply_text(
                f"❌ <b>{ticker}</b> verisi çekilemedi.", parse_mode="HTML"
            )
            return

        actual_ticker = stock_data.get("ticker", ticker)
        sig_result = detect_buy_signals(stock_data["history_df"], market="stock", symbol=actual_ticker)
        signals = sig_result["signals"]
        rsi = calculate_rsi(stock_data["history_df"])
        macd = calculate_macd(stock_data["history_df"])
        company = stock_data["info"].get("shortName", actual_ticker)

        message = (
            f"🔍 <b>{company} ({actual_ticker}) — Sinyal Analizi</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
            f"💵 Fiyat: <b>{stock_data['price']:,.2f}</b>\n"
            f"📊 RSI: <b>{rsi}</b> — {get_rsi_status(rsi)}\n"
        )
        if macd:
            message += f"📉 MACD: <b>{macd['macd']}</b> | Signal: <b>{macd['signal']}</b>\n"

        if signals:
            message += "\n🚀 <b>AL SİNYALLERİ:</b>\n"
            for s in signals:
                message += f"  {s}\n"
            message += f"\n🎯 <b>Hedef / Vade Analizi:</b>\n"
            message += f"  ⏱️ Vade: <b>{sig_result['vade']}</b>\n"
            message += f"  🛑 Stop Loss: <b>{sig_result['sl']:,.2f}</b>\n"
            message += f"  ✅ TP1 (Direnç 1): <b>{sig_result['tp1']:,.2f}</b>\n"
            message += f"  ✅ TP2 (Direnç 2): <b>{sig_result['tp2']:,.2f}</b>\n"
            message += f"  🤖 ML Skor: <b>{sig_result.get('ml_score', 0):.2f}</b>\n"
            message += f"  🎭 Rejim: <b>{sig_result.get('regime', 'UNKNOWN')}</b>\n"
        else:
            message += "\n⚪ <i>Şu an aktif AL sinyali yok.</i>\n"

        message += f"\n⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
        await msg.reply_text(message, parse_mode="HTML")
        return

    # --- Tüm listeyi tara ---
    await msg.reply_text(
        f"🔍 <i>{len(SCAN_STOCKS)} BIST hissesi taranıyor...\n"
        "Bu biraz sürebilir.</i>",
        parse_mode="HTML",
    )

    buy_signals = []  # (isim, fiyat, sinyaller)

    # Hisse tarama
    for ticker in SCAN_STOCKS:
        try:
            data = await fetch_stock_data_async(ticker)
            if data is None:
                continue

            sig_result = detect_buy_signals(data["history_df"], market="stock", symbol=ticker)
            signals = sig_result["signals"]
            if signals:
                actual_ticker = data.get("ticker", ticker)
                name = data["info"].get("shortName", actual_ticker)
                price = data["price"]
                rsi = calculate_rsi(data["history_df"])
                buy_signals.append({
                    "name": f"📈 {name} ({actual_ticker})",
                    "price": f"{price:,.2f}",
                    "rsi": rsi,
                    "signals": signals,
                    "vade": sig_result["vade"],
                    "sl": sig_result["sl"],
                    "tp1": sig_result["tp1"],
                    "tp2": sig_result["tp2"],
                    "ml_score": sig_result.get("ml_score"),
                    "regime": sig_result.get("regime"),
                })
        except Exception as e:
            logger.warning(f"⚠️ Sinyal tarama hatası ({ticker}): {e}")

    # Sonuçları gönder
    if buy_signals:
        message = (
            f"🚀 <b>HİSSE AL SİNYALİ RAPORU</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Taranan: {len(SCAN_STOCKS)} BIST hissesi\n"
            f"Sinyal bulunan: <b>{len(buy_signals)}</b>\n\n"
        )
        for item in buy_signals:
            message += f"<b>{item['name']}</b>\n"
            message += f"  💵 Fiyat: {item['price']} | RSI: {item['rsi']}\n"
            for s in item["signals"]:
                message += f"  {s}\n"
            message += f"  ⏱️ Vade: {item['vade']} | 🛑 SL: {item['sl']:,.2f} | ✅ TP1: {item['tp1']:,.2f} | ✅ TP2: {item['tp2']:,.2f}\n"
            message += f"  🤖 ML: {item.get('ml_score', 0):.2f} | 🎭 Regime: {item.get('regime', 'UNKNOWN')}\n\n"

        message += f"⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
    else:
        message = (
            "🔍 <b>HİSSE AL SİNYALİ RAPORU</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Taranan: {len(SCAN_STOCKS)} BIST hissesi\n\n"
            "⚪ <i>Şu an hiçbir hissede aktif AL sinyali bulunamadı.</i>\n\n"
            f"⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
        )

    await msg.reply_text(message, parse_mode="HTML")


# ============================================================================
# 8c. KRİPTO AL SİNYALİ TARAMA KOMUTU (/kriptosinyal)
# ============================================================================
async def kriptosinyal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /kriptosinyal komutu — kripto paraları AL sinyali için tarar.
    """
    msg = update.effective_message

    await msg.reply_text(
        f"🔍 <i>{len(SCAN_CRYPTO)} kripto taranıyor...\n"
        "Bu biraz sürebilir.</i>",
        parse_mode="HTML",
    )

    buy_signals = []  # (isim, fiyat, sinyaller)

    # Kripto tarama
    for symbol in SCAN_CRYPTO:
        try:
            data = await fetch_crypto_data(symbol)
            if data is None:
                continue

            sig_result = detect_buy_signals(data["ohlcv_df"], market="crypto", symbol=symbol)
            signals = sig_result["signals"]
            if signals:
                rsi = calculate_rsi(data["ohlcv_df"])
                buy_signals.append({
                    "name": f"🪙 {symbol}",
                    "price": f"${format_crypto_price(data['price'])}",
                    "rsi": rsi,
                    "signals": signals,
                    "vade": sig_result["vade"],
                    "sl": sig_result["sl"],
                    "tp1": sig_result["tp1"],
                    "tp2": sig_result["tp2"],
                    "ml_score": sig_result.get("ml_score"),
                    "regime": sig_result.get("regime"),
                })
        except Exception as e:
            logger.warning(f"⚠️ Sinyal tarama hatası ({symbol}): {e}")

    # Sonuçları gönder
    if buy_signals:
        message = (
            f"🚀 <b>KRİPTO AL SİNYALİ RAPORU</b>\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"Taranan: {len(SCAN_CRYPTO)} kripto\n"
            f"Sinyal bulunan: <b>{len(buy_signals)}</b>\n\n"
        )
        for item in buy_signals:
            message += f"<b>{item['name']}</b>\n"
            message += f"  💵 Fiyat: {item['price']} | RSI: {item['rsi']}\n"
            for s in item["signals"]:
                message += f"  {s}\n"
            message += f"  ⏱️ Vade: {item['vade']} | 🛑 SL: {format_crypto_price(item['sl'])} | ✅ TP1: {format_crypto_price(item['tp1'])} | ✅ TP2: {format_crypto_price(item['tp2'])}\n"
            message += f"  🤖 ML: {item.get('ml_score', 0):.2f} | 🎭 Regime: {item.get('regime', 'UNKNOWN')}\n\n"

        message += f"⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
    else:
        message = (
            "🔍 <b>KRİPTO AL SİNYALİ RAPORU</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Taranan: {len(SCAN_CRYPTO)} kripto\n\n"
            "⚪ <i>Şu an hiçbir kripto parada aktif AL sinyali bulunamadı.</i>\n\n"
            f"⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
        )

    await msg.reply_text(message, parse_mode="HTML")


# ============================================================================
# 8d. TÜMÜNÜ TARA KOMUTU (/tamtara)
# ============================================================================
async def tamtara_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /tamtara komutu — hem hisseleri hem de kripto paraları AL sinyali için tarar.
    Sırayla önce hisseleri, sonra kriptoları tarar.
    """
    msg = update.effective_message

    await msg.reply_text(
        f"🔍 <i>{len(SCAN_STOCKS)} BIST hissesi ve {len(SCAN_CRYPTO)} kripto taranıyor...\n"
        "Lütfen bekleyin, bu işlem biraz zaman alabilir.</i>",
        parse_mode="HTML",
    )

    # 1. Hisse Tarama
    buy_signals_stock = []
    for ticker in SCAN_STOCKS:
        try:
            data = await fetch_stock_data_async(ticker)
            if data is None:
                continue

            sig_result = detect_buy_signals(data["history_df"], market="stock", symbol=ticker)
            signals = sig_result["signals"]
            if signals:
                actual_ticker = data.get("ticker", ticker)
                name = data["info"].get("shortName", actual_ticker)
                rsi = calculate_rsi(data["history_df"])
                buy_signals_stock.append({
                    "name": f"📈 {name} ({actual_ticker})",
                    "price": f"{data['price']:,.2f}",
                    "rsi": rsi,
                    "signals": signals,
                    "vade": sig_result["vade"],
                    "sl": sig_result["sl"],
                    "tp1": sig_result["tp1"],
                    "tp2": sig_result["tp2"],
                    "ml_score": sig_result.get("ml_score"),
                    "regime": sig_result.get("regime"),
                })
        except Exception as e:
            logger.warning(f"⚠️ Sinyal tarama hatası ({ticker}): {e}")

    # 2. Kripto Tarama
    buy_signals_crypto = []
    for symbol in SCAN_CRYPTO:
        try:
            data = await fetch_crypto_data(symbol)
            if data is None:
                continue

            sig_result = detect_buy_signals(data["ohlcv_df"], market="crypto", symbol=symbol)
            signals = sig_result["signals"]
            if signals:
                rsi = calculate_rsi(data["ohlcv_df"])
                buy_signals_crypto.append({
                    "name": f"🪙 {symbol}",
                    "price": f"${format_crypto_price(data['price'])}",
                    "rsi": rsi,
                    "signals": signals,
                    "vade": sig_result["vade"],
                    "sl": sig_result["sl"],
                    "tp1": sig_result["tp1"],
                    "tp2": sig_result["tp2"],
                    "ml_score": sig_result.get("ml_score"),
                    "regime": sig_result.get("regime"),
                })
        except Exception as e:
            logger.warning(f"⚠️ Sinyal tarama hatası ({symbol}): {e}")

    # --- Mesajı Oluştur ---
    message = (
        f"🚀 <b>GENEL AL SİNYALİ RAPORU</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"Taranan: {len(SCAN_STOCKS)} Hisse, {len(SCAN_CRYPTO)} Kripto\n\n"
    )

    # Hisse Sinyalleri
    if buy_signals_stock:
        message += "📊 <b>HİSSE SENETLERİ:</b>\n"
        for item in buy_signals_stock:
            message += f"<b>{item['name']}</b>\n"
            message += f"  💵 Fiyat: {item['price']} | RSI: {item['rsi']}\n"
            for s in item["signals"]:
                message += f"  {s}\n"
            message += f"  ⏱️ Vade: {item['vade']} | 🛑 SL: {item['sl']:,.2f} | ✅ TP1: {item['tp1']:,.2f} | ✅ TP2: {item['tp2']:,.2f}\n"
            message += f"  🤖 ML: {item.get('ml_score', 0):.2f} | 🎭 Regime: {item.get('regime', 'UNKNOWN')}\n\n"
    else:
        message += "📊 <b>HİSSE SENETLERİ:</b> Aktif sinyal bulunamadı.\n\n"

    # Kripto Sinyalleri
    if buy_signals_crypto:
        message += "🪙 <b>KRİPTOLAR:</b>\n"
        for item in buy_signals_crypto:
            message += f"<b>{item['name']}</b>\n"
            message += f"  💵 Fiyat: {item['price']} | RSI: {item['rsi']}\n"
            for s in item["signals"]:
                message += f"  {s}\n"
            message += f"  ⏱️ Vade: {item['vade']} | 🛑 SL: {format_crypto_price(item['sl'])} | ✅ TP1: {format_crypto_price(item['tp1'])} | ✅ TP2: {format_crypto_price(item['tp2'])}\n"
            message += f"  🤖 ML: {item.get('ml_score', 0):.2f} | 🎭 Regime: {item.get('regime', 'UNKNOWN')}\n\n"
    else:
        message += "🪙 <b>KRİPTOLAR:</b> Aktif sinyal bulunamadı.\n\n"

    message += f"⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
    
    await msg.reply_text(message, parse_mode="HTML")


# ============================================================================
# 9. ARKA PLAN ZAMANLAYICISI VE OTOMATİK UYARI MEKANİZMASI
# ============================================================================
async def scheduled_check(app: Application) -> None:
    """
    Her 15 dakikada bir çalışan zamanlayıcı fonksiyonu.
    İzleme listesindeki varlıkları kontrol eder ve AL sinyali/uyarı gönderir.

    Uyarı Koşulları:
    - AL sinyali tespit edilen hisse/kripto → detaylı sinyal raporu
    - Temettü tarihi 7 gün içinde → Temettü uyarısı
    """
    if not CHAT_ID:
        logger.warning("⚠️ CHAT_ID ayarlanmamış, uyarılar gönderilemez.")
        return

    # Önce bekleyen sinyalleri sonuçlandır (paper trading döngüsü)
    try:
        import paper
        await paper.settle_signals()
    except Exception as e:
        logger.error(f"❌ Paper trading değerlendirmesi hatası: {e}")

    alerts = []
    buy_signal_items = []
    pending_cooldown = []  # (sembol_anahtarı, sinyaller) — gönderim sonrası cooldown'a sokulur
    div_alert_keys = []  # gönderilen temettü uyarılarının cooldown anahtarları
    logger.info("🔄 Zamanlayıcı: AL sinyali taraması başlıyor...")

    # Update portfolio state before checking signals
    _update_portfolio_state()

    # --- Kripto AL Sinyali Taraması ---
    for symbol in SCAN_CRYPTO:
        try:
            data = await fetch_crypto_data(symbol)
            if data is None:
                continue

            sig_result = detect_buy_signals(data["ohlcv_df"], market="crypto", symbol=symbol)
            # Cooldown'da olan sinyaller eleilir — aynı uyarı tekrar spam yapmaz
            signals = filter_cooldown(f"kripto:{symbol}", sig_result["signals"])
            if not signals:
                continue

            # Portfolio risk gate
            if ENABLE_PORTFOLIO_RISK:
                gate = portfolio_gate({
                    "symbol": f"kripto:{symbol}",
                    "entry_num": data["price"],
                    "sl": sig_result["sl"],
                    "tp1": sig_result["tp1"],
                    "quantity": risk_result.quantity if (risk_result := calculate_position_risk(data["price"], sig_result["sl"])).is_calculable else 0,
                }, _portfolio_state)
                if not gate.allowed:
                    logger.info(f"🚫 Portfolio risk gate blocked {symbol}: {gate.message}")
                    continue

            rsi = calculate_rsi(data["ohlcv_df"])
            risk_result = calculate_position_risk(data["price"], sig_result["sl"])
            buy_signal_items.append({
                "name": f"🪙 {symbol}",
                "key": f"kripto:{symbol}",
                "market": "crypto",
                "price": f"${format_crypto_price(data['price'])}",
                "entry_num": data["price"],
                "rsi": rsi,
                "signals": signals,
                "skor": sig_result.get("score"),
                "rejim": sig_result.get("trend"),
                "regime": sig_result.get("regime"),
                "ml_score": sig_result.get("ml_score"),
                "rr": sig_result.get("rr"),
                "vade": sig_result["vade"],
                "sl": sig_result["sl"],
                "tp1": sig_result["tp1"],
                "tp2": sig_result["tp2"],
                "risk": risk_result.as_dict(),
                "sector": "CRYPTO",
            })
            pending_cooldown.append((f"kripto:{symbol}", signals))
        except Exception as e:
            logger.error(f"❌ Zamanlayıcı kripto hatası ({symbol}): {e}")

    # --- Hisse AL Sinyali Taraması ---
    for ticker in SCAN_STOCKS:
        try:
            stock_data = await fetch_stock_data_async(ticker)
            if stock_data is None:
                continue

            actual_ticker = stock_data.get("ticker", ticker)
            sig_result = detect_buy_signals(stock_data["history_df"], market="stock", symbol=actual_ticker)
            signals = filter_cooldown(f"hisse:{actual_ticker}", sig_result["signals"])
            if not signals:
                continue

            # Portfolio risk gate
            if ENABLE_PORTFOLIO_RISK:
                risk_result = calculate_position_risk(stock_data["price"], sig_result["sl"])
                gate = portfolio_gate({
                    "symbol": f"hisse:{actual_ticker}",
                    "entry_num": stock_data["price"],
                    "sl": sig_result["sl"],
                    "tp1": sig_result["tp1"],
                    "quantity": risk_result.quantity if risk_result.is_calculable else 0,
                }, _portfolio_state)
                if not gate.allowed:
                    logger.info(f"🚫 Portfolio risk gate blocked {actual_ticker}: {gate.message}")
                    continue

            name = stock_data["info"].get("shortName", actual_ticker)
            rsi = calculate_rsi(stock_data["history_df"])
            risk_result = calculate_position_risk(
                stock_data["price"], sig_result["sl"]
            )
            # Get sector from config
            from portfolio_risk import get_portfolio_config
            sector = get_portfolio_config().sector_mapping.get(actual_ticker, "UNKNOWN")

            buy_signal_items.append({
                "name": f"📈 {name} ({actual_ticker})",
                "key": f"hisse:{actual_ticker}",
                "market": "stock",
                "price": f"{stock_data['price']:,.2f}",
                "entry_num": stock_data["price"],
                "rsi": rsi,
                "signals": signals,
                "skor": sig_result.get("score"),
                "rejim": sig_result.get("trend"),
                "regime": sig_result.get("regime"),
                "ml_score": sig_result.get("ml_score"),
                "rr": sig_result.get("rr"),
                "vade": sig_result["vade"],
                "sl": sig_result["sl"],
                "tp1": sig_result["tp1"],
                "tp2": sig_result["tp2"],
                "risk": risk_result.as_dict(),
                "sector": sector,
                "portfolio_risk_pct": risk_result.risk_amount / 100000 * 100 if risk_result.is_calculable else 0,
            })
            pending_cooldown.append((f"hisse:{actual_ticker}", signals))
        except Exception as e:
            logger.error(f"❌ Zamanlayıcı hisse hatası ({ticker}): {e}")

    # --- Temettü Kontrolleri (izleme listesinden) ---
    for ticker in WATCHLIST_STOCKS:
        try:
            div_data = await fetch_dividend_data_async(ticker)
            if div_data and div_data["ex_date"]:
                days_until = (div_data["ex_date"] - datetime.now()).days
                if 0 < days_until <= 7:
                    # Aynı temettü tarihi için tekrar uyarı gönderme
                    div_key = f"temettu:{ticker}:{div_data['ex_date']:%Y-%m-%d}"
                    if _cooldown_active(div_key):
                        continue
                    alerts.append(
                        f"💰 <b>TEMETTÜ YAKLAŞIYOR</b> — {ticker}\n"
                        f"   Tarih: {div_data['ex_date'].strftime('%d.%m.%Y')} "
                        f"({days_until} gün kaldı)\n"
                        f"   Verim: %{div_data['dividend_yield']:.2f}"
                    )
                    div_alert_keys.append(div_key)
        except Exception as e:
            logger.error(f"❌ Zamanlayıcı temettü hatası ({ticker}): {e}")

    # --- AL Sinyali Mesajı Gönder ---
    if buy_signal_items:
        message = (
            "🚀 <b>OTOMATİK AL SİNYALİ</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
        )
        for item in buy_signal_items:
            message += f"<b>{item['name']}</b>\n"
            message += f"  💵 Fiyat: {item['price']} | RSI: {item['rsi']}\n"
            message += (
                f"  🎯 Skor: {item.get('skor', '?')} | "
                f"ML: {item.get('ml_score', 0):.2f} | "
                f"Rejim: {item.get('rejim') or '—'} | "
                f"Regime: {item.get('regime') or '—'} | "
                f"R/R: {item['rr'] if item.get('rr') is not None else '—'}\n"
            )
            entry_num = item.get("entry_num")
            if entry_num and item.get("sl"):
                message += sizing_hint(float(entry_num), float(item["sl"]))
            for s in item["signals"]:
                message += f"  {s}\n"

            if "🪙" in item["name"]:
                message += f"  ⏱️ Vade: {item['vade']} | 🛑 SL: {format_crypto_price(item['sl'])} | ✅ TP1: {format_crypto_price(item['tp1'])} | ✅ TP2: {format_crypto_price(item['tp2'])}\n\n"
            else:
                message += f"  ⏱️ Vade: {item['vade']} | 🛑 SL: {item['sl']:,.2f} | ✅ TP1: {item['tp1']:,.2f} | ✅ TP2: {item['tp2']:,.2f}\n\n"

        message += f"⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"

        try:
            await app.bot.send_message(
                chat_id=CHAT_ID, text=message, parse_mode="HTML"
            )
            logger.info(f"🚀 {len(buy_signal_items)} AL sinyali gönderildi.")
            # Yalnızca BAŞARILI gönderimden sonra cooldown başlat
            for sym_key, sigs in pending_cooldown:
                mark_signals_sent(sym_key, sigs)
            # Paper trading takibine al
            try:
                import paper
                paper.record_signals(buy_signal_items)
            except Exception as e:
                logger.error(f"❌ Sinyal kaydı hatası: {e}")
        except Exception as e:
            logger.error(f"❌ AL sinyali gönderilemedi: {e}")

    # --- Temettü Uyarıları Gönder ---
    if alerts:
        header = (
            "🚨 <b>OTOMATİK UYARI</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
        )
        footer = f"\n\n⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
        full_message = header + "\n\n".join(alerts) + footer

        try:
            await app.bot.send_message(
                chat_id=CHAT_ID, text=full_message, parse_mode="HTML"
            )
            logger.info(f"📨 {len(alerts)} temettü uyarısı gönderildi.")
            _cooldown_mark(div_alert_keys)
        except Exception as e:
            logger.error(f"❌ Uyarı gönderilemedi: {e}")

    if not buy_signal_items and not alerts:
        logger.info("✅ Tarama tamamlandı, sinyal/uyarı yok.")


# ============================================================================
# 8e. ML MODEL EĞİTİM KOMUTU (/mltrain)
# ============================================================================
async def mltrain_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /mltrain komutu — ML modellerini geçmiş veriyle eğitir.
    /mltrain              → Tüm SCAN listeleriyle eğitir (uzun sürebilir)
    /mltrain quick        → Hızlı eğitim (az veri)
    """
    msg = update.effective_message
    quick_mode = context.args and context.args[0].lower() == "quick"

    await msg.reply_text(
        f"🤖 <b>ML Model Eğitimi Başlatılıyor...</b>\n"
        f"Mod: {'Hızlı' if quick_mode else 'Tam'}\n"
        f"Bu işlem birkaç dakika sürebilir.",
        parse_mode="HTML",
    )

    try:
        from ml_model import RegimeAwareModelEnsemble, prepare_training_data

        symbols_stock = SCAN_STOCKS[:10] if quick_mode else SCAN_STOCKS
        symbols_crypto = SCAN_CRYPTO[:5] if quick_mode else SCAN_CRYPTO

        await msg.reply_text("📊 Hisse verileri toplanıyor...", parse_mode="HTML")
        stock_features = prepare_training_data(symbols_stock, "stock", lookback_days=365 if not quick_mode else 180)

        await msg.reply_text("🪙 Kripto verileri toplanıyor...", parse_mode="HTML")
        crypto_features = prepare_training_data(symbols_crypto, "crypto", lookback_days=180 if not quick_mode else 90)

        # Combine features by regime
        from signals_advanced import MarketRegime
        all_features = {r: [] for r in MarketRegime}
        for regime, feats in stock_features.items():
            all_features[regime].extend(feats)
        for regime, feats in crypto_features.items():
            all_features[regime].extend(feats)

        ensemble = RegimeAwareModelEnsemble(ML_MODEL_TYPE)
        results = ensemble.train(all_features, min_samples_per_regime=20 if quick_mode else 30)

        response = "✅ <b>ML Eğitimi Tamamlandı</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        for regime, metrics in results.items():
            response += f"📊 <b>{regime}</b>\n"
            for k, v in metrics.items():
                response += f"  {k}: {v}\n"
            response += "\n"

        await msg.reply_text(response, parse_mode="HTML")

    except Exception as e:
        logger.error(f"❌ ML eğitim hatası: {e}")
        await msg.reply_text(f"❌ Eğitim hatası: {e}", parse_mode="HTML")


# ============================================================================
# 8f. PORTFÖY DURUM KOMUTU (/portfoy)
# ============================================================================
async def portfoy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /portfoy komutu — mevcut portföy risk metriklerini gösterir.
    """
    msg = update.effective_message

    _update_portfolio_state()
    metrics = calculate_portfolio_metrics(_portfolio_state)

    message = (
        "📊 <b>Portföy Risk Durumu</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💰 Equity: ${metrics.get('total_notional', 0):,.2f}\n"
        f"📈 Peak Equity: ${_portfolio_state.peak_equity:,.2f}\n"
        f"📉 Drawdown: %{metrics.get('drawdown_pct', 0):.2f}\n"
        f"⚖️ Toplam Risk: ${metrics.get('total_risk', 0):,.2f}\n"
        f"📊 VaR (95%): ${metrics.get('var_95', 0):,.2f}\n"
        f"🔢 Açık Pozisyon: {metrics.get('total_positions', 0)}\n\n"
    )

    sector_exp = metrics.get('sector_exposure', {})
    if sector_exp:
        message += "🏭 <b>Sektör Dağılımı:</b>\n"
        for sector, notional in sorted(sector_exp.items(), key=lambda x: -x[1]):
            pct = notional / metrics.get('total_notional', 1) * 100
            message += f"  {sector}: ${notional:,.0f} (%{pct:.1f})\n"

    await msg.reply_text(message, parse_mode="HTML")


# ============================================================================
# 10. ANA ÇALIŞTIRMA DÖNGÜSÜ VE LOGLAMA
# ============================================================================
def main() -> None:
    """Bot uygulamasını ayağa kaldıran ana fonksiyon."""
    # Token kontrolü
    if not BOT_TOKEN:
        logger.error(
            "❌ BOT_TOKEN bulunamadı! .env dosyasını kontrol edin.\n"
            "Örnek: BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        )
        return

    logger.info("🚀 Finans Botu başlatılıyor...")

    # Application oluştur
    app = Application.builder().token(BOT_TOKEN).build()

    # Komut işleyicilerini ekle
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("kripto", kripto_command))
    app.add_handler(CommandHandler("hisse", hisse_command))
    app.add_handler(CommandHandler("temettu", temettu_command))
    app.add_handler(CommandHandler("chatid", chatid_command))
    app.add_handler(CommandHandler("izle", izle_command))
    app.add_handler(CommandHandler("sinyal", sinyal_command))
    app.add_handler(CommandHandler("kriptosinyal", kriptosinyal_command))
    app.add_handler(CommandHandler("tamtara", tamtara_command))
    app.add_handler(CommandHandler("performans", performans_command))
    app.add_handler(CommandHandler("mltrain", mltrain_command))
    app.add_handler(CommandHandler("portfoy", portfoy_command))

    # Arka plan zamanlayıcısını hazırla (henüz başlatma, event loop yok)
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        scheduled_check,
        trigger=IntervalTrigger(minutes=SCHEDULER_INTERVAL_MINUTES),
        args=[app],
        id="watchlist_check",
        name="AL Sinyali Taraması",
        replace_existing=True,
        max_instances=1,   # tarama 15 dk'yı aşarsa üst üste binmesin
        coalesce=True,     # geciken tetiklemeler tek koşuya birleşsin
    )

    # Bot komutları menüsünü ayarla ve scheduler'ı event loop içinde başlat
    async def post_init(application: Application) -> None:
        await application.bot.set_my_commands([
            BotCommand("start", "Botu başlat"),
            BotCommand("help", "Yardım ve komut listesi"),
            BotCommand("kripto", "Kripto fiyat ve analiz (örn: /kripto BTC)"),
            BotCommand("hisse", "Hisse fiyat ve analiz (örn: /hisse AAPL)"),
            BotCommand("temettu", "Temettü bilgileri (örn: /temettu AAPL)"),
            BotCommand("sinyal", "Hisse AL sinyali tara"),
            BotCommand("kriptosinyal", "Kripto AL sinyali tara"),
            BotCommand("tamtara", "Hem hisse hem kripto tara"),
            BotCommand("performans", "Canlı sinyal performansı"),
            BotCommand("mltrain", "ML modellerini eğit"),
            BotCommand("portfoy", "Portföy risk durumu"),
            BotCommand("izle", "İzleme ve tarama listesi"),
            BotCommand("chatid", "Chat ID'nizi öğrenin"),
        ])
        logger.info("✅ Bot komutları menüsü ayarlandı.")

        # Scheduler'ı burada başlat — artık event loop çalışıyor
        scheduler.start()
        logger.info(
            f"⏰ Zamanlayıcı başlatıldı — Her {SCHEDULER_INTERVAL_MINUTES} dakikada bir kontrol."
        )

    app.post_init = post_init

    # Botu çalıştır
    logger.info("✅ Bot hazır! Telegram'da komut gönderebilirsiniz.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
