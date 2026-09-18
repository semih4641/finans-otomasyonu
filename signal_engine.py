"""
AL Sinyali Motoru
==================
Teknik sinyallerin tespit edilmesi, konfluans skorlaması, kalite kapıları,
ML ensemble yönetimi ve sinyal cooldown mekanizması.
"""

import json
import logging
import math
from datetime import datetime, timedelta

import pandas as pd

from indicators import _completed_candles, _rsi_series
from signals_advanced import extract_features, generate_advanced_signals, SignalFeatures
from ml_model import RegimeAwareModelEnsemble
from risk import calculate_position_risk, format_position_risk

logger = logging.getLogger(__name__)

# ============================================================================
# SİNYAL KONFİGÜRASYONU
# ============================================================================

# Aynı sinyal tekrar spam yapmasın: sembol+sinyal türü başına sessizlik süresi (saat)
SIGNAL_COOLDOWN_HOURS = 24
# Cooldown durumunun kalıcı olarak saklandığı dosya
SIGNAL_STATE_FILE = "signal_state.json"

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
# Independent BIST validation has not established reliable discrimination.
# Keep the model score visible while technical setups determine signal admission.
ML_FILTER_SIGNALS = False

# ============================================================================
# ML MODEL ENSEMBLE (lazy init)
# ============================================================================
_ml_ensemble: RegimeAwareModelEnsemble | None = None
_ml_models_loaded = False
_ml_training_active = False


def _get_ml_ensemble() -> RegimeAwareModelEnsemble:
    global _ml_ensemble, _ml_models_loaded
    if _ml_ensemble is None:
        try:
            _ml_ensemble = RegimeAwareModelEnsemble.load_saved(ML_MODEL_TYPE)
        except Exception as exc:
            logger.warning("Aktif ML model paketi yüklenemedi: %s", exc)
            _ml_ensemble = RegimeAwareModelEnsemble(ML_MODEL_TYPE)
        _ml_models_loaded = bool(_ml_ensemble.global_model or _ml_ensemble.models)
    return _ml_ensemble


# ============================================================================
# SİNYAL TEKRAR ÖNLEME (COOLDOWN)
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
# YARDIMCI FONKSİYONLAR
# ============================================================================
def format_ml_score(result: dict) -> str:
    """Distinguish an actual auxiliary model score from the neutral fallback."""
    if not result.get("ml_available"):
        return "— (uygun model/veri yok)"
    suffix = "filtre" if ML_FILTER_SIGNALS else "yardımcı puan"
    return f"{result['ml_score']:.2f} ({suffix})"


def _min_score(market: str, trend: str) -> int:
    """Piyasa türü ve rejime göre gereken minimum konfluans skoru."""
    if market == "crypto":
        return MIN_SIGNAL_SCORE_CRYPTO
    return MIN_SIGNAL_SCORE_BEAR if trend == "AYI" else MIN_SIGNAL_SCORE


def sizing_hint(entry: float, sl: float) -> str:
    """Return risk-only sizing information; never place an order."""
    return format_position_risk(calculate_position_risk(entry, sl))


# ============================================================================
# AL SİNYALİ TESPİT MOTORU
# ============================================================================
def detect_buy_signals(df: pd.DataFrame, market: str = "stock", symbol: str = "", *, use_ml: bool | None = None) -> dict:
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

        if (len(close) < 30 or not pd.to_numeric(close.tail(30), errors="coerce").map(
                lambda value: pd.notna(value) and math.isfinite(value) and value > 0).all()):
            return {"signals": [], "tp1": 0.0, "tp2": 0.0, "sl": 0.0, "vade": "",
                    "score": 0, "trend": "", "rr": None, "ml_score": 0.5, "regime": "UNKNOWN", "features": None,
                    "decision_reason": "En az 30 geçerli kapanış gerekli; veri yetersiz veya geçersiz."}

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
        if len(close) >= 201:
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
            avg_volume = volume.iloc[-21:-1].mean()
            curr_volume = volume.iloc[-1]
            if avg_volume > 0 and curr_volume > avg_volume * 1.5:
                ratio = curr_volume / avg_volume
                signals.append(f"🔊 Hacim Patlaması (hacim ortalamanın {ratio:.1f}x üzerinde)")
                base_signals.append("🔊 Hacim Patlaması")

    except Exception as e:
        logger.error(f"❌ Temel sinyal tespiti hatası: {e}")
        return {"signals": [], "tp1": 0.0, "tp2": 0.0, "sl": 0.0, "vade": "",
                "score": 0, "trend": "", "rr": None, "ml_score": 0.5,
                "ml_available": False, "regime": "UNKNOWN", "features": None,
                "decision_reason": "Veri analiz edilemedi; teknik değerlendirme üretilemedi."}

    # --- Gelişmiş Özellikler ve ML Scoring ---
    ml_score = 0.5
    ml_available = False
    regime = "UNKNOWN"
    features: SignalFeatures | None = None
    advanced_signals = []
    score_reasons = []

    if (signals and (USE_ML_SCORING if use_ml is None else use_ml)
            and market == "stock" and symbol.upper().endswith(".IS") and len(completed) >= 201):
        try:
            # Extract features for advanced analysis
            features = extract_features(df, symbol, market)
            regime = features.regime.value

            # Generate advanced signals
            advanced_result = generate_advanced_signals(df, symbol, market)
            advanced_signals = advanced_result.get("base_signals", [])
            score_reasons = advanced_result.get("score_reasons", [])

            # ML Model prediction
            ensemble = _get_ml_ensemble()
            if ensemble.global_model or features.regime in ensemble.models:
                ml_prob, model_used = ensemble.predict(features)
                if model_used != "no_model" and math.isfinite(ml_prob) and 0 <= ml_prob <= 1:
                    ml_score = ml_prob
                    ml_available = True
                logger.debug(f"ML prediction ({model_used}): {ml_prob:.3f} for {symbol}")

        except Exception as e:
            logger.error(f"❌ Gelişsin analizi hatası: {e}")

    result = {
        "signals": signals, "tp1": 0.0, "tp2": 0.0, "sl": 0.0, "vade": "",
        "score": 0, "trend": "", "rr": None,
        "ml_score": ml_score,
        "ml_available": ml_available,
        "score_reasons": score_reasons,
        "regime": regime,
        "features": features,
        "decision_reason": "Temel teknik koşullarda yeni bir sinyal oluşmadı.",
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

            # ML filters eligible technical setups; it must not create new
            # setups absent from its own training population.

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
            if not all(math.isfinite(float(value)) for value in (curr_price, atr, risk, sl, tp1, tp2)) or risk <= 0 or sl <= 0:
                gated_reason = f"geçersiz stop mesafesi (risk={risk})"
            elif score < threshold:
                gated_reason = f"konfluans {score} < {threshold} [{trend}]"
            elif ML_FILTER_SIGNALS and ml_available and ml_score < ML_MIN_CONFIDENCE:
                gated_reason = f"ML güven skoru düşük: {ml_score:.2f} < {ML_MIN_CONFIDENCE}"

            if gated_reason:
                result["decision_reason"] = (
                    f"Teknik koşulların toplam puanı {score}; bu eğilimde en az {threshold} gerekiyor."
                    if score < threshold else
                    "Model puanı etkin filtre eşiğinin altında." if ML_FILTER_SIGNALS and ml_available and ml_score < ML_MIN_CONFIDENCE
                    else "Geçerli bir stop mesafesi hesaplanamadı."
                )
                logger.info(f"🚫 Sinyal bastırıldı: {gated_reason}")
                signals.clear()
                result.update({"signals": [], "tp1": 0.0, "tp2": 0.0, "sl": 0.0, "vade": ""})
            else:
                result["decision_reason"] = "Teknik puan ve stop/ hedef koşulları sağlandı."
                result["tp1"] = float(tp1)
                result["tp2"] = float(tp2)
                result["sl"] = float(sl)
                result["vade"] = vade

        except Exception as e:
            logger.error(f"❌ TP/SL hesaplama hatası: {e}")
            result.update({"signals": [], "tp1": 0.0, "tp2": 0.0, "sl": 0.0, "vade": ""})
            result["decision_reason"] = "Stop ve hedef seviyeleri güvenilir biçimde hesaplanamadı."

    return result
