"""
Telegram Finans Botu
====================
Hisse senedi, kripto para ve temettü takibi yapan,
teknik indikatörleri (RSI, MACD) hesaplayan asenkron Telegram botu.

Kütüphaneler: python-telegram-bot (v20+), yfinance, ccxt, pandas, apscheduler
"""

import os
import sys
import logging
import math
from datetime import datetime

# Handlers import this module by name. Script startup must share that same
# module, otherwise Python creates a second configuration/portfolio instance.
if __name__ == "__main__":
    sys.modules["bot"] = sys.modules[__name__]

import pandas as pd
from dotenv import load_dotenv

from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, CallbackQueryHandler
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

import paper
from portfolio_risk import PortfolioState
from risk import get_risk_config

# ============================================================================
# 1. KONFİGÜRASYON VE ORTAM DEĞİŞKENLERİ
# ============================================================================
load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

# İzleme listesi
WATCHLIST_CRYPTO = ["BTC/USDT", "ETH/USDT", "DOT/USDT"]
WATCHLIST_STOCKS = ["THYAO.IS", "ASELS.IS", "TUPRS.IS"]

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

# Automatic scans and model training focus on BIST. Manual crypto commands remain available.
BIST_ONLY = True

# --- Portfolio Risk Ayarları ---
ENABLE_PORTFOLIO_RISK = True

logger = logging.getLogger(__name__)

# ============================================================================
# 2. PORTFÖY DURUMU
# ============================================================================
_portfolio_state = PortfolioState(
    positions=[],
    equity=0.0,
    peak_equity=0.0,
    daily_pnl=0.0,
    account_size=0.0,
    price_data={},
)

def _update_portfolio_state() -> bool:
    """Rebuild the immutable snapshot using sized, settled cash P/L."""
    global _portfolio_state
    try:
        open_positions = paper._load_open()
        account_size = get_risk_config().account_size or 0.0
        equity = peak_equity = account_size
        daily_pnl = 0.0
        if paper.SETTLED_CSV.exists():
            df = pd.read_csv(paper.SETTLED_CSV, encoding="utf-8-sig")
            if not df.empty:
                df = df.sort_values("closed_at", kind="stable")
                raw_amounts = df.get("net_pnl_amount", pd.Series(index=df.index, dtype=float))
                amounts = pd.to_numeric(raw_amounts, errors="coerce")
                if (raw_amounts.notna() & amounts.isna()).any():
                    raise ValueError("Sonuçlanan işlem tutarı sayısal değil")
                # Old unsized signals have no cash P/L; never compound their
                # percentage return as if each invested the entire account.
                if {"entry", "quantity", "net_pnl_pct"}.issubset(df.columns):
                    fallback = (pd.to_numeric(df["entry"], errors="coerce")
                                * pd.to_numeric(df["quantity"], errors="coerce")
                                * (1 + paper.COST_RATE)
                                * pd.to_numeric(df["net_pnl_pct"], errors="coerce") / 100)
                    amounts = amounts.fillna(fallback)
                amounts = amounts.fillna(0.0)
                if not amounts.map(math.isfinite).all():
                    raise ValueError("Sonuçlanan işlem tutarı sonlu değil")
                curve = account_size + amounts.cumsum()
                equity = float(curve.iloc[-1])
                peak_equity = max(account_size, float(curve.max()))
                today = datetime.now().date().isoformat()
                daily_pnl = float(amounts[df["closed_at"].astype(str).str[:10] == today].sum())
        _portfolio_state = PortfolioState(
            positions=open_positions, equity=equity, peak_equity=peak_equity,
            daily_pnl=daily_pnl, account_size=account_size,
            price_data={},
        )
        return True
    except Exception as e:
        logger.warning(f"⚠️ Portfolio state güncellenemedi: {e}")
        return False


# ============================================================================
# 3. DIŞ MODÜLLERDEN RE-EXPORT (Geriye Dönük Uyumluluk)
# ============================================================================
# backtest.py, train_bist.py ve test dosyaları için gerekli fonksiyonlar
from signal_engine import (
    detect_buy_signals,
    _get_ml_ensemble,
    _signal_category,
    SIGNAL_SCORES_STOCK,
    ML_FILTER_SIGNALS
)


# ============================================================================
# 4. ANA ÇALIŞTIRMA DÖNGÜSÜ
# ============================================================================
def main() -> None:
    """Bot uygulamasını ayağa kaldıran ana fonksiyon."""
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[logging.StreamHandler(), logging.FileHandler("finans_bot.log", encoding="utf-8")],
    )
    # Token kontrolü
    if not BOT_TOKEN:
        logger.error(
            "❌ BOT_TOKEN bulunamadı! .env dosyasını kontrol edin.\n"
            "Örnek: BOT_TOKEN=123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11"
        )
        return

    logger.info("🚀 Finans Botu başlatılıyor...")

    # Komut handler'larını içeri aktar (döngüsel importları önlemek için burada yapıyoruz)
    from telegram_handlers import (
        start_command, help_command, chatid_command,
        kripto_command, hisse_command, temettu_command,
        izle_command, performans_command,
        sinyal_command, kriptosinyal_command, tamtara_command,
        mltrain_command, portfoy_command, grafik_command,
        hisseler_command, stock_menu_callback
    )
    from scheduler import scheduled_check

    # Application oluştur
    app = Application.builder().token(BOT_TOKEN).build()

    # Komut işleyicilerini ekle
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("chatid", chatid_command))
    app.add_handler(CommandHandler("kripto", kripto_command))
    app.add_handler(CommandHandler("hisse", hisse_command))
    app.add_handler(CommandHandler("hisseler", hisseler_command))
    app.add_handler(CallbackQueryHandler(stock_menu_callback, pattern=r"^stock(?:menu|info):"))
    app.add_handler(CommandHandler("temettu", temettu_command))
    app.add_handler(CommandHandler("izle", izle_command))
    app.add_handler(CommandHandler("sinyal", sinyal_command))
    app.add_handler(CommandHandler("kriptosinyal", kriptosinyal_command))
    app.add_handler(CommandHandler("tamtara", tamtara_command))
    app.add_handler(CommandHandler("performans", performans_command))
    app.add_handler(CommandHandler("mltrain", mltrain_command, block=False))
    app.add_handler(CommandHandler("portfoy", portfoy_command))
    app.add_handler(CommandHandler("grafik", grafik_command))

    # Arka plan zamanlayıcısını hazırla
    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        scheduled_check,
        trigger=IntervalTrigger(minutes=SCHEDULER_INTERVAL_MINUTES),
        args=[app],
        id="watchlist_check",
        name="AL Sinyali Taraması",
        replace_existing=True,
        max_instances=1,
        coalesce=True,
    )

    # Bot komutları menüsünü ayarla ve scheduler'ı başlat
    async def post_init(application: Application) -> None:
        await application.bot.set_my_commands([
            BotCommand("start", "Botu başlat"),
            BotCommand("help", "Yardım ve komut listesi"),
            BotCommand("kripto", "Kripto fiyat ve analiz (örn: /kripto BTC)"),
            BotCommand("hisseler", "Hisse kodlarından seçerek analiz aç"),
            BotCommand("hisse", "Hisse menüsü veya analiz (örn: /hisse THYAO.IS)"),
            BotCommand("temettu", "Temettü bilgileri (örn: /temettu AAPL)"),
            BotCommand("sinyal", "Hisse AL sinyali tara"),
            BotCommand("kriptosinyal", "Kripto AL sinyali tara"),
            BotCommand("tamtara", "Hem hisse hem kripto tara"),
            BotCommand("performans", "Canlı sinyal performansı"),
            BotCommand("mltrain", "ML modellerini eğit"),
            BotCommand("portfoy", "Portföy risk durumu"),
            BotCommand("grafik", "Equity curve grafiğini çiz"),
            BotCommand("izle", "İzleme ve tarama listesi"),
            BotCommand("chatid", "Chat ID'nizi öğrenin"),
        ])
        logger.info("✅ Bot komutları menüsü ayarlandı.")

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
