"""
Telegram Komut İşleyicileri
============================
Tüm Telegram bot komutları, yetkilendirme decorator'ü,
mesaj bölme yardımcısı ve equity curve grafik komutu.
"""

import asyncio
import io
import logging
import math
from datetime import datetime
from functools import wraps
from html import escape

import pandas as pd

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ContextTypes

from indicators import calculate_rsi, calculate_macd, get_rsi_status, format_crypto_price
from data_fetcher import fetch_crypto_data, fetch_stock_data_async, fetch_dividend_data_async
from signal_engine import detect_buy_signals, sizing_hint, format_ml_score
from risk import calculate_position_risk
from portfolio_risk import calculate_portfolio_metrics

logger = logging.getLogger(__name__)

# ============================================================================
# YETKİLENDİRME DECORATOR'Ü
# ============================================================================
TELEGRAM_MAX_LENGTH = 4000  # 96 karakter güvenlik payı (limit: 4096)


def authorized(func):
    """CHAT_ID kontrolü — yalnızca yetkili kullanıcı komut çalıştırabilir.

    /start, /help ve /chatid herkese açıktır; diğer tüm komutlar
    yalnızca .env'deki CHAT_ID'ye sahip kullanıcıya yanıt verir.
    """
    @wraps(func)
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE):
        from bot import CHAT_ID
        if CHAT_ID and str(update.effective_chat.id) != str(CHAT_ID):
            logger.warning(f"⛔ Yetkisiz erişim denemesi: chat_id={update.effective_chat.id}")
            if getattr(update, "callback_query", None):
                await update.callback_query.answer("Bu menüye erişim yetkiniz yok.", show_alert=True)
            return
        return await func(update, context)
    return wrapper


# ============================================================================
# MESAJ BÖLME YARDIMCISI (Telegram 4096 karakter limiti)
# ============================================================================
async def send_long_message(target, chat_id_or_msg, text: str, parse_mode="HTML"):
    """4096 karakteri aşan mesajları otomatik böler ve gönderir.

    target: bot nesnesi (bot.send_message) veya message nesnesi (msg.reply_text)
    chat_id_or_msg: chat_id (int/str) veya None (target bir message ise)
    """
    # Bot nesnesi ile mi yoksa message nesnesi ile mi çağrılıyor?
    if hasattr(target, "send_message"):
        # Bot nesnesi — chat_id gerekli
        async def _send(chunk):
            await target.send_message(chat_id=chat_id_or_msg, text=chunk, parse_mode=parse_mode)
    else:
        # Message nesnesi — reply_text kullan
        async def _send(chunk):
            await target.reply_text(chunk, parse_mode=parse_mode)

    while text:
        if len(text) <= TELEGRAM_MAX_LENGTH:
            await _send(text)
            break
        # Satır sonundan böl
        cut = text[:TELEGRAM_MAX_LENGTH].rfind("\n")
        if cut < 100:
            cut = TELEGRAM_MAX_LENGTH
        await _send(text[:cut])
        text = text[cut:].lstrip("\n")


# ============================================================================
# BAŞLANGIÇ VE MENÜ KOMUTLARI (/start, /help, /chatid)
# ============================================================================
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Botu başlatan /start komutu."""
    msg = update.effective_message
    welcome_message = (
        "🤖 <b>Finans Takip Botu</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        "Borsa İstanbul hisselerini anlamana yardımcı oluyorum.\n"
        "Fiyat hareketlerini, teknik koşulları ve takip edilecek seviyeleri açıklıyorum.\n\n"
        "Başlamak için aşağıdaki hisse kodlarından birine dokun.\n"
        "Raporda kapanış tarihini ve göstergelerin ne anlama geldiğini görebilirsin.\n\n"
        "📊 <b>Neler yapabilirim?</b>\n"
        "• Kripto ve hisse anlık fiyat sorgulama\n"
        "• RSI ve MACD teknik analiz\n"
        "• Temettü bilgisi ve takvimi\n"
        "• Otomatik RSI ve temettü uyarıları\n\n"
        "Komutları görmek için /help yazın."
    )
    await msg.reply_text(welcome_message, parse_mode="HTML", reply_markup=stock_menu_markup())


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
        "  /hisseler → Hisse kodlarına dokunarak analiz aç\n"
        "  /hisse THYAO.IS →  Açıklamalı günlük hisse analizi\n"
        "  /hisse ASELS.IS →  Veri tarihi, göstergeler ve teknik seviyeler\n\n"
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
        "  /portfoy       →  Portföy risk durumu ve metrikler\n"
        "  /grafik        →  Equity curve grafiği\n\n"
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
# KRİPTO VE HİSSE SORGU KOMUTLARI (/kripto, /hisse)
# ============================================================================
STOCK_MENU_PAGE_SIZE = 12


def stock_menu_symbols():
    from bot import SCAN_STOCKS
    return sorted({symbol.upper() for symbol in SCAN_STOCKS if symbol.upper().endswith(".IS")})


def stock_menu_markup(page=0):
    symbols = stock_menu_symbols()
    pages = max(1, (len(symbols) + STOCK_MENU_PAGE_SIZE - 1) // STOCK_MENU_PAGE_SIZE)
    page = max(0, min(page, pages - 1))
    selected = symbols[page * STOCK_MENU_PAGE_SIZE:(page + 1) * STOCK_MENU_PAGE_SIZE]
    buttons = [InlineKeyboardButton(symbol[:-3], callback_data=f"stockinfo:{symbol}") for symbol in selected]
    rows = [buttons[index:index + 3] for index in range(0, len(buttons), 3)]
    navigation = []
    if page > 0:
        navigation.append(InlineKeyboardButton("◀ Önceki", callback_data=f"stockmenu:{page - 1}"))
    if page + 1 < pages:
        navigation.append(InlineKeyboardButton("Sonraki ▶", callback_data=f"stockmenu:{page + 1}"))
    if navigation:
        rows.append(navigation)
    return InlineKeyboardMarkup(rows)


def stock_menu_text(page=0):
    count = len(stock_menu_symbols())
    pages = max(1, (count + STOCK_MENU_PAGE_SIZE - 1) // STOCK_MENU_PAGE_SIZE)
    page = max(0, min(page, pages - 1))
    if not count:
        return "Henüz menüye eklenmiş BIST hissesi yok."
    return (f"📈 <b>BIST hisse menüsü</b> · {page + 1}/{pages}\n\n"
            "Bilgi almak istediğin hisse koduna dokun. Kapanış tarihi, açıklamalı göstergeler ve teknik seviyeler gösterilir.")


@authorized
async def hisseler_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(stock_menu_text(), parse_mode="HTML", reply_markup=stock_menu_markup())


@authorized
async def stock_menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if query is None:
        return
    action, _, value = str(query.data or "").partition(":")
    if action == "stockmenu" and value.isdigit() and len(value) <= 6:
        await query.answer()
        page = int(value)
        # Repeated clicks on the same page are harmless.
        from telegram.error import BadRequest
        try:
            await query.edit_message_text(stock_menu_text(page), parse_mode="HTML", reply_markup=stock_menu_markup(page))
        except BadRequest as exc:
            if "message is not modified" not in str(exc).lower():
                raise
        return
    if action != "stockinfo" or value not in stock_menu_symbols():
        await query.answer("Bu hisse menüde bulunmuyor. /hisseler ile listeyi yenileyin.", show_alert=True)
        return
    await query.answer("Analiz hazırlanıyor…")
    if update.effective_message is not None:
        await _send_stock_info(update.effective_message, value)


@authorized
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


@authorized
async def hisse_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /hisse <ticker> komutu.
    Örnek: /hisse AAPL, /hisse THYAO.IS
    """
    msg = update.effective_message
    if not context.args:
        await hisseler_command(update, context)
        return

    ticker = context.args[0].upper()
    await _send_stock_info(msg, ticker)


async def _send_stock_info(msg, ticker):
    """Use one analysis path for typed symbols and menu selections."""

    await msg.reply_text(f"⏳ <i>{escape(ticker)} verisi çekiliyor...</i>", parse_mode="HTML")

    data = await fetch_stock_data_async(ticker)
    if data is None:
        await msg.reply_text(
            f"❌ <b>{escape(ticker)}</b> verisi çekilemedi.\n"
            "Hisse kodunu kontrol edin (örn: AAPL, MSFT, THYAO.IS).",
            parse_mode="HTML",
        )
        return

    if str(data.get("ticker", "")).upper().endswith(".IS"):
        from analysis_report import build_stock_report
        await send_long_message(msg, None, build_stock_report(data))
        await msg.reply_text("Başka bir hisseyi inceleyebilirsin.", reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("📈 Hisse menüsüne dön", callback_data="stockmenu:0")]
        ]))
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
# TEMETTÜ SORGU KOMUTU (/temettu)
# ============================================================================
@authorized
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
# İZLEME LİSTESİ KOMUTU (/izle)
# ============================================================================
@authorized
async def izle_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/izle komutu — mevcut izleme listesini gösterir."""
    from bot import WATCHLIST_CRYPTO, WATCHLIST_STOCKS, SCAN_STOCKS, SCHEDULER_INTERVAL_MINUTES
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
    await send_long_message(msg, None, message)


# ============================================================================
# PERFORMANS KOMUTU (/performans)
# ============================================================================
@authorized
async def performans_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/performans komutu — canlı paper-trading istatistiklerini gösterir."""
    msg = update.effective_message
    import paper
    await send_long_message(msg, None, paper.performance_summary())


# ============================================================================
# AL SİNYALİ TARAMA KOMUTU (/sinyal)
# ============================================================================
@authorized
async def sinyal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /sinyal komutu — hisseleri AL sinyali için tarar.
    /sinyal         → Tüm SCAN_STOCKS listesini tarar
    /sinyal ASELS   → Tek bir hisseyi tarar
    """
    from bot import SCAN_STOCKS
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
            message += f"  🤖 ML Skor: <b>{format_ml_score(sig_result)}</b>\n"
            message += f"  🎭 Rejim: <b>{sig_result.get('regime', 'UNKNOWN')}</b>\n"
        else:
            message += "\n⚪ <i>Şu an aktif AL sinyali yok.</i>\n"

        message += f"\n⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
        await send_long_message(msg, None, message)
        return

    # --- Tüm listeyi tara ---
    await msg.reply_text(
        f"🔍 <i>{len(SCAN_STOCKS)} BIST hissesi taranıyor...\n"
        "Bu biraz sürebilir.</i>",
        parse_mode="HTML",
    )

    buy_signals = []

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
                    "ml_available": sig_result.get("ml_available", False),
                    "regime": sig_result.get("regime"),
                })
        except Exception as e:
            logger.warning(f"⚠️ Sinyal tarama hatası ({ticker}): {e}")

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
            message += f"  🤖 ML: {format_ml_score(item)} | 🎭 Regime: {item.get('regime', 'UNKNOWN')}\n\n"

        message += f"⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
    else:
        message = (
            "🔍 <b>HİSSE AL SİNYALİ RAPORU</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Taranan: {len(SCAN_STOCKS)} BIST hissesi\n\n"
            "⚪ <i>Şu an hiçbir hissede aktif AL sinyali bulunamadı.</i>\n\n"
            f"⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
        )

    await send_long_message(msg, None, message)


# ============================================================================
# KRİPTO AL SİNYALİ TARAMA KOMUTU (/kriptosinyal)
# ============================================================================
@authorized
async def kriptosinyal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /kriptosinyal komutu — kripto paraları AL sinyali için tarar.
    """
    from bot import SCAN_CRYPTO
    msg = update.effective_message

    await msg.reply_text(
        f"🔍 <i>{len(SCAN_CRYPTO)} kripto taranıyor...\n"
        "Bu biraz sürebilir.</i>",
        parse_mode="HTML",
    )

    buy_signals = []

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
                    "ml_available": sig_result.get("ml_available", False),
                    "regime": sig_result.get("regime"),
                })
        except Exception as e:
            logger.warning(f"⚠️ Sinyal tarama hatası ({symbol}): {e}")

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
            message += f"  🤖 ML: {format_ml_score(item)} | 🎭 Regime: {item.get('regime', 'UNKNOWN')}\n\n"

        message += f"⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
    else:
        message = (
            "🔍 <b>KRİPTO AL SİNYALİ RAPORU</b>\n"
            "━━━━━━━━━━━━━━━━━━━━\n\n"
            f"Taranan: {len(SCAN_CRYPTO)} kripto\n\n"
            "⚪ <i>Şu an hiçbir kripto parada aktif AL sinyali bulunamadı.</i>\n\n"
            f"⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"
        )

    await send_long_message(msg, None, message)


# ============================================================================
# TÜMÜNÜ TARA KOMUTU (/tamtara)
# ============================================================================
@authorized
async def tamtara_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /tamtara komutu — hem hisseleri hem de kripto paraları AL sinyali için tarar.
    """
    from bot import SCAN_STOCKS, SCAN_CRYPTO
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
                    "ml_available": sig_result.get("ml_available", False),
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
                    "ml_available": sig_result.get("ml_available", False),
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
            message += f"  🤖 ML: {format_ml_score(item)} | 🎭 Regime: {item.get('regime', 'UNKNOWN')}\n\n"
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
            message += f"  🤖 ML: {format_ml_score(item)} | 🎭 Regime: {item.get('regime', 'UNKNOWN')}\n\n"
    else:
        message += "🪙 <b>KRİPTOLAR:</b> Aktif sinyal bulunamadı.\n\n"

    message += f"⏰ <i>{datetime.now().strftime('%d.%m.%Y %H:%M')}</i>"

    await send_long_message(msg, None, message)


# ============================================================================
# ML MODEL EĞİTİM KOMUTU (/mltrain)
# ============================================================================
@authorized
async def mltrain_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /mltrain komutu — ML modellerini geçmiş veriyle eğitir.
    /mltrain              → Tüm SCAN listeleriyle eğitir (uzun sürebilir)
    /mltrain quick        → Hızlı eğitim (az veri)
    """
    import signal_engine
    from bot import SCAN_STOCKS

    msg = update.effective_message
    if signal_engine._ml_training_active:
        await msg.reply_text("⏳ Model eğitimi zaten devam ediyor.")
        return
    signal_engine._ml_training_active = True
    quick_mode = context.args and context.args[0].lower() == "quick"

    try:
        await msg.reply_text(
            f"🤖 <b>ML Model Eğitimi Başlatılıyor...</b>\n"
            f"Mod: {'Hızlı' if quick_mode else 'Tam'}\n"
            f"Bu işlem birkaç dakika sürebilir.",
            parse_mode="HTML",
        )
        from ml_model import RegimeAwareModelEnsemble, prepare_training_data

        bist_symbols = [symbol for symbol in SCAN_STOCKS if symbol.endswith(".IS")]
        symbols_stock = bist_symbols[:10] if quick_mode else bist_symbols

        await msg.reply_text("📊 BIST günlük verileri ve teknik sinyal sonuçları hazırlanıyor...", parse_mode="HTML")
        stock_features = await prepare_training_data(symbols_stock, "stock", lookback_days=1825)

        # Combine features by regime
        from signals_advanced import MarketRegime
        all_features = {r: [] for r in MarketRegime}
        for regime, feats in stock_features.items():
            all_features[regime].extend(feats)

        ensemble = RegimeAwareModelEnsemble(signal_engine.ML_MODEL_TYPE)
        results = await asyncio.to_thread(ensemble.train, all_features, min_samples_per_regime=20 if quick_mode else 30)
        if not results:
            await msg.reply_text("⚠️ Model eğitilemedi: yeterli etiketli veri bulunamadı. Mevcut model korunuyor.")
            return
        signal_engine._ml_ensemble = ensemble
        signal_engine._ml_models_loaded = True

        response = "✅ <b>ML Eğitimi Tamamlandı</b>\n━━━━━━━━━━━━━━━━━━━━\n\n"
        for regime, metrics in results.items():
            response += f"📊 <b>{regime}</b>\n"
            for k, v in metrics.items():
                response += f"  {k}: {v}\n"
            response += "\n"

        await send_long_message(msg, None, response)

    except Exception as e:
        logger.error(f"❌ ML eğitim hatası: {e}")
        await msg.reply_text(f"❌ Eğitim hatası: {e}")
    finally:
        signal_engine._ml_training_active = False


# ============================================================================
# PORTFÖY DURUM KOMUTU (/portfoy)
# ============================================================================
@authorized
async def portfoy_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/portfoy komutu — mevcut portföy risk metriklerini gösterir."""
    import bot
    msg = update.effective_message

    if not bot._update_portfolio_state():
        await msg.reply_text("❌ Portföy kayıtları okunamadı; güncel durum hesaplanamıyor.")
        return
    portfolio = bot._portfolio_state
    metrics = calculate_portfolio_metrics(portfolio)

    message = (
        "📊 <b>Portföy Risk Durumu</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n\n"
        f"💰 Hesap Değeri: {portfolio.equity:,.2f}\n"
        f"📈 En Yüksek Hesap Değeri: {portfolio.peak_equity:,.2f}\n"
        f"💼 Pozisyon Tutarı: {metrics.get('total_notional', 0):,.2f}\n"
        f"📉 Drawdown: %{metrics.get('drawdown_pct', 0):.2f}\n"
        f"⚖️ Toplam Risk: {metrics.get('total_risk', 0):,.2f}\n"
        f"📊 VaR (95%): {metrics.get('var_95', 0):,.2f}\n"
        f"🔢 Açık Pozisyon: {metrics.get('total_positions', 0)}\n\n"
    )

    sector_exp = metrics.get('sector_exposure', {})
    if sector_exp:
        message += "🏭 <b>Sektör Dağılımı:</b>\n"
        for sector, notional in sorted(sector_exp.items(), key=lambda x: -x[1]):
            pct = notional / metrics.get('total_notional', 1) * 100
            message += f"  {sector}: {notional:,.0f} (%{pct:.1f})\n"

    await send_long_message(msg, None, message)


# ============================================================================
# EQUİTY CURVE GRAFİK KOMUTU (/grafik)
# ============================================================================
@authorized
async def grafik_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """/grafik komutu — equity curve grafiğini Telegram'a resim olarak gönderir."""
    import paper

    msg = update.effective_message
    await msg.reply_text("📊 <i>Grafik hazırlanıyor...</i>", parse_mode="HTML")

    try:
        # Sonuçlanan işlem verisi var mı?
        if not paper.SETTLED_CSV.exists():
            await msg.reply_text("⚠️ Henüz sonuçlanan işlem yok. Grafik oluşturulamıyor.")
            return

        df = pd.read_csv(paper.SETTLED_CSV, encoding="utf-8-sig")
        if df.empty or "net_pnl_pct" not in df.columns:
            await msg.reply_text("⚠️ Yeterli işlem verisi yok. Grafik oluşturulamıyor.")
            return

        from risk import get_risk_config
        account_size = get_risk_config().account_size or 100_000.0

        # Equity curve hesapla
        df = df.sort_values("closed_at", kind="stable")
        raw_amounts = df.get("net_pnl_amount", pd.Series(index=df.index, dtype=float))
        amounts = pd.to_numeric(raw_amounts, errors="coerce")
        if (raw_amounts.notna() & amounts.isna()).any():
            raise ValueError("Sonuçlanan işlem tutarı sayısal değil")
        if "entry" in df.columns and "quantity" in df.columns:
            fallback = (pd.to_numeric(df["entry"], errors="coerce")
                        * pd.to_numeric(df["quantity"], errors="coerce")
                        * (1 + paper.COST_RATE)
                        * pd.to_numeric(df["net_pnl_pct"], errors="coerce") / 100)
            amounts = amounts.fillna(fallback)
        amounts = amounts.fillna(0.0)
        if not amounts.map(math.isfinite).all():
            raise ValueError("Sonuçlanan işlem tutarı sonlu değil")
        equity = account_size + amounts.cumsum()
        dates = pd.to_datetime(df["closed_at"])

        # Peak equity ve drawdown
        peak = equity.cummax().clip(lower=account_size)
        drawdown_pct = (equity - peak) / peak * 100

        # matplotlib ile profesyonel grafik oluştur
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[3, 1],
                                        gridspec_kw={"hspace": 0.08})

        # Dark tema
        fig.set_facecolor("#1a1a2e")
        for ax in (ax1, ax2):
            ax.set_facecolor("#16213e")
            ax.tick_params(colors="#e0e0e0", labelsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            for spine in ax.spines.values():
                spine.set_color("#333366")

        # Equity curve
        ax1.plot(dates, equity, color="#00d2ff", linewidth=1.8, label="Hesap Değeri")
        ax1.plot(dates, peak, color="#555588", linewidth=1, linestyle="--", alpha=0.7, label="En Yüksek")
        ax1.fill_between(dates, equity, peak, where=(equity < peak),
                         color="#ff4444", alpha=0.15, label="Drawdown Bölgesi")
        ax1.fill_between(dates, account_size, equity,
                         where=(equity >= account_size),
                         color="#00d2ff", alpha=0.08)
        ax1.axhline(y=account_size, color="#666699", linewidth=0.8, linestyle=":", alpha=0.5)
        ax1.set_ylabel("Hesap Değeri (₺)", color="#e0e0e0", fontsize=11)
        ax1.legend(loc="upper left", facecolor="#1a1a2e", edgecolor="#333366",
                   labelcolor="#e0e0e0", fontsize=9)
        ax1.set_title("📈 Paper Trading Equity Curve", color="#e0e0e0", fontsize=14, fontweight="bold", pad=12)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax1.set_xticklabels([])

        # Drawdown
        ax2.fill_between(dates, drawdown_pct, 0, color="#ff4444", alpha=0.4)
        ax2.plot(dates, drawdown_pct, color="#ff6666", linewidth=1)
        ax2.set_ylabel("Drawdown %", color="#e0e0e0", fontsize=10)
        ax2.set_xlabel("Tarih", color="#e0e0e0", fontsize=10)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate(rotation=30)

        # İstatistik kutusu
        total_pnl = float(amounts.sum())
        total_pct = total_pnl / account_size * 100
        max_dd = float(drawdown_pct.min()) if len(drawdown_pct) else 0
        win_rate = (pd.to_numeric(df["net_pnl_pct"], errors="coerce") > 0).mean() * 100
        stats_text = (
            f"Toplam P/L: ₺{total_pnl:,.0f} ({total_pct:+.2f}%)\n"
            f"Max DD: {max_dd:.2f}% | Win Rate: {win_rate:.0f}%\n"
            f"İşlem: {len(df)}"
        )
        ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes,
                 fontsize=9, color="#ccccdd", ha="right", va="bottom",
                 bbox=dict(boxstyle="round,pad=0.4", facecolor="#1a1a2e",
                           edgecolor="#333366", alpha=0.9))

        # BytesIO'ya kaydet
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)

        await msg.reply_photo(
            photo=buf,
            caption=f"📊 Equity Curve — {datetime.now().strftime('%d.%m.%Y %H:%M')}"
        )

    except ImportError:
        await msg.reply_text(
            "❌ matplotlib yüklü değil.\n"
            "<code>pip install matplotlib</code> ile yükleyin.",
            parse_mode="HTML"
        )
    except Exception as e:
        logger.error(f"❌ Grafik oluşturma hatası: {e}")
        await msg.reply_text(f"❌ Grafik oluşturulamadı: {e}")
