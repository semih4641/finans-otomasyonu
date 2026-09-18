"""
Arka Plan Zamanlayıcı
======================
15 dakikada bir çalışan otomatik AL sinyali taraması,
temettü kontrolleri ve paper trading değerlendirmesi.
"""

import asyncio
import logging
from dataclasses import replace
from datetime import datetime
from zoneinfo import ZoneInfo

from market_data import BIST_TIMEZONE, completed_bist_daily
from risk import calculate_position_risk, open_position_gate, daily_loss_gate, get_risk_config
from portfolio_risk import PortfolioState, portfolio_gate

logger = logging.getLogger(__name__)


def _candidate_gate(position: dict, portfolio: PortfolioState):
    """Check position count and daily loss before portfolio exposure."""
    config = get_risk_config()
    for decision in (open_position_gate(len(portfolio.positions), config),
                     daily_loss_gate(portfolio.daily_pnl, config)):
        if not decision.allowed:
            return decision
    return portfolio_gate(position, portfolio, risk_config=config)


async def _load_open_position_data(portfolio: PortfolioState) -> dict:
    """Load open exposures before scanning so correlation is order-independent."""
    from data_fetcher import fetch_crypto_data, fetch_stock_data_async

    async def fetch_position(position):
        key = position["symbol"]
        symbol = key.split(":", 1)[-1]
        market = position.get("market") or ("crypto" if key.startswith("kripto:") else "stock")
        data = await (fetch_crypto_data(symbol) if market == "crypto" else fetch_stock_data_async(symbol))
        if data is not None:
            portfolio.price_data[symbol] = data["ohlcv_df" if market == "crypto" else "history_df"]
        return key, data

    results = await asyncio.gather(*(fetch_position(pos) for pos in portfolio.positions), return_exceptions=True)
    cached = {}
    for result in results:
        if isinstance(result, Exception):
            logger.warning("⚠️ Açık pozisyon fiyat geçmişi alınamadı: %s", result)
        else:
            key, data = result
            cached[key] = data
    return cached


async def scheduled_check(app) -> None:
    """
    Her 15 dakikada bir çalışan zamanlayıcı fonksiyonu.
    İzleme listesindeki varlıkları kontrol eder ve AL sinyali/uyarı gönderir.

    Uyarı Koşulları:
    - AL sinyali tespit edilen hisse/kripto → detaylı sinyal raporu
    - Temettü tarihi 7 gün içinde → Temettü uyarısı
    """
    # Lazy imports to avoid circular dependencies
    import bot
    from bot import (
        CHAT_ID, SCAN_STOCKS, SCAN_CRYPTO, WATCHLIST_STOCKS,
        BIST_ONLY, ENABLE_PORTFOLIO_RISK,
    )
    from data_fetcher import fetch_crypto_data, fetch_stock_data_async, fetch_dividend_data_async
    from signal_engine import (
        detect_buy_signals, filter_cooldown, mark_signals_sent,
        sizing_hint, format_ml_score, _signal_category, _cooldown_active, _cooldown_mark,
    )
    from indicators import calculate_rsi, format_crypto_price
    from telegram_handlers import send_long_message
    import paper

    if not CHAT_ID:
        logger.warning("⚠️ CHAT_ID ayarlanmamış, uyarılar gönderilemez.")
        return

    # Önce bekleyen sinyalleri sonuçlandır (paper trading döngüsü)
    try:
        await paper.settle_signals()
    except Exception as e:
        logger.error(f"❌ Paper trading değerlendirmesi hatası: {e}")

    alerts = []
    buy_signal_items = []
    pending_cooldown = []  # (sembol_anahtarı, sinyaller) — gönderim sonrası cooldown'a sokulur
    div_alert_keys = []  # gönderilen temettü uyarılarının cooldown anahtarları
    logger.info("🔄 Zamanlayıcı: AL sinyali taraması başlıyor...")

    # Update portfolio state before checking signals
    if not bot._update_portfolio_state() and ENABLE_PORTFOLIO_RISK:
        logger.error("❌ Güncel portföy durumu olmadan otomatik tarama yapılamaz.")
        return
    # Refresh replaces the immutable state; read it only after the update.
    refreshed_state = bot._portfolio_state
    scan_portfolio = replace(refreshed_state, positions=list(refreshed_state.positions),
                             price_data=dict(refreshed_state.price_data))
    reserved_symbols = {pos.get("symbol") for pos in scan_portfolio.positions}
    open_data = await _load_open_position_data(scan_portfolio) if ENABLE_PORTFOLIO_RISK else {}

    # --- Kripto AL Sinyali Taraması ---
    for symbol in ([] if BIST_ONLY else SCAN_CRYPTO):
        try:
            data = open_data.get(f"kripto:{symbol}") or await fetch_crypto_data(symbol)
            if data is None:
                continue
            scan_portfolio.price_data[symbol] = data["ohlcv_df"]
            if f"kripto:{symbol}" in reserved_symbols:
                continue

            sig_result = detect_buy_signals(data["ohlcv_df"], market="crypto", symbol=symbol)
            # Cooldown'da olan sinyaller eleilir — aynı uyarı tekrar spam yapmaz
            signals = filter_cooldown(f"kripto:{symbol}", sig_result["signals"])
            if not signals:
                continue

            # Portfolio risk gate
            if ENABLE_PORTFOLIO_RISK:
                candidate = {
                    "symbol": f"kripto:{symbol}",
                    "sector": "CRYPTO",
                    "entry": data["price"],
                    "entry_num": data["price"],
                    "sl": sig_result["sl"],
                    "tp1": sig_result["tp1"],
                    "quantity": risk_result.quantity if (risk_result := calculate_position_risk(data["price"], sig_result["sl"])).is_calculable else 0,
                }
                gate = _candidate_gate(candidate, scan_portfolio)
                if not gate.allowed:
                    logger.info(f"🚫 Portfolio risk gate blocked {symbol}: {gate.message}")
                    continue
                scan_portfolio.positions.append(candidate)
                reserved_symbols.add(candidate["symbol"])

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
                "ml_available": sig_result.get("ml_available", False),
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
    scan_time = datetime.now(ZoneInfo(BIST_TIMEZONE))
    for ticker in SCAN_STOCKS:
        try:
            stock_data = open_data.get(f"hisse:{ticker}") or await fetch_stock_data_async(ticker)
            if stock_data is None:
                continue

            actual_ticker = stock_data.get("ticker", ticker)
            history = stock_data["history_df"]
            reference_price = stock_data["price"]
            signal_at = None
            if actual_ticker.upper().endswith(".IS"):
                # Only today's completed daily setup can be scheduled for the
                # next session open. Intraday and stale setups remain manual.
                history = completed_bist_daily(history, now=scan_time)
                if history.empty:
                    continue
                last_bar = history.index[-1]
                local_bar = (last_bar.tz_localize(BIST_TIMEZONE) if last_bar.tzinfo is None
                             else last_bar.tz_convert(BIST_TIMEZONE))
                if local_bar.date() != scan_time.date():
                    continue
                close_column = "Close" if "Close" in history.columns else "close"
                reference_price = float(history[close_column].iloc[-1])
                signal_at = local_bar.isoformat()
            scan_portfolio.price_data[actual_ticker] = history
            if f"hisse:{actual_ticker}" in reserved_symbols:
                continue
            sig_result = detect_buy_signals(history, market="stock", symbol=actual_ticker)
            signals = filter_cooldown(f"hisse:{actual_ticker}", sig_result["signals"])
            if not signals:
                continue

            # Portfolio risk gate
            if ENABLE_PORTFOLIO_RISK:
                risk_result = calculate_position_risk(reference_price, sig_result["sl"])
                candidate = {
                    "symbol": f"hisse:{actual_ticker}",
                    "entry": reference_price,
                    "entry_num": reference_price,
                    "sl": sig_result["sl"],
                    "tp1": sig_result["tp1"],
                    "quantity": risk_result.quantity if risk_result.is_calculable else 0,
                }
                gate = _candidate_gate(candidate, scan_portfolio)
                if not gate.allowed:
                    logger.info(f"🚫 Portfolio risk gate blocked {actual_ticker}: {gate.message}")
                    continue
                scan_portfolio.positions.append(candidate)
                reserved_symbols.add(candidate["symbol"])

            name = stock_data["info"].get("shortName", actual_ticker)
            rsi = calculate_rsi(history)
            risk_result = calculate_position_risk(
                reference_price, sig_result["sl"]
            )
            # Get sector from config
            from portfolio_risk import get_portfolio_config
            sector = get_portfolio_config().sector_mapping.get(actual_ticker, "UNKNOWN")

            buy_signal_items.append({
                "name": f"📈 {name} ({actual_ticker})",
                "key": f"hisse:{actual_ticker}",
                "market": "stock",
                "price": f"{reference_price:,.2f}",
                "entry_num": reference_price,
                "signal_at": signal_at,
                "rsi": rsi,
                "signals": signals,
                "skor": sig_result.get("score"),
                "rejim": sig_result.get("trend"),
                "regime": sig_result.get("regime"),
                "ml_score": sig_result.get("ml_score"),
                "ml_available": sig_result.get("ml_available", False),
                "rr": sig_result.get("rr"),
                "vade": sig_result["vade"],
                "sl": sig_result["sl"],
                "tp1": sig_result["tp1"],
                "tp2": sig_result["tp2"],
                "risk": risk_result.as_dict(),
                "sector": sector,
                "portfolio_risk_pct": risk_result.risk_amount / scan_portfolio.account_size * 100 if risk_result.is_calculable and scan_portfolio.account_size > 0 else 0,
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
            price_label = "Referans kapanış" if item.get("signal_at") else "Fiyat"
            message += f"  💵 {price_label}: {item['price']} | RSI: {item['rsi']}\n"
            if item.get("signal_at"):
                message += "  ⏳ Sanal giriş: sonraki seans açılışı bekleniyor.\n"
            message += (
                f"  🎯 Skor: {item.get('skor', '?')} | "
                f"ML: {format_ml_score(item)} | "
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
            await send_long_message(app.bot, CHAT_ID, message)
            logger.info(f"🚀 {len(buy_signal_items)} AL sinyali gönderildi.")
            # Yalnızca BAŞARILI gönderimden sonra cooldown başlat
            for sym_key, sigs in pending_cooldown:
                mark_signals_sent(sym_key, sigs)
            # Paper trading takibine al
            try:
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
            await send_long_message(app.bot, CHAT_ID, full_message)
            logger.info(f"📨 {len(alerts)} temettü uyarısı gönderildi.")
            _cooldown_mark(div_alert_keys)
        except Exception as e:
            logger.error(f"❌ Uyarı gönderilemedi: {e}")

    if not buy_signal_items and not alerts:
        logger.info("✅ Tarama tamamlandı, sinyal/uyarı yok.")
