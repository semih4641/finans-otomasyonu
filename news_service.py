"""Shared low-frequency KAP refresh, observations and authorized-chat alerts."""
import asyncio
from collections import defaultdict
from datetime import datetime
from html import escape
import logging

from kap_news import NewsStore, current_time, news_enabled, refresh_news
from news_learning import measure_event, price_frame
from access_control import additional_chat_ids

logger = logging.getLogger(__name__)
_job_lock = asyncio.Lock()


async def update_learning(store, now):
    state = store.state()
    last = state.get("last_learning_attempt")
    if last and (now - datetime.fromisoformat(last)).total_seconds() < 6 * 3600:
        return
    store.set_state(last_learning_attempt=now.isoformat())
    events = store.learning_events()
    if not events:
        return
    from data_fetcher import fetch_stock_data_async
    groups = defaultdict(list)
    for event in events:
        groups[event["symbol"]].append(event)
    benchmark = None
    try:
        quote = await fetch_stock_data_async("XU100.IS")
        if quote:
            benchmark = price_frame(quote["history_df"], now)
    except Exception:
        logger.warning("Haber ölçümü için BIST 100 verisi alınamadı.")
    failed, measured, unavailable = 0, 0, 0
    for symbol, symbol_events in groups.items():
        try:
            quote = await fetch_stock_data_async(symbol)
            if not quote:
                failed += 1
                continue
            prices = price_frame(quote["history_df"], now)
            observations = []
            for event in symbol_events:
                result = measure_event(event, prices, benchmark, now)
                observations.extend(result)
                if not result:
                    unavailable += 1
            store.save_observations(observations)
            measured += 1
        except Exception:
            failed += 1
            logger.warning("Haber sonrası fiyat ölçülemedi: %s", symbol)
    notes = []
    if failed:
        notes.append(f"{failed} hissenin fiyatı alınamadı; önceki ölçümler korundu.")
    if unavailable:
        notes.append(f"{unavailable} açıklama için seanslar henüz oluşmadı veya geçmiş veri yetersiz.")
    if benchmark is None:
        notes.append("BIST 100 verisi alınamadı; piyasa karşılaştırması eksik olabilir.")
    values = {"learning_warning": " ".join(notes)}
    if measured:
        values["last_learning"] = now.isoformat()
    store.set_state(**values)


async def scheduled_news_check(app):
    if not news_enabled():
        return
    from bot import CHAT_ID, SCAN_STOCKS
    from telegram_handlers import send_long_message
    if not CHAT_ID or _job_lock.locked():
        return
    async with _job_lock:
        store, now = NewsStore(), current_time()
        try:
            await refresh_news(SCAN_STOCKS, store=store, now=now)
            recipients = {str(CHAT_ID).strip()} | additional_chat_ids()
            store.prepare_deliveries(recipients)
            # Bound delivery per recipient. A failed chat cannot block others.
            # A crash between delivery and ack can duplicate that one message.
            for chat_id in sorted(recipients):
                for row in store.pending_deliveries(chat_id, limit=10):
                    published = datetime.fromisoformat(row["published_at"])
                    text = (f"📰 <b>YENİ KAP AÇIKLAMASI</b>\n"
                            f"<b>{escape(row['symbols'])}</b> · {published:%d.%m.%Y %H:%M}\n"
                            f"{escape(row['title'])}\n{escape(row['summary'][:600])}\n"
                            f"Konu: {escape(row['category'])}\n"
                            f'<a href="https://www.kap.org.tr/tr/Bildirim/{row["id"]}">Resmi açıklama ve ekleri</a>\n'
                            "Bu bildirim AL/SAT önerisi değildir. Geçmiş gözlemler: /haberogren")
                    try:
                        await send_long_message(app.bot, chat_id, text)
                    except Exception as exc:
                        logger.warning("KAP alıcı gönderimi başarısız; tekrar denenecek: %s", type(exc).__name__)
                        break
                    store.acknowledge_delivery(row["id"], chat_id)
            await update_learning(store, now)
        except Exception as exc:
            logger.error("KAP takip döngüsü tamamlanamadı: %s", type(exc).__name__)
