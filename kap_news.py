"""Low-frequency public KAP feed, durable news archive and notification outbox.

This uses the public website's read-only listing, not the licensed KAP REST
service. No full disclosure/PDF scraping or large historical crawling is done.
"""
import asyncio
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from html import escape
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
from zoneinfo import ZoneInfo

import httpx

ZONE = ZoneInfo("Europe/Istanbul")
FEED_URL = "https://www.kap.org.tr/tr/api/disclosure/list/main"
POLL_SECONDS = 1800
MAX_FEED_BYTES = 8_000_000
logger = logging.getLogger(__name__)
_refresh_lock = asyncio.Lock()


def current_time():
    return datetime.now(ZONE)


def news_enabled():
    return os.getenv("KAP_ENABLED", "true").lower() in {"1", "true", "yes"}


def category_for(title, summary):
    text = (title + " " + summary).casefold().replace("i̇", "i")
    for category, words in (
        ("Devre kesici / tedbir", ("devre kesici", "işlem sırası", "vbts", "volatilite bazlı")),
        ("Finansal sonuçlar", ("finansal rapor", "finansal tablo", "bilanço", "faaliyet raporu")),
        ("Kâr payı", ("kar payı", "kâr payı", "temettü")),
        ("Pay geri alımı", ("geri alım", "geri alınan")),
        ("Yeni iş / sözleşme", ("yeni iş", "sözleşme", "ihale")),
        ("Sermaye işlemi", ("sermaye artır", "sermaye azalt", "bedelli", "bedelsiz")),
    ):
        if any(word in text for word in words):
            return category
    return "Diğer açıklamalar"


def flag(value):
    return value is True or str(value).lower() in {"true", "y", "yes", "1"}


@dataclass(frozen=True)
class Disclosure:
    index: int
    symbols: tuple[str, ...]
    published_at: str
    title: str
    summary: str
    category: str
    amended: bool = False
    blocked: bool = False


def parse_feed(payload, tracked, now):
    if not isinstance(payload, list) or len(payload) > 20000:
        raise ValueError("KAP bildirim listesi beklenen biçimde değil.")
    watched = {symbol.upper().removesuffix(".IS") for symbol in tracked}
    results = []
    for item in payload:
        if not isinstance(item, dict) or not isinstance(item.get("disclosureBasic"), dict):
            raise ValueError("KAP bildirim şeması değişmiş olabilir.")
        raw = item["disclosureBasic"]
        index = raw.get("disclosureIndex")
        if isinstance(index, bool) or not str(index).isdigit() or int(index) <= 0:
            raise ValueError("KAP bildirim numarası geçersiz.")
        if flag(raw.get("isBlocked")):
            results.append(Disclosure(int(index), (), "", "", "", "", blocked=True))
            continue
        codes = set(re.findall(r"[A-Z0-9]+", str(raw.get("stockCode") or "").upper()))
        codes.update(re.findall(r"[A-Z0-9]+", str(raw.get("relatedStocks") or "").upper()))
        symbols = tuple(sorted(code + ".IS" for code in watched & codes))
        if not symbols:
            continue
        # Feed times are local wall times, not UTC. Never infer a missing date.
        published = datetime.strptime(str(raw.get("publishDate")), "%d.%m.%Y %H:%M:%S").replace(tzinfo=ZONE)
        if published > now + timedelta(minutes=5):
            raise ValueError("KAP bildirimi gelecekte görünüyor; yerel saat/kaynak kontrol edilmeli.")
        title = " ".join(str(raw.get("title") or "").split())[:250]
        summary = " ".join(str(raw.get("summary") or "").split())[:1500]
        if not title:
            raise ValueError("KAP bildirim başlığı eksik.")
        detail = item.get("disclosureDetail") or {}
        if not isinstance(detail, dict):
            raise ValueError("KAP bildirim ayrıntısı beklenen biçimde değil.")
        amended = (flag(raw.get("isChanged")) or bool(detail.get("relatedDisclosureIndex"))
                   or bool(raw.get("relatedDisclosureOid")) or "düzelt" in title.casefold())
        results.append(Disclosure(int(index), symbols, published.isoformat(), title, summary,
                                  category_for(title, summary), amended=amended))
    return results


async def fetch_disclosures(start, end, *, client=None):
    body = {"fromDate": start.strftime("%d.%m.%Y"), "toDate": end.strftime("%d.%m.%Y"),
            "disclosureTypes": None, "memberTypes": ["IGS"], "mkkMemberOid": None}

    async def request(http):
        async with http.stream("POST", FEED_URL, json=body) as response:
            response.raise_for_status()
            parts, length = [], 0
            async for part in response.aiter_bytes():
                length += len(part)
                if length > MAX_FEED_BYTES:
                    raise ValueError("KAP yanıtı güvenli boyut sınırını aştı.")
                parts.append(part)
        return json.loads(b"".join(parts))

    if client is not None:
        return await request(client)
    async with httpx.AsyncClient(timeout=25, follow_redirects=False,
                                 headers={"Accept": "application/json", "Accept-Language": "tr"}) as http:
        return await request(http)


class NewsStore:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else Path(os.getenv("BOT_DATA_DIR", "backtest_out")) / "kap_news.sqlite3"

    @contextmanager
    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(self.path, timeout=20)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS news (
                    id INTEGER PRIMARY KEY, published_at TEXT NOT NULL, first_seen TEXT NOT NULL,
                    title TEXT NOT NULL, summary TEXT NOT NULL, category TEXT NOT NULL,
                    amended INTEGER NOT NULL, notified INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS news_symbols (
                    news_id INTEGER REFERENCES news(id) ON DELETE CASCADE, symbol TEXT NOT NULL,
                    PRIMARY KEY(news_id, symbol));
                CREATE TABLE IF NOT EXISTS observations (
                    news_id INTEGER REFERENCES news(id) ON DELETE CASCADE, symbol TEXT NOT NULL,
                    horizon INTEGER NOT NULL, entry_day TEXT NOT NULL, exit_day TEXT NOT NULL,
                    raw_return REAL NOT NULL, benchmark_return REAL, measured_at TEXT NOT NULL,
                    PRIMARY KEY(news_id, symbol, horizon));
                CREATE TABLE IF NOT EXISTS news_deliveries (
                    news_id INTEGER REFERENCES news(id) ON DELETE CASCADE,
                    chat_id TEXT NOT NULL, delivered INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(news_id, chat_id));
            """)
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def state(self):
        with self.connect() as db:
            return dict(db.execute("SELECT key,value FROM state").fetchall())

    def set_state(self, **values):
        with self.connect() as db:
            db.executemany("INSERT OR REPLACE INTO state VALUES (?,?)", [(k, str(v)) for k, v in values.items()])

    def ingest(self, records, now, *, seed=False):
        added = 0
        with self.connect() as db:
            for item in records:
                if item.blocked:
                    db.execute("DELETE FROM news WHERE id=?", (item.index,))
                    continue
                existing = db.execute("SELECT * FROM news WHERE id=?", (item.index,)).fetchone()
                if existing:
                    if item.amended or (existing["title"], existing["summary"], existing["published_at"]) != (item.title, item.summary, item.published_at):
                        # A changed historical text cannot be treated as known at original publication.
                        db.execute("UPDATE news SET title=?,summary=?,amended=1 WHERE id=?", (item.title, item.summary, item.index))
                        db.execute("DELETE FROM observations WHERE news_id=?", (item.index,))
                    continue
                db.execute("INSERT INTO news VALUES (?,?,?,?,?,?,?,?)", (
                    item.index, item.published_at, now.isoformat(), item.title, item.summary,
                    item.category, int(item.amended), int(seed)))
                db.executemany("INSERT INTO news_symbols VALUES (?,?)", [(item.index, s) for s in item.symbols])
                added += 1
            db.executemany("INSERT OR REPLACE INTO state VALUES (?,?)", [
                ("last_success", now.isoformat()), ("last_error", ""), ("initialised", "1")])
        return added

    def latest(self, symbol=None, limit=6, *, pending=False):
        with self.connect() as db:
            where, args = [], []
            if symbol:
                where.append("n.id IN (SELECT news_id FROM news_symbols WHERE symbol=?)")
                args.append(symbol)
            if pending:
                where.append("n.notified=0")
            query = "SELECT n.*, GROUP_CONCAT(s.symbol) AS symbols FROM news n JOIN news_symbols s ON n.id=s.news_id"
            if where:
                query += " WHERE " + " AND ".join(where)
            query += (" GROUP BY n.id ORDER BY n.published_at ASC,n.id ASC LIMIT ?" if pending
                      else " GROUP BY n.id ORDER BY n.published_at DESC,n.id DESC LIMIT ?")
            return [dict(row) for row in db.execute(query, [*args, limit])]

    def acknowledge(self, ids):
        with self.connect() as db:
            db.executemany("UPDATE news SET notified=1 WHERE id=?", [(i,) for i in ids])

    def prepare_deliveries(self, recipients):
        """Snapshot current recipients for new news; never replay the old archive."""
        recipients = sorted({str(chat) for chat in recipients})
        if not recipients:
            return
        with self.connect() as db:
            placeholders = ",".join("?" for _ in recipients)
            db.execute(f"DELETE FROM news_deliveries WHERE chat_id NOT IN ({placeholders})", recipients)
            db.executemany("""INSERT OR IGNORE INTO news_deliveries(news_id,chat_id)
                SELECT id,? FROM news WHERE notified=0""", [(chat,) for chat in recipients])
            db.execute("UPDATE news SET notified=1 WHERE notified=0")

    def pending_deliveries(self, chat_id, limit=10):
        with self.connect() as db:
            return [dict(row) for row in db.execute("""
                SELECT n.*,GROUP_CONCAT(s.symbol) AS symbols FROM news n
                JOIN news_symbols s ON s.news_id=n.id
                JOIN news_deliveries d ON d.news_id=n.id
                WHERE d.chat_id=? AND d.delivered=0
                GROUP BY n.id ORDER BY n.published_at,n.id LIMIT ?
            """, (str(chat_id), limit))]

    def acknowledge_delivery(self, news_id, chat_id):
        with self.connect() as db:
            db.execute("UPDATE news_deliveries SET delivered=1 WHERE news_id=? AND chat_id=?",
                       (news_id, str(chat_id)))

    def pending_delivery_count(self):
        with self.connect() as db:
            return db.execute("SELECT COUNT(*) FROM news_deliveries WHERE delivered=0").fetchone()[0]

    def learning_events(self):
        with self.connect() as db:
            return [dict(row) for row in db.execute("""
                SELECT n.*,s.symbol FROM news n JOIN news_symbols s ON n.id=s.news_id
                WHERE n.amended=0 ORDER BY n.published_at,n.id
            """)]

    def save_observations(self, rows):
        with self.connect() as db:
            db.executemany("""INSERT INTO observations VALUES
                (:news_id,:symbol,:horizon,:entry_day,:exit_day,:raw_return,:benchmark_return,:measured_at)
                ON CONFLICT(news_id,symbol,horizon) DO UPDATE SET
                    benchmark_return=CASE
                        WHEN excluded.benchmark_return IS NULL
                          AND excluded.entry_day=observations.entry_day
                          AND excluded.exit_day=observations.exit_day
                          AND excluded.raw_return=observations.raw_return
                        THEN observations.benchmark_return ELSE excluded.benchmark_return END,
                    entry_day=excluded.entry_day, exit_day=excluded.exit_day,
                    raw_return=excluded.raw_return, measured_at=excluded.measured_at
            """, rows)

    def observations(self, symbol=None):
        with self.connect() as db:
            query = """SELECT o.*,n.category,n.published_at FROM observations o
                       JOIN news n ON n.id=o.news_id WHERE n.amended=0"""
            args = []
            if symbol:
                query += " AND o.symbol=?"
                args.append(symbol)
            return [dict(row) for row in db.execute(query + " ORDER BY o.entry_day,o.news_id", args)]


async def refresh_news(tracked, *, store=None, now=None):
    store, now = store or NewsStore(), now or current_time()
    if not news_enabled():
        return store.state()
    async with _refresh_lock:
        state = store.state()
        last = state.get("last_attempt")
        if last and (now - datetime.fromisoformat(last)).total_seconds() < POLL_SECONDS:
            return state
        store.set_state(last_attempt=now.isoformat())
        seed = state.get("initialised") != "1"
        last_success = datetime.fromisoformat(state["last_success"]) if state.get("last_success") else None
        start = max(now.date() - timedelta(days=6), last_success.date() - timedelta(days=1)) if last_success else now.date() - timedelta(days=6)
        try:
            payload = await fetch_disclosures(start, now.date())
            records = parse_feed(payload, tracked, now)
            store.ingest(records, now, seed=seed)
            if seed:
                store.set_state(coverage_start=start.isoformat())
            if last_success and now.date() - last_success.date() > timedelta(days=6):
                store.set_state(coverage_gap="Son kesintide 7 günden eski bildirimler tamamlanamadı.")
        except (httpx.HTTPError, ValueError, TypeError) as exc:
            # Never turn unavailable data into an empty successful feed.
            store.set_state(last_error=f"KAP akışına erişilemedi ({type(exc).__name__}); önbellek gösteriliyor.")
            logger.warning("KAP akışı alınamadı: %s", type(exc).__name__)
        return store.state()


def render_news(store, symbol=None):
    state = store.state()
    lines = ["📰 <b>KAP açıklamaları</b>", "Kaynak: KAP herkese açık bildirim listesi · başlık ve resmi özet"]
    if not news_enabled():
        lines.append("Takip kapalı; yalnız kayıtlı arşiv gösteriliyor.")
    if state.get("last_success"):
        when = datetime.fromisoformat(state["last_success"])
        lines.append(f"Son başarılı kontrol: {when:%d.%m.%Y %H:%M} (İstanbul)")
    if state.get("last_error"):
        lines.append("⚠️ " + escape(state["last_error"]))
    if state.get("coverage_gap"):
        lines.append("⚠️ " + escape(state["coverage_gap"]))
    rows = store.latest(symbol)
    for row in rows:
        published = datetime.fromisoformat(row["published_at"])
        lines.extend(["", f"<b>{escape(row['symbols'])}</b> · {published:%d.%m.%Y %H:%M}",
                      f"{escape(row['title'])} · {escape(row['category'])}",
                      escape(row["summary"][:220]),
                      f'<a href="https://www.kap.org.tr/tr/Bildirim/{row["id"]}">KAP açıklamasını aç</a>'])
        if row["amended"]:
            lines.append("Düzeltme/ilişkili açıklama; öğrenme istatistiğine alınmadı.")
    if not rows:
        lines.append("Kayıtlı arşivde eşleşen açıklama yok; bu, hiç haber olmadığı anlamına gelmez.")
    lines.extend(["", "Tam metin ve ekler otomatik yorumlanmaz. Başlık kategorisi olumlu/olumsuz yatırım kararı değildir."])
    return "\n".join(lines)
