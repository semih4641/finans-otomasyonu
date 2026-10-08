"""KAP integration tests use synthetic disclosures and HTTP mocks only."""
import asyncio
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import pandas as pd

with patch("dotenv.load_dotenv"), patch.dict(os.environ, {}, clear=True):
    import bot
    import telegram_handlers as handlers

import kap_news as kap
import news_learning as learning
import news_service as service

NOW = datetime(2026, 10, 7, 19, tzinfo=kap.ZONE)


def payload(index=10, published="06.10.2026 19:30:00", **changes):
    return {"disclosureBasic": {"disclosureIndex": index, "publishDate": published,
            "title": "Yeni İş İlişkisi", "summary": "Sözleşme & <resmi açıklama>",
            "stockCode": "ASELS", "relatedStocks": None, "isBlocked": False, **changes},
            "disclosureDetail": {}}


def prices(start="2026-10-01", count=30, base=100.):
    close = [base + i for i in range(count)]
    return pd.DataFrame({"Open": close, "High": [v + 2 for v in close],
                         "Low": [v - 2 for v in close], "Close": [v + 1 for v in close],
                         "Volume": [1000.] * count},
                        index=pd.bdate_range(start, periods=count, tz=kap.ZONE))


class FeedTests(unittest.TestCase):
    def test_stock_and_related_symbols_are_exact_matches(self):
        rows = kap.parse_feed([payload(stockCode="ASELSX", relatedStocks="THYAO, ASELS")],
                              ["ASELS.IS", "THYAO.IS", "ASE.IS"], NOW)
        self.assertEqual(rows[0].symbols, ("ASELS.IS", "THYAO.IS"))
        self.assertEqual(rows[0].category, "Yeni iş / sözleşme")
        self.assertTrue(rows[0].published_at.endswith("+03:00"))

    def test_schema_errors_never_become_an_empty_success(self):
        for raw in ({"error": "maintenance"}, [{}], [payload(published="bad")],
                    [payload(index=True)], [payload(published="07.10.2030 19:30:00")]):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                kap.parse_feed(raw, ["ASELS.IS"], NOW)

    def test_related_revisions_are_excluded_from_learning(self):
        raw = payload()
        raw["disclosureDetail"]["relatedDisclosureIndex"] = 9
        self.assertTrue(kap.parse_feed([raw], ["ASELS.IS"], NOW)[0].amended)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = kap.NewsStore(Path(self.directory.name) / "news.sqlite3")

    def test_seed_is_silent_and_repeated_refresh_does_not_duplicate(self):
        event = kap.parse_feed([payload()], ["ASELS.IS"], NOW)
        self.store.ingest(event, NOW, seed=True)
        self.store.ingest(event, NOW + timedelta(hours=1))
        self.assertEqual(len(self.store.latest()), 1)
        self.assertFalse(self.store.latest(pending=True))
        self.store.ingest(kap.parse_feed([payload(11)], ["ASELS.IS"], NOW), NOW)
        self.assertEqual(len(self.store.latest(pending=True)), 1)
        self.store.acknowledge([11])
        self.assertFalse(self.store.latest(pending=True))

    def test_blocked_disclosure_and_observations_are_removed(self):
        self.store.ingest(kap.parse_feed([payload()], ["ASELS.IS"], NOW), NOW)
        self.store.save_observations([{"news_id": 10, "symbol": "ASELS.IS", "horizon": 1,
            "entry_day": "2026-10-07", "exit_day": "2026-10-07", "raw_return": 1.,
            "benchmark_return": None, "measured_at": NOW.isoformat()}])
        self.store.ingest(kap.parse_feed([payload(isBlocked=True)], [], NOW), NOW)
        self.assertFalse(self.store.latest())
        self.assertFalse(self.store.observations())

    def test_changed_text_invalidates_historical_interpretation(self):
        self.store.ingest(kap.parse_feed([payload()], ["ASELS.IS"], NOW), NOW)
        self.store.ingest(kap.parse_feed([payload(summary="Changed")], ["ASELS.IS"], NOW), NOW)
        self.assertTrue(self.store.latest()[0]["amended"])
        self.assertFalse(self.store.learning_events())

    def test_render_escapes_external_text_and_links_only_to_official_id(self):
        self.store.ingest(kap.parse_feed([payload()], ["ASELS.IS"], NOW), NOW)
        text = kap.render_news(self.store)
        self.assertIn("&amp; &lt;resmi açıklama&gt;", text)
        self.assertIn("https://www.kap.org.tr/tr/Bildirim/10", text)

    def test_new_revision_flag_excludes_unchanged_text(self):
        self.store.ingest(kap.parse_feed([payload()], ["ASELS.IS"], NOW), NOW)
        self.store.ingest(kap.parse_feed([payload(isChanged=True)], ["ASELS.IS"], NOW), NOW)
        self.assertFalse(self.store.learning_events())

    def test_transient_missing_benchmark_preserves_only_matching_measurement(self):
        self.store.ingest(kap.parse_feed([payload()], ["ASELS.IS"], NOW), NOW)
        row = {"news_id": 10, "symbol": "ASELS.IS", "horizon": 1,
               "entry_day": "2026-10-07", "exit_day": "2026-10-07", "raw_return": 1.,
               "benchmark_return": 0.5, "measured_at": NOW.isoformat()}
        self.store.save_observations([row])
        self.store.save_observations([{**row, "benchmark_return": None}])
        self.assertEqual(self.store.observations()[0]["benchmark_return"], 0.5)
        self.store.save_observations([{**row, "raw_return": 2., "benchmark_return": None}])
        self.assertIsNone(self.store.observations()[0]["benchmark_return"])


class LearningTests(unittest.TestCase):
    def test_after_close_news_uses_next_session_and_waits_for_full_horizon(self):
        data = learning.price_frame(prices(), NOW)
        event = {"id": 10, "symbol": "ASELS.IS", "published_at": "2026-10-06T19:30:00+03:00"}
        rows = learning.measure_event(event, data, data, NOW)
        self.assertEqual([row["horizon"] for row in rows], [1])
        self.assertEqual(rows[0]["entry_day"], "2026-10-07")
        self.assertAlmostEqual(rows[0]["raw_return"], (105 / 104 - 1) * 100)
        self.assertEqual(rows[0]["raw_return"], rows[0]["benchmark_return"])

    def test_weekend_publication_waits_for_next_available_session(self):
        data = learning.price_frame(prices(), NOW)
        event = {"id": 1, "symbol": "ASELS.IS", "published_at": "2026-10-03T12:00:00+03:00"}
        self.assertEqual(learning.measure_event(event, data, None, NOW)[0]["entry_day"], "2026-10-05")

    def test_future_price_changes_cannot_change_completed_measurements(self):
        source = prices()
        changed = source.copy()
        changed.loc[changed.index.date > NOW.date(), ["Open", "High", "Low", "Close"]] *= 100
        event = {"id": 1, "symbol": "ASELS.IS", "published_at": "2026-10-02T12:00:00+03:00"}
        self.assertEqual(learning.measure_event(event, learning.price_frame(source, NOW), None, NOW),
                         learning.measure_event(event, learning.price_frame(changed, NOW), None, NOW))

    def test_truncated_history_does_not_invent_a_late_entry(self):
        data = learning.price_frame(prices(), NOW)
        event = {"id": 1, "symbol": "ASELS.IS", "published_at": "2026-09-01T12:00:00+03:00"}
        self.assertFalse(learning.measure_event(event, data, None, NOW))

    def test_overlap_is_removed_across_categories_but_not_other_stocks(self):
        base = {"news_id": 1, "symbol": "ASELS.IS", "horizon": 5, "entry_day": "2026-10-01",
                "exit_day": "2026-10-07", "category": "A"}
        rows = [base, {**base, "news_id": 2, "category": "B"},
                {**base, "news_id": 3, "symbol": "THYAO.IS"},
                {**base, "news_id": 4, "entry_day": "2026-10-08", "exit_day": "2026-10-14"}]
        groups = learning.grouped_observations(rows)
        self.assertEqual(len(groups[("A", 5)]), 3)
        self.assertNotIn(("B", 5), groups)

    def test_small_samples_have_no_reported_success_probability(self):
        row = {"news_id": 1, "symbol": "ASELS.IS", "horizon": 5, "entry_day": "2026-10-01",
               "exit_day": "2026-10-07", "category": "A", "raw_return": 90, "benchmark_return": 1}
        store = SimpleNamespace(observations=lambda symbol: [row], state=lambda: {})
        text = learning.render_learning(store)
        self.assertIn("1/10", text)
        self.assertNotIn("Pozitif sonuç:", text)

    def test_benchmark_requires_exact_same_entry_and_exit_dates(self):
        data = learning.price_frame(prices(), NOW)
        event = {"id": 1, "symbol": "ASELS.IS", "published_at": "2026-10-06T19:00:00+03:00"}
        benchmark = data.iloc[:-1]
        self.assertIsNone(learning.measure_event(event, data, benchmark, NOW)[0]["benchmark_return"])


class AsyncNewsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = kap.NewsStore(Path(self.directory.name) / "news.sqlite3")
        self.env = patch.dict(os.environ, {}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)
        lock = patch.object(kap, "_refresh_lock", asyncio.Lock())
        lock.start()
        self.addCleanup(lock.stop)
        recipients = patch.object(service, "additional_chat_ids", return_value=set())
        recipients.start()
        self.addCleanup(recipients.stop)

    async def test_official_request_uses_observed_date_format_and_no_redirects(self):
        def handle(request):
            self.assertEqual(request.method, "POST")
            self.assertEqual(json.loads(request.content)["fromDate"], "07.10.2026")
            return httpx.Response(200, json=[payload()])
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            response = await kap.fetch_disclosures(NOW.date(), NOW.date(), client=client)
        self.assertEqual(response, [payload()])

    async def test_failed_feed_preserves_cache_and_obeys_shared_retry_cooldown(self):
        with patch.object(kap, "fetch_disclosures", AsyncMock(return_value=[payload()])) as fetch:
            await kap.refresh_news(["ASELS.IS"], store=self.store, now=NOW)
            await kap.refresh_news(["ASELS.IS"], store=self.store, now=NOW + timedelta(minutes=1))
        self.assertEqual(fetch.await_count, 1)
        with patch.object(kap, "fetch_disclosures", AsyncMock(side_effect=ValueError("bad schema"))):
            state = await kap.refresh_news(["ASELS.IS"], store=self.store, now=NOW + timedelta(hours=1))
        self.assertTrue(state["last_error"])
        self.assertEqual(state["last_success"], NOW.isoformat())
        self.assertEqual(len(self.store.latest()), 1)

    async def test_simultaneous_commands_share_single_feed_request(self):
        with patch.object(kap, "fetch_disclosures", AsyncMock(return_value=[payload()])) as fetch:
            await asyncio.gather(*(kap.refresh_news(["ASELS.IS"], store=self.store, now=NOW) for _ in range(5)))
        self.assertEqual(fetch.await_count, 1)

    async def test_telegram_failure_keeps_notification_for_next_run(self):
        self.store.ingest(kap.parse_feed([payload()], ["ASELS.IS"], NOW), NOW)
        app = SimpleNamespace(bot=SimpleNamespace())
        with patch.object(bot, "CHAT_ID", "test"), patch.object(service, "NewsStore", return_value=self.store), \
             patch.object(service, "refresh_news", AsyncMock()), patch.object(service, "update_learning", AsyncMock()), \
             patch.object(service, "_job_lock", asyncio.Lock()), \
             patch.object(handlers, "send_long_message", AsyncMock(side_effect=RuntimeError("offline"))):
            await service.scheduled_news_check(app)
        self.assertEqual(len(self.store.pending_deliveries("test")), 1)
        with patch.object(bot, "CHAT_ID", "test"), patch.object(service, "NewsStore", return_value=self.store), \
             patch.object(service, "refresh_news", AsyncMock()), patch.object(service, "update_learning", AsyncMock()), \
             patch.object(service, "_job_lock", asyncio.Lock()), patch.object(handlers, "send_long_message", AsyncMock()) as send:
            await service.scheduled_news_check(app)
            await service.scheduled_news_check(app)
        self.assertEqual(send.await_count, 1)
        self.assertFalse(self.store.pending_deliveries("test"))

    async def test_multiple_recipients_retry_independently_and_deduplicate_owner(self):
        self.store.ingest(kap.parse_feed([payload()], ["ASELS.IS"], NOW), NOW)
        async def deliver(target, chat, text):
            if chat == "friend":
                raise RuntimeError("blocked")
        with patch.object(bot, "CHAT_ID", "owner"), patch.object(service, "NewsStore", return_value=self.store), \
             patch.object(service, "additional_chat_ids", return_value={"owner", "friend"}), \
             patch.object(service, "refresh_news", AsyncMock()), patch.object(service, "update_learning", AsyncMock()), \
             patch.object(service, "_job_lock", asyncio.Lock()), \
             patch.object(handlers, "send_long_message", AsyncMock(side_effect=deliver)) as send:
            await service.scheduled_news_check(SimpleNamespace(bot=object()))
            self.assertEqual([call.args[1] for call in send.await_args_list], ["friend", "owner"])
            send.reset_mock()
            send.side_effect = None
            await service.scheduled_news_check(SimpleNamespace(bot=object()))
            self.assertEqual([call.args[1] for call in send.await_args_list], ["friend"])
        self.assertEqual(self.store.pending_delivery_count(), 0)

    async def test_migration_seed_and_removed_recipient_do_not_replay_archive(self):
        self.store.ingest(kap.parse_feed([payload()], ["ASELS.IS"], NOW), NOW, seed=True)
        self.store.prepare_deliveries({"owner", "friend"})
        self.assertEqual(self.store.pending_delivery_count(), 0)
        self.store.ingest(kap.parse_feed([payload(11)], ["ASELS.IS"], NOW), NOW)
        self.store.prepare_deliveries({"owner", "friend"})
        self.assertEqual(self.store.pending_delivery_count(), 2)
        self.store.acknowledge_delivery(11, "owner")
        self.store.prepare_deliveries({"owner"})
        self.store.prepare_deliveries({"owner", "friend", "new"})
        self.assertEqual(self.store.pending_delivery_count(), 0)

    async def test_unauthorized_chat_cannot_fetch_news_or_status(self):
        update = SimpleNamespace(effective_chat=SimpleNamespace(id="unknown"))
        with patch.object(bot, "CHAT_ID", "owner"), patch.object(handlers, "additional_chat_ids", return_value=set()), \
             patch.object(kap, "refresh_news", AsyncMock()) as fetch:
            for command in (handlers.haber_command, handlers.haberogren_command, handlers.kapdurum_command):
                await command(update, SimpleNamespace(args=[]))
        fetch.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
