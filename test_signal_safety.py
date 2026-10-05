"""Offline safety regressions: no credentials, model files or service calls."""

import os
import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from zoneinfo import ZoneInfo

import pandas as pd

with patch("dotenv.load_dotenv"), patch.dict(os.environ, {}, clear=True):
    import bot
    import scheduler
    import signal_engine
    import analysis_report

from portfolio_risk import PortfolioState
from signal_safety import SafetyConfig, assess_signal_safety, validated_ohlcv


def bars(count=80):
    frame = pd.DataFrame({"open": [100.] * count, "high": [101.] * count,
                          "low": [99.] * count, "close": [100.] * count,
                          "volume": [1000.] * count},
                         index=pd.bdate_range(end="2026-10-05", periods=count,
                                              tz="Europe/Istanbul"))
    frame.attrs["completed_only"] = True
    return frame


def anomalous_bars():
    frame = bars()
    frame.iloc[-1] = [100, 108, 99, 107, 6000]
    return frame


class SignalSafetyTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {}, clear=True)
        env.start()
        self.addCleanup(env.stop)

    def test_numeric_strings_normalized_without_mutating_source(self):
        source = bars().astype(str).rename(columns=str.title)
        original = source.copy(deep=True)
        clean = validated_ohlcv(source)
        pd.testing.assert_frame_equal(source, original)
        self.assertTrue(clean.attrs["completed_only"])
        self.assertEqual(clean["close"].iloc[-1], 100.)
        self.assertFalse(assess_signal_safety(clean)["blocked"])

    def test_invalid_ohlcv_is_rejected_before_model_or_indicators(self):
        corruptions = [("high", 98), ("low", 102), ("open", 200),
                       ("close", float("nan")), ("volume", -1),
                       ("volume", float("inf")), ("high", 0)]
        for column, value in corruptions:
            with self.subTest(column=column, value=value):
                data = bars()
                data.loc[data.index[-1], column] = value
                with patch.object(signal_engine, "_get_ml_ensemble") as model:
                    result = signal_engine.detect_buy_signals(data, symbol="TEST.IS")
                self.assertFalse(result["signals"])
                self.assertEqual(result["sl"], 0)
                self.assertIn("INVALID_DATA", result["safety"]["codes"])
                model.assert_not_called()

    def test_old_corrupt_bar_cannot_poison_full_history_indicators(self):
        data = bars(250)
        data.iloc[0, data.columns.get_loc("high")] = float("inf")
        result = signal_engine.detect_buy_signals(data, use_ml=False)
        self.assertIn("INVALID_DATA", result["safety"]["codes"])

    def test_missing_columns_and_bad_timestamps_fail_closed(self):
        duplicate = bars()
        duplicate.index = duplicate.index[:-1].append(duplicate.index[-2:-1])
        no_dates = bars().reset_index(drop=True)
        for frame in (bars().drop(columns="volume"), bars().iloc[::-1], duplicate, no_dates):
            with self.subTest(columns=list(frame.columns), index_type=type(frame.index)):
                result = signal_engine.detect_buy_signals(frame, use_ml=False)
                self.assertIn("INVALID_DATA", result["safety"]["codes"])

    def test_joint_price_volume_spike_blocks_and_explains(self):
        result = signal_engine.detect_buy_signals(anomalous_bars(), symbol="TEST.IS", use_ml=False)
        self.assertFalse(result["signals"])
        self.assertEqual(result["sl"], 0)
        self.assertIn("PRICE_VOLUME_ANOMALY", result["safety"]["codes"])
        self.assertEqual(result["safety"]["metrics"]["volume_ratio"], 6)
        self.assertIn("sinyal durduruldu", result["decision_reason"])

    def test_large_intraday_range_with_flat_close_also_blocks(self):
        frame = bars()
        frame.iloc[-1] = [100, 106, 94, 100, 6000]
        result = signal_engine.detect_buy_signals(frame, use_ml=False)
        self.assertIn("PRICE_VOLUME_ANOMALY", result["safety"]["codes"])

    def test_volume_alone_preserves_technical_signal_and_gives_warning(self):
        frame = bars()
        frame.iloc[-1, frame.columns.get_loc("volume")] = 6000
        with patch.object(signal_engine, "SIGNAL_SCORES_STOCK", {"🔊 Hacim Patlaması": 2}):
            result = signal_engine.detect_buy_signals(frame, use_ml=False)
        self.assertTrue(result["signals"])
        self.assertFalse(result["safety"]["blocked"])
        self.assertTrue(result["safety"]["warnings"])

    def test_daily_stock_anomaly_threshold_not_applied_to_hourly_crypto(self):
        result = signal_engine.detect_buy_signals(anomalous_bars(), market="crypto", use_ml=False)
        self.assertFalse(result["safety"]["blocked"])

    def test_zero_volume_blocks_new_signals(self):
        frame = bars()
        frame.iloc[-1, frame.columns.get_loc("volume")] = 0
        result = signal_engine.detect_buy_signals(frame, use_ml=False)
        self.assertIn("NO_VOLUME", result["safety"]["codes"])

    def test_baseline_excludes_current_bar_and_requires_past_volume(self):
        data = anomalous_bars()
        self.assertEqual(assess_signal_safety(data)["metrics"]["volume_ratio"], 6)
        data.iloc[:-1, data.columns.get_loc("volume")] = 0
        self.assertIn("INSUFFICIENT_VOLUME", assess_signal_safety(data)["codes"])

    def test_unfinished_extreme_bar_does_not_trigger_guard(self):
        data = anomalous_bars()
        data.attrs["completed_only"] = False
        data.iloc[-1, data.columns.get_loc("volume")] = float("inf")
        result = signal_engine.detect_buy_signals(data, use_ml=False)
        self.assertFalse(result["safety"]["blocked"])
        self.assertEqual(result["safety"]["metrics"]["volume_ratio"], 1)

    def test_live_stale_data_rejected_but_historical_evaluation_is_reproducible(self):
        frame = bars()
        for now in ("2026-10-13T19:00:00+03:00", "2030-01-01T19:00:00+03:00"):
            result = signal_engine.detect_buy_signals(frame, symbol="TEST.IS", use_ml=False, now=now)
            self.assertIn("STALE_DATA", result["safety"]["codes"])
        historic = signal_engine.detect_buy_signals(frame, symbol="TEST.IS", use_ml=False)
        self.assertFalse(historic["safety"]["blocked"])

    def test_normal_weekend_not_mislabeled_stale(self):
        self.assertFalse(assess_signal_safety(bars(), now="2026-10-11T12:00:00+03:00")["blocked"])

    def test_crypto_clock_is_utc_for_naive_exchange_timestamps(self):
        data = bars()
        data.index = pd.date_range(end="2026-10-05T10:00", periods=len(data), freq="h")
        self.assertFalse(assess_signal_safety(data, "crypto", now="2026-10-05T15:00+03:00")["blocked"])
        self.assertIn("STALE_DATA", assess_signal_safety(data, "crypto", now="2026-10-05T17:00+03:00")["codes"])
        self.assertIn("FUTURE_DATA", assess_signal_safety(data, "crypto", now="2026-10-05T09:00Z")["codes"])

    def test_invalid_settings_fail_closed(self):
        for raw in ("nan", "inf", "oops", "0", "-1"):
            with self.subTest(raw=raw), patch.dict(os.environ, {"ANOMALY_VOLUME_RATIO": raw}):
                result = signal_engine.detect_buy_signals(bars(), use_ml=False)
                self.assertIn("INVALID_SAFETY_CONFIG", result["safety"]["codes"])

    def test_config_changes_threshold_without_disabling_data_validation(self):
        with patch.dict(os.environ, {"ANOMALY_VOLUME_RATIO": "7"}):
            result = signal_engine.detect_buy_signals(anomalous_bars(), use_ml=False)
        self.assertFalse(result["safety"]["blocked"])

    def test_live_and_historical_completed_decision_match(self):
        data = anomalous_bars()
        historical = signal_engine.detect_buy_signals(data, symbol="TEST.IS", use_ml=False)
        live = signal_engine.detect_buy_signals(data, symbol="TEST.IS", use_ml=False,
                                               now="2026-10-05T19:00+03:00")
        self.assertEqual(live, historical)

    def test_future_prices_cannot_change_past_backtest_decisions(self):
        import backtest
        data = bars(90)
        data.iloc[40, data.columns.get_loc("volume")] = 2000
        data.iloc[60, data.columns.get_loc("volume")] = 2000
        changed = data.copy()
        changed.iloc[70:] = [1000, 1001, 999, 1000, 500000]
        with patch.object(signal_engine, "SIGNAL_SCORES_STOCK", {"🔊 Hacim Patlaması": 2}):
            original_events = backtest.walk_forward(data, cooldown_bars=1)
            changed_events = backtest.walk_forward(changed, cooldown_bars=1)
        original_past = [event for event in original_events if event["decision_idx"] < 69]
        changed_past = [event for event in changed_events if event["decision_idx"] < 69]
        self.assertTrue(original_past)
        self.assertEqual(original_past, changed_past)

    def test_reference_price_identifies_same_completed_candle_as_signal(self):
        data = bars()
        data.attrs["completed_only"] = False
        data.iloc[-1] = [900, 1000, 800, 950, 1000]
        result = signal_engine.detect_buy_signals(data, use_ml=False)
        self.assertEqual(result["reference_price"], 100)
        self.assertEqual(result["candle_time"], data.index[-2].isoformat())

    def test_report_shows_block_reason_and_no_targets(self):
        payload = {"ticker": "TEST.IS", "info": {}, "history_df": anomalous_bars()}
        report = analysis_report.build_stock_report(payload, "2026-10-05T19:00+03:00")
        self.assertIn("Koruma kontrolü sinyali durdurdu", report)
        self.assertIn("Aşırı fiyat-hacim", report)
        self.assertNotIn("Senaryo hedefleri:", report)
        self.assertLess(len(report), 4000)

    def test_stale_report_avoids_presenting_old_indicators_as_current(self):
        payload = {"ticker": "TEST.IS", "info": {}, "history_df": bars()}
        report = analysis_report.build_stock_report(payload, "2026-11-05T19:00+03:00")
        self.assertIn("31 takvim günü eski", report)
        self.assertNotIn("RSI (14)", report)


class ScheduledProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_anomalous_candidate_is_not_recorded_and_warning_deduplicated(self):
        frame = anomalous_bars()
        quote = {"ticker": "TEST.IS", "history_df": frame, "price": 107, "info": {}}
        app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
        sent = set()
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(bot, "CHAT_ID", "offline"), patch.object(bot, "BIST_ONLY", True), \
             patch.object(bot, "SCAN_STOCKS", ["TEST.IS"]), patch.object(bot, "WATCHLIST_STOCKS", []), \
             patch.object(bot, "ENABLE_PORTFOLIO_RISK", False), \
             patch.object(bot, "_update_portfolio_state", return_value=True), \
             patch.object(bot, "_portfolio_state", PortfolioState([], 10000, 10000, 0, 10000)), \
             patch.object(scheduler, "datetime", wraps=datetime) as clock, \
             patch("data_fetcher.fetch_stock_data_async", AsyncMock(return_value=quote)), \
             patch("signal_engine._cooldown_active", side_effect=lambda key: key in sent), \
             patch("signal_engine._cooldown_mark", side_effect=sent.update), \
             patch("paper.settle_signals", AsyncMock()), patch("paper.record_signals") as record:
            clock.now.return_value = datetime(2026, 10, 5, 19, tzinfo=ZoneInfo("Europe/Istanbul"))
            await scheduler.scheduled_check(app)
            await scheduler.scheduled_check(app)
        record.assert_not_called()
        app.bot.send_message.assert_awaited_once()
        self.assertIn("SİNYAL KORUMA UYARISI", app.bot.send_message.await_args.kwargs["text"])
        self.assertIn("TEST.IS", app.bot.send_message.await_args.kwargs["text"])


if __name__ == "__main__":
    unittest.main()
