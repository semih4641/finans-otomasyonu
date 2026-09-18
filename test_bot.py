"""Offline integration regressions; never read credentials or contact services."""

import asyncio
import os
import tempfile
import threading
import unittest
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

with patch("dotenv.load_dotenv"), patch.dict(os.environ, {}, clear=True):
    import bot
    import data_fetcher
    import signal_engine
    import scheduler
    import telegram_handlers
    import paper

from risk import RiskConfig
from portfolio_risk import PortfolioConfig, PortfolioState
from signals_advanced import MarketRegime


def frame(size=80):
    close = np.linspace(100, 120, size)
    data = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": np.full(size, 100.0)},
                        index=pd.date_range("2025-01-01", periods=size))
    data.iloc[-2, data.columns.get_loc("volume")] = 1000.0
    return data


class SignalIntegrationTests(unittest.TestCase):
    def test_missing_model_does_not_invent_probability_or_block_valid_signal(self):
        ensemble = SimpleNamespace(global_model=None, models={})
        with patch.object(signal_engine, "_get_ml_ensemble", return_value=ensemble), \
             patch.object(signal_engine, "SIGNAL_SCORES_STOCK", {"🔊 Hacim Patlaması": 2}), \
             patch.object(signal_engine, "generate_advanced_signals", return_value={"adjusted_score": 99, "base_score": 1}):
            result = signal_engine.detect_buy_signals(frame(220), symbol="TEST.IS")
        self.assertTrue(result["signals"])
        self.assertEqual(result["score"], 2)
        self.assertEqual(result["ml_score"], 0.5)
        self.assertFalse(result["ml_available"])
        self.assertGreater(result["tp1"], result["sl"])

    def test_trained_model_confidence_still_gates_signals(self):
        ensemble = SimpleNamespace(global_model=True, models={}, predict=Mock(return_value=(0.1, "global")))
        with patch.object(signal_engine, "_get_ml_ensemble", return_value=ensemble), \
             patch.object(signal_engine, "ML_FILTER_SIGNALS", True), \
             patch.object(signal_engine, "SIGNAL_SCORES_STOCK", {"🔊 Hacim Patlaması": 5}):
            result = signal_engine.detect_buy_signals(frame(220), symbol="TEST.IS")
        self.assertTrue(result["ml_available"])
        self.assertFalse(result["signals"])

    def test_diagnostic_ml_score_does_not_veto_a_technical_signal(self):
        ensemble = SimpleNamespace(global_model=True, models={}, predict=Mock(return_value=(0.1, "global")))
        with patch.object(signal_engine, "_get_ml_ensemble", return_value=ensemble), \
             patch.object(signal_engine, "ML_FILTER_SIGNALS", False), \
             patch.object(signal_engine, "SIGNAL_SCORES_STOCK", {"🔊 Hacim Patlaması": 2}):
            result = signal_engine.detect_buy_signals(frame(220), symbol="TEST.IS")
        self.assertTrue(result["signals"])
        self.assertEqual(result["ml_score"], 0.1)
        self.assertTrue(result["ml_available"])

    def test_short_bist_history_never_reaches_model_trained_on_sma200_history(self):
        with patch.object(signal_engine, "_get_ml_ensemble") as model:
            signal_engine.detect_buy_signals(frame(80), symbol="TEST.IS")
        model.assert_not_called()

    def test_two_valid_sma200_observations_are_required_for_crossover(self):
        data = frame(201)
        data["volume"] = 100.0
        with patch.object(signal_engine, "USE_ML_SCORING", False):
            result = signal_engine.detect_buy_signals(data)
        self.assertFalse(result["signals"])

    def test_invalid_prices_and_targets_never_emit_signals(self):
        for column, value in (("close", float("nan")), ("close", float("inf")), ("high", float("inf"))):
            with self.subTest(column=column, value=value):
                data = frame()
                data.iloc[-2, data.columns.get_loc(column)] = value
                with patch.object(signal_engine, "USE_ML_SCORING", False), \
                     patch.object(signal_engine, "SIGNAL_SCORES_STOCK", {"🔊 Hacim Patlaması": 3}):
                    result = signal_engine.detect_buy_signals(data)
                self.assertFalse(result["signals"])


class PortfolioSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.ledger = Path(self.directory.name) / "settled.csv"
        self.config = RiskConfig(account_size=10000)
        self.patches = [patch.object(paper, "SETTLED_CSV", self.ledger),
                        patch.object(paper, "_load_open", return_value=[]),
                        patch("bot.get_risk_config", return_value=self.config),
                        patch.object(bot, "_portfolio_state", PortfolioState([{"symbol": "old"}], 1, 2, -1, 1))]
        for patcher in self.patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_empty_positions_clear_previous_frozen_snapshot(self):
        self.assertTrue(bot._update_portfolio_state())
        self.assertEqual(bot._portfolio_state.positions, [])
        self.assertEqual(bot._portfolio_state.equity, 10000)
        self.assertEqual(bot._portfolio_state.daily_pnl, 0)

    def test_cash_pnl_reconstructs_peak_and_daily_loss_in_time_order(self):
        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        pd.DataFrame([
            {"closed_at": f"{today}T11:00:00", "net_pnl_amount": -700, "net_pnl_pct": -7},
            {"closed_at": f"{yesterday}T11:00:00", "net_pnl_amount": 500, "net_pnl_pct": 50},
        ]).to_csv(self.ledger, index=False)
        self.assertTrue(bot._update_portfolio_state())
        self.assertEqual(bot._portfolio_state.equity, 9800)
        self.assertEqual(bot._portfolio_state.peak_equity, 10500)
        self.assertEqual(bot._portfolio_state.daily_pnl, -700)

    def test_legacy_unsized_returns_are_not_treated_as_full_account_returns(self):
        pd.DataFrame([{"closed_at": "2025-01-01", "net_pnl_pct": 50}]).to_csv(self.ledger, index=False)
        self.assertTrue(bot._update_portfolio_state())
        self.assertEqual(bot._portfolio_state.equity, 10000)

    def test_unreadable_snapshot_reports_failure(self):
        with patch.object(paper, "_load_open", side_effect=RuntimeError("invalid ledger")):
            self.assertFalse(bot._update_portfolio_state())

    def test_invalid_cash_pnl_is_not_silently_treated_as_zero(self):
        pd.DataFrame([{"closed_at": "2025-01-01", "net_pnl_amount": "invalid"}]).to_csv(self.ledger, index=False)
        self.assertFalse(bot._update_portfolio_state())

    def test_count_and_daily_loss_gate_before_exposure(self):
        config = RiskConfig(account_size=10000, max_open_positions=1)
        with patch("scheduler.get_risk_config", return_value=config), \
             patch("scheduler.portfolio_gate") as exposure:
            decision = scheduler._candidate_gate({}, PortfolioState([{}], 10000, 10000, 0, 10000))
            self.assertFalse(decision.allowed)
            decision = scheduler._candidate_gate({}, PortfolioState([], 9800, 10000, -200, 10000))
            self.assertFalse(decision.allowed)
            self.assertEqual(decision.code, "DAILY_LOSS_LIMIT")
            exposure.assert_not_called()


class AsyncIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        for name, value in (("_stock_cache", {}), ("_stock_cache_ts", {}),
                            ("_stock_fetch_tasks", {}), ("_STOCK_SEMAPHORE", asyncio.Semaphore(4))):
            patcher = patch.object(data_fetcher, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        for name, value in (("_ml_training_active", False), ("_ml_ensemble", None),
                            ("_ml_models_loaded", False)):
            patcher = patch.object(signal_engine, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        patcher = patch.object(bot, "BIST_ONLY", False)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_scan_uses_snapshot_rebuilt_from_real_ledger(self):
        old = PortfolioState([], 1, 2, -1, 1)
        app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "settled.csv"
            pd.DataFrame([{"closed_at": datetime.now().isoformat(),
                           "net_pnl_amount": -250}]).to_csv(ledger, index=False)
            with patch.object(bot, "CHAT_ID", "test"), \
                 patch.object(bot, "SCAN_STOCKS", []), patch.object(bot, "SCAN_CRYPTO", []), \
                 patch.object(bot, "WATCHLIST_STOCKS", []), \
                 patch.object(bot, "ENABLE_PORTFOLIO_RISK", True), \
                 patch.object(bot, "_portfolio_state", old), \
                 patch.object(bot, "get_risk_config", return_value=RiskConfig(account_size=10000)), \
                 patch.object(paper, "SETTLED_CSV", ledger), \
                 patch.object(paper, "_load_open", return_value=[]), \
                 patch.object(paper, "settle_signals", AsyncMock()), \
                 patch.object(scheduler, "_load_open_position_data", AsyncMock(return_value={})) as load:
                await scheduler.scheduled_check(app)
                used_state = load.await_args.args[0]
                self.assertEqual(used_state.equity, 9750)
                self.assertEqual(used_state.peak_equity, 10000)
                self.assertEqual(used_state.daily_pnl, -250)
                self.assertIsNot(used_state, bot._portfolio_state)
                self.assertIsNot(used_state.positions, bot._portfolio_state.positions)

    async def test_portfolio_command_displays_real_refreshed_snapshot(self):
        message = SimpleNamespace(reply_text=AsyncMock())
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "settled.csv"
            pd.DataFrame([{"closed_at": datetime.now().isoformat(),
                           "net_pnl_amount": -250}]).to_csv(ledger, index=False)
            with patch.object(bot, "CHAT_ID", None), \
                 patch.object(bot, "_portfolio_state", PortfolioState([], 1, 2, -1, 1)), \
                 patch.object(bot, "get_risk_config", return_value=RiskConfig(account_size=10000)), \
                 patch.object(paper, "SETTLED_CSV", ledger), \
                 patch.object(paper, "_load_open", return_value=[]):
                await telegram_handlers.portfoy_command(
                    SimpleNamespace(effective_message=message), SimpleNamespace())
        text = message.reply_text.await_args.args[0]
        self.assertIn("Hesap Değeri: 9,750.00", text)
        self.assertIn("En Yüksek Hesap Değeri: 10,000.00", text)
        self.assertIn("Drawdown: %2.50", text)

    async def test_legacy_chart_preserves_cash_sizing_and_first_trade_drawdown(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        message = SimpleNamespace(reply_text=AsyncMock(), reply_photo=AsyncMock())
        charts = []
        create_subplots = plt.subplots

        def capture_chart(*args, **kwargs):
            chart = create_subplots(*args, **kwargs)
            charts.append(chart)
            return chart

        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "settled.csv"
            # Legacy rows have sized percentage returns but no cash-P/L column.
            pd.DataFrame([
                {"closed_at": "2025-01-02", "entry": 100, "quantity": 1, "net_pnl_pct": 5},
                {"closed_at": "2025-01-01", "entry": 100, "quantity": 5, "net_pnl_pct": -10},
            ]).to_csv(ledger, index=False)
            with patch.object(bot, "CHAT_ID", None), patch.object(paper, "SETTLED_CSV", ledger), \
                 patch("risk.get_risk_config", return_value=RiskConfig(account_size=1000)), \
                 patch.object(plt, "subplots", side_effect=capture_chart), \
                 warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Glyph .* missing from font")
                await telegram_handlers.grafik_command(
                    SimpleNamespace(effective_message=message), SimpleNamespace())
        message.reply_photo.assert_awaited_once()
        self.assertTrue(message.reply_photo.await_args.kwargs["photo"].getvalue().startswith(b"\x89PNG"))
        _, (equity_axis, drawdown_axis) = charts[0]
        first_loss = 50 * (1 + paper.COST_RATE)
        np.testing.assert_allclose(equity_axis.lines[0].get_ydata(),
                                   [1000 - first_loss, 1000 - 45 * (1 + paper.COST_RATE)])
        self.assertAlmostEqual(drawdown_axis.lines[0].get_ydata()[0], -first_loss / 1000 * 100)

    async def test_bist_daily_scan_waits_for_completed_current_session(self):
        data = frame(220)
        data.index = pd.date_range(end="2026-09-17", periods=len(data), tz="Europe/Istanbul")
        data.attrs["completed_only"] = True
        quote = {"price": 999.0, "history_df": data, "ticker": "TEST.IS", "info": {}}
        app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
        for now in (datetime(2026, 9, 17, 12, tzinfo=ZoneInfo("Europe/Istanbul")),
                    datetime(2026, 9, 18, 19, tzinfo=ZoneInfo("Europe/Istanbul"))):
            with self.subTest(now=now), patch.object(bot, "CHAT_ID", "test"), \
                 patch.object(bot, "BIST_ONLY", True), \
                 patch.object(bot, "SCAN_STOCKS", ["TEST.IS"]), patch.object(bot, "WATCHLIST_STOCKS", []), \
                 patch.object(bot, "_update_portfolio_state", return_value=True), \
                 patch.object(bot, "_portfolio_state", PortfolioState([], 10000, 10000, 0, 10000)), \
                 patch.object(scheduler, "datetime", wraps=datetime) as clock, \
                 patch("data_fetcher.fetch_stock_data_async", AsyncMock(return_value=quote)), \
                 patch("signal_engine.detect_buy_signals") as detect, \
                 patch("paper.settle_signals", AsyncMock()), patch("paper.record_signals") as record:
                clock.now.return_value = now
                await scheduler.scheduled_check(app)
                detect.assert_not_called()
                record.assert_not_called()
        app.bot.send_message.assert_not_called()

    async def test_bist_after_close_uses_decision_close_and_next_session_entry(self):
        data = frame(220)
        data.index = pd.date_range(end="2026-09-17", periods=len(data), tz="Europe/Istanbul")
        data.attrs["completed_only"] = True
        quote = {"price": 999.0, "history_df": data, "ticker": "TEST.IS", "info": {}}
        app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
        result = {"signals": ["signal"], "sl": 116, "tp1": 128, "tp2": 136,
                  "vade": "test", "score": 3, "trend": "test", "ml_score": 0.5, "rr": 2.0}
        with patch.object(bot, "CHAT_ID", "test"), patch.object(bot, "BIST_ONLY", True), \
             patch.object(bot, "SCAN_STOCKS", ["TEST.IS"]), patch.object(bot, "WATCHLIST_STOCKS", []), \
             patch.object(bot, "_update_portfolio_state", return_value=True), \
             patch.object(bot, "_portfolio_state", PortfolioState([], 10000, 10000, 0, 10000)), \
             patch.object(scheduler, "datetime", wraps=datetime) as clock, \
             patch.object(scheduler, "_candidate_gate", return_value=SimpleNamespace(allowed=True)) as gate, \
             patch.object(scheduler, "calculate_position_risk", return_value=SimpleNamespace(
                 quantity=25, risk_amount=100, is_calculable=True, as_dict=lambda: {"quantity": 25})) as size, \
             patch("data_fetcher.fetch_stock_data_async", AsyncMock(return_value=quote)), \
             patch("signal_engine.detect_buy_signals", return_value=result), \
             patch("signal_engine.filter_cooldown", side_effect=lambda symbol, signals: signals), \
             patch("signal_engine.sizing_hint", return_value=""), patch("signal_engine.mark_signals_sent"), \
             patch("paper.settle_signals", AsyncMock()), patch("paper.record_signals") as record:
            clock.now.return_value = datetime(2026, 9, 17, 19, tzinfo=ZoneInfo("Europe/Istanbul"))
            await scheduler.scheduled_check(app)
        item = record.call_args.args[0][0]
        self.assertEqual(item["entry_num"], 120)
        self.assertEqual(item["signal_at"], "2026-09-17T00:00:00+03:00")
        self.assertEqual(gate.call_args.args[0]["entry"], 120)
        self.assertTrue(all(call.args[0] == 120 for call in size.call_args_list))
        text = app.bot.send_message.await_args.kwargs["text"]
        self.assertIn("Referans kapanış: 120.00", text)
        self.assertIn("sonraki seans açılışı bekleniyor", text)

    async def test_concurrent_requests_share_download_and_cached_result(self):
        download = Mock(return_value={"price": 10}, __name__="download")
        results = await asyncio.gather(*(data_fetcher._cached_fetch(download, "TEST") for _ in range(8)))
        self.assertTrue(all(result == {"price": 10} for result in results))
        self.assertEqual(await data_fetcher._cached_fetch(download, "TEST"), {"price": 10})
        download.assert_called_once()
        self.assertFalse(data_fetcher._stock_fetch_tasks)

    async def test_bist_automatic_scan_does_not_fetch_crypto(self):
        state = PortfolioState([], 10000, 10000, 0, 10000)
        app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
        with patch.object(bot, "CHAT_ID", "test"), patch.object(bot, "BIST_ONLY", True), \
             patch.object(bot, "SCAN_CRYPTO", ["BTC/USDT"]), \
             patch.object(bot, "SCAN_STOCKS", []), patch.object(bot, "WATCHLIST_STOCKS", []), \
             patch.object(bot, "_update_portfolio_state", return_value=True), \
             patch.object(bot, "_portfolio_state", state), \
             patch("data_fetcher.fetch_crypto_data", AsyncMock()) as crypto, \
             patch("paper.settle_signals", AsyncMock()):
            await scheduler.scheduled_check(app)
        crypto.assert_not_called()
        app.bot.send_message.assert_not_called()

    async def test_failed_download_can_be_retried(self):
        download = Mock(side_effect=[RuntimeError("temporary"), {"price": 10}], __name__="download")
        with self.assertRaises(RuntimeError):
            await data_fetcher._cached_fetch(download, "TEST")
        self.assertEqual(await data_fetcher._cached_fetch(download, "TEST"), {"price": 10})
        self.assertEqual(download.call_count, 2)

    async def test_training_awaits_data_runs_off_loop_and_activates_model(self):
        main_thread = threading.get_ident()
        trained_threads = []
        def train(*args, **kwargs):
            trained_threads.append(threading.get_ident())
            return {"global": {"f1": 0.75}}
        ensemble = SimpleNamespace(train=Mock(side_effect=train))
        collect = AsyncMock(return_value={MarketRegime.UNKNOWN: [{"label": 1}]})
        message = SimpleNamespace(reply_text=AsyncMock())
        with patch("ml_model.prepare_training_data", collect), \
             patch("ml_model.RegimeAwareModelEnsemble", return_value=ensemble):
            await telegram_handlers.mltrain_command(SimpleNamespace(effective_message=message), SimpleNamespace(args=["quick"]))
        self.assertEqual(collect.await_count, 1)
        self.assertTrue(all(symbol.endswith(".IS") for symbol in collect.await_args.args[0]))
        self.assertIs(signal_engine._ml_ensemble, ensemble)
        self.assertTrue(signal_engine._ml_models_loaded)
        self.assertNotEqual(trained_threads[0], main_thread)
        self.assertFalse(signal_engine._ml_training_active)

    async def test_unsuccessful_training_preserves_active_model(self):
        old = object()
        signal_engine._ml_ensemble = old
        message = SimpleNamespace(reply_text=AsyncMock())
        with patch("ml_model.prepare_training_data", AsyncMock(return_value={})), \
             patch("ml_model.RegimeAwareModelEnsemble", return_value=SimpleNamespace(train=Mock(return_value={}))):
            await telegram_handlers.mltrain_command(SimpleNamespace(effective_message=message), SimpleNamespace(args=[]))
        self.assertIs(signal_engine._ml_ensemble, old)
        self.assertFalse(signal_engine._ml_training_active)

    async def test_same_scan_reserves_each_accepted_position(self):
        data = {"price": 100.0, "ohlcv_df": frame()}
        config = RiskConfig(account_size=10000, max_open_positions=1)
        state = PortfolioState([], 10000, 10000, 0, 10000)
        result = {"signals": ["signal"], "sl": 96, "tp1": 108, "tp2": 116,
                  "vade": "test", "score": 3, "trend": "test", "ml_score": 0.5, "rr": 2.0}
        app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
        with patch.object(bot, "CHAT_ID", "test"), \
             patch.object(bot, "SCAN_CRYPTO", ["A/USDT", "B/USDT"]), \
             patch.object(bot, "SCAN_STOCKS", []), patch.object(bot, "WATCHLIST_STOCKS", []), \
             patch.object(bot, "_update_portfolio_state", return_value=True), \
             patch.object(bot, "_portfolio_state", state), \
             patch("scheduler.get_risk_config", return_value=config), \
             patch("data_fetcher.fetch_crypto_data", AsyncMock(return_value=data)), \
             patch("signal_engine.detect_buy_signals", return_value=result), \
             patch("signal_engine.filter_cooldown", side_effect=lambda symbol, signals: signals), \
             patch("scheduler.portfolio_gate", return_value=SimpleNamespace(allowed=True)), \
             patch("risk.calculate_position_risk", return_value=SimpleNamespace(
                 quantity=25, risk_amount=100, is_calculable=True, as_dict=lambda: {"quantity": 25})), \
             patch("signal_engine.sizing_hint", return_value=""), \
             patch("signal_engine.mark_signals_sent") as mark, \
             patch("paper.settle_signals", AsyncMock()), \
             patch("paper.record_signals") as record:
            await scheduler.scheduled_check(app)
        self.assertEqual(len(record.call_args.args[0]), 1)
        self.assertEqual(record.call_args.args[0][0]["key"], "kripto:A/USDT")
        self.assertEqual(mark.call_count, 1)
        self.assertEqual(state.positions, [])

    async def test_real_correlation_gate_is_independent_of_open_position_scan_order(self):
        data = {"price": 100.0, "ohlcv_df": frame()}
        config = RiskConfig(account_size=10000)
        portfolio_config = PortfolioConfig(max_sector_exposure_pct=100,
                                            max_correlation_exposure_pct=35)
        position = {"symbol": "kripto:B/USDT", "market": "crypto", "sector": "CRYPTO",
                    "entry": 100, "quantity": 25, "sl": 96}
        result = {"signals": ["signal"], "sl": 96, "tp1": 108, "tp2": 116,
                  "vade": "test", "score": 3, "trend": "test", "ml_score": 0.5, "rr": 2.0}
        for symbols in (["A/USDT", "B/USDT"], ["B/USDT", "A/USDT"]):
            with self.subTest(symbols=symbols):
                state = PortfolioState([position], 10000, 10000, 0, 10000)
                app = SimpleNamespace(bot=SimpleNamespace(send_message=AsyncMock()))
                with patch.object(bot, "CHAT_ID", "test"), \
                     patch.object(bot, "SCAN_CRYPTO", symbols), \
                     patch.object(bot, "SCAN_STOCKS", []), patch.object(bot, "WATCHLIST_STOCKS", []), \
                     patch.object(bot, "_update_portfolio_state", return_value=True), \
                     patch.object(bot, "_portfolio_state", state), \
                     patch("scheduler.get_risk_config", return_value=config), \
                     patch("risk.RISK_CONFIG", config), \
                     patch("portfolio_risk.PORTFOLIO_CONFIG", portfolio_config), \
                     patch("data_fetcher.fetch_crypto_data", AsyncMock(return_value=data)) as fetch, \
                     patch("signal_engine.detect_buy_signals", return_value=result), \
                     patch("signal_engine.filter_cooldown", side_effect=lambda symbol, signals: signals), \
                     patch("signal_engine.mark_signals_sent") as mark, \
                     patch("paper.settle_signals", AsyncMock()), \
                     patch("paper.record_signals") as record:
                    await scheduler.scheduled_check(app)
                self.assertEqual(fetch.await_args_list[0].args, ("B/USDT",))
                self.assertEqual(fetch.await_count, 2)
                app.bot.send_message.assert_not_called()
                record.assert_not_called()
                mark.assert_not_called()


if __name__ == "__main__":
    unittest.main()
