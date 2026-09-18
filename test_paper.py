import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd

with patch("dotenv.load_dotenv"), patch.dict("os.environ", {}, clear=True):
    import paper


def position(**changes):
    item = {
        "symbol": "crypto:TEST/USDT", "market": "crypto", "name": "TEST",
        "opened_at": "2026-01-01T03:00:00+03:00", "entry": 100.0,
        "sl": 90.0, "tp1": 110.0, "tp2": 120.0, "quantity": 10.0,
        "risk_amount": 100.0, "score": 5, "ml_score": 0.8,
        "regime": "RANGE",
    }
    item.update(changes)
    return item


def frame(count=4):
    return pd.DataFrame(
        {"low": [99.0] * count, "high": [101.0] * count, "close": [100.0] * count},
        index=pd.date_range("2026-01-01", periods=count, freq="h", tz="UTC"),
    )


def pending_position(**changes):
    return position(symbol="hisse:TEST.IS", market="stock",
                    opened_at="2026-01-01T19:00:00+03:00",
                    signal_at="2026-01-01T00:00:00+03:00", status="pending",
                    reference_entry=100.0, execution_model="bist_next_open_v1", **changes)


def bist_frame(opening=105, high=106, low=104, close=105):
    return pd.DataFrame(
        {"open": [100, opening], "high": [101, high],
         "low": [99, low], "close": [100, close]},
        index=pd.date_range("2026-01-01", periods=2, freq="D", tz="Europe/Istanbul"),
    )


class PaperDataBoundaryTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_data_module_is_used_for_stock_and_crypto(self):
        import data_fetcher
        data = frame()
        with patch.object(data_fetcher, "fetch_stock_data_async", new=AsyncMock(return_value={"history_df": data})) as stock, \
             patch.object(data_fetcher, "fetch_crypto_data", new=AsyncMock(return_value={"ohlcv_df": data})) as crypto:
            self.assertIs(await paper.fetch_frame_async("stock", "TEST.IS"), data)
            self.assertIs(await paper.fetch_frame_async("crypto", "TEST/USDT"), data)
        stock.assert_awaited_once_with("TEST.IS")
        crypto.assert_awaited_once_with("TEST/USDT")


class PaperSettlementTests(unittest.TestCase):
    def setUp(self):
        self.horizon = patch.object(paper, "HORIZON_BARS", {"stock": 4, "crypto": 4})
        self.horizon.start()
        self.addCleanup(self.horizon.stop)

    def test_open_position_does_not_timeout_before_full_horizon(self):
        self.assertIsNone(paper.settle_one(position(), frame(3)))
        row = paper.settle_one(position(), frame(4))
        self.assertEqual(row["outcome"], "MTM-TIMEOUT")
        self.assertEqual(row["bars_held"], 4)
        self.assertEqual(row["exit_price"], 100)

    def test_timeout_uses_horizon_close_and_ignores_later_targets(self):
        data = frame(5)
        data.iloc[3] = [101, 103, 102]
        data.iloc[4] = [90, 130, 120]
        row = paper.settle_one(position(), data)
        self.assertEqual(row["outcome"], "MTM-TIMEOUT")
        self.assertEqual(row["exit_price"], 102)
        self.assertEqual(row["bars_held"], 4)

    def test_stop_wins_if_target_and_stop_are_touched_on_same_bar(self):
        data = frame(1)
        data.iloc[0] = [85, 125, 105]
        row = paper.settle_one(position(), data)
        self.assertEqual(row["outcome"], "SL")
        self.assertEqual(row["exit_price"], 90)

    def test_settlement_persists_actual_cash_profit_after_both_fees(self):
        data = frame(1)
        data.iloc[0] = [99, 111, 110]
        row = paper.settle_one(position(), data)
        self.assertEqual(row["outcome"], "TP1")
        self.assertAlmostEqual(row["net_pnl_amount"], 96.85, places=6)
        self.assertAlmostEqual(row["net_pnl_pct"], (109.835 / 100.15 - 1) * 100, places=3)

    def test_legacy_unsized_position_does_not_invent_cash_profit(self):
        for quantity in (None, 0, -1, float("nan"), float("inf"), True):
            with self.subTest(quantity=quantity):
                row = paper.settle_one(position(quantity=quantity), frame())
                self.assertIsNone(row["net_pnl_amount"])

    def test_naive_crypto_timestamps_are_utc_and_pre_entry_bars_are_ignored(self):
        data = frame(3)
        data.index = data.index.tz_localize(None)
        data.iloc[0] = [80, 130, 100]
        opened = position(opened_at="2026-01-01T04:00:00+03:00")
        self.assertIsNone(paper.settle_one(opened, data))
        data.iloc[1] = [99, 125, 120]
        row = paper.settle_one(opened, data)
        self.assertEqual(row["outcome"], "TP2")
        self.assertEqual(row["bars_held"], 1)

    def test_invalid_prices_and_nonmonotonic_bars_are_rejected(self):
        for invalid in (float("nan"), float("inf"), -1):
            data = frame()
            data.iloc[0, 0] = invalid
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                paper.settle_one(position(), data)
        with self.assertRaises(ValueError):
            paper.settle_one(position(), frame().iloc[::-1])
        with self.assertRaises(ValueError):
            paper.settle_one(position(entry=0), frame())


class PaperLedgerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        paths = {
            "BASE": self.base, "OPEN_FILE": self.base / "open.json",
            "SETTLED_CSV": self.base / "settled.csv", "EQUITY_CSV": self.base / "equity.json",
            "METRICS_JSON": self.base / "metrics.json",
            "HORIZON_BARS": {"stock": 4, "crypto": 4},
        }
        for name, value in paths.items():
            patcher = patch.object(paper, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(paper, "_save_metrics")
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_record_sizing_and_duplicate_guard(self):
        signal = {"key": "stock:TEST", "entry_num": 100, "sl": 90,
                  "tp1": 110, "tp2": 120, "quantity": None,
                  "risk": {"quantity": 2, "risk_amount": 20}}
        self.assertEqual(paper.record_signals([signal, signal]), 1)
        saved = paper._load_open()
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["quantity"], 2)
        self.assertEqual(saved[0]["risk_amount"], 20)
        self.assertIsNotNone(pd.Timestamp(saved[0]["opened_at"]).tzinfo)
        self.assertEqual(paper.record_signals([signal]), 0)

    def test_invalid_signal_quantity_cannot_break_valid_records(self):
        signal = {"key": "stock:TEST", "entry_num": 100, "sl": 90, "tp1": 110, "tp2": 120}
        invalid = [{**signal, "quantity": qty} for qty in (float("nan"), float("inf"), -1, True)]
        self.assertEqual(paper.record_signals(invalid + [{**signal, "quantity": 2}]), 1)
        self.assertEqual(paper._load_open()[0]["quantity"], 2)

    def test_corrupt_open_file_is_preserved(self):
        paper.OPEN_FILE.write_text("{broken", encoding="utf-8")
        with self.assertRaises(RuntimeError):
            paper.record_signals([{"key": "stock:NEW", "entry_num": 100}])
        self.assertEqual(paper.OPEN_FILE.read_text(encoding="utf-8"), "{broken")

    def test_interrupted_json_replace_preserves_previous_file(self):
        paper._save_open([position()])
        before = paper.OPEN_FILE.read_bytes()
        with patch.object(paper.os, "replace", side_effect=OSError("simulated replace failure")):
            with self.assertRaises(OSError):
                paper._save_open([])
        self.assertEqual(paper.OPEN_FILE.read_bytes(), before)
        self.assertEqual(list(self.base.iterdir()), [paper.OPEN_FILE])

    def test_settlement_migrates_old_header_and_preserves_cash_fields(self):
        old_fields = [name for name in paper.SETTLED_FIELDS if name not in {"quantity", "risk_amount", "net_pnl_amount"}]
        old = paper.settle_one(position(symbol="crypto:OLD/USDT"), frame())
        with paper.SETTLED_CSV.open("w", newline="", encoding="utf-8-sig") as target:
            writer = csv.DictWriter(target, fieldnames=old_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(old)
        paper._save_open([position()])
        with patch.object(paper, "fetch_frame_async", new=AsyncMock(return_value=frame())):
            self.assertEqual(asyncio.run(paper.settle_signals()), 0)
        with paper.SETTLED_CSV.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            rows = list(reader)
            self.assertEqual(reader.fieldnames, paper.SETTLED_FIELDS)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["symbol"], "crypto:OLD/USDT")
        self.assertEqual(float(rows[0]["net_pnl_pct"]), old["net_pnl_pct"])
        self.assertEqual(rows[0]["quantity"], "")
        self.assertEqual(float(rows[1]["quantity"]), 10)
        self.assertEqual(float(rows[1]["risk_amount"]), 100)
        self.assertAlmostEqual(float(rows[1]["net_pnl_amount"]), -3)
        self.assertEqual(paper._load_open(), [])

    def test_retry_after_csv_commit_does_not_duplicate_settlement(self):
        paper._save_open([position()])
        with patch.object(paper, "fetch_frame_async", new=AsyncMock(return_value=frame())), \
             patch.object(paper, "_save_open", side_effect=OSError("simulated crash")):
            with self.assertRaises(OSError):
                asyncio.run(paper.settle_signals())
        self.assertEqual(len(paper._load_settled_rows()), 1)
        with patch.object(paper, "fetch_frame_async", new=AsyncMock()) as fetch:
            self.assertEqual(asyncio.run(paper.settle_signals()), 0)
            fetch.assert_not_called()
        self.assertEqual(len(paper._load_settled_rows()), 1)
        self.assertEqual(paper._load_open(), [])
        self.assertEqual(len(json.loads(paper.EQUITY_CSV.read_text(encoding="utf-8"))), 1)

    def test_failed_fetch_retains_open_position(self):
        paper._save_open([position()])
        with patch.object(paper, "fetch_frame_async", new=AsyncMock(side_effect=ValueError("offline fixture"))):
            self.assertEqual(asyncio.run(paper.settle_signals()), 1)
        self.assertEqual(paper._load_open(), [position()])
        self.assertFalse(paper.SETTLED_CSV.exists())

    def test_new_bist_setup_is_pending_and_requires_decision_candle(self):
        signal = {"key": "hisse:TEST.IS", "market": "stock", "entry_num": 100,
                  "sl": 90, "tp1": 110, "tp2": 120, "quantity": 10}
        self.assertEqual(paper.record_signals([signal]), 0)
        self.assertEqual(paper.record_signals([{**signal, "signal_at": "2026-01-01T00:00:00+03:00"}]), 1)
        saved = paper._load_open()[0]
        self.assertEqual(saved["status"], "pending")
        self.assertEqual(saved["reference_entry"], 100)
        self.assertEqual(saved["execution_model"], "bist_next_open_v1")
        self.assertNotIn("filled_at", saved)

    def test_pending_entry_waits_for_next_completed_daily_bar(self):
        from market_data import completed_bist_daily
        pending = pending_position()
        paper._save_open([pending])
        data = bist_frame(high=121)
        with patch.object(paper, "fetch_frame_async", new=AsyncMock(return_value=data)), \
             patch("market_data.completed_bist_daily", side_effect=lambda df: completed_bist_daily(df, now="2026-01-02T12:00:00+03:00")):
            self.assertEqual(asyncio.run(paper.settle_signals()), 1)
        self.assertEqual(paper._load_open(), [pending])
        self.assertFalse(paper.SETTLED_CSV.exists())

    def test_next_open_fill_is_persisted_and_does_not_increase_reserved_risk(self):
        pending = pending_position()
        paper._save_open([pending])
        data = bist_frame()
        with patch.object(paper, "fetch_frame_async", new=AsyncMock(return_value=data)):
            self.assertEqual(asyncio.run(paper.settle_signals()), 1)
        filled = paper._load_open()[0]
        self.assertEqual(filled["status"], "open")
        self.assertEqual(filled["entry"], 105)
        self.assertEqual(filled["opened_at"], pending["opened_at"])
        self.assertEqual(filled["filled_at"], data.index[1].isoformat())
        self.assertLess(filled["quantity"], pending["quantity"])
        cost = paper.COST_RATE
        original_risk = (100 * (1 + cost) - 90 * (1 - cost)) * 10
        filled_risk = (105 * (1 + cost) - 90 * (1 - cost)) * filled["quantity"]
        self.assertLessEqual(filled_risk, original_risk + 1e-9)
        self.assertLessEqual(filled["entry"] * filled["quantity"], 1000)
        # Repeating settlement does not fill or resize it again.
        with patch.object(paper, "fetch_frame_async", new=AsyncMock(return_value=data)):
            self.assertEqual(asyncio.run(paper.settle_signals()), 1)
        self.assertEqual(paper._load_open(), [filled])

    def test_next_open_entry_bar_is_included_in_tp_and_cost_calculation(self):
        pending = pending_position()
        paper._save_open([pending])
        with patch.object(paper, "fetch_frame_async", new=AsyncMock(return_value=bist_frame(high=111))):
            self.assertEqual(asyncio.run(paper.settle_signals()), 0)
        row = paper._load_settled_rows()[0]
        self.assertEqual(row["outcome"], "TP1")
        self.assertEqual(float(row["entry"]), 105)
        self.assertEqual(int(row["bars_held"]), 1)
        self.assertEqual(row["opened_at"], pending["opened_at"])
        self.assertAlmostEqual(float(row["net_pnl_pct"]), (110 * (1 - paper.COST_RATE) / (105 * (1 + paper.COST_RATE)) - 1) * 100, places=3)

    def test_invalid_opening_gap_is_cancelled_without_counting_a_trade(self):
        for opening in (89, 90, 110, 115):
            with self.subTest(opening=opening):
                paper._save_open([pending_position()])
                with patch.object(paper, "fetch_frame_async", new=AsyncMock(return_value=bist_frame(opening=opening))):
                    self.assertEqual(asyncio.run(paper.settle_signals()), 0)
                self.assertFalse(paper.SETTLED_CSV.exists())
                self.assertEqual(paper._load_cancelled()[0]["reason"], "invalid_entry_gap")
                (self.base / "live_cancelled.json").unlink()

    def test_late_notification_cannot_fill_at_an_open_that_already_happened(self):
        pending = {**pending_position(), "opened_at": "2026-01-02T12:00:00+03:00"}
        paper._save_open([pending])
        with patch.object(paper, "fetch_frame_async", new=AsyncMock(return_value=bist_frame())):
            self.assertEqual(asyncio.run(paper.settle_signals()), 0)
        self.assertEqual(paper._load_cancelled()[0]["reason"], "missed_entry")
        self.assertFalse(paper.SETTLED_CSV.exists())

    def test_cancelled_entry_recovers_after_audit_commit_without_fetching_again(self):
        paper._save_open([pending_position()])
        with patch.object(paper, "fetch_frame_async", new=AsyncMock(return_value=bist_frame(opening=115))), \
             patch.object(paper, "_save_open", side_effect=OSError("simulated crash")):
            with self.assertRaises(OSError):
                asyncio.run(paper.settle_signals())
        self.assertEqual(len(paper._load_cancelled()), 1)
        with patch.object(paper, "fetch_frame_async", new=AsyncMock()) as fetch:
            self.assertEqual(asyncio.run(paper.settle_signals()), 0)
            fetch.assert_not_called()
        self.assertEqual(len(paper._load_cancelled()), 1)
        self.assertEqual(paper._load_open(), [])

    def test_legacy_bist_record_keeps_existing_entry_semantics(self):
        legacy = position(symbol="hisse:TEST.IS", market="stock", opened_at="2026-01-01T19:00:00+03:00")
        paper._save_open([legacy])
        with patch.object(paper, "fetch_frame_async", new=AsyncMock(return_value=bist_frame(high=111))):
            self.assertEqual(asyncio.run(paper.settle_signals()), 0)
        row = paper._load_settled_rows()[0]
        self.assertEqual(float(row["entry"]), 100)
        self.assertEqual(row["filled_at"], "")


if __name__ == "__main__":
    unittest.main()
