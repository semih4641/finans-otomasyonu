"""Offline regression tests: importing the engine cannot initialize the live bot."""

import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd


# Keep real module boundaries; isolate credentials and external calls only.
# Replacing bot with a stub used to conceal missing production imports.
with patch("dotenv.load_dotenv"), patch.dict(os.environ, {}, clear=True):
    import backtest


def signal(categories=("A",)):
    return {
        "signals": list(categories), "tp1": 110.0, "tp2": 120.0,
        "sl": 90.0, "score": 3, "trend": "BOGA", "rr": 2.0,
    }


def bars(count):
    return pd.DataFrame({
        "open": [100.0] * count, "high": [105.0] * count,
        "low": [95.0] * count, "close": [101.0] * count,
        "volume": [1000.0] * count,
    })


def event():
    return {
        "decision_idx": 0, "entry_idx": 1, "entry": 100.0,
        "tp1": 110.0, "tp2": 120.0, "sl": 90.0,
        "score": 3, "trend": "BOGA", "rr": 2.0, "categories": ["A"],
    }


class WalkForwardTests(unittest.TestCase):
    def test_daily_cooldown_expires_at_next_bar_and_last_entry_is_included(self):
        data = bars(38)
        seen_lengths = []

        def detect(history, **kwargs):
            seen_lengths.append(len(history))
            self.assertFalse(kwargs["use_ml"])
            self.assertFalse(history.attrs["completed_only"])
            return signal()

        with patch.object(backtest, "detect_buy_signals", side_effect=detect):
            events = backtest.walk_forward(data, cooldown_bars=1)
        self.assertEqual(seen_lengths, [35, 36, 37, 38])
        self.assertEqual([item["entry_idx"] for item in events], [34, 35, 36, 37])
        self.assertEqual(events[-1]["decision_idx"], 36)

    def test_hourly_cooldown_expires_at_exactly_24_bars(self):
        with patch.object(backtest, "detect_buy_signals", return_value=signal()):
            events = backtest.walk_forward(bars(60), cooldown_bars=24)
        self.assertEqual([item["decision_idx"] for item in events], [33, 57])

    def test_fresh_category_is_not_suppressed_by_older_category(self):
        with patch.object(backtest, "detect_buy_signals", side_effect=[signal(), signal(("A", "B"))]):
            events = backtest.walk_forward(bars(36), cooldown_bars=24)
        self.assertEqual([item["categories"] for item in events], [["A"], ["B"]])

    def test_entry_uses_next_open_and_falls_back_to_decision_close(self):
        data = bars(35)
        data.loc[34, "open"] = 103.0
        with patch.object(backtest, "detect_buy_signals", return_value=signal()):
            events = backtest.walk_forward(data, cooldown_bars=1)
            fallback = backtest.walk_forward(data.drop(columns="open"), cooldown_bars=1)
        self.assertEqual(events[0]["entry"], 103.0)
        self.assertEqual(fallback[0]["entry"], 101.0)


class EvaluationTests(unittest.TestCase):
    def test_incomplete_horizon_excludes_both_early_winners_and_unresolved_trades(self):
        for high in (105, 121):
            with self.subTest(high=high):
                data = bars(3)
                data.loc[1, "high"] = high
                self.assertEqual(backtest.evaluate([event()], data, horizon=3), [])

    def test_holding_period_starts_at_actual_entry_bar(self):
        data = bars(5)
        data.loc[3, "high"] = 111
        trade = backtest.evaluate([{**event(), "entry_idx": 3}], data, horizon=2)[0]
        self.assertEqual(trade["bars_held"], 1)

    def test_opening_gaps_outside_setup_are_not_counted_as_trades(self):
        for entry in (89.0, 90.0, 110.0, 115.0, float("nan")):
            with self.subTest(entry=entry):
                candidate = {**event(), "entry": entry}
                self.assertEqual(backtest.evaluate([candidate], bars(3), horizon=2), [])

    def test_stop_wins_same_bar_ambiguity(self):
        data = bars(3)
        data.loc[1, ["low", "high"]] = [89, 121]
        trade = backtest.evaluate([event()], data, horizon=2, cost_rate=0.0015)[0]
        self.assertEqual(trade["outcome"], "SL")
        self.assertEqual(trade["pnl_pct"], -10.0)
        self.assertLess(trade["net_pnl_pct"], -10.0)
        self.assertEqual(trade["bars_held"], 1)

    def test_timeout_marks_window_close_with_both_costs(self):
        data = bars(4)
        data.loc[2, "close"] = 104
        trade = backtest.evaluate([event()], data, horizon=2, cost_rate=0.0015)[0]
        self.assertEqual(trade["outcome"], "MTM-TIMEOUT")
        self.assertEqual(trade["pnl_pct"], 4.0)
        expected = ((104 * 0.9985) / (100 * 1.0015) - 1) * 100
        self.assertAlmostEqual(trade["net_pnl_pct"], round(expected, 3))
        self.assertEqual(trade["bars_held"], 2)

    def test_invalid_windows_and_costs_are_rejected(self):
        for horizon in (0, -1, 1.5, True):
            with self.subTest(horizon=horizon), self.assertRaises(ValueError):
                backtest.evaluate([], bars(3), horizon=horizon)
        for cost in (-0.1, float("nan"), float("inf"), 1.0):
            with self.subTest(cost=cost), self.assertRaises(ValueError):
                backtest.evaluate([], bars(3), horizon=2, cost_rate=cost)

    def test_summary_counts_initial_loss_in_drawdown(self):
        trades = [{
            "outcome": "SL", "net_pnl_pct": pnl, "bars_held": 1, "categories": "A",
        } for pnl in [-5.0, -3.0, 2.0]]
        summary = backtest.summarize(trades)
        self.assertEqual(summary["GENEL"]["max_drawdown_pct"], 8.0)
        self.assertEqual(summary["A"]["max_drawdown_pct"], 8.0)


class CommandDefaultsTests(unittest.TestCase):
    def test_default_and_all_scans_follow_bist_scope(self):
        for arguments in ([], ["--all"]):
            with self.subTest(arguments=arguments), \
                 patch.object(sys, "argv", ["backtest.py", *arguments]), \
                 patch.object(backtest, "BIST_ONLY", True), \
                 patch.object(backtest, "run_backtest") as run:
                backtest.main()
                stocks, cryptos = run.call_args.args[:2]
                self.assertTrue(stocks)
                self.assertTrue(all(symbol.endswith(".IS") for symbol in stocks))
                self.assertEqual(cryptos, [])

    def test_explicit_crypto_selection_is_preserved(self):
        with patch.object(sys, "argv", ["backtest.py", "--stocks", "", "--crypto", "BTC/USDT"]), \
             patch.object(backtest, "run_backtest") as run:
            backtest.main()
        self.assertEqual(run.call_args.args[:2], ([], ["BTC/USDT"]))


if __name__ == "__main__":
    unittest.main()
