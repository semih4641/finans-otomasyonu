"""BIST daily data, strategy-label alignment and model compatibility tests."""
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

with patch("dotenv.load_dotenv"), patch.dict(os.environ, {}, clear=True):
    import signal_engine
    import indicators
    import backtest
    import paper
import ml_model
from market_data import completed_bist_daily, completed_candles
from signals_advanced import extract_features
from train_bist import temporal_holdout


def daily_bars(count=280):
    values = 100 + np.sin(np.arange(count) / 8) * 5 + np.arange(count) * 0.03
    data = pd.DataFrame({"open": values, "high": values + 1, "low": values - 1,
                         "close": values, "volume": np.full(count, 1000.0)},
                        index=pd.date_range("2024-01-01", periods=count, tz="Europe/Istanbul"))
    return data


class BistTests(unittest.TestCase):
    def test_training_backtest_and_paper_agree_on_next_open_outcomes(self):
        setup = {"signals": ["candidate"], "sl": 90, "tp1": 110, "tp2": 120}
        for outcome, low, high in (("TP1", 99, 111), ("TP2", 99, 121),
                                   ("SL", 89, 121), ("MTM-TIMEOUT", 99, 106)):
            with self.subTest(outcome=outcome):
                data = daily_bars(204)
                data[["open", "close"]] = 100.0
                data["low"], data["high"] = 99.0, 101.0
                data.iloc[201:, data.columns.get_loc("open")] = 105.0
                data.iloc[201:, data.columns.get_loc("close")] = 104.0
                data.iloc[201:, data.columns.get_loc("high")] = 106.0
                data.iloc[201, data.columns.get_loc("low")] = low
                data.iloc[201, data.columns.get_loc("high")] = high
                data.attrs["completed_only"] = True
                groups = ml_model.build_bist_samples(data, "TEST.IS", Mock(return_value=setup), horizon=3)
                samples = [row for group in groups.values() for row in group]
                self.assertEqual(len(samples), 1)
                pending = {**setup, "symbol": "hisse:TEST.IS", "market": "stock",
                           "status": "pending", "execution_model": "bist_next_open_v1",
                           "signal_at": data.index[200].isoformat(),
                           "opened_at": (data.index[200] + pd.Timedelta(hours=19)).isoformat(),
                           "entry": 100, "reference_entry": 100, "quantity": 10}
                filled, reason = paper._resolve_pending_entry(pending, data)
                self.assertIsNone(reason)
                with patch.dict(paper.HORIZON_BARS, stock=3):
                    settled = paper.settle_one(filled, data)
                event = {**setup, "entry": 105, "decision_idx": 200, "entry_idx": 201,
                         "score": 3, "trend": "BOGA", "rr": 2, "categories": ["candidate"]}
                historical = backtest.evaluate([event], data, horizon=3, cost_rate=paper.COST_RATE)[0]
                for result in (samples[0], settled, historical):
                    self.assertEqual(result["entry"], 105)
                    self.assertEqual(result["outcome"], outcome)
                    self.assertEqual(result["bars_held"], 3 if outcome == "MTM-TIMEOUT" else 1)
                    self.assertAlmostEqual(result["net_pnl_pct"], settled["net_pnl_pct"], places=3)

    def test_today_before_close_is_excluded_but_previous_session_is_kept(self):
        data = daily_bars(3)
        result = completed_bist_daily(data, "2024-01-03T17:59:00+03:00")
        self.assertEqual(len(result), 2)
        self.assertEqual(len(completed_candles(result)), 2)
        self.assertEqual(indicators.calculate_rsi(result), None)

    def test_latest_closed_day_is_used_after_close_and_on_weekend(self):
        data = daily_bars(5)
        for moment in ("2024-01-05T18:20:00+03:00", "2024-01-06T12:00:00+03:00"):
            with self.subTest(moment=moment):
                self.assertEqual(len(completed_bist_daily(data, moment)), 5)

    def test_future_dates_are_excluded_and_utc_clock_is_converted(self):
        data = daily_bars(5)
        self.assertEqual(len(completed_bist_daily(data, "2024-01-03T15:20:00Z")), 3)

    def test_advanced_and_base_analysis_use_same_last_completed_session(self):
        data = completed_bist_daily(daily_bars(), "2025-01-01")
        features = extract_features(data, "TEST.IS")
        self.assertEqual(features.timestamp, data.index[-1].to_pydatetime())
        self.assertEqual(features.rsi, indicators.calculate_rsi(data))

    def test_macd_model_input_is_invariant_to_nominal_share_price(self):
        data = daily_bars()
        expensive = data.copy()
        expensive[["open", "high", "low", "close"]] *= 100
        cheap_feature = extract_features(data, "CHEAP.IS")
        expensive_feature = extract_features(expensive, "EXPENSIVE.IS")
        self.assertAlmostEqual(cheap_feature.macd_hist_pct, expensive_feature.macd_hist_pct, places=6)
        self.assertIn("macd_hist_pct", ml_model.FEATURE_COLUMNS)
        self.assertNotIn("macd_hist", ml_model.FEATURE_COLUMNS)

    def test_training_uses_actual_targets_not_old_fixed_percentages(self):
        data = daily_bars(205)
        data[["open", "close"]] = 100.0
        data["high"], data["low"] = 104.0, 99.0
        detector = Mock(return_value={"signals": ["candidate"], "sl": 96, "tp1": 108, "tp2": 116})
        groups = ml_model.build_bist_samples(data, "TEST.IS", detector, horizon=3)
        rows = [row for group in groups.values() for row in group]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], 0)  # +3% was touched, actual TP1 was not.
        self.assertEqual((rows[0]["sl"], rows[0]["tp1"]), (96, 108))
        detector.assert_called_once()
        self.assertFalse(detector.call_args.kwargs["use_ml"])
        self.assertTrue(detector.call_args.args[0].attrs["completed_only"])

    def test_non_candidates_and_entry_gaps_beyond_target_are_not_training_samples(self):
        for setup in ({"signals": []}, {"signals": ["x"], "sl": 90, "tp1": 95, "tp2": 99}):
            rows = ml_model.build_bist_samples(daily_bars(205), "TEST.IS", Mock(return_value=setup), horizon=3)
            self.assertEqual(sum(map(len, rows.values())), 0)

    def test_bist_model_is_not_used_for_other_markets(self):
        data = daily_bars()
        data.iloc[-2, data.columns.get_loc("volume")] = 10000
        with patch.object(signal_engine, "_get_ml_ensemble") as load:
            signal_engine.detect_buy_signals(data, symbol="AAPL")
            signal_engine.detect_buy_signals(data, market="crypto", symbol="BTC/USDT")
        load.assert_not_called()

    def test_legacy_mixed_model_is_rejected_before_accessing_estimator(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pkl"
            path.write_bytes(pickle.dumps({"schema_version": 1}))
            with self.assertRaisesRegex(ValueError, "BIST"):
                ml_model.SignalMLModel.load(path)

    def test_holdout_purges_unresolved_labels_and_keeps_dates_together(self):
        rows = [{"timestamp": date, "label_end_time": date + pd.Timedelta(days=10)}
                for date in pd.date_range("2024-01-01", periods=100)] * 2
        train, test, cutoff = temporal_holdout(rows)
        self.assertTrue(all(row["label_end_time"] < cutoff for row in train))
        self.assertTrue(all(row["timestamp"] >= cutoff for row in test))
        self.assertEqual(len(test), 40)
        self.assertLess(len(train) + len(test), len(rows))


if __name__ == "__main__":
    unittest.main()
