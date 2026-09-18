"""Offline signal and ML regressions; live bot imports and model writes are stubbed."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd

import ml_model
import signal_engine
from signals_advanced import (
    MarketRegime, SignalFeatures, calculate_adx, calculate_bollinger_bands,
    detect_volume_anomaly, extract_features, score_signal_advanced,
)


def bars(count=80):
    return pd.DataFrame({
        "open": np.full(count, 100.0), "high": np.full(count, 101.0),
        "low": np.full(count, 99.0), "close": np.full(count, 100.0),
        "volume": np.full(count, 1000.0),
    }, index=pd.date_range("2025-01-01", periods=count, tz="UTC"))


def samples(count=90):
    dates = pd.date_range("2025-01-01", periods=count, tz="UTC")
    return [{
        "rsi": float(20 + 50 * (i % 2)), "adx": float(i), "label": i % 2,
        "timestamp": date, "label_end_time": date + pd.Timedelta(days=4),
    } for i, date in enumerate(dates)]


class SignalTests(unittest.TestCase):
    def test_unfinished_candle_is_excluded_once_for_every_feature(self):
        data = bars()
        data.iloc[-2, data.columns.get_loc("close")] = 104.0
        data.iloc[-2, data.columns.get_loc("high")] = 105.0
        completed = extract_features(data.iloc[:-1], "X", analyze_completed_only=False)
        data.iloc[-1] = [1e6, 2e6, 1, 1e6, 1e12]
        self.assertEqual(extract_features(data, "X"), completed)
        self.assertEqual(completed.timestamp, data.index[-2].to_pydatetime())
        self.assertEqual(completed.bb_position, calculate_bollinger_bands(data)["bb_position"])
        self.assertEqual(completed.adx, calculate_adx(data)["adx"])

    def test_directional_movement_distinguishes_rising_and_falling_markets(self):
        data = bars()
        for direction in (1, -1):
            with self.subTest(direction=direction):
                close = 200 + direction * np.arange(len(data))
                data["close"], data["high"], data["low"] = close, close + 1, close - 1
                result = calculate_adx(data)
                self.assertGreater(result["adx"], 90)
                self.assertGreater(result["di_plus" if direction == 1 else "di_minus"], 0)
                self.assertEqual(result["di_minus" if direction == 1 else "di_plus"], 0)

    def test_flat_prices_produce_neutral_finite_indicators(self):
        data = bars()
        data["high"] = data["low"] = data["close"]
        feature = extract_features(data, "FLAT")
        self.assertEqual((feature.rsi, feature.adx, feature.bb_position), (50.0, 0.0, 0.5))
        self.assertEqual(feature.obv_trend, "FLAT")
        for name, value in asdict(feature).items():
            if isinstance(value, (int, float)):
                self.assertTrue(np.isfinite(value), name)

    def test_volume_spike_uses_preceding_candles_as_baseline(self):
        data = bars()
        data.iloc[-2, data.columns.get_loc("volume")] = 4000.0
        data.iloc[-2, data.columns.get_loc("close")] = 101.0
        result = detect_volume_anomaly(data)
        self.assertEqual(result["ratio"], 4.0)
        self.assertTrue(result["anomaly"])

    def test_short_and_empty_history_do_not_raise_or_invent_rsi(self):
        for size in (0, 1, 2, 5):
            with self.subTest(size=size):
                feature = extract_features(bars(size), "SHORT")
                self.assertIsNone(feature.rsi)
                self.assertEqual(feature.regime, MarketRegime.UNKNOWN)

    def test_zero_directional_index_and_zero_rsi_are_valid_scores(self):
        base = SignalFeatures(adx=40, di_plus=30, di_minus=0, rsi=0, obv_trend="UP")
        score, _ = score_signal_advanced(base, 2, "stock")
        self.assertEqual(score, 4)


class ModelTests(unittest.TestCase):
    def test_encoding_is_shared_and_preserves_zeros_and_finite_defaults(self):
        feature = SignalFeatures(rsi=0, bb_position=0, volume_ratio=0, adx=np.nan,
                                 macd_hist=np.inf, obv_trend="UP", regime=MarketRegime.BEAR_TRENDING)
        row = ml_model._feature_row(feature)
        self.assertEqual(row, ml_model._feature_row(asdict(feature)))
        values = dict(zip(ml_model.FEATURE_COLUMNS, row))
        self.assertEqual([values[key] for key in ("rsi", "bb_position", "volume_ratio")], [0, 0, 0])
        self.assertTrue(np.isfinite(row).all())

    def test_untrained_models_are_not_reported_as_available(self):
        ensemble = ml_model.RegimeAwareModelEnsemble("lr")
        ensemble.models[MarketRegime.UNKNOWN] = ml_model.SignalMLModel("lr")
        ensemble.global_model = ml_model.SignalMLModel("lr")
        self.assertEqual(ensemble.predict(SignalFeatures()), (0.5, "no_model"))
        self.assertEqual(ensemble.global_model.predict_proba(SignalFeatures()), 0.5)

    def test_untrained_regime_uses_fitted_global_model(self):
        ensemble = ml_model.RegimeAwareModelEnsemble("lr")
        ensemble.models[MarketRegime.UNKNOWN] = ml_model.SignalMLModel("lr")
        ensemble.global_model = ml_model.SignalMLModel("lr")
        ensemble.global_model.train_walk_forward(samples(), n_splits=3, purge_gap=0)
        probability, source = ensemble.predict(SignalFeatures(rsi=70))
        self.assertEqual(source, "global")
        self.assertTrue(0 <= probability <= 1)

    def test_grouped_symbols_use_time_order_and_purge_overlapping_outcomes(self):
        first = samples(60)
        second = [dict(sample, symbol="OTHER") for sample in first]
        rows = list(reversed(first)) + second
        for train, test in ml_model.SignalMLModel._walk_forward_splits(rows, 3, 0):
            train_times = {rows[i]["timestamp"] for i in train}
            test_times = {rows[i]["timestamp"] for i in test}
            self.assertFalse(train_times & test_times)
            self.assertLess(max(rows[i]["label_end_time"] for i in train), min(test_times))
            for timestamp in test_times:
                self.assertEqual(sum(rows[i]["timestamp"] == timestamp for i in test), 2)

    def test_training_requires_outcome_labels(self):
        model = ml_model.SignalMLModel("lr")
        for rows in ([SignalFeatures()] * 60, [{"rsi": 10}] * 60, [{"label": 2}] * 60):
            with self.subTest(rows_type=type(rows[0])):
                with self.assertRaises(ValueError):
                    model.train_walk_forward(rows)

    def test_failed_retraining_preserves_previous_fitted_state(self):
        model = ml_model.SignalMLModel("lr")
        self.assertTrue(model.train_walk_forward(samples(), n_splits=3, purge_gap=0))
        previous = model.model, model.scaler, model.metadata
        invalid = samples()
        invalid[-1]["label_end_time"] = invalid[-1]["timestamp"] - pd.Timedelta(days=1)
        self.assertEqual(model.train_walk_forward(invalid), {})
        self.assertEqual((model.model, model.scaler, model.metadata), previous)
        self.assertEqual(model.train_walk_forward([dict(row, label=0) for row in samples()]), {})
        self.assertEqual((model.model, model.scaler, model.metadata), previous)

    def test_partial_time_metadata_cannot_fall_back_to_positional_splits(self):
        rows = [{"label": i % 2, "label_end_time": "2025-01-01"} for i in range(60)]
        self.assertEqual(ml_model.SignalMLModel("lr").train_walk_forward(rows), {})

    def test_single_class_validation_does_not_report_fabricated_auc(self):
        rows = [{"rsi": i % 2, "label": i % 2 if i < 30 else 1} for i in range(90)]
        model = ml_model.SignalMLModel("lr")
        metrics = model.train_walk_forward(rows, n_splits=2, purge_gap=0)
        self.assertIn("f1", metrics)
        self.assertNotIn("roc_auc", metrics)
        self.assertEqual(model.metadata.n_samples, 90)

    def test_ensemble_keeps_previous_global_when_training_has_no_valid_folds(self):
        ensemble = ml_model.RegimeAwareModelEnsemble("lr")
        previous = ml_model.SignalMLModel("lr")
        ensemble.global_model = previous
        with patch.object(ml_model.SignalMLModel, "save") as save:
            result = ensemble.train({MarketRegime.UNKNOWN: [dict(row, label=0) for row in samples()]})
        self.assertEqual(result, {})
        self.assertIs(ensemble.global_model, previous)
        save.assert_not_called()

    def test_best_model_handles_zero_f1_and_missing_metrics(self):
        ensemble = ml_model.RegimeAwareModelEnsemble("lr")
        model = ml_model.SignalMLModel("lr")
        model.metadata = ml_model.ModelMetadata("lr", "all", "", 90, 15, {})
        ensemble.global_model = model
        self.assertIsNone(ensemble.get_best_model())
        model.metadata.metrics["f1"] = 0.0
        self.assertIs(ensemble.get_best_model(), model)


class ModelBundleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.model_dir = Path(directory.name) / "models"
        patcher = patch.object(ml_model, "MODEL_DIR", self.model_dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def train(self, **kwargs):
        ensemble = ml_model.RegimeAwareModelEnsemble("lr")
        self.assertIn("global", ensemble.train({MarketRegime.BULL_TRENDING: samples()}, **kwargs))
        return ensemble

    def reload_engine(self):
        with patch.object(signal_engine, "ML_MODEL_TYPE", "lr"), \
             patch.object(signal_engine, "_ml_ensemble", None), \
             patch.object(signal_engine, "_ml_models_loaded", False):
            ensemble = signal_engine._get_ml_ensemble()
            return ensemble, signal_engine._ml_models_loaded

    def test_restart_uses_exact_active_set_without_resurrecting_stale_models(self):
        ensemble = self.train()
        old_manifest = json.loads((self.model_dir / "active.json").read_text())
        legacy = ensemble.models[MarketRegime.BULL_TRENDING].save(self.model_dir / "legacy.pkl")
        self.assertIn(MarketRegime.BULL_TRENDING, ensemble.models)

        # A smaller training universe has no eligible regime-specific model.
        self.assertIn("global", ensemble.train({MarketRegime.BULL_TRENDING: samples()},
                                               min_samples_per_regime=1000))
        reloaded, loaded = self.reload_engine()
        self.assertTrue(loaded)
        self.assertEqual(ensemble.models, {})
        self.assertEqual(reloaded.models, {})
        for regime in MarketRegime:
            features = SignalFeatures(rsi=70, regime=regime)
            self.assertEqual(ensemble.predict(features), reloaded.predict(features))
        self.assertTrue(legacy.exists())
        self.assertTrue((self.model_dir / old_manifest["bundle"] / "global.pkl").exists())

    def test_validation_training_does_not_create_or_modify_model_files(self):
        ensemble = self.train(save_models=False)
        self.assertFalse(self.model_dir.exists())
        ensemble = self.train()
        before = {path.relative_to(self.model_dir): path.read_bytes()
                  for path in self.model_dir.rglob("*") if path.is_file()}
        ensemble.train({MarketRegime.UNKNOWN: samples()}, save_models=False)
        after = {path.relative_to(self.model_dir): path.read_bytes()
                 for path in self.model_dir.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_unsuccessful_global_training_preserves_active_bundle_and_memory(self):
        ensemble = self.train()
        previous = ensemble.global_model, ensemble.models
        manifest = (self.model_dir / "active.json").read_bytes()
        with patch.object(ml_model.SignalMLModel, "save") as save:
            result = ensemble.train({MarketRegime.UNKNOWN: [dict(row, label=0) for row in samples()]})
        self.assertEqual(result, {})
        save.assert_not_called()
        self.assertIs(ensemble.global_model, previous[0])
        self.assertIs(ensemble.models, previous[1])
        self.assertEqual((self.model_dir / "active.json").read_bytes(), manifest)

    def test_failed_file_write_or_manifest_replace_keeps_previous_active_bundle(self):
        save = ml_model.SignalMLModel.save
        def fail_regime_write(model, path=None):
            if model.regime is not None:
                raise OSError("regime write failed")
            return save(model, path)

        for failure in ("file", "manifest"):
            with self.subTest(failure=failure):
                ensemble = self.train()
                previous = ensemble.global_model, ensemble.models
                manifest = (self.model_dir / "active.json").read_bytes()
                patcher = (patch.object(ml_model.SignalMLModel, "save", autospec=True, side_effect=fail_regime_write)
                           if failure == "file" else
                           patch.object(ml_model.os, "replace", side_effect=OSError("replace failed")))
                with patcher, self.assertRaises(OSError):
                    ensemble.train({MarketRegime.UNKNOWN: samples()})
                self.assertIs(ensemble.global_model, previous[0])
                self.assertIs(ensemble.models, previous[1])
                self.assertEqual((self.model_dir / "active.json").read_bytes(), manifest)
                reloaded = ml_model.RegimeAwareModelEnsemble.load_saved("lr")
                self.assertEqual(set(reloaded.models), set(ensemble.models))
                self.assertEqual(reloaded.predict(SignalFeatures(rsi=70)),
                                 ensemble.predict(SignalFeatures(rsi=70)))

    def test_invalid_bundle_does_not_fall_back_to_obsolete_legacy_files(self):
        ensemble = self.train()
        ensemble.global_model.save(self.model_dir / "legacy.pkl")
        manifest_path = self.model_dir / "active.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["models"]["global"] = "missing.pkl"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            ml_model.RegimeAwareModelEnsemble.load_saved("lr")
        reloaded, loaded = self.reload_engine()
        self.assertFalse(loaded)
        self.assertEqual(reloaded.predict(SignalFeatures()), (0.5, "no_model"))

    def test_existing_standalone_bist_models_load_without_a_manifest(self):
        ensemble = self.train(save_models=False)
        ensemble.global_model.save(self.model_dir / "lr_global.pkl")
        ensemble.models[MarketRegime.BULL_TRENDING].save(self.model_dir / "lr_bull.pkl")
        ml_model.SignalMLModel("lr", MarketRegime.UNKNOWN).save(self.model_dir / "unfitted.pkl")
        reloaded, loaded = self.reload_engine()
        self.assertTrue(loaded)
        self.assertEqual(set(reloaded.models), {MarketRegime.BULL_TRENDING})
        self.assertEqual(reloaded.predict(SignalFeatures(rsi=70)), ensemble.predict(SignalFeatures(rsi=70)))
        self.assertFalse((self.model_dir / "active.json").exists())


class TrainingDataTests(unittest.IsolatedAsyncioTestCase):
    async def test_preparation_uses_signal_engine_without_importing_bot_startup(self):
        groups = {MarketRegime.UNKNOWN: samples()}
        data = bars()
        with patch.dict("sys.modules", {"bot": None}), \
             patch.object(ml_model, "fetch_bist_history", return_value=data), \
             patch.object(ml_model, "build_bist_samples", return_value=groups) as build:
            result = await ml_model.prepare_training_data(["TEST.IS"])
        self.assertEqual(result[MarketRegime.UNKNOWN], groups[MarketRegime.UNKNOWN])
        self.assertIs(build.call_args.args[2], signal_engine.detect_buy_signals)

    async def test_training_rejects_non_bist_universes_before_downloading(self):
        stub = types.ModuleType("bot")
        stub.detect_buy_signals = Mock()
        with patch.dict("sys.modules", {"bot": stub}), patch.object(ml_model, "fetch_bist_history") as fetch:
            for symbols, market in ((["BTC/USDT"], "crypto"), (["AAPL"], "stock")):
                with self.assertRaises(ValueError):
                    await ml_model.prepare_training_data(symbols, market)
            fetch.assert_not_called()

    async def prepare(self, data, **kwargs):
        detector = Mock(return_value={"signals": ["test"], "sl": 98, "tp1": 103, "tp2": 106})
        with patch.object(ml_model, "MIN_TRAINING_BARS", 50):
            result = ml_model.build_bist_samples(data, "TEST.IS", detector, **kwargs)
        self.assertTrue(all(call.kwargs["use_ml"] is False for call in detector.call_args_list))
        return [sample for bucket in result.values() for sample in bucket]

    async def test_features_timestamps_full_horizon_and_last_finished_bar_align(self):
        data = bars(55)
        with patch.object(ml_model, "extract_features", return_value=SignalFeatures(rsi=0, volume_ratio=0)) as extract:
            rows = await self.prepare(data, horizon=3)
        self.assertEqual(len(rows), 2)
        self.assertEqual([len(call.args[0]) for call in extract.call_args_list], [50, 51])
        self.assertTrue(all(call.kwargs["analyze_completed_only"] is False for call in extract.call_args_list))
        self.assertEqual([row["timestamp"] for row in rows], [data.index[49], data.index[50]])
        self.assertEqual([row["label_end_time"] for row in rows], [data.index[52], data.index[53]])
        self.assertEqual((rows[0]["rsi"], rows[0]["volume_ratio"]), (0, 0))

    async def test_target_precedes_later_stop_and_entry_uses_next_open(self):
        data = bars(54)
        data.iloc[50, data.columns.get_loc("high")] = 104.0
        data.iloc[50, data.columns.get_loc("close")] = 103.0
        data.iloc[51, data.columns.get_loc("low")] = 97.0
        rows = await self.prepare(data, horizon=3)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["label"], 1)

    async def test_stop_wins_when_both_barriers_are_hit_in_same_candle(self):
        data = bars(54)
        data.iloc[50, data.columns.get_loc("high")] = 104.0
        data.iloc[50, data.columns.get_loc("low")] = 97.0
        self.assertEqual((await self.prepare(data, horizon=3))[0]["label"], 0)

    async def test_without_open_entry_uses_last_observed_close(self):
        data = bars(54).drop(columns="open")
        data.iloc[50, data.columns.get_loc("close")] = 103.0
        data.iloc[50, data.columns.get_loc("high")] = 104.0
        self.assertEqual((await self.prepare(data, horizon=3))[0]["label"], 1)

    async def test_unfinished_outcome_and_nonfinite_prices_cannot_label_samples(self):
        data = bars(54)
        data.iloc[-1, data.columns.get_loc("high")] = 1000.0
        rows = await self.prepare(data, horizon=3)
        self.assertEqual(rows[0]["label"], 0)
        data.iloc[50, data.columns.get_loc("low")] = np.nan
        self.assertEqual(await self.prepare(data, horizon=3), [])

    async def test_lookback_filters_decisions_but_retains_indicator_warmup(self):
        data = bars(60)
        rows = await self.prepare(data, horizon=1, lookback_days=3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["timestamp"], data.index[55])
        self.assertIsNotNone(rows[0]["rsi"])

    async def test_undated_data_is_rejected_instead_of_inventing_training_times(self):
        with self.assertRaises(ValueError):
            await self.prepare(bars(100).reset_index(drop=True), horizon=3)


if __name__ == "__main__":
    unittest.main()
