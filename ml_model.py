"""ML Model for signal scoring: walk-forward training, regime-aware models, feature engineering."""

from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from signals_advanced import SignalFeatures, MarketRegime, extract_features
from market_data import completed_candles, fetch_bist_history
from paper import COST_RATE, HORIZON_BARS

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models") / "bist_v2"
MODEL_SCHEMA_VERSION = 2
BUNDLE_MANIFEST_VERSION = 1
ACTIVE_MODEL_MANIFEST = "active.json"
BIST_HORIZON = HORIZON_BARS["stock"]
MIN_TRAINING_BARS = 201

FEATURE_COLUMNS = [
    "adx", "di_plus", "di_minus", "rsi", "macd_hist_pct",
    "atr_pct", "bb_width", "bb_position",
    "volume_ratio", "obv_trend_encoded",
    "vwap_distance", "regime_encoded",
    "higher_highs", "lower_lows", "trend_strength",
]

FEATURE_DEFAULTS = {column: 0.0 for column in FEATURE_COLUMNS}
FEATURE_DEFAULTS.update(rsi=50.0, bb_position=0.5, volume_ratio=1.0,
                        regime_encoded=float(list(MarketRegime).index(MarketRegime.UNKNOWN)))


def _feature_row(features: SignalFeatures | dict[str, Any]) -> list[float]:
    """Use identical, finite inputs for training and inference, retaining valid zeros."""
    get = features.get if isinstance(features, dict) else lambda name, default=None: getattr(features, name, default)
    regime = get("regime", MarketRegime.UNKNOWN)
    try:
        regime_index = list(MarketRegime).index(MarketRegime(regime))
    except (ValueError, TypeError):
        regime_index = list(MarketRegime).index(MarketRegime.UNKNOWN)
    encoded = {
        "obv_trend_encoded": get("obv_trend_encoded", 1 if get("obv_trend") == "UP" else 0),
        "regime_encoded": get("regime_encoded", regime_index),
    }
    row = []
    for column in FEATURE_COLUMNS:
        value = encoded.get(column, get(column))
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = FEATURE_DEFAULTS[column]
        if not np.isfinite(value):
            value = FEATURE_DEFAULTS[column]
        row.append(value)
    return row

REGIME_MODELS = {
    MarketRegime.BULL_TRENDING: "bull_trending",
    MarketRegime.BULL_RANGING: "bull_ranging",
    MarketRegime.BEAR_TRENDING: "bear_trending",
    MarketRegime.BEAR_RANGING: "bear_ranging",
    MarketRegime.HIGH_VOLATILITY: "high_vol",
    MarketRegime.LOW_VOLATILITY: "low_vol",
    MarketRegime.UNKNOWN: "unknown",
}


@dataclass
class ModelMetadata:
    model_type: str
    regime: str
    trained_at: str
    n_samples: int
    n_features: int
    metrics: dict[str, float]
    feature_importance: dict[str, float] = field(default_factory=dict)


class SignalMLModel:
    """Walk-forward trained ML model for signal quality prediction."""

    def __init__(self, model_type: str = "rf", regime: Optional[MarketRegime] = None):
        self.model_type = model_type
        self.regime = regime
        self.model: Any = None
        self.scaler = StandardScaler()
        self.metadata: Optional[ModelMetadata] = None
        self._init_model()

    def _init_model(self):
        if self.model_type == "rf":
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=8,
                min_samples_split=20,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gb":
            self.model = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=5,
                learning_rate=0.05,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
            )
        elif self.model_type == "lr":
            self.model = LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Bilinmeyen model tipi: {self.model_type}")

    def _encode_features(self, features_list: list) -> tuple[np.ndarray, np.ndarray]:
        """Encode inputs; training separately requires explicit outcome labels."""
        rows = [_feature_row(features) for features in features_list]
        labels = [features.get("label", 0) if isinstance(features, dict) else 0
                  for features in features_list]
        return np.asarray(rows, dtype=float).reshape(-1, len(FEATURE_COLUMNS)), np.asarray(labels)

    @staticmethod
    def _walk_forward_splits(features_list: list[dict[str, Any]], n_splits: int, purge_gap: int):
        """Keep simultaneous samples together and purge unresolved training outcomes."""
        timestamped = any("timestamp" in sample or "label_end_time" in sample for sample in features_list)
        if not timestamped:
            yield from TimeSeriesSplit(n_splits=n_splits, gap=purge_gap).split(features_list)
            return
        if not all("timestamp" in sample and "label_end_time" in sample for sample in features_list):
            raise ValueError("Zaman damgalı eğitim için timestamp ve label_end_time gereklidir")
        timestamps = pd.to_datetime([sample["timestamp"] for sample in features_list], utc=True)
        label_end_times = pd.to_datetime([sample["label_end_time"] for sample in features_list], utc=True)
        if timestamps.isna().any() or label_end_times.isna().any() or (label_end_times < timestamps).any():
            raise ValueError("Geçersiz eğitim zaman aralığı")
        unique_times = timestamps.unique().sort_values()
        for train_times, test_times in TimeSeriesSplit(n_splits=n_splits, gap=purge_gap).split(unique_times):
            test_start = unique_times[test_times[0]]
            train_idx = np.flatnonzero(timestamps.isin(unique_times[train_times]) & (label_end_times < test_start))
            test_idx = np.flatnonzero(timestamps.isin(unique_times[test_times]))
            yield train_idx, test_idx

    def train_walk_forward(
        self,
        features_list: list[dict[str, Any]],
        n_splits: int = 5,
        purge_gap: int = 5,
    ) -> dict[str, float]:
        """Validate on future observations and exclude overlapping outcome windows.

        Explicit binary outcome labels are mandatory. For undated caller-supplied
        rows, preserve their chronological order and set purge_gap to at least
        their forward label horizon.
        """
        if n_splits < 2 or purge_gap < 0:
            raise ValueError("n_splits en az 2, purge_gap en az 0 olmalıdır")
        if any(not isinstance(sample, dict) or sample.get("label") not in (0, 1)
               for sample in features_list):
            raise ValueError("Eğitim örnekleri gerçekleşmiş sonuca dayanan 0/1 label içermelidir")
        X, y = self._encode_features(features_list)

        if len(X) < 50:
            logger.warning(f"⚠️ Yetersiz veri: {len(X)} örnek (min 50 gerekli)")
            return {}

        if len(np.unique(y)) < 2:
            logger.warning("Eğitim için hem başarılı hem başarısız örnekler gereklidir")
            return {}

        # Invalid or undersized splits must not leave a partially retrained model.
        try:
            splits = list(self._walk_forward_splits(features_list, n_splits, purge_gap))
        except ValueError as exc:
            logger.warning("Walk-forward bölünemedi: %s", exc)
            return {}
        scores = {"precision": [], "recall": [], "f1": [], "roc_auc": []}

        for fold, (train_idx, test_idx) in enumerate(splits):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2:
                continue

            fold_scaler = StandardScaler()
            fold_model = clone(self.model)
            X_train_scaled = fold_scaler.fit_transform(X_train)
            X_test_scaled = fold_scaler.transform(X_test)

            fold_model.fit(X_train_scaled, y_train)
            y_pred = fold_model.predict(X_test_scaled)
            y_proba = fold_model.predict_proba(X_test_scaled)[:, 1] if hasattr(fold_model, "predict_proba") else y_pred

            scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
            scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
            scores["f1"].append(f1_score(y_test, y_pred, zero_division=0))
            # AUC is undefined for one-class validation sets, not a measured 0.5.
            if len(np.unique(y_test)) == 2:
                scores["roc_auc"].append(roc_auc_score(y_test, y_proba))

            logger.info(f"  Fold {fold+1}: P={scores['precision'][-1]:.3f} R={scores['recall'][-1]:.3f} F1={scores['f1'][-1]:.3f}")

        avg_scores = {k: round(float(np.mean(v)), 4) for k, v in scores.items() if v}
        if not avg_scores:
            logger.warning("Yeterli sınıf çeşitliliği olan doğrulama bölümü bulunamadı")
            return {}

        scaler = StandardScaler()
        model = clone(self.model)
        X_scaled = scaler.fit_transform(X)
        model.fit(X_scaled, y)
        self.scaler, self.model = scaler, model

        if hasattr(self.model, "feature_importances_"):
            importance = dict(zip(FEATURE_COLUMNS, self.model.feature_importances_))
        elif hasattr(self.model, "coef_"):
            importance = dict(zip(FEATURE_COLUMNS, np.abs(self.model.coef_[0])))
        else:
            importance = {}

        self.metadata = ModelMetadata(
            model_type=self.model_type,
            regime=self.regime.value if self.regime else "all",
            trained_at=datetime.now().isoformat(),
            n_samples=len(X),
            n_features=len(FEATURE_COLUMNS),
            metrics=avg_scores,
            feature_importance=importance,
        )

        logger.info(f"✅ Model eğitildi ({self.regime.value if self.regime else 'global'}): {avg_scores}")
        return avg_scores

    def is_fitted(self) -> bool:
        """Whether both model and preprocessing are ready for inference."""
        if self.model is None:
            return False
        try:
            check_is_fitted(self.model)
            check_is_fitted(self.scaler)
        except NotFittedError:
            return False
        return True

    def predict_proba(self, features: SignalFeatures) -> float:
        """Predict signal probability, returning neutral confidence until fitted."""
        if not self.is_fitted():
            return 0.5
        row = np.asarray([_feature_row(features)], dtype=float)
        row_scaled = self.scaler.transform(row)
        if hasattr(self.model, "predict_proba"):
            positive = np.flatnonzero(self.model.classes_ == 1)
            if not len(positive):
                return 0.0
            return float(self.model.predict_proba(row_scaled)[0, positive[0]])
        return float(self.model.predict(row_scaled)[0])

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model and metadata."""
        if path is None:
            regime_str = self.regime.value if self.regime else "global"
            path = MODEL_DIR / f"{self.model_type}_{regime_str}.pkl"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            pickle.dump({
                "schema_version": MODEL_SCHEMA_VERSION,
                "training_scope": "BIST",
                "model": self.model,
                "scaler": self.scaler,
                "metadata": self.metadata,
                "model_type": self.model_type,
                "regime": self.regime.value if self.regime else None,
            }, f)

        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "schema_version": MODEL_SCHEMA_VERSION,
                "training_scope": "BIST",
                "target": "technical_tp1_before_stop",
                "feature_columns": FEATURE_COLUMNS,
                "model_type": self.model_type,
                "regime": self.regime.value if self.regime else "all",
                "trained_at": self.metadata.trained_at if self.metadata else None,
                "metrics": self.metadata.metrics if self.metadata else {},
                "feature_importance": self.metadata.feature_importance if self.metadata else {},
            }, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Model kaydedildi: {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "SignalMLModel":
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        if data.get("schema_version") != MODEL_SCHEMA_VERSION or data.get("training_scope") != "BIST":
            raise ValueError("Model BIST özellik/hedef sürümüyle uyumlu değil; yeniden eğitim gerekli")

        instance = cls(data["model_type"], MarketRegime(data["regime"]) if data["regime"] else None)
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.metadata = data["metadata"]
        logger.info(f"📂 Model yüklendi: {path}")
        return instance


class RegimeAwareModelEnsemble:
    """Ensemble of regime-specific models."""

    def __init__(self, model_type: str = "rf"):
        self.model_type = model_type
        self.models: dict[MarketRegime, SignalMLModel] = {}
        self.global_model: Optional[SignalMLModel] = None

    def train(
        self,
        features_by_regime: dict[MarketRegime, list[dict[str, Any]]],
        min_samples_per_regime: int = 30,
        *, save_models: bool = True,
    ) -> dict[str, dict]:
        """Replace the active set only after a complete, successful training run."""
        all_features = [f for feats in features_by_regime.values() for f in feats]
        if len(all_features) < 50:
            return {}
        global_model = SignalMLModel(self.model_type, None)
        global_metrics = global_model.train_walk_forward(all_features)
        if not global_metrics:
            return {}

        results = {"global": global_metrics}
        models = {}

        for regime, features in features_by_regime.items():
            if len(features) < min_samples_per_regime:
                logger.info(f"⚠️ {regime.value}: yetersiz örnek ({len(features)}), global modele dahil edilecek")
                continue

            model = SignalMLModel(self.model_type, regime)
            metrics = model.train_walk_forward(features)
            if metrics:
                models[regime] = model
                results[regime.value] = metrics

        # Publishing may fail (for example, a full disk). Keep both the old
        # manifest and the in-memory ensemble in that case.
        if save_models:
            self._publish_bundle(global_model, models)
        self.global_model, self.models = global_model, models
        return results

    def _publish_bundle(self, global_model, models) -> None:
        """Write immutable files, then atomically point readers at the full set."""
        model_dir = Path(MODEL_DIR)
        model_dir.mkdir(parents=True, exist_ok=True)
        bundle_dir = Path(tempfile.mkdtemp(prefix="bundle-", dir=model_dir))
        files = {"global": "global.pkl"}
        global_model.save(bundle_dir / files["global"])
        for regime, model in models.items():
            files[regime.value] = f"{regime.value}.pkl"
            model.save(bundle_dir / files[regime.value])
        manifest = {
            "manifest_version": BUNDLE_MANIFEST_VERSION,
            "schema_version": MODEL_SCHEMA_VERSION,
            "training_scope": "BIST",
            "feature_columns": FEATURE_COLUMNS,
            "model_type": self.model_type,
            "bundle": bundle_dir.name,
            "models": files,
        }
        # A unique temporary manifest also makes concurrent publishers safe:
        # each successful replacement refers to one whole training run.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=model_dir,
                                         prefix=".active-", suffix=".json", delete=False) as handle:
            temporary_manifest = Path(handle.name)
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary_manifest, model_dir / ACTIVE_MODEL_MANIFEST)
        finally:
            temporary_manifest.unlink(missing_ok=True)

    @classmethod
    def load_saved(cls, model_type: str = "rf", model_dir: Optional[Path] = None):
        """Load exactly the published set; use legacy files only without a manifest.

        An invalid published bundle is an error, never a reason to resurrect
        obsolete standalone models. Existing files are not deleted or rewritten.
        """
        model_dir = Path(MODEL_DIR if model_dir is None else model_dir)
        ensemble = cls(model_type)
        manifest_path = model_dir / ACTIVE_MODEL_MANIFEST
        if not manifest_path.exists():
            for path in sorted(model_dir.glob("*.pkl")):
                try:
                    model = SignalMLModel.load(path)
                    if model.model_type != model_type or not model.is_fitted():
                        continue
                    if model.regime is None:
                        ensemble.global_model = model
                    else:
                        ensemble.models[model.regime] = model
                except Exception as exc:
                    logger.warning("Model yüklenemedi %s: %s", path, exc)
            return ensemble

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (manifest.get("manifest_version") != BUNDLE_MANIFEST_VERSION
                or manifest.get("schema_version") != MODEL_SCHEMA_VERSION
                or manifest.get("training_scope") != "BIST"
                or manifest.get("feature_columns") != FEATURE_COLUMNS
                or manifest.get("model_type") != model_type):
            raise ValueError("Aktif model paketi BIST model yapılandırmasıyla uyumsuz")
        bundle = manifest.get("bundle")
        files = manifest.get("models")
        if not isinstance(bundle, str) or not isinstance(files, dict) or "global" not in files:
            raise ValueError("Aktif model paketi eksik")
        bundle_dir = (model_dir / bundle).resolve()
        if bundle_dir.parent != model_dir.resolve():
            raise ValueError("Geçersiz model paketi yolu")
        for regime_name, filename in files.items():
            regime = None if regime_name == "global" else MarketRegime(regime_name)
            if not isinstance(filename, str):
                raise ValueError("Geçersiz model dosyası")
            path = (bundle_dir / filename).resolve()
            if path.parent != bundle_dir:
                raise ValueError("Geçersiz model dosyası yolu")
            model = SignalMLModel.load(path)
            if model.regime != regime or model.model_type != model_type or not model.is_fitted():
                raise ValueError("Aktif model paketi içeriği manifest ile uyuşmuyor")
            if regime is None:
                ensemble.global_model = model
            else:
                ensemble.models[regime] = model
        return ensemble

    def predict(self, features: SignalFeatures) -> tuple[float, str]:
        """Predict using regime-specific model or global fallback."""
        model = self.models.get(features.regime)
        model_used = "regime_specific"

        if model is None or not model.is_fitted():
            model = self.global_model
            model_used = "global"

        if model is None or not model.is_fitted():
            return 0.5, "no_model"

        return model.predict_proba(features), model_used

    def get_best_model(self) -> Optional[SignalMLModel]:
        """Return the model with best F1 score."""
        best_model = None
        best_f1 = -1.0

        for model in list(self.models.values()) + ([self.global_model] if self.global_model else []):
            if model and model.metadata and model.metadata.metrics.get("f1", -1) > best_f1:
                best_f1 = model.metadata.metrics["f1"]
                best_model = model

        return best_model


def build_bist_samples(df, symbol, signal_fn, *, lookback_days=1825, horizon=BIST_HORIZON):
    """Label only the same technical setups traded by the bot, with ML disabled.

    Prices and features end at the decision candle. The next open is the entry;
    targets and stops are the bot's actual ATR/support levels. A full future
    horizon is required and purged from validation boundaries.
    """
    if not symbol.upper().endswith(".IS"):
        raise ValueError("Model eğitimi yalnız BIST hisselerini kabul eder")
    if not isinstance(horizon, int) or horizon < 1 or not isinstance(lookback_days, int) or lookback_days < 1:
        raise ValueError("horizon ve lookback_days pozitif tam sayılar olmalıdır")
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.hasnans or not df.index.is_unique:
        raise ValueError("Eğitim için benzersiz mum zamanları gereklidir")
    df = completed_candles(df.sort_index()).copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df.columns = [str(column).lower() for column in df.columns]
    df.attrs["completed_only"] = True
    result = {regime: [] for regime in MarketRegime}
    if len(df) < MIN_TRAINING_BARS + horizon:
        return result
    cutoff = df.index[-1] - pd.Timedelta(days=lookback_days)
    for i in range(MIN_TRAINING_BARS, len(df) - horizon + 1):
        window = df.iloc[:i]
        timestamp = window.index[-1]
        if timestamp < cutoff:
            continue
        setup = signal_fn(window, market="stock", symbol=symbol, use_ml=False)
        if not setup.get("signals"):
            continue
        future = df.iloc[i:i+horizon]
        entry = float(future["open"].iloc[0] if "open" in future else window["close"].iloc[-1])
        sl, tp1, tp2 = (float(setup[key]) for key in ("sl", "tp1", "tp2"))
        prices = future[["high", "low", "close"]].to_numpy(dtype=float)
        if (not np.isfinite([entry, sl, tp1, tp2]).all() or not 0 < sl < entry < tp1 <= tp2
                or not np.isfinite(prices).all() or (prices <= 0).any()
                or (prices[:, 0] < prices[:, 1]).any()):
            continue
        label, exit_price, bars_held = 0, float(future["close"].iloc[-1]), horizon
        outcome = "MTM-TIMEOUT"
        for offset, (high, low, _) in enumerate(prices):
            if low <= sl:
                exit_price, bars_held, outcome = sl, offset + 1, "SL"
                break
            if high >= tp1:
                label, bars_held = 1, offset + 1
                exit_price, outcome = (tp2, "TP2") if high >= tp2 else (tp1, "TP1")
                break
        features = extract_features(window, symbol, "stock", analyze_completed_only=False)
        row = dict(zip(FEATURE_COLUMNS, _feature_row(features)))
        row.update(symbol=symbol, market="stock", timestamp=timestamp,
                   label_end_time=future.index[-1], label=label,
                   entry=entry, sl=sl, tp1=tp1, tp2=tp2, outcome=outcome,
                   exit_price=exit_price, bars_held=bars_held,
                   net_pnl_pct=(exit_price * (1 - COST_RATE) / (entry * (1 + COST_RATE)) - 1) * 100)
        result[features.regime].append(row)
    return result


async def prepare_training_data(symbols, market="stock", lookback_days=1825, horizon=BIST_HORIZON):
    """Download five years of BIST bars and prepare samples outside the event loop."""
    import asyncio
    from signal_engine import detect_buy_signals
    if market != "stock" or any(not symbol.upper().endswith(".IS") for symbol in symbols):
        raise ValueError("BIST modeline kripto veya yabancı hisse verisi eklenemez")
    result = {regime: [] for regime in MarketRegime}
    for symbol in symbols:
        try:
            data = await asyncio.to_thread(fetch_bist_history, symbol, "5y")
            rows = await asyncio.to_thread(build_bist_samples, data, symbol, detect_buy_signals,
                                          lookback_days=lookback_days, horizon=horizon)
            for regime, samples in rows.items():
                result[regime].extend(samples)
            logger.info("BIST eğitim verisi %s: %d aday", symbol, sum(map(len, rows.values())))
        except Exception as exc:
            logger.warning("%s BIST eğitim verisi hazırlanamadı: %s", symbol, exc)
    for rows in result.values():
        rows.sort(key=lambda row: row["timestamp"])
    return result
