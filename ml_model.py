"""ML Model for signal scoring: walk-forward training, regime-aware models, feature engineering."""

from __future__ import annotations

import json
import logging
import pickle
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from signals_advanced import SignalFeatures, MarketRegime, extract_features

warnings.filterwarnings("ignore", category=UserWarning)
logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLUMNS = [
    "adx", "di_plus", "di_minus", "rsi", "macd_hist",
    "atr_pct", "bb_width", "bb_position",
    "volume_ratio", "obv_trend_encoded",
    "vwap_distance", "regime_encoded",
    "higher_highs", "lower_lows", "trend_strength",
]

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
        """Convert SignalFeatures list or dict list to X, y arrays."""
        X_rows = []
        y_rows = []

        for f in features_list:
            # Handle both SignalFeatures objects and dicts
            if isinstance(f, dict):
                get = f.get
                adx = get("adx", 0) or 0
                di_plus = get("di_plus", 0) or 0
                di_minus = get("di_minus", 0) or 0
                rsi = get("rsi", 50) or 50
                macd_hist = get("macd_hist", 0) or 0
                atr_pct = get("atr_pct", 0) or 0
                bb_width = get("bb_width", 0) or 0
                bb_position = get("bb_position", 0.5) or 0.5
                volume_ratio = get("volume_ratio", 1) or 1
                obv_trend = get("obv_trend_encoded", 0) or 0
                vwap_distance = get("vwap_distance", 0) or 0
                regime_encoded = get("regime_encoded", 0) or 0
                higher_highs = get("higher_highs", 0) or 0
                lower_lows = get("lower_lows", 0) or 0
                trend_strength = get("trend_strength", 0) or 0
                label = get("label", 0) or 0
            else:
                adx = f.adx or 0
                di_plus = f.di_plus or 0
                di_minus = f.di_minus or 0
                rsi = f.rsi or 50
                macd_hist = f.macd_hist or 0
                atr_pct = f.atr_pct or 0
                bb_width = f.bb_width or 0
                bb_position = f.bb_position or 0.5
                volume_ratio = f.volume_ratio or 1
                obv_trend = 1 if f.obv_trend == "UP" else 0
                vwap_distance = f.vwap_distance or 0
                regime_encoded = list(MarketRegime).index(f.regime)
                higher_highs = f.higher_highs
                lower_lows = f.lower_lows
                trend_strength = f.trend_strength
                label = 1 if f.rsi and f.rsi < 40 and f.macd_hist and f.macd_hist > 0 else 0

            row = {
                "adx": adx,
                "di_plus": di_plus,
                "di_minus": di_minus,
                "rsi": rsi,
                "macd_hist": macd_hist,
                "atr_pct": atr_pct,
                "bb_width": bb_width,
                "bb_position": bb_position,
                "volume_ratio": volume_ratio,
                "obv_trend_encoded": obv_trend,
                "vwap_distance": vwap_distance,
                "regime_encoded": regime_encoded,
                "higher_highs": higher_highs,
                "lower_lows": lower_lows,
                "trend_strength": trend_strength,
            }
            X_rows.append([row[c] for c in FEATURE_COLUMNS])
            y_rows.append(label)

        return np.array(X_rows), np.array(y_rows)

    def train_walk_forward(
        self,
        features_list: list[SignalFeatures],
        n_splits: int = 5,
        purge_gap: int = 5,
    ) -> dict[str, float]:
        """Walk-forward training with purging to prevent leakage."""
        X, y = self._encode_features(features_list)

        if len(X) < 50:
            logger.warning(f"⚠️ Yetersiz veri: {len(X)} örnek (min 50 gerekli)")
            return {}

        tscv = TimeSeriesSplit(n_splits=n_splits, gap=purge_gap)
        scores = {"precision": [], "recall": [], "f1": [], "roc_auc": []}

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if len(np.unique(y_train)) < 2:
                continue

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            self.model.fit(X_train_scaled, y_train)
            y_pred = self.model.predict(X_test_scaled)
            y_proba = self.model.predict_proba(X_test_scaled)[:, 1] if hasattr(self.model, "predict_proba") else y_pred

            scores["precision"].append(precision_score(y_test, y_pred, zero_division=0))
            scores["recall"].append(recall_score(y_test, y_pred, zero_division=0))
            scores["f1"].append(f1_score(y_test, y_pred, zero_division=0))
            try:
                scores["roc_auc"].append(roc_auc_score(y_test, y_proba))
            except Exception:
                scores["roc_auc"].append(0.5)

            logger.info(f"  Fold {fold+1}: P={scores['precision'][-1]:.3f} R={scores['recall'][-1]:.3f} F1={scores['f1'][-1]:.3f}")

        avg_scores = {k: round(np.mean(v), 4) for k, v in scores.items() if v}

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

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

    def predict_proba(self, features: SignalFeatures) -> float:
        """Predict probability of successful signal."""
        if self.model is None:
            return 0.5

        row = np.array([[
            features.adx or 0,
            features.di_plus or 0,
            features.di_minus or 0,
            features.rsi or 50,
            features.macd_hist or 0,
            features.atr_pct or 0,
            features.bb_width or 0,
            features.bb_position or 0.5,
            features.volume_ratio or 1,
            1 if features.obv_trend == "UP" else 0,
            features.vwap_distance or 0,
            list(MarketRegime).index(features.regime),
            features.higher_highs,
            features.lower_lows,
            features.trend_strength,
        ]])

        row_scaled = self.scaler.transform(row)

        if hasattr(self.model, "predict_proba"):
            return float(self.model.predict_proba(row_scaled)[0, 1])
        return float(self.model.predict(row_scaled)[0])

    def save(self, path: Optional[Path] = None) -> Path:
        """Save model and metadata."""
        if path is None:
            regime_str = self.regime.value if self.regime else "global"
            path = MODEL_DIR / f"{self.model_type}_{regime_str}.pkl"

        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "metadata": self.metadata,
                "model_type": self.model_type,
                "regime": self.regime.value if self.regime else None,
            }, f)

        meta_path = path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
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
        features_by_regime: dict[MarketRegime, list[SignalFeatures]],
        min_samples_per_regime: int = 30,
    ) -> dict[str, dict]:
        """Train regime-specific models and a global fallback."""
        results = {}

        for regime, features in features_by_regime.items():
            if len(features) < min_samples_per_regime:
                logger.info(f"⚠️ {regime.value}: yetersiz örnek ({len(features)}), global modele dahil edilecek")
                continue

            model = SignalMLModel(self.model_type, regime)
            metrics = model.train_walk_forward(features)
            if metrics:
                model.save()
                self.models[regime] = model
                results[regime.value] = metrics

        all_features = [f for feats in features_by_regime.values() for f in feats]
        if len(all_features) >= 50:
            self.global_model = SignalMLModel(self.model_type, None)
            metrics = self.global_model.train_walk_forward(all_features)
            if metrics:
                self.global_model.save()
                results["global"] = metrics

        return results

    def predict(self, features: SignalFeatures) -> tuple[float, str]:
        """Predict using regime-specific model or global fallback."""
        model = self.models.get(features.regime)
        model_used = "regime_specific"

        if model is None:
            model = self.global_model
            model_used = "global"

        if model is None:
            return 0.5, "no_model"

        return model.predict_proba(features), model_used

    def get_best_model(self) -> Optional[SignalMLModel]:
        """Return the model with best F1 score."""
        best_model = None
        best_f1 = 0.0

        for model in list(self.models.values()) + ([self.global_model] if self.global_model else []):
            if model and model.metadata and model.metadata.metrics.get("f1", 0) > best_f1:
                best_f1 = model.metadata.metrics["f1"]
                best_model = model

        return best_model


async def prepare_training_data(
    symbols: list[str],
    market: str,
    lookback_days: int = 365,
    horizon: int = 20,
) -> dict[MarketRegime, list[SignalFeatures]]:
    """Prepare labeled training data from historical price data."""
    from bot import fetch_stock_data_async, fetch_crypto_data, detect_buy_signals

    features_by_regime: dict[MarketRegime, list[SignalFeatures]] = {r: [] for r in MarketRegime}

    async def _collect():
        for symbol in symbols:
            try:
                if market == "crypto":
                    data = await fetch_crypto_data(symbol)
                    if not data:
                        continue
                    df = data["ohlcv_df"]
                else:
                    data = await fetch_stock_data_async(symbol)
                    if not data:
                        continue
                    df = data["history_df"]

                if len(df) < 100:
                    continue

                n = len(df)
                for i in range(50, n - horizon):
                    window = df.iloc[:i]
                    features = extract_features(window, symbol, market)

                    future = df.iloc[i:i+horizon]
                    entry = features.rsi  # dummy, will use close
                    close_col = "close" if "close" in future.columns else "Close"
                    entry_price = future[close_col].iloc[0]
                    max_high = future["high" if "high" in future.columns else "High"].max()
                    min_low = future["low" if "low" in future.columns else "Low"].min()

                    tp_hit = (max_high - entry_price) / entry_price > 0.03
                    sl_hit = (entry_price - min_low) / entry_price > 0.02

                    label = 1 if tp_hit and not sl_hit else 0

                    features_dict = {
                        "adx": features.adx or 0,
                        "di_plus": features.di_plus or 0,
                        "di_minus": features.di_minus or 0,
                        "rsi": features.rsi or 50,
                        "macd_hist": features.macd_hist or 0,
                        "atr_pct": features.atr_pct or 0,
                        "bb_width": features.bb_width or 0,
                        "bb_position": features.bb_position or 0.5,
                        "volume_ratio": features.volume_ratio or 1,
                        "obv_trend_encoded": 1 if features.obv_trend == "UP" else 0,
                        "vwap_distance": features.vwap_distance or 0,
                        "regime_encoded": list(MarketRegime).index(features.regime),
                        "higher_highs": features.higher_highs,
                        "lower_lows": features.lower_lows,
                        "trend_strength": features.trend_strength,
                        "label": label,
                    }

                    features_by_regime[features.regime].append(features_dict)

            except Exception as e:
                logger.warning(f"⚠️ {symbol} veri hatası: {e}")

    await _collect()

    return features_by_regime