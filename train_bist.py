"""Train and validate BIST technical signals without starting the Telegram bot.

Run: python train_bist.py --period 5y
History and reports are written to backtest_out/bist_training; candidate models
are versioned separately from the previously shipped mixed-market models.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from market_data import completed_bist_daily, fetch_bist_history
from ml_model import (BIST_HORIZON, FEATURE_COLUMNS, MODEL_DIR, RegimeAwareModelEnsemble,
                      SignalMLModel, build_bist_samples)
from signals_advanced import MarketRegime


def temporal_holdout(rows, fraction=0.2):
    """Hold out the latest dates and purge all unresolved training labels."""
    timestamps = sorted({pd.Timestamp(row["timestamp"]) for row in rows})
    if len(timestamps) < 20:
        raise ValueError("Bağımsız dönem testi için en az 20 farklı sinyal tarihi gerekli")
    cutoff = timestamps[int(len(timestamps) * (1 - fraction))]
    train = [row for row in rows if pd.Timestamp(row["label_end_time"]) < cutoff]
    test = [row for row in rows if pd.Timestamp(row["timestamp"]) >= cutoff]
    return train, test, cutoff


def evaluate_holdout(model, rows, threshold=0.55):
    """Report future-period results at the pre-existing bot threshold."""
    if isinstance(model, RegimeAwareModelEnsemble):
        probabilities = []
        for row in rows:
            regime = list(MarketRegime)[int(row["regime_encoded"])]
            estimator = model.models.get(regime)
            if estimator is None or not estimator.is_fitted():
                estimator = model.global_model
            if estimator is None or not estimator.is_fitted():
                raise ValueError("Test örneği için eğitilmiş model yok")
            X, _ = estimator._encode_features([row])
            probabilities.append(estimator.model.predict_proba(estimator.scaler.transform(X))[0, 1])
        probabilities, y = np.asarray(probabilities), np.asarray([row["label"] for row in rows])
    else:
        X, y = model._encode_features(rows)
        probabilities = model.model.predict_proba(model.scaler.transform(X))[:, 1]
    selected = probabilities >= threshold
    returns = np.array([row["net_pnl_pct"] for row in rows])
    winners = float(returns[selected & (returns > 0)].sum())
    losers = float(-returns[selected & (returns < 0)].sum())
    return {
        "samples": len(rows), "selected": int(selected.sum()), "threshold": threshold,
        "baseline_target_rate": float(y.mean()),
        "selected_target_rate": float(y[selected].mean()) if selected.any() else None,
        "baseline_mean_net_return_pct": float(returns.mean()),
        "selected_mean_net_return_pct": float(returns[selected].mean()) if selected.any() else None,
        "selected_profit_factor": winners / losers if losers else None,
        "roc_auc": float(roc_auc_score(y, probabilities)) if len(np.unique(y)) == 2 else None,
        "brier_score": float(brier_score_loss(y, probabilities)),
        "note": "Örtüşen sanal sinyaller; portföy getirisi veya yatırım tavsiyesi değildir.",
    }


def validate_samples(rows):
    """Evaluate the same regime/global selection used at inference."""
    train, test, cutoff = temporal_holdout(rows)
    groups = {regime: [] for regime in MarketRegime}
    for row in train:
        groups[list(MarketRegime)[int(row["regime_encoded"])]].append(row)
    model = RegimeAwareModelEnsemble("rf")
    cv = model.train(groups, save_models=False)
    if not cv or not test or model.global_model is None:
        raise ValueError("Geçerli model/doğrulama örneği üretilemedi; model yayımlanmadı")
    return {"training_samples": len(train), "holdout_start": str(cutoff),
            "purged_samples": len(rows) - len(train) - len(test),
            "cross_validation": cv, "holdout": evaluate_holdout(model, test),
            "evaluated_model": "regime_ensemble_with_global_fallback"}


def run(symbols, period="5y", output=Path("backtest_out/bist_training"), refresh=False):
    from signal_engine import detect_buy_signals
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    datasets = {regime: [] for regime in MarketRegime}
    coverage, failures = [], {}
    for symbol in symbols:
        try:
            if not symbol.endswith(".IS"):
                raise ValueError("Yalnız BIST .IS sembolleri kabul edilir")
            cache = output / f"{symbol}_{period}.csv"
            if cache.exists() and not refresh:
                data = pd.read_csv(cache, index_col=0, parse_dates=True)
                data.index = pd.to_datetime(data.index, utc=True).tz_convert("Europe/Istanbul")
                data = completed_bist_daily(data)
            else:
                data = fetch_bist_history(symbol, period)
                data.to_csv(cache)
            groups = build_bist_samples(data, symbol, detect_buy_signals)
            for regime, rows in groups.items():
                datasets[regime].extend(rows)
            count = sum(map(len, groups.values()))
            coverage.append({"symbol": symbol, "bars": len(data), "samples": count,
                             "start": str(data.index.min()), "end": str(data.index.max())})
            print(f"{symbol}: {len(data)} bar, {count} sinyal", flush=True)
        except Exception as exc:
            failures[symbol] = str(exc)
            print(f"{symbol}: veri hazırlanamadı: {exc}", flush=True)

    rows = sorted([row for values in datasets.values() for row in values], key=lambda row: row["timestamp"])
    pd.DataFrame(rows).to_csv(output / "samples.csv", index=False)
    validation = validate_samples(rows)
    report = {"scope": "BIST", "period": period, "symbols": coverage, "failures": failures,
              "horizon_bars": BIST_HORIZON, "feature_columns": FEATURE_COLUMNS,
              **validation,
              "limitations": ["Bugünkü takip listesi kullanılır; hayatta kalma yanlılığı olabilir.",
                              "Sinyaller örtüşebilir; sermaye dağılımı simülasyonu değildir.",
                              "Yahoo günlük OHLC verisi; fiyat boşlukları ve tavan/taban gerçekleşmeleri tam modellenmez."]}
    # Report the untouched test period before fitting on all observed outcomes.
    (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(validation["holdout"], ensure_ascii=False, indent=2), flush=True)
    ensemble = RegimeAwareModelEnsemble("rf")
    results = ensemble.train(datasets)
    report["full_training"] = results
    report["model_directory"] = str(MODEL_DIR)
    (output / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def validate_saved_samples(output=Path("backtest_out/bist_training")):
    """Repeat the fixed validation procedure from saved inputs, without downloads."""
    output = Path(output)
    report_path = output / "validation.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["feature_columns"] != FEATURE_COLUMNS or report["horizon_bars"] != BIST_HORIZON:
        raise ValueError("Kaydedilmiş örneklerin özellikleri/vadesi uyumsuz; veriyi yeniden oluşturun")
    frame = pd.read_csv(output / "samples.csv")
    for column in ("timestamp", "label_end_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if "evaluated_model" not in report:
        report["initial_global_holdout"] = report["holdout"]
    report.update(validate_samples(frame.to_dict("records")))
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["holdout"], ensure_ascii=False, indent=2), flush=True)
    return report


def main():
    from bot import SCAN_STOCKS
    parser = argparse.ArgumentParser(description="BIST teknik sinyal eğitimi ve bağımsız dönem testi")
    parser.add_argument("--symbols", default=",".join(SCAN_STOCKS))
    parser.add_argument("--period", default="5y", choices=["3y", "5y", "10y"])
    parser.add_argument("--refresh", action="store_true", help="Kaydedilmiş fiyatları yeniden indir")
    parser.add_argument("--validate-only", action="store_true", help="Kaydedilmiş örneklerden bağımsız dönem testini tekrarla")
    args = parser.parse_args()
    logging.getLogger("bot").setLevel(logging.ERROR)
    if args.validate_only:
        validate_saved_samples()
    else:
        run([symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip()], args.period, refresh=args.refresh)


if __name__ == "__main__":
    main()
