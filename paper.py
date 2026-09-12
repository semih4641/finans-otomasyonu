"""
Faz 4: Paper Trading Takip Sistemi (Gelişmiş)
==============================================
- Sinyal kaydı ve otomatik TP/SL değerlendirmesi
- Gelişmiş performans metrikleri (Sharpe, Sortino, Calmar, Max DD)
- Regime-bazlı performans analizi
- Sinyal kalite analizi (ML skoruna göre)
- Equity curve ve drawdown takibi
- Canlı risk metrikleri entegrasyonu
"""

import csv
import json
import logging
from datetime import datetime, date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from portfolio_risk import PortfolioState, calculate_portfolio_metrics

logger = logging.getLogger(__name__)

BASE = Path("backtest_out")
OPEN_FILE = BASE / "live_open.json"
SETTLED_CSV = BASE / "live_settled.csv"
EQUITY_CSV = BASE / "equity_curve.csv"
METRICS_JSON = BASE / "live_metrics.json"

HORIZON_BARS = {"stock": 60, "crypto": 120}
COST_RATE = 0.0015

FIELDS = [
    "symbol", "market", "name", "opened_at", "entry",
    "tp1", "tp2", "sl", "score", "trend", "rr", "categories",
    "ml_score", "regime", "portfolio_risk_pct", "sector",
]
SETTLED_FIELDS = FIELDS + ["closed_at", "outcome", "exit_price", "pnl_pct", "net_pnl_pct", "bars_held"]


def _load_open() -> list:
    try:
        return json.loads(OPEN_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as e:
        logger.warning(f"⚠️ Açık sinyal listesi okunamadı: {e}")
        return []


def _save_open(items: list) -> None:
    OPEN_FILE.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")


def _load_equity() -> list:
    try:
        return json.loads(EQUITY_CSV.read_text(encoding="utf-8")) if EQUITY_CSV.exists() else []
    except Exception:
        return []


def _save_equity(equity_points: list) -> None:
    EQUITY_CSV.write_text(json.dumps(equity_points, ensure_ascii=False, indent=1), encoding="utf-8")


def _update_equity_curve(settled_rows: list) -> None:
    """Update equity curve with new settled trades."""
    equity_points = _load_equity()
    current_equity = equity_points[-1]["equity"] if equity_points else 100000.0

    for row in settled_rows:
        net_pnl = row.get("net_pnl_pct", 0) / 100
        current_equity *= (1 + net_pnl)
        equity_points.append({
            "date": row["closed_at"][:10],
            "equity": round(current_equity, 2),
            "trade": row["symbol"],
            "outcome": row["outcome"],
            "pnl_pct": row["net_pnl_pct"],
        })

    _save_equity(equity_points)


def calculate_advanced_metrics(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate comprehensive performance metrics."""
    if df.empty:
        return {}

    returns = df["net_pnl_pct"] / 100
    equity = (1 + returns).cumprod() * 100000

    total_return = (equity.iloc[-1] / 100000 - 1) * 100
    n_trades = len(df)
    win_rate = (df["outcome"].isin(["TP1", "TP2"]).sum() / n_trades * 100) if n_trades > 0 else 0

    avg_win = df.loc[df["net_pnl_pct"] > 0, "net_pnl_pct"].mean() if (df["net_pnl_pct"] > 0).any() else 0
    avg_loss = df.loc[df["net_pnl_pct"] < 0, "net_pnl_pct"].mean() if (df["net_pnl_pct"] < 0).any() else 0
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    sortino = (returns.mean() / returns[returns < 0].std() * np.sqrt(252)) if (returns[returns < 0].std() > 0) else 0

    peak = equity.cummax()
    drawdown = (peak - equity) / peak * 100
    max_drawdown = drawdown.max()
    calmar = (total_return / max_drawdown) if max_drawdown > 0 else 0

    expectancy = returns.mean() * 100
    avg_bars = df["bars_held"].mean() if "bars_held" in df else 0

    return {
        "total_trades": int(n_trades),
        "win_rate": round(win_rate, 1),
        "total_return_pct": round(total_return, 2),
        "avg_win_pct": round(avg_win, 2),
        "avg_loss_pct": round(avg_loss, 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "∞",
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        "max_drawdown_pct": round(max_drawdown, 2),
        "expectancy_pct": round(expectancy, 3),
        "avg_bars_held": round(avg_bars, 1),
        "best_trade_pct": round(df["net_pnl_pct"].max(), 2),
        "worst_trade_pct": round(df["net_pnl_pct"].min(), 2),
    }


def calculate_regime_metrics(df: pd.DataFrame) -> dict[str, dict]:
    """Calculate metrics broken down by market regime."""
    if df.empty or "regime" not in df.columns:
        return {}

    metrics = {}
    for regime, sub in df.groupby("regime"):
        if len(sub) < 3:
            continue
        metrics[regime] = calculate_advanced_metrics(sub)
    return metrics


def calculate_score_metrics(df: pd.DataFrame) -> dict[str, dict]:
    """Calculate metrics broken down by signal score."""
    if df.empty or "score" not in df.columns:
        return {}

    metrics = {}
    for score in sorted(df["score"].dropna().unique()):
        sub = df[df["score"] == score]
        if len(sub) < 3:
            continue
        metrics[f"score_{int(score)}"] = calculate_advanced_metrics(sub)
    return metrics


def calculate_ml_metrics(df: pd.DataFrame) -> dict[str, dict]:
    """Calculate metrics broken down by ML score buckets."""
    if df.empty or "ml_score" not in df.columns:
        return {}

    df = df.copy()
    df["ml_bucket"] = pd.cut(df["ml_score"], bins=[0, 0.3, 0.5, 0.7, 1.0], labels=["low", "med", "high", "vhigh"])

    metrics = {}
    for bucket, sub in df.groupby("ml_bucket", observed=True):
        if len(sub) < 3:
            continue
        metrics[bucket] = calculate_advanced_metrics(sub)
    return metrics


def record_signals(items: list) -> int:
    """Başarıyla gönderilen sinyalleri açık listede tutar."""
    if not items:
        return 0
    BASE.mkdir(parents=True, exist_ok=True)
    opened = _load_open()
    known = {o["symbol"] for o in opened}
    now_iso = datetime.now().isoformat(timespec="seconds")
    added = 0
    for it in items:
        key = it.get("key")
        entry = it.get("entry_num")
        if not key or key in known or entry is None:
            continue
        if not (it.get("tp1") and it.get("sl")):
            continue
        opened.append({
            "symbol": key,
            "market": it.get("market", "stock"),
            "name": it.get("name", key),
            "opened_at": now_iso,
            "entry": float(entry),
            "tp1": float(it["tp1"]),
            "tp2": float(it.get("tp2") or it["tp1"]),
            "sl": float(it["sl"]),
            "score": it.get("score", it.get("skor", 0)),
            "trend": it.get("trend", it.get("rejim", "")),
            "rr": it.get("rr"),
            "categories": "|".join(it.get("signals", [])),
            "ml_score": it.get("ml_score", 0.5),
            "regime": it.get("regime", "UNKNOWN"),
            "portfolio_risk_pct": it.get("portfolio_risk_pct", 0),
            "sector": it.get("sector", "UNKNOWN"),
        })
        known.add(key)
        added += 1
    if added:
        _save_open(opened)
        logger.info(f"📝 {added} sinyal paper-trading takibine alındı.")
    return added


async def fetch_frame_async(market: str, symbol: str):
    import bot as b
    if market == "crypto":
        d = await b.fetch_crypto_data(symbol)
        return d["ohlcv_df"] if d else None
    d = await b.fetch_stock_data_async(symbol)
    return d["history_df"] if d else None


def _start_index(df: pd.DataFrame, opened_at: str) -> int | None:
    try:
        ts = pd.Timestamp(opened_at)
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            naive = idx.tz_localize(None) if idx.tz is not None else idx
            mask = naive >= ts.tz_localize(None) if ts.tzinfo is None else idx >= ts
            hits = pd.Series(mask).to_numpy().nonzero()[0]
            return int(hits[0]) if len(hits) else None
    except Exception:
        pass
    return None


def settle_one(pos: dict, df: pd.DataFrame) -> dict | None:
    market = pos.get("market", "stock")
    horizon = HORIZON_BARS.get(market, 60)
    n = len(df)
    start = _start_index(df, pos["opened_at"])

    try:
        ts = pd.Timestamp(pos["opened_at"])
        if isinstance(df.index, pd.DatetimeIndex):
            last = df.index[-1]
            last_naive = last.tz_localize(None) if last.tz is not None else last
            ts_naive = ts.tz_localize(None) if ts.tzinfo is None else ts
            if bool(ts_naive > last_naive):
                return None
    except Exception:
        pass

    if start is None:
        start = 0

    end = min(n, start + horizon)
    lows, highs, closes = df["low"], df["high"], df["close"]
    outcome, exit_price, exit_bar = ("MTM-TIMEOUT", None, None)

    for j in range(start, end):
        if lows.iloc[j] <= pos["sl"]:
            outcome, exit_price, exit_bar = "SL", pos["sl"], j
            break
        if highs.iloc[j] >= pos["tp1"]:
            outcome, exit_price, exit_bar = (
                ("TP2", pos["tp2"], j) if highs.iloc[j] >= pos["tp2"]
                else ("TP1", pos["tp1"], j)
            )
            break

    if outcome == "MTM-TIMEOUT":
        exit_price = float(closes.iloc[end - 1])
        exit_bar = end - 1

    gross = (exit_price / pos["entry"] - 1.0) * 100
    net = ((exit_price * (1 - COST_RATE)) / (pos["entry"] * (1 + COST_RATE)) - 1.0) * 100
    row = {**pos, "outcome": outcome,
           "exit_price": round(float(exit_price), 6),
           "pnl_pct": round(gross, 3),
           "net_pnl_pct": round(net, 3),
           "bars_held": int(exit_bar - start) if exit_bar is not None else None,
           "closed_at": datetime.now().isoformat(timespec="seconds")}
    return row


async def settle_signals() -> int:
    """Tüm açık sinyalleri değerlendirir; sonuçlananları CSV'e yazar."""
    opened = _load_open()
    if not opened:
        return 0
    BASE.mkdir(parents=True, exist_ok=True)
    remaining = []
    settled_rows = []

    for pos in opened:
        market = pos.get("market", "stock")
        symbol = pos["symbol"].split(":", 1)[-1]
        try:
            df = await fetch_frame_async(market, symbol)
            if df is None or len(df) < 3:
                remaining.append(pos)
                continue
            if isinstance(df.columns[0], str) and df.columns[0].istitle():
                df.columns = [str(c).lower() for c in df.columns]
            row = settle_one(pos, df)
            if row is None:
                remaining.append(pos)
                continue
            settled_rows.append(row)
        except Exception as e:
            logger.warning(f"⚠️ Sinyal değerlendirilemedi ({pos['symbol']}): {e}")
            remaining.append(pos)

    if settled_rows:
        exists = SETTLED_CSV.exists()
        with open(SETTLED_CSV, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=SETTLED_FIELDS, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerows(settled_rows)

        _update_equity_curve(settled_rows)
        _save_metrics(settled_rows)

        _save_open(remaining)
        logger.info(f"📊 {len(settled_rows)} sinyal sonuçlandı ({len(remaining)} açık kaldı).")
    return len(remaining)


def _save_metrics(new_rows: list) -> None:
    """Calculate and save comprehensive metrics."""
    if not SETTLED_CSV.exists():
        return

    df = pd.read_csv(SETTLED_CSV, encoding="utf-8-sig")
    if df.empty:
        return

    metrics = {
        "updated_at": datetime.now().isoformat(),
        "overall": calculate_advanced_metrics(df),
        "by_regime": calculate_regime_metrics(df),
        "by_score": calculate_score_metrics(df),
        "by_ml_score": calculate_ml_metrics(df),
        "equity_curve": _load_equity()[-100:],  # last 100 points
    }

    METRICS_JSON.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")


def performance_summary() -> str:
    """Gelişmiş performans özeti (/performans komutu için)."""
    if not SETTLED_CSV.exists():
        return "📉 Henüz sonuçlanmış sinyal yok."

    df = pd.read_csv(SETTLED_CSV, encoding="utf-8-sig")
    if df.empty:
        return "📉 Henüz sonuçlanmış sinyal yok."

    overall = calculate_advanced_metrics(df)

    def line(sub, label):
        wins = sub["outcome"].isin(["TP1", "TP2"]).sum()
        sls = (sub["outcome"] == "SL").sum()
        net = sub["net_pnl_pct"].mean()
        return f"{label}: {len(sub)} | 🏆 {wins} (%{wins/len(sub)*100:.1f}) | 🛑 {sls} | NET %{net:+.3f}"

    parts = ["📈 <b>Canlı Sinyal Performansı</b>\n━━━━━━━━━━━━━━━━━━━━"]
    parts.append(line(df, "GENEL"))

    for mkt, sub in df.groupby("market"):
        parts.append(line(sub, {"stock": "📊 Hisse", "crypto": "🪙 Kripto"}.get(mkt, mkt)))

    if overall:
        parts.append("\n📊 <b>Gelişmiş Metrikler</b>")
        parts.append(f"  Toplam Getiri: %{overall['total_return_pct']:+.2f}")
        parts.append(f"  Sharpe: {overall['sharpe_ratio']} | Sortino: {overall['sortino_ratio']} | Calmar: {overall['calmar_ratio']}")
        parts.append(f"  Max DD: %{overall['max_drawdown_pct']:.2f} | Profit Factor: {overall['profit_factor']}")
        parts.append(f"  Expectancy: %{overall['expectancy_pct']:+.3f} | Ort. Tutma: {overall['avg_bars_held']} bar")
        parts.append(f"  En İyi: %{overall['best_trade_pct']:+.2f} | En Kötü: %{overall['worst_trade_pct']:+.2f}")

    if "regime" in df.columns:
        parts.append("\n🎯 <b>Rejim Bazlı</b>")
        for regime, sub in df.groupby("regime"):
            if len(sub) >= 3:
                m = calculate_advanced_metrics(sub)
                parts.append(f"  {regime}: {len(sub)} trade | WR %{m['win_rate']:.1f} | PF {m['profit_factor']} | DD %{m['max_drawdown_pct']:.1f}")

    if "score" in df.columns:
        parts.append("\n🎯 <b>Skor Bazlı</b>")
        for score in sorted(df["score"].dropna().unique()):
            sub = df[df["score"] == score]
            if len(sub) >= 3:
                m = calculate_advanced_metrics(sub)
                parts.append(f"  Skor {int(score)}: {len(sub)} trade | WR %{m['win_rate']:.1f} | NET %{m['expectancy_pct']:+.3f}")

    open_n = len(_load_open())
    parts.append(f"\n⏳ Bekleyen: {open_n}")
    parts.append(f"💰 Maliyet: tek yön %{COST_RATE*100:.2f}")
    return "\n".join(parts)


def get_equity_curve() -> list:
    """Return equity curve data for plotting."""
    return _load_equity()


def get_live_metrics() -> dict:
    """Return latest calculated metrics."""
    try:
        return json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return {}