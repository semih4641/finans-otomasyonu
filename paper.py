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
import math
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    "quantity", "risk_amount",
    "status", "signal_at", "filled_at", "reference_entry", "execution_model",
]
SETTLED_FIELDS = FIELDS + ["closed_at", "outcome", "exit_price", "pnl_pct", "net_pnl_pct", "bars_held", "net_pnl_amount"]


def _load_open() -> list:
    try:
        items = json.loads(OPEN_FILE.read_text(encoding="utf-8"))
        if not isinstance(items, list) or any(
            not isinstance(item, dict) or not item.get("symbol") or not item.get("opened_at")
            for item in items
        ):
            raise ValueError("açık sinyal dosyası geçerli bir pozisyon listesi değil")
        return items
    except FileNotFoundError:
        return []
    except (OSError, ValueError) as e:
        raise RuntimeError("Açık sinyal dosyası okunamadı; mevcut kayıtlar korunuyor.") from e


def _write_json(path: Path, value: Any) -> None:
    """Replace JSON only after a complete, flushed write in the same directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            json.dump(value, f, ensure_ascii=False, indent=1, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _save_open(items: list) -> None:
    _write_json(OPEN_FILE, items)


def _load_equity() -> list:
    try:
        return json.loads(EQUITY_CSV.read_text(encoding="utf-8")) if EQUITY_CSV.exists() else []
    except Exception:
        return []


def _save_equity(equity_points: list) -> None:
    _write_json(EQUITY_CSV, equity_points)


def _update_equity_curve(settled_rows: list) -> None:
    """Rebuild the normalized return curve from the committed trade ledger."""
    equity_points = []
    current_equity = 100000.0

    for row in settled_rows:
        net_pnl = float(row.get("net_pnl_pct") or 0) / 100
        current_equity *= (1 + net_pnl)
        equity_points.append({
            "date": row["closed_at"][:10],
            "equity": round(current_equity, 2),
            "trade": row["symbol"],
            "outcome": row["outcome"],
            "pnl_pct": float(row["net_pnl_pct"]),
        })

    _save_equity(equity_points)


def _load_settled_rows() -> list[dict]:
    if not SETTLED_CSV.exists() or SETTLED_CSV.stat().st_size == 0:
        return []
    with SETTLED_CSV.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if not {"symbol", "opened_at", "net_pnl_pct"}.issubset(reader.fieldnames or []):
            raise ValueError("Sonuçlanmış sinyal dosyasının başlıkları geçersiz.")
        return list(reader)


def _save_settled_rows(rows: list[dict]) -> None:
    """Atomically rewrite the ledger, upgrading older CSV column layouts."""
    SETTLED_CSV.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", newline="", encoding="utf-8-sig", dir=SETTLED_CSV.parent, delete=False) as f:
            temporary = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=SETTLED_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, SETTLED_CSV)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


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
    gains = df.loc[df["net_pnl_pct"] > 0, "net_pnl_pct"].sum()
    losses = -df.loc[df["net_pnl_pct"] < 0, "net_pnl_pct"].sum()
    profit_factor = gains / losses if losses else (float("inf") if gains else 0.0)

    sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
    sortino = (returns.mean() / returns[returns < 0].std() * np.sqrt(252)) if (returns[returns < 0].std() > 0) else 0

    peak = equity.cummax().clip(lower=100000)
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
    df["ml_bucket"] = pd.cut(df["ml_score"], bins=[0, 0.3, 0.5, 0.7, 1.0], labels=["low", "med", "high", "vhigh"], include_lowest=True)

    metrics = {}
    for bucket, sub in df.groupby("ml_bucket", observed=True):
        if len(sub) < 3:
            continue
        metrics[bucket] = calculate_advanced_metrics(sub)
    return metrics


def record_signals(items: list) -> int:
    """Keep sent signals; new BIST setups wait for the next session's open."""
    if not items:
        return 0
    BASE.mkdir(parents=True, exist_ok=True)
    opened = _load_open()
    known = {o["symbol"] for o in opened}
    now_iso = datetime.now().astimezone().isoformat(timespec="seconds")
    added = 0
    for it in items:
        key = it.get("key")
        entry = it.get("entry_num")
        if not key or key in known or entry is None:
            continue
        try:
            entry, tp1, tp2, sl = map(float, (entry, it.get("tp1"), it.get("tp2") or it.get("tp1"), it.get("sl")))
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (entry, tp1, tp2, sl)) or not 0 < sl < entry < tp1 <= tp2:
            continue
        sizing = it.get("risk") or {}
        if not isinstance(sizing, dict):
            continue
        quantity = it.get("quantity")
        if quantity is None:
            quantity = sizing.get("quantity")
        if quantity is not None:
            if isinstance(quantity, bool):
                continue
            try:
                quantity = float(quantity)
            except (TypeError, ValueError, OverflowError):
                continue
            if not math.isfinite(quantity) or quantity <= 0 or not math.isfinite(entry * quantity):
                continue
        record = {
            "symbol": key,
            "market": it.get("market", "stock"),
            "name": it.get("name", key),
            "opened_at": now_iso,
            "entry": entry,
            "tp1": tp1,
            "tp2": tp2,
            "sl": sl,
            "score": it.get("score", it.get("skor", 0)),
            "trend": it.get("trend", it.get("rejim", "")),
            "rr": it.get("rr"),
            "categories": "|".join(it.get("signals", [])),
            "ml_score": it.get("ml_score", 0.5),
            "regime": it.get("regime", "UNKNOWN"),
            "portfolio_risk_pct": it.get("portfolio_risk_pct", 0),
            "sector": it.get("sector", "UNKNOWN"),
            "quantity": quantity,
            "risk_amount": (entry - sl) * quantity if quantity is not None else None,
        }
        if record["market"] == "stock" and str(key).upper().endswith(".IS"):
            try:
                signal_at = _bist_timestamp(it.get("signal_at"))
                if signal_at.date() > _bist_timestamp(now_iso).date():
                    raise ValueError("Karar mumu gelecekte olamaz")
            except (TypeError, ValueError):
                logger.warning("BIST sinyali karar mumu olmadan kaydedilmedi: %s", key)
                continue
            record.update(status="pending", signal_at=signal_at.isoformat(),
                          reference_entry=entry, execution_model="bist_next_open_v1")
        opened.append(record)
        known.add(key)
        added += 1
    if added:
        _save_open(opened)
        logger.info(f"📝 {added} sinyal paper-trading takibine alındı.")
    return added


async def fetch_frame_async(market: str, symbol: str):
    import data_fetcher
    if market == "crypto":
        d = await data_fetcher.fetch_crypto_data(symbol)
        return d["ohlcv_df"] if d else None
    d = await data_fetcher.fetch_stock_data_async(symbol)
    return d["history_df"] if d else None


def _bist_timestamp(value) -> pd.Timestamp:
    from market_data import BIST_TIMEZONE
    if value is None:
        raise ValueError("BIST karar zamanı gerekli")
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("BIST karar zamanı geçersiz")
    return timestamp.tz_localize(BIST_TIMEZONE) if timestamp.tzinfo is None else timestamp.tz_convert(BIST_TIMEZONE)


def _resolve_pending_entry(pos: dict, df: pd.DataFrame) -> tuple[dict | None, str | None]:
    """Resolve a daily opening only after its bar completes, without backdating.

    The immutable opened_at identifies the notification. filled_at identifies
    the actual entry bar, which must be a later session than registration.
    """
    if pos.get("status") != "pending" or pos.get("execution_model") != "bist_next_open_v1":
        return pos, None
    if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("BIST giriş verisi sıralı ve benzersiz olmalı")
    dates = [_bist_timestamp(timestamp).date() for timestamp in df.index]
    decision_day = _bist_timestamp(pos["signal_at"]).date()
    if decision_day not in dates:
        raise ValueError("BIST karar mumu veri geçmişinde bulunamadı")
    candidates = [index for index, day in enumerate(dates) if day > decision_day]
    if not candidates:
        return pos, None
    start = candidates[0]
    if _bist_timestamp(pos["opened_at"]).date() >= dates[start]:
        return None, "missed_entry"
    entry = float(df["open"].iloc[start])
    sl, tp1, tp2 = (float(pos[key]) for key in ("sl", "tp1", "tp2"))
    if not all(math.isfinite(value) and value > 0 for value in (entry, sl, tp1, tp2)):
        raise ValueError("BIST açılış fiyatı geçersiz")
    if not sl < entry < tp1 <= tp2:
        return None, "invalid_entry_gap"
    resolved = {**pos, "status": "open", "entry": entry,
                "filled_at": df.index[start].isoformat()}
    if pos.get("quantity") is not None:
        reference = float(pos["reference_entry"])
        quantity = float(pos["quantity"])
        # Never increase the original size, cash reservation or loss at stop.
        cash_budget = reference * (1 + COST_RATE) * quantity
        risk_budget = (reference * (1 + COST_RATE) - sl * (1 - COST_RATE)) * quantity
        quantity = min(quantity, cash_budget / (entry * (1 + COST_RATE)),
                       risk_budget / (entry * (1 + COST_RATE) - sl * (1 - COST_RATE)))
        if not math.isfinite(quantity) or quantity <= 0:
            return None, "invalid_quantity"
        resolved["quantity"] = quantity
        resolved["risk_amount"] = (entry - sl) * quantity
    return resolved, None


def _load_cancelled() -> list[dict]:
    """Unfilled setups are audited separately and never counted as trades."""
    path = BASE / "live_cancelled.json"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    if not isinstance(rows, list) or any(
        not isinstance(row, dict) or not row.get("symbol") or not row.get("opened_at")
        for row in rows
    ):
        raise ValueError("İptal edilen sinyal kayıtları geçersiz")
    return rows


def _start_index(df: pd.DataFrame, opened_at: str) -> int | None:
    try:
        ts = pd.Timestamp(opened_at)
        idx = df.index
        if isinstance(idx, pd.DatetimeIndex):
            if idx.tz is not None and ts.tzinfo is None:
                # Legacy records used the machine's local wall clock.
                ts = pd.Timestamp(ts.to_pydatetime().astimezone())
            elif idx.tz is None and ts.tzinfo is not None:
                ts = ts.tz_convert(datetime.now().astimezone().tzinfo).tz_localize(None)
            mask = idx >= ts
            hits = pd.Series(mask).to_numpy().nonzero()[0]
            return int(hits[0]) if len(hits) else None
    except Exception:
        pass
    return None


def settle_one(pos: dict, df: pd.DataFrame) -> dict | None:
    market = pos.get("market", "stock")
    horizon = HORIZON_BARS.get(market, 60)
    if df.empty:
        return None
    try:
        entry, sl, tp1, tp2 = map(float, (pos["entry"], pos["sl"], pos["tp1"], pos["tp2"]))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Paper pozisyonunun giriş/stop/hedef fiyatları geçersiz.") from exc
    if not all(math.isfinite(value) for value in (entry, sl, tp1, tp2)) or not 0 < sl < entry < tp1 <= tp2:
        raise ValueError("Paper pozisyonunun giriş/stop/hedef fiyatları geçersiz.")
    if not isinstance(df.index, pd.DatetimeIndex) or not df.index.is_monotonic_increasing or not df.index.is_unique:
        raise ValueError("Paper verisi sıralı ve benzersiz zaman damgaları içermeli.")
    if market == "crypto" and df.index.tz is None:
        # Exchange millisecond timestamps are UTC even when pandas left them naive.
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    n = len(df)
    if pos.get("status") == "pending":
        return None
    start = _start_index(df, pos.get("filled_at") or pos["opened_at"])
    if start is None:
        return None

    end = min(n, start + horizon)
    lows, highs, closes = df["low"], df["high"], df["close"]
    values = df.iloc[start:end][["low", "high", "close"]].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any() or (values[:, 0] > values[:, 1]).any():
        raise ValueError("Paper verisi sonlu ve pozitif OHLC fiyatları içermeli.")
    outcome, exit_price, exit_bar = ("MTM-TIMEOUT", None, None)

    for j in range(start, end):
        if lows.iloc[j] <= sl:
            outcome, exit_price, exit_bar = "SL", sl, j
            break
        if highs.iloc[j] >= tp1:
            outcome, exit_price, exit_bar = (
                ("TP2", tp2, j) if highs.iloc[j] >= tp2
                else ("TP1", tp1, j)
            )
            break

    if outcome == "MTM-TIMEOUT":
        if n - start < horizon:
            return None
        exit_price = float(closes.iloc[end - 1])
        exit_bar = end - 1

    gross = (exit_price / entry - 1.0) * 100
    net = ((exit_price * (1 - COST_RATE)) / (entry * (1 + COST_RATE)) - 1.0) * 100
    try:
        quantity = float(pos.get("quantity"))
        amount = quantity * (exit_price * (1 - COST_RATE) - entry * (1 + COST_RATE))
        valid_quantity = not isinstance(pos.get("quantity"), bool) and math.isfinite(quantity) and quantity > 0
        net_pnl_amount = round(amount, 6) if valid_quantity and math.isfinite(amount) else None
    except (TypeError, ValueError, OverflowError):
        net_pnl_amount = None
    row = {**pos, "outcome": outcome,
           "exit_price": round(float(exit_price), 6),
           "pnl_pct": round(gross, 3),
           "net_pnl_pct": round(net, 3),
           "net_pnl_amount": net_pnl_amount,
           "bars_held": int(exit_bar - start + 1) if exit_bar is not None else None,
           "closed_at": datetime.now().astimezone().isoformat(timespec="seconds")}
    return row


async def settle_signals() -> int:
    """Tüm açık sinyalleri değerlendirir; sonuçlananları CSV'e yazar."""
    opened = _load_open()
    if not opened:
        return 0
    BASE.mkdir(parents=True, exist_ok=True)
    committed = _load_settled_rows()
    committed_keys = {(row["symbol"], row["opened_at"]) for row in committed}
    cancelled = _load_cancelled()
    cancelled_keys = {(row["symbol"], row["opened_at"]) for row in cancelled}
    new_cancelled = []
    entries_filled = False
    remaining = []
    settled_rows = []

    for pos in opened:
        if (pos["symbol"], pos["opened_at"]) in committed_keys | cancelled_keys:
            continue
        market = pos.get("market", "stock")
        symbol = pos["symbol"].split(":", 1)[-1]
        try:
            df = await fetch_frame_async(market, symbol)
            if df is None or df.empty:
                remaining.append(pos)
                continue
            df = df.rename(columns=lambda column: str(column).lower())
            if pos.get("execution_model") == "bist_next_open_v1":
                from market_data import completed_bist_daily
                df = completed_bist_daily(df)
                resolved, reason = _resolve_pending_entry(pos, df)
                if reason:
                    new_cancelled.append({**pos, "status": "cancelled", "reason": reason,
                                          "cancelled_at": datetime.now().astimezone().isoformat(timespec="seconds")})
                    cancelled_keys.add((pos["symbol"], pos["opened_at"]))
                    continue
                entries_filled |= resolved != pos
                pos = resolved
            row = settle_one(pos, df)
            if row is None:
                remaining.append(pos)
                continue
            settled_rows.append(row)
            committed_keys.add((pos["symbol"], pos["opened_at"]))
        except Exception as e:
            logger.warning(f"⚠️ Sinyal değerlendirilemedi ({pos['symbol']}): {e}")
            remaining.append(pos)

    if settled_rows:
        committed.extend(settled_rows)
        _save_settled_rows(committed)
    if new_cancelled:
        _write_json(BASE / "live_cancelled.json", cancelled + new_cancelled)
    if entries_filled or settled_rows or len(remaining) != len(opened):
        # The ledger is authoritative if a crash occurs between these two writes.
        _save_open(remaining)
        try:
            _update_equity_curve(committed)
            _save_metrics(settled_rows)
        except Exception:
            logger.exception("Paper sonuçları kaydedildi, türetilen performans raporu güncellenemedi.")
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

    _write_json(METRICS_JSON, metrics)


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
