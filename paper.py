"""
Faz 4: Paper Trading Takip Sistemi
==================================
Otomatik gönderilen her sinyali kaydeder; sonraki taramalarda fiyat
geçmişine bakarak TP1 / TP2 / SL sonucunu otomatik işaretler. Böylece
gerçek para devreye alınmadan stratejinin CANLI doğrulaması birikir.

Dosyalar:
    backtest_out/live_open.json     — sonuçlanmamış sinyaller
    backtest_out/live_settled.csv   — sonuçlanmış sinyal geçmişi

Değerlendirme kuralı backtest ile aynıdır (muhafazakâr):
    - Aynı barda hem SL hem TP'ye dokunuş → SL sayılır
    - Sonuçlanmayanlar horizon sonunda mark-to-market edilir
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

BASE = Path("backtest_out")
OPEN_FILE = BASE / "live_open.json"
SETTLED_CSV = BASE / "live_settled.csv"

# Sonuç penceresi: hisse günlük bar, kripto saatlik bar
HORIZON_BARS = {"stock": 60, "crypto": 120}
COST_RATE = 0.0015  # tek yön %0.15 (backtest varsayımıyla uyumlu)

FIELDS = [
    "symbol", "market", "name", "opened_at", "entry",
    "tp1", "tp2", "sl", "score", "trend", "rr", "categories",
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


def record_signals(items: list) -> int:
    """Başarıyla gönderilen sinyalleri açık listede tutar (aynı sembol tek kayıt)."""
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
            "tp2": float(it["tp2"]),
            "sl": float(it["sl"]),
            "score": it.get("score"),
            "trend": it.get("trend", ""),
            "rr": it.get("rr"),
            "categories": "|".join(it.get("signals", [])),
        })
        known.add(key)
        added += 1
    if added:
        _save_open(opened)
        logger.info(f"📝 {added} sinyal paper-trading takibine alındı.")
    return added


async def fetch_frame_async(market: str, symbol: str):
    """İlgili piyasa için OHLCV çerçevesi getirir (async ortamda)."""
    import bot as b
    if market == "crypto":
        d = await b.fetch_crypto_data(symbol)
        return d["ohlcv_df"] if d else None
    d = await b.fetch_stock_data_async(symbol)
    return d["history_df"] if d else None


def _start_index(df: pd.DataFrame, opened_at: str) -> int | None:
    """Açılış zamanından sonraki ilk bar indeksini bulur; bulunamazsa None."""
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
    """Tek bir açık sinyali mevcut çerçeve üzerinde sonuçlandırır."""
    market = pos.get("market", "stock")
    horizon = HORIZON_BARS.get(market, 60)
    n = len(df)
    start = _start_index(df, pos["opened_at"])

    # Açılış zamanı tüm barlardan yeni ise henüz değerlendirilemez
    try:
        ts = pd.Timestamp(pos["opened_at"])
        if isinstance(df.index, pd.DatetimeIndex):
            last = df.index[-1]
            last_naive = last.tz_localize(None) if last.tz is not None else last
            ts_naive = ts.tz_localize(None) if ts.tzinfo is None else ts
            if bool(ts_naive > last_naive):
                return None  # açık kalmalı
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
    """Tüm açık sinyalleri değerlendirir; sonuçlananları CSV'e yazar. Kalan sayısını döndürür."""
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
        _save_open(remaining)
        logger.info(f"📊 {len(settled_rows)} sinyal sonuçlandı "
                    f"({len(remaining)} açık kaldı).")
    return len(remaining)


def performance_summary() -> str:
    """Sonuçlanmış sinyallerden özet metin üretir (/performans komutu için)."""
    if not SETTLED_CSV.exists():
        return "📉 Henüz sonuçlanmış sinyal yok. Otomatik taramalar sonuç ürettikçe burada birikecek."
    df = pd.read_csv(SETTLED_CSV, encoding="utf-8-sig")
    if df.empty:
        return "📉 Henüz sonuçlanmış sinyal yok."

    def line(sub, label):
        wins = sub["outcome"].isin(["TP1", "TP2"]).sum()
        sls = (sub["outcome"] == "SL").sum()
        net = sub["net_pnl_pct"].mean()
        return f"{label}: {len(sub)} sinyal | 🏆 {wins} (%{wins/len(sub)*100:.1f}) | 🛑 {sls} | Ort NET %{net:+.3f}"

    parts = ["📈 <b>Canlı Sinyal Performansı</b>\n━━━━━━━━━━━━━━━━━━━━"]
    parts.append(line(df, "GENEL"))
    for mkt, sub in df.groupby("market"):
        parts.append(line(sub, {"stock": "📊 Hisse", "crypto": "🪙 Kripto"}.get(mkt, mkt)))
    open_n = len(_load_open())
    parts.append(f"\n⏳ Değerlendirmeyi bekleyen: {open_n}")
    parts.append(f"💰 Maliyet varsayımı: tek yön %{COST_RATE*100:.2f}")
    return "\n".join(parts)
