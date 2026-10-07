"""Descriptive event outcomes, not forecasts or proof of a news effect."""
from collections import defaultdict
from datetime import datetime
from html import escape
import math
from statistics import median

import pandas as pd

from market_data import BIST_TIMEZONE, completed_bist_daily
from signal_safety import validated_ohlcv

HORIZONS = (1, 5, 20)
MIN_EXAMPLES = 10


def price_frame(frame, now):
    data = validated_ohlcv(completed_bist_daily(frame, now))
    index = data.index
    data.index = (index.tz_localize(BIST_TIMEZONE) if index.tz is None else index.tz_convert(BIST_TIMEZONE)).normalize()
    if not data.index.is_unique:
        raise ValueError("Günlük fiyatlarda yinelenen seans var.")
    return data


def measure_event(event, prices, benchmark, now):
    """Use NEXT session open and h-th session close, never same-day pre-news prices.

    The caller supplies adjusted, validated, completed daily bars. This is a
    historical description, including seeded history, not a trade backtest.
    """
    published = pd.Timestamp(event["published_at"])
    day = published.tz_convert(BIST_TIMEZONE).normalize()
    if prices.empty or prices.index[0] > day:
        return []  # Entry coverage cannot be inferred from a truncated history.
    future = prices.loc[prices.index > day]
    if future.empty:
        return []
    entry = float(future["open"].iloc[0])
    rows = []
    for horizon in HORIZONS:
        if len(future) < horizon:
            continue
        entry_date, exit_date = future.index[0], future.index[horizon - 1]
        raw = (float(future["close"].iloc[horizon - 1]) / entry - 1) * 100
        market_return = None
        if benchmark is not None and entry_date in benchmark.index and exit_date in benchmark.index:
            market_return = (float(benchmark.loc[exit_date, "close"]) / float(benchmark.loc[entry_date, "open"]) - 1) * 100
        if not math.isfinite(raw) or (market_return is not None and not math.isfinite(market_return)):
            continue
        rows.append({"news_id": event["id"], "symbol": event["symbol"], "horizon": horizon,
                     "entry_day": entry_date.date().isoformat(), "exit_day": exit_date.date().isoformat(),
                     "raw_return": raw, "benchmark_return": market_return, "measured_at": now.isoformat()})
    return rows


def grouped_observations(rows):
    """For each horizon, retain non-overlapping windows per stock, across topics."""
    groups = defaultdict(list)
    last_exit = {}
    for row in sorted(rows, key=lambda item: (item["entry_day"], item["news_id"], item["symbol"])):
        key = (row["symbol"], row["horizon"])
        if row["entry_day"] <= last_exit.get(key, ""):
            continue
        last_exit[key] = row["exit_day"]
        groups[(row["category"], row["horizon"])].append(row)
    return groups


def render_learning(store, symbol=None):
    rows = store.observations(symbol)
    groups = grouped_observations(rows)
    state = store.state()
    lines = ["📚 <b>Haber sonrası fiyat gözlemleri</b>",
             "Kapsam: " + escape(symbol or "Takip listesindeki hisseler"),
             "Ölçüm: yayından SONRAKİ seans açılışı → 1/5/20. seans kapanışı.",
             "Aynı hissede örtüşen ölçüm pencereleri tekrar sayılmaz."]
    if state.get("coverage_start"):
        lines.append("Haber arşivi başlangıcı: " + escape(state["coverage_start"]))
    if state.get("last_learning"):
        moment = datetime.fromisoformat(state["last_learning"])
        lines.append(f"Son fiyat ölçümü: {moment:%d.%m.%Y %H:%M} (İstanbul)")
    if state.get("learning_warning"):
        lines.append("⚠️ " + escape(state["learning_warning"]))
    for warning in ("last_error", "coverage_gap"):
        if state.get(warning):
            lines.append("⚠️ " + escape(state[warning]))
    if not groups:
        lines.append("\nHenüz sonucu ölçülmüş uygun örnek yok. Yeni haberler ve tamamlanan seanslar geldikçe birikir.")
    for (category, horizon), samples in sorted(groups.items()):
        size = len(samples)
        lines.append(f"\n<b>{escape(category)} · {horizon} seans</b>")
        if size < MIN_EXAMPLES:
            lines.append(f"{size}/{MIN_EXAMPLES} örtüşmeyen pencere; istatistik göstermek için örnek az.")
            continue
        changes = [row["raw_return"] for row in samples]
        positive = sum(value > 0 for value in changes) / size * 100
        lines.append(f"n={size} · Medyan değişim: %{median(changes):+.2f} · Pozitif sonuç: %{positive:.1f}")
        excess = [row["raw_return"] - row["benchmark_return"] for row in samples if row["benchmark_return"] is not None]
        if len(excess) >= MIN_EXAMPLES:
            lines.append(f"BIST 100'e göre medyan fark: {median(excess):+.2f} yüzde puan (n={len(excess)}).")
        else:
            lines.append(f"BIST 100 karşılaştırması: yeterli eşleşme yok ({len(excess)}/{MIN_EXAMPLES}).")
    lines.extend(["", "Bu betimsel gözlem bir tahmin veya haberin fiyatı değiştirdiğinin kanıtı değildir. "
                  "Komisyon, kayma ve haberin yayımlandığı günün ilk tepkisi dahil değildir. "
                  "Haberler teknik AL sinyalini veya ML modelini otomatik değiştirmez."])
    return "\n".join(lines)
