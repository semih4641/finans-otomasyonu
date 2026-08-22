"""
Faz 3: Backtest Motoru
======================
Üretimdeki sinyal mantığını (bot.detect_buy_signals) geçmiş veri üzerinde
walk-forward yöntemiyle çalıştırır ve her kurulumun TP1 / TP2 / SL
sonucunu ölçer. Böylece skor ağırlıkları ve eşikler gerçek veriyle
değerlendirilebilir.

Çalıştırma:
    python backtest.py                          # varsayılan örneklem
    python backtest.py --all                    # tüm SCAN listeleri
    python backtest.py --stocks THYAO.IS --crypto SOL/USDT
    python backtest.py --horizon 90 --period 2y

Notlar:
- Walk-forward: her bar için yalnızca o ana kadarki veri kullanılır;
  canlıdaki gibi oluşmakta olan son mum detect içinde atılır.
- Cooldown: canlıdaki SIGNAL_COOLDOWN_HOURS davranışı taklit edilir
  (günlük barda 1 bar, saatlik barda 24 bar).
- Muhafazakâr değerlendirme: aynı barda hem SL hem TP seviyesine
  dokunulduysa SL sayılır; TP1'e ve TP2'ye aynı barda dokunulduysa
  yalnızca TP1 kazanılır.
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd

from bot import (
    SCAN_STOCKS,
    SCAN_CRYPTO,
    CRYPTO_EXCHANGES,
    detect_buy_signals,
    _signal_category,
)

# Walk-forward binlerce bastırma kararı üretir; rapor gürültüsünü engelle
logging.getLogger("bot").setLevel(logging.ERROR)

# Varsayılan parametreler
DEFAULT_PERIOD = "1y"
DEFAULT_HORIZON = 60          # sonuç penceresi (bar)
COOLDOWN_BARS_DAILY = 1       # günlük veride ~24 saat
COOLDOWN_BARS_HOURLY = 24     # saatlik veride ~24 saat
MIN_BARS_FOR_SCAN = 35        # detect'in en az ihtiyaç duyduğu geçmiş

# Varsayılan işlem maliyeti (baz puan, tek yön): 10 bp komisyon + 5 bp kayma
DEFAULT_FEE_BPS = 10
DEFAULT_SLIP_BPS = 5


def net_return_pct(entry: float, exit_price: float, cost_rate: float) -> float:
    """Girişte ve çıkışta komisyon+kayma uygulanmış net getiri (%)."""
    return ((exit_price * (1 - cost_rate)) / (entry * (1 + cost_rate)) - 1.0) * 100


# ============================================================================
# VERİ YÜKLEYİCİLER
# ============================================================================
def load_stock(ticker_symbol: str, period: str) -> pd.DataFrame | None:
    """Yahoo Finance'tan düzeltilmiş OHLCV indirir, sütun adlarını küçük harfe çevirir."""
    try:
        import yfinance as yf
        candidates = [ticker_symbol]
        if not ticker_symbol.endswith(".IS"):
            candidates.append(f"{ticker_symbol}.IS")
        for symbol in candidates:
            df = yf.Ticker(symbol).history(period=period, auto_adjust=True)
            if df is None or df.empty:
                continue
            df.columns = [str(c).lower() for c in df.columns]
            print(f"  ✅ {symbol}: {len(df)} bar")
            return df
        print(f"  ❌ {ticker_symbol}: veri yok")
    except Exception as e:
        print(f"  ❌ {ticker_symbol}: {e}")
    return None


def load_crypto(pair: str, limit: int = 1000) -> pd.DataFrame | None:
    """Bot ile aynı borsa sırasını kullanarak saatlik OHLCV indirir."""
    import ccxt
    for exchange_id in CRYPTO_EXCHANGES:
        exchange = None
        try:
            exchange = getattr(ccxt, exchange_id)({"enableRateLimit": True})
            ohlcv = exchange.fetch_ohlcv(pair, timeframe="1h", limit=limit)
            if not ohlcv:
                continue
            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df.set_index("timestamp", inplace=True)
            print(f"  ✅ {pair} ({exchange_id}): {len(df)} bar")
            return df
        except Exception as e:
            print(f"  ⚠️ {exchange_id} başarısız ({pair}): {e}")
        finally:
            if exchange:
                try:
                    exchange.close()
                except Exception:
                    pass
    print(f"  ❌ {pair}: hiçbir borsadan çekilemedi")
    return None


# ============================================================================
# WALK-FORWARD SİNYAL SİMÜLASYONU
# ============================================================================
def walk_forward(df: pd.DataFrame, cooldown_bars: int, market: str = "stock") -> list:
    """
    Geçmişi bar bar tarar; canlıdakiyle birebir mantıkla sinyal üretir.
    Dönen her olay: karar barı, giriş fiyatı, TP/SL ve meta alanları taşır.
    """
    events = []
    last_fire: dict = {}  # kategori -> son karar barı
    n = len(df)
    for i in range(MIN_BARS_FOR_SCAN, n):
        res = detect_buy_signals(df.iloc[:i], market=market)
        if not res["signals"]:
            continue

        decision_idx = i - 2  # dilimde atılan 'oluşan' mum öncesi son kapanmış bar
        cats = {_signal_category(s) for s in res["signals"]}

        # Cooldown: kategorilerden biri hâlâ sessizlik süresindeyse kurulumu sayma
        if any(
            decision_idx - last_fire.get(c, -(10**9)) <= cooldown_bars for c in cats
        ):
            continue
        for c in cats:
            last_fire[c] = decision_idx

        events.append({
            "decision_idx": decision_idx,
            "entry": float(df["close"].iloc[decision_idx]),
            "tp1": res["tp1"],
            "tp2": res["tp2"],
            "sl": res["sl"],
            "score": res.get("score", 0),
            "trend": res.get("trend", ""),
            "rr": res.get("rr"),
            "categories": sorted(cats),
        })
    return events


# ============================================================================
# SONUÇ DEĞERLENDİRME
# ============================================================================
def evaluate(events: list, df: pd.DataFrame, horizon: int, cost_rate: float = 0.0) -> list:
    """Her olayın TP1 → TP2 → SL yolculuğunu ileriye dönük pencerelerde takip eder.

    TIMEOUT kalan kurulumlar pencere sonunda mark-to-market edilir — böylece
    gerçekleşmemiş zarar/kar raporda görünür olur.
    """
    trades = []
    lows, highs, closes = df["low"], df["high"], df["close"]
    n = len(df)

    for ev in events:
        start = ev["decision_idx"] + 1
        end = min(n, start + horizon)
        outcome, exit_price, exit_bar = "TIMEOUT", None, None

        for j in range(start, end):
            # Muhafazakâr sıra: önce SL
            if lows.iloc[j] <= ev["sl"]:
                outcome, exit_price, exit_bar = "SL", ev["sl"], j
                break
            if highs.iloc[j] >= ev["tp1"]:
                # Aynı pencerede TP2'ye de uzandı mı?
                if highs.iloc[j] >= ev["tp2"]:
                    outcome, exit_price, exit_bar = "TP2", ev["tp2"], j
                else:
                    outcome, exit_price, exit_bar = "TP1", ev["tp1"], j
                break

        # Sonuçsuz kurulumları pencere sonu kapanışıyla mark-to-market et
        mtm = False
        if outcome == "TIMEOUT" and end - 1 >= start:
            exit_price = float(closes.iloc[end - 1])
            exit_bar = end - 1
            mtm = True

        pnl_gross = ((exit_price / ev["entry"]) - 1.0) * 100 if exit_price else 0.0
        pnl_net = net_return_pct(ev["entry"], exit_price, cost_rate) if exit_price else 0.0
        trades.append({
            "entry": round(ev["entry"], 6),
            "tp1": round(ev["tp1"], 6),
            "tp2": round(ev["tp2"], 6),
            "sl": round(ev["sl"], 6),
            "score": ev["score"],
            "trend": ev["trend"],
            "rr": ev["rr"],
            "categories": "|".join(ev["categories"]),
            "outcome": ("MTM-" + outcome) if mtm else outcome,
            "pnl_pct": round(pnl_gross, 3),
            "net_pnl_pct": round(pnl_net, 3),
            "bars_held": (exit_bar - ev["decision_idx"]) if exit_bar is not None else None,
        })
    return trades


def summarize(trades: list) -> dict:
    """Genel ve kategori bazlı istatistik üretir."""
    if not trades:
        return {}
    df = pd.DataFrame(trades)

    def stats(sub: pd.DataFrame) -> dict:
        total = len(sub)
        wins = sub["outcome"].isin(["TP1", "TP2"]).sum()
        sls = (sub["outcome"] == "SL").sum()
        timeouts = sub["outcome"].str.startswith("MTM").sum()
        avg_net = sub["net_pnl_pct"].mean() if "net_pnl_pct" in sub else None
        return {
            "toplam": total,
            "kazanc": int(wins),
            "zarar": int(sls),
            "sonucsuz": int(timeouts),
            "kazanc_orani": round(wins / total * 100, 1) if total else 0.0,
            "ort_net_yuzde": round(avg_net, 3) if pd.notna(avg_net) else None,
            "ort_bar": round(sub["bars_held"].dropna().mean(), 1) if sub["bars_held"].notna().any() else None,
        }

    summary = {"GENEL": stats(df)}
    exploded = df.assign(kat=df["categories"].str.split("|")).explode("kat")
    for kat, sub in exploded.groupby("kat"):
        summary[kat] = stats(sub)
    return summary


def format_report(summaries: dict, cost_rate: float = 0.0) -> str:
    lines = [f"💰 Maliyet varsayımı: tek yön %{cost_rate * 100:.3f} (komisyon+kayma)"]
    for scope, s in summaries.items():
        lines.append(f"\n📊 {scope}")
        lines.append(
            f"   Kurulum: {s['toplam']} | 🏆 Kazanç: {s['kazanc']} "
            f"({s['kazanc_orani']}%) | 🛑 Zarar: {s['zarar']} | ⏳ Sonuçsuz: {s['sonucsuz']}"
        )
        if s.get("ort_net_yuzde") is not None:
            lines.append(f"   Ortalama NET P/L: %{s['ort_net_yuzde']}")
        if s.get("ort_bar") is not None:
            lines.append(f"   Ortalama tutma süresi: {s['ort_bar']} bar")
    return "\n".join(lines)


# ============================================================================
# ANA AKIŞ
# ============================================================================
def run_backtest(stocks: list, cryptos: list, period: str, horizon: int,
                 out_csv: str, cost_rate: float = 0.0):
    all_trades = []
    summaries = {}

    print(f"💰 Maliyet modeli: tek yön %{cost_rate * 100:.3f} (giriş+çıkış uygulanır)")

    if stocks:
        print("\n=== HİSSE TARAMASI (günlük bar) ===")
        stock_trades = []
        for t in stocks:
            print(f"• {t}")
            df = load_stock(t, period)
            if df is None or len(df) < MIN_BARS_FOR_SCAN + 5:
                continue
            events = walk_forward(df, cooldown_bars=COOLDOWN_BARS_DAILY, market="stock")
            trades = evaluate(events, df, horizon, cost_rate)
            for tr in trades:
                tr["symbol"] = t
            stock_trades += trades
        if stock_trades:
            summaries["HİSSE (genel)"] = summarize(stock_trades)["GENEL"]
            all_trades += [("HISSE", "", tr) for tr in stock_trades]
            for k, v in summarize(stock_trades).items():
                if k != "GENEL":
                    summaries[f"HİSSE · {k}"] = v

    if cryptos:
        print("\n=== KRİPTO TARAMASI (saatlik bar) ===")
        crypto_trades = []
        for pair in cryptos:
            print(f"• {pair}")
            df = load_crypto(pair)
            if df is None or len(df) < MIN_BARS_FOR_SCAN + 5:
                continue
            events = walk_forward(df, cooldown_bars=COOLDOWN_BARS_HOURLY, market="crypto")
            trades = evaluate(events, df, horizon, cost_rate)
            for tr in trades:
                tr["symbol"] = pair
            crypto_trades += trades
        if crypto_trades:
            summaries["KRİPTO (genel)"] = summarize(crypto_trades)["GENEL"]
            all_trades += [("KRIPTO", "", tr) for tr in crypto_trades]
            for k, v in summarize(crypto_trades).items():
                if k != "GENEL":
                    summaries[f"KRİPTO · {k}"] = v

    print("\n" + "=" * 60)
    if not summaries:
        print("Hiç kurulum üretilmedi — parametreleri genişletmeyi dene.")
        return

    for scope in sorted(summaries):
        s = summaries[scope]
        print(f"\n📊 {scope}")
        print(
            f"   Kurulum: {s['toplam']} | 🏆 {s['kazanc']} (%{s['kazanc_orani']})"
            f" | 🛑 {s['zarar']} | ⏳ {s['sonucsuz']}"
        )
        if s.get("ort_net_yuzde") is not None:
            print(f"   Ort. NET P/L: %{s['ort_net_yuzde']}")
        if s.get("ort_bar") is not None:
            print(f"   Ort. tutma: {s['ort_bar']} bar")

    rows = [{"market": m, "symbol": sym, **tr} for m, sym, tr in all_trades]
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n💾 İşlem detayları: {out_csv}")
    print(f"⏰ Rapor: {datetime.now().strftime('%d.%m.%Y %H:%M')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Faz 3 Backtest Motoru")
    parser.add_argument("--stocks", default="AAPL,MSFT,THYAO.IS",
                        help="Virgülle ayrılmış hisse listesi")
    parser.add_argument("--crypto", default="BTC/USDT",
                        help="Virgülle ayrılmış kripto parite listesi")
    parser.add_argument("--all", action="store_true",
                        help="Tüm SCAN_STOCKS ve SCAN_CRYPTO listesini tara")
    parser.add_argument("--period", default=DEFAULT_PERIOD, help="yfinance dönem aralığı")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON,
                        help="Sonuç penceresi (bar)")
    parser.add_argument("--out", default="backtest_out/trades.csv", help="CSV çıktı yolu")
    parser.add_argument("--fee-bps", type=int, default=DEFAULT_FEE_BPS,
                        help="Tek yön komisyon (baz puan, varsayılan 10 = %%0.10)")
    parser.add_argument("--slip-bps", type=int, default=DEFAULT_SLIP_BPS,
                        help="Tek yön kayma/slippage (baz puan, varsayılan 5)")
    args = parser.parse_args()

    stocks = SCAN_STOCKS if args.all else [s.strip() for s in args.stocks.split(",") if s.strip()]
    cryptos = SCAN_CRYPTO if args.all else [c.strip() for c in args.crypto.split(",") if c.strip()]
    cost_rate = (args.fee_bps + args.slip_bps) / 10000.0

    run_backtest(stocks, cryptos, args.period, args.horizon, args.out, cost_rate)


if __name__ == "__main__":
    main()
