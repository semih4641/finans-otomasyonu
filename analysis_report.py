"""Readable BIST analysis from the same completed bars used by the strategy."""

from html import escape
import math

import pandas as pd

from indicators import calculate_rsi, calculate_macd
from market_data import BIST_TIMEZONE, completed_bist_daily
from signal_engine import detect_buy_signals
from signal_safety import validated_ohlcv


def build_stock_report(data: dict, now=None) -> str:
    symbol = str(data["ticker"])
    name = str((data.get("info") or {}).get("shortName") or symbol)[:100]
    heading = f"📊 <b>{escape(name)} ({escape(symbol[:30])})</b>"
    current = pd.Timestamp.now(tz=BIST_TIMEZONE) if now is None else pd.Timestamp(now)
    current = current.tz_localize(BIST_TIMEZONE) if current.tzinfo is None else current.tz_convert(BIST_TIMEZONE)
    try:
        frame = completed_bist_daily(data["history_df"], current)
        frame = validated_ohlcv(frame)
    except (ValueError, TypeError, KeyError, AttributeError):
        return heading + "\n\nVeri kontrolü geçilemedi. Geçerli ve tamamlanmış fiyat geçmişi olmadan analiz üretilemiyor. Daha sonra tekrar deneyin."

    last = frame.index[-1]
    last = last.tz_localize(BIST_TIMEZONE) if last.tzinfo is None else last.tz_convert(BIST_TIMEZONE)
    close = float(frame["close"].iloc[-1])
    age = (current.date() - last.date()).days
    lines = [heading, "", f"<b>Analizin dayandığı kapanış:</b> {close:,.2f} TL",
             f"<b>Mum tarihi:</b> {last:%d.%m.%Y} · Günlük",
             "Kaynak: Yahoo Finance · düzeltilmiş fiyat geçmişi",
             f"Rapor zamanı: {current:%d.%m.%Y %H:%M} (İstanbul)",
             "Kapanış fiyatı, anlık işlem fiyatı değildir."]
    if age:
        lines.append(f"Son mum {age} takvim günü önceye ait; hafta sonu/tatil veya veri gecikmesi olabilir.")
    if len(frame) < 30:
        lines.extend(["", "<b>Değerlendirme yapılamadı</b>",
                      f"{len(frame)} tamamlanmış mum var; temel analiz için en az 30 mum gerekli."])
        return "\n".join(lines)

    # Passing the completed frame prevents indicators from discarding another bar.
    result = detect_buy_signals(frame, market="stock", symbol=symbol, now=current)
    safety = result.get("safety", {})
    lines.extend(["", "<b>Veri ve hareket kontrolü</b>"])
    if safety.get("blocked"):
        lines.append("⚠️ Koruma kontrolü sinyali durdurdu.")
        lines.extend("• " + escape(str(reason)) for reason in safety.get("reasons", []))
    elif safety:
        lines.append("Tanımlı veri ve aşırı fiyat-hacim kontrollerinde engel bulunmadı.")
    lines.extend("• " + escape(str(reason)) for reason in safety.get("warnings", []))
    if safety.get("metrics"):
        metrics = safety["metrics"]
        lines.append(f"Hacim / önceki 20 mum medyanı: {metrics['volume_ratio']:.1f}x; "
                     f"kapanış değişimi: %{metrics['change_pct']:.1f}.")
    lines.append("Bu kontrol manipülasyonu kanıtlamaz veya güvenli yatırım garantisi vermez.")
    if "STALE_DATA" in safety.get("codes", []):
        lines.append("Güncel veri gelene kadar teknik gösterge ve hedef üretilmedi.")
        return "\n".join(lines)
    rsi = calculate_rsi(frame)
    macd = calculate_macd(frame)
    lines.extend(["", "<b>Göstergeler ne söylüyor?</b>"])
    if rsi is not None and math.isfinite(rsi):
        explanation = ("Son fiyat hareketlerinde düşüş baskısı yüksek; tek başına dönüş teyidi değildir." if rsi < 30
                       else "Son fiyat hareketlerinde yükseliş baskısı yüksek; tek başına satış işareti değildir." if rsi > 70
                       else "30–70 aralığında; belirgin bir aşırı bölge göstermiyor.")
        lines.append(f"• RSI (14): {rsi:.1f} — {explanation}")
    if macd and math.isfinite(float(macd["histogram"])):
        description = ("sinyal çizgisinin üstünde" if macd["histogram"] > 0 else
                       "sinyal çizgisinin altında" if macd["histogram"] < 0 else "sinyal çizgisiyle aynı seviyede")
        lines.append(f"• MACD: {description}. Bu durum yeni bir kesişim olduğu anlamına gelmez.")
    if len(frame) >= 200:
        average = float(frame["close"].tail(200).mean())
        trend = "üzerinde" if close > average else "altında" if close < average else "ile aynı seviyede"
        lines.append(f"• Son 200 kapanışın ortalaması: {average:,.2f} TL. Fiyat bu ortalamanın {trend}." if close != average
                     else f"• Son 200 kapanışın ortalaması: {average:,.2f} TL. Fiyat bu ortalamayla aynı seviyede.")
    else:
        lines.append("• 200 günlük eğilim: yeterli geçmiş yok.")
    if len(frame) < 201:
        lines.append("Uzun dönem eğilimini ve ortalama kesişimlerini değerlendirmek için geçmiş henüz sınırlı.")

    lines.extend(["", "<b>Teknik kurulum</b>"])
    if result.get("signals"):
        lines.append("Stratejinin teknik koşulları sağlandı.")
        lines.extend("• " + escape(str(reason)[:220]) for reason in result["signals"][:5])
        lines.extend([f"Koşulların bozulma seviyesi (stop): {result['sl']:,.2f} TL",
                      f"Senaryo hedefleri: {result['tp1']:,.2f} / {result['tp2']:,.2f} TL",
                      "Hedefler gerçekleşme garantisi taşımaz. Seviyeler bu kapanışa göre hesaplandı; yeni fiyatla yeniden değerlendirilmelidir."])
    else:
        lines.extend(["Şu anda onaylanmış teknik kurulum yok.",
                      "Neden: " + escape(str(result.get("decision_reason", "Koşullar sağlanmadı."))[:300]),
                      "Bu sonuç satış önerisi değildir."])
    lines.extend(["", "<b>Takip edilecekler</b>",
                  "Bir sonraki kapanışta koşulların sürüp sürmediğini ve şirketin yeni açıklamalarını kontrol edin.",
                  "Bu rapor günlük fiyat/hacim verisini yorumlar; şirket haberleri ve finansal tablolar bu değerlendirmeye dahil değildir."])
    lines.append(f"KAP başlık ve özetleri: /haber {escape(symbol.removesuffix('.IS'))}")
    return "\n".join(lines)
