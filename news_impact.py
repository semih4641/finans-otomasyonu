"""Conservative, deterministic scenarios from KAP titles/short summaries.

Not a language model, valuation model or price forecast. No external calls.
Unknown, negated or revised disclosures must not acquire a directional score.
"""
from dataclasses import dataclass
from html import escape
import re


@dataclass(frozen=True)
class Impact:
    outlook: str
    reason: str
    checks: str


def normalized(value):
    return " ".join(str(value or "").casefold().replace("i̇", "i").split())


def assess_impact(row):
    text = normalized(row.get("title")) + " " + normalized(row.get("summary"))
    unknown = Impact(
        "Yön belirsiz",
        "Başlık ve kısa özet, şirkete ekonomik etkisini belirlemek için yeterli değil.",
        "Tam açıklama, tutar, zamanlama ve şirketin büyüklüğüyle karşılaştırma gerekli.")
    if row.get("amended") or "düzelt" in text:
        return Impact("Yön belirsiz — düzeltme/ilişkili açıklama",
                      "Önceki açıklamayla neyin değiştiği karşılaştırılmadan yön yorumu yapılamaz.",
                      "İlk açıklama ve düzeltmenin tam metni birlikte incelenmeli.")
    # Conservative abstention: Turkish negation can reverse a headline's meaning.
    # This is deliberately broader than a sentiment parser, not comprehensive NLP.
    if re.search(r"değil|yok|olumsuz|redd|vazgeç|sonlandır|ertelen|iptal|fesih|"
                 r"\b(?:gerçekleş|imzalan|feshedil|dağıtıl|alın|bulun|sonuçlan|yapıl|ol)(?:ma|me)(?:d|y|m)", text):
        return Impact("Yön belirsiz — olumsuzluk/iptal ifadesi",
                      "Özetteki ifade bir iptal, ret veya bunların olmadığını anlatıyor olabilir; otomatik olumlu yorum yapılmadı.",
                      "İşlemin gerçekten iptal edilip edilmediği ve parasal etkisi tam metinden doğrulanmalı.")
    if any(term in text for term in ("konkordato", "iflas", "temerrüt", "idari para cezası", "dava")):
        return Impact("Hukuki/finansal risk açısından dikkat",
                      "Böyle bir yükümlülük gerçekleşir ve şirket için önemli tutardaysa nakit akışı veya faaliyetler üzerinde baskı yaratabilir; sonuç kesinleşmiş sayılmaz.",
                      "Şirketin olayda tarafı, süreç/itiraz aşaması, tutar ve olası karşılıklar tam metinden doğrulanmalı.")
    if any(term in text for term in ("devre kesici", "vbts", "volatilite bazlı", "brüt takas", "işlem sırası")):
        return Impact("Oynaklık/işlem koşulları açısından dikkat",
                      "Devre kesici veya tedbir bildirimi tek başına yükseliş, düşüş ya da manipülasyon kanıtı değildir.",
                      "Uygulamanın türü, süresi ve emir/işlem kısıtları kontrol edilmeli; mevcut durum yalnız bu başlıktan anlaşılamaz.")
    if "bedelsiz" in text:
        return Impact("Otomatik değer artışı anlamına gelmez",
                      "Bedelsiz artırımda pay sayısı ve pay başına fiyatın birlikte değerlendirilmesi gerekir; tek başına şirketin toplam değerini artırmaz.",
                      "Oran, onay ve hak kullanım tarihi; aynı açıklamada bedelli bölüm olup olmadığı kontrol edilmeli.")
    if "bedelli" in text or "rüçhan" in text:
        return Impact("Koşullara bağlı / çift yönlü",
                      "Yeni kaynak şirketi destekleyebilir; katılmayan ortağın pay oranı seyrelme riski taşır. Doğrudan olumlu veya olumsuz sayılamaz.",
                      "Fon kullanım amacı, ihraç fiyatı, oran ve rüçhan koşulları gerekli.")
    if "sermaye" in text:
        return Impact("Yön belirsiz — sermaye işlemi",
                      "Sermaye tavanı izni, gerçekleşmiş sermaye artırımı ile aynı şey değildir; başlık tek başına finansal etkiyi göstermez.",
                      "İşlem türü, gerçekleşme/onay aşaması ve ortakların hakları kontrol edilmeli.")
    if any(term in text for term in ("kar payı", "kâr payı", "temettü")):
        return Impact("Nakit dağıtım koşullarına bağlı",
                      "Dağıtım kararı varsa nakit getiri sağlayabilir; hak kullanımındaki fiyat düzeltmesi nedeniyle bedelsiz ek kazanç olarak okunmamalı.",
                      "Dağıtım yapılıp yapılmadığı, pay başına tutar, onay, tarih ve nakit akışı gerekli.")
    if "geri alım" in text or "geri alınan" in text:
        return Impact("Koşullu destek potansiyeli",
                      "Şirketin fiilen pay alması talebi destekleyebilir; program ilanı, alımın gerçekleştiği anlamına gelmez ve fiyat garantisi vermez.",
                      "Gerçekleşen miktar, fiyat, program bütçesi ve şirketin nakit durumu gerekli.")
    if any(term in text for term in ("finansal rapor", "finansal tablo", "bilanço", "faaliyet raporu")):
        return Impact("Yön belirsiz — rakamlar gerekli",
                      "Raporun yayımlanması tek başına olumlu haber değildir; kâr, nakit akışı ve beklentilerle fark bilinmeden fiyat yönü çıkarılamaz.",
                      "Dönemsel karşılaştırma, tek seferlik kalemler, borç ve piyasa beklentisi gerekli; tablolar bu yorumda okunmadı.")
    if any(term in text for term in ("yeni iş", "sözleşme", "ihale")):
        return Impact("Koşullu olumlu potansiyel — gerçekleşme teyidi gerekli",
                      "Yeni iş kesinleşir, önemli büyüklükte ve kârlı olursa gelir beklentisini destekleyebilir. Başvuru/görüşme, imzalanmış sözleşme değildir.",
                      "Tutarın yıllık ciroya oranı, kâr marjı, teslim/tahsilat takvimi ve haberin önceden fiyatlanıp fiyatlanmadığı gerekli.")
    return unknown


def render_impact(row):
    impact = assess_impact(row)
    lines = ["🔎 <b>Olası hisse etkisi — kural tabanlı ön yorum</b>",
             "<b>Görünüm:</b> " + escape(impact.outlook),
             "<b>Neden:</b> " + escape(impact.reason),
             "<b>Kontrol:</b> " + escape(impact.checks),
             "Dayanak: yalnız başlık/kısa özet; tam metin okunmadı. Fiyat yönüne güven düşük; hedef fiyat veya AL/SAT sinyali değildir."]
    if len(str(row.get("symbols") or "").split(",")) > 1:
        lines.append("Birden fazla ilgili hisse var; olayın her şirkete etkisi aynı olmayabilir.")
    return "\n".join(lines)
