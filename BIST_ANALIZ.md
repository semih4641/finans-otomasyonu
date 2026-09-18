# BIST teknik analiz ve model değerlendirmesi

Değerlendirme: 17 Eylül 2026. Fiyat verisinin son günü: 16 Eylül 2026.

İlk model çalışmasında 99 otomatik test geçti. Beş yeni BIST modelinin bot tarafından
yüklenmesi ve gerçek fiyat geçmişinden sonlu puan üretmesi ayrıca kontrol edildi.

## Uygulanan değişiklikler

- Otomatik tarama ve model eğitimi BIST hisselerine odaklandı. Eğitimde kripto
  veya yabancı hisse örneği kullanılmaz.
- Model yalnız teknik stratejinin kabul ettiği sinyallerden öğrenir. Etiketler
  sabit yüzde hedeflerden değil, botun gerçek stop ve hedef seviyelerinden üretilir.
- Giriş sonraki barın açılışıdır. Aynı barda hedef ve stop varsa stop önceliklidir.
  Vade 60 işlem barıdır; yeterli ileri veri olmayan sinyaller etiketlenmez.
- Hacim karşılaştırması önceki 20 barı kullanır. Modelde MACD fiyat yüzdesine
  çevrilir; pahalı ve ucuz hisseler aynı ölçekte karşılaştırılır.
- Kapanmış BIST günlük mumları korunur. Gün içindeki güncel mum filtrelenir;
  İstanbul saati kullanılır. Normal kapanış için veri gecikmesi payı uygulanır.
- Yeni modeller `models/bist_v2/` altında saklanır. Eski, uyumsuz modeller
  korunur ve yeni özelliklerle kullanılmaz.

## Modüler yapı sonrası doğrulama

Backtest ve sanal işlem veri bağlantıları gerçek modüllerle kontrol edilir;
otomatik tarama ve `/portfoy` yenilenen portföy nesnesini kullanır. Yeni BIST
sinyalleri kapanıştan sonra kaydedilip sonraki seans açılışını bekler. Günlük
veri nedeniyle sanal giriş, giriş gününün mumu tamamlandığında doğrulanır.
Geçersiz açılış boşlukları iptal edilir; bekleyen işlemler risk limitlerinde
yer tutar. Model paketleri atomik manifest üzerinden etkinleştirilir.

Eğitim, backtest ve sanal işlem için ortak sentetik senaryolar; TP1, TP2,
aynı mumda stop/hedef ve vade sonu sonuçlarının maliyetler dahil eşleştiğini
kontrol eder. Backtest tam sonuç penceresi bulunmayan adayları dışlar; yakın
tarihli erken kazananları tek başına ölçüme almaz.

Bu yazılım kontrolleri yeni model performansı ölçümü değildir. Aşağıdaki
bağımsız dönem sonuçları önceki ölçüme aittir; ML filtresi kapalı kalır.

## Ölçüm

34 hissenin beş yıla kadar mevcut Yahoo Finance geçmişi incelendi. Yeni halka
arz edilenlerin geçmişi daha kısadır. AAGYO için 201 barlık başlangıç geçmişi
olmadığından eğitim örneği üretilmedi. Diğer 33 hisseden 1.012 uygun sinyal çıktı.

746 sinyal eğitimde kullanıldı; test sınırına taşan sonuç pencereleri nedeniyle
58 sinyal çıkarıldı. Son 208 sinyal eğitimden ayrı tutuldu. Testteki karar tarihleri
15 Eylül 2025–22 Haziran 2026 aralığında; sonraki fiyatlar 60 barlık sonuçları
değerlendirmek için kullanıldı. Model ve eşik bu test sonuçlarına göre ayarlanmadı.

Botun rejim modeli ve global yedek model seçimini birlikte uygulayan test:

| Ölçüm | Teknik sinyallerin tamamı | ML puanı ≥ 0,55 olanlar |
| --- | ---: | ---: |
| Sinyal sayısı | 208 | 46 |
| Stop öncesi hedefe ulaşma | %48,08 | %50,00 |
| Ortalama net sanal işlem sonucu | %0,91 | %1,28 |

ML ROC-AUC: **0,467**. Brier skoru: **0,259**. Seçilen alt kümede sonuçlar
biraz daha iyi görünse de genel ayrıştırma gücü ve sınırlı örneklem, modelin
üstünlüğünü doğrulamıyor. Bu nedenle **ML puanı yardımcı bilgi olarak açık,
ML ile sinyal engelleme kapalıdır**. Teknik kurallar sinyal kabulünü belirler.

Bu rakamlar örtüşebilen sanal işlemlerin sonuçlarıdır; portföy getirisi değildir.
Giriş ve çıkışta yüzde 0,15 varsayımsal maliyet vardır. Takip listesi bugünkü
liste olduğundan hayatta kalma yanlılığı olabilir. Günlük OHLC, fiyat boşlukları,
tavan/taban likiditesi ve gerçek emir gerçekleşmesini tam olarak modellemez.

## Dosyalar ve tekrar çalıştırma

- `backtest_out/bist_training/validation.json`: ayrıntılı doğrulama ve kapsam.
- `backtest_out/bist_training/samples.csv`: kullanılan sinyaller ve sonuçları.
- `models/bist_v2/`: bütün uygun geçmiş verilerle yeniden eğitilmiş modeller.

```powershell
python train_bist.py --period 5y --refresh
python train_bist.py --validate-only
python -m unittest discover -v
```

`--validate-only` aynı testi tekrarlamak içindir; yeni bağımsız kanıt sağlamaz.
Sonraki performans doğrulaması, henüz görülmemiş yeni bir dönemde yapılmalıdır.
