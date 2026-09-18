# Finans Takip Botu

Telegram üzerinden Borsa İstanbul hisselerini izleyen Python uygulaması. RSI,
MACD, ADX ve diğer indikatörlerle sinyal üretir; geçmiş veri testi, ML puanlaması
ve sanal işlem takibi sunar. Borsaya emir gönderme uygulaması içermez.

## Kurulum

Python 3.10 veya üstü gerekir. Komutları proje dizininde çalıştırın:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`.env` henüz yoksa `.env.example` dosyasını `.env` adıyla kopyalayın.
Mevcut `.env` dosyasının üzerine yazmayın. `BOT_TOKEN` ve `CHAT_ID` değerlerini
doldurun. Risk hesabı için `ACCOUNT_SIZE` gerekir; boş bırakıldığında miktar
üretilmez ve portföy kontrolleri açıkken otomatik yeni pozisyon kabul edilmez.
Diğer risk sınırları `.env.example` içinde açıklanmıştır.

```powershell
python bot.py
```

Botu çalıştırmak Telegram bağlantısını ve 15 dakikalık otomatik taramayı başlatır.
Tarama listeleri ve portföy risk kontrolü `bot.py`, ML puanlama ve filtreleme
ayarları `signal_engine.py` üzerinden yönetilir.
Varsayılan `BIST_ONLY=True` otomatik kripto taramasını kapatır. Elle kripto
komutları kullanılabilir; BIST modeli bu verilere uygulanmaz.

## Kullanım

| Komut | İşlev |
| --- | --- |
| `/hisse THYAO.IS` | Yeni başlayanlar için açıklamalı BIST analizi |
| `/hisseler`, `/hisse` | Kod yazmadan hisse seçilebilen, sayfalı BIST menüsü |
| `/kripto BTC` | Kripto fiyatı ve teknik göstergeler |
| `/temettu AAPL` | Temettü bilgileri |
| `/sinyal`, `/kriptosinyal`, `/tamtara` | Elle sinyal taraması |
| `/performans` | Sanal sinyal sonuçları |
| `/portfoy` | Sanal hesap ve pozisyon riskleri |
| `/grafik` | Gerçekleşmiş parasal sonuçlarla hesap değeri ve düşüş grafiği |
| `/mltrain quick`, `/mltrain` | Hızlı veya tam model eğitimi |
| `/izle`, `/help`, `/chatid` | İzleme listeleri ve yardım |

`/hisse` BIST raporu, analizde kullanılan kapanış fiyatını ve mum tarihini
gösterir. RSI/MACD değerleri kısa açıklamalarla sunulur; teknik kurulum yoksa
neden sağlanmadığı, varsa hangi koşulların oluştuğu ve stop/hedef seviyeleri
yazılır. Anlık fiyat ile analizin dayandığı kapanış birbirine karıştırılmaz.
Eksik veya tutarsız OHLC verisinde analiz üretilmez. Eski mumların takvim yaşı
açıkça gösterilir; tatil ile veri gecikmesi otomatik olarak ayırt edilmez.
Bu rapor şirket haberlerini ve finansal tabloları değerlendirmez.

`/start` mesajında da hisse düğmeleri bulunur. Menü, `SCAN_STOCKS` listesinden
otomatik oluşturulur; 12 hisse içeren sayfalarda önceki/sonraki düğmeleriyle
gezinilir. Bir koda dokununca aynı açıklamalı rapor açılır; raporun altından
menüye dönülebilir. Mevcut sohbet erişim kontrolü menü seçimlerinde de uygulanır.

Geçmiş veri testi:

```powershell
python backtest.py --stocks THYAO.IS,ASELS.IS --crypto "" --horizon 60
python backtest.py --all --period 2y --fee-bps 10 --slip-bps 5
```

Bu komutlar piyasa verisi indirir ve varsayılan olarak
`backtest_out/trades.csv` üretir. Komisyon ve kayma hem girişte hem çıkışta
hesaba katılır.
Veri sonunda tam sonuç penceresi bulunmayan adaylar istatistiklere alınmaz;
bu, erken kazananları seçip henüz açık kalan işlemleri dışlama yanlılığını önler.

Otomatik BIST taraması yalnız aynı günün tamamlanmış günlük mumundan sinyal
üretir. Yeni sanal kayıt sonraki seans açılışını bekler; bildirimdeki fiyat
referans kapanıştır. Günlük veri kullanıldığı için açılıştaki sanal gerçekleşme
ve o günün TP/SL sonucu, giriş gününün mumu tamamlandığında kaydedilir.
Bekleyen kayıtlar pozisyon ve risk limitlerinde yer tutar. Açılış stop/hedef
aralığı dışındaysa işlem iptal edilir; fiyat boşluğu riski artırıyorsa miktar
azaltılır. İptaller `backtest_out/live_cancelled.json` içinde tutulur, işlem
başarısına dahil edilmez. Eski kayıtların giriş fiyatları değiştirilmez.

BIST modelini Telegram'ı başlatmadan eğitmek ve bağımsız dönemde değerlendirmek:

```powershell
python train_bist.py --period 5y
python train_bist.py --validate-only
```

İlk komut fiyatları indirir; sonraki çalıştırmalar yerel fiyat önbelleğini kullanır.
Yeni tarihleri almak için `--refresh` ekleyin. İkinci komut kaydedilmiş eğitim
örneklerini kullanır, veri indirmez. Model dosyaları `models/bist_v2/`, fiyatlar,
örnekler ve değerlendirme `backtest_out/bist_training/` altında tutulur.
Son ölçüm ve etkin ayarlar [BIST_ANALIZ.md](BIST_ANALIZ.md) içinde açıklanmıştır.

## Kodun yapısı

| Dosya | Sorumluluk |
| --- | --- |
| `bot.py` | Uygulama başlangıcı, tarama ayarları ve portföy durumu |
| `data_fetcher.py` | Piyasa verisi erişimi ve istek önbelleği |
| `indicators.py` | Temel göstergeler ve fiyat biçimleme |
| `analysis_report.py` | Veri kontrolü ve yeni başlayanlar için açıklamalı hisse raporu |
| `signal_engine.py` | Teknik sinyal kuralları, ML puanı ve bildirim tekrar sınırı |
| `scheduler.py` | Otomatik tarama, risk kontrolleri ve sanal işlem döngüsü |
| `telegram_handlers.py` | Telegram komutları, erişim kontrolü, mesajlar ve grafik |
| `signals_advanced.py` | İndikatörler ve piyasa rejimi özellikleri |
| `ml_model.py` | Özellik kodlama, zaman sıralı eğitim ve model saklama |
| `market_data.py` | BIST günlük kapanış verisi ve tamamlanmış mum kontrolü |
| `train_bist.py` | BIST eğitim komutu ve bağımsız dönem karşılaştırması |
| `risk.py` | Pozisyon büyüklüğü, açık pozisyon ve günlük zarar sınırları |
| `portfolio_risk.py` | Toplam risk, sektör, korelasyon ve düşüş sınırları |
| `paper.py` | Sanal pozisyon kaydı, sonuçlandırma ve performans raporu |
| `backtest.py` | Geçmişte adım adım sinyal ve sonuç değerlendirmesi |

## Doğrulama

```powershell
python -m unittest discover -v
```

Testler sentetik veri ve geçici dosyalar kullanır; Telegram'a mesaj göndermez,
piyasa verisi indirmez, model dosyalarını veya kişisel `.env` dosyasını okumaz.
İndikatör sınırları, BIST kapanışları, eğitim/doğrulama ayrımı, asenkron çağrılar, portföy
limitleri, CSV uyumluluğu ve işlem vadesi için regresyon kontrolleri içerir.
Gerçek komut satırı giriş noktaları, güncellenmiş portföyün kullanılması,
model paketinin yeniden yüklenmesi ve eğitim/backtest/sanal işlemin aynı
sentetik fiyatlarda aynı sonucu üretmesi de kontrol edilir.

## Bu incelemede düzeltilenler

- ADX yön hesabı ve aynı analizde son mumun birden çok kez çıkarılması düzeltildi.
- Geçerli sıfır özellik değerleri ML girdisinde korunur; eksik model geleneksel
  sinyal puanını yapay bir güven katsayısıyla değiştirmez.
- Model veri toplama çağrıları beklenir; model eğitimi ayrı iş parçacığında
  çalışır ve başarılı sonuç çalışan botta etkinleşir.
- Portföy durumu her taramada yeniden kurulur. Hesap değeri, başlangıç sermayesi
  ve miktarı bilinen işlemlerin parasal net sonucuyla hesaplanır.
- Aynı taramadaki adaylar birlikte değerlendirilir; açık pozisyon sayısı ve
  günlük zarar sınırları da otomatik taramaya uygulanır.
- Sektör ve korelasyon yoğunluğu hesap değerine göre ölçülür. İlk pozisyonun
  otomatik olarak yüzde 100 sektör yoğunluğu sayılması önlendi.
- Sanal işlemler TP/SL gerçekleşmeden vadelerinden önce kapatılmaz. İlk işlem
  zararı düşüş hesabına, kazanç/kayıp toplamları kâr faktörüne dahil edilir.
- Sanal işlem kayıtları atomik yazılır; eski CSV sütunları yeni biçime taşınır.
  Miktar, risk tutarı ve parasal net sonuç yeni kayıtlarda saklanır.
- Backtest tekrar bildirim süresi canlı taramayla eşitlendi; son giriş barı
  ve geçersiz pencere/maliyet parametreleri düzeltildi.
- Aynı sembole eşzamanlı veri istekleri tek indirmeyi paylaşır. NumPy ve
  scikit-learn eksik bağımlılıkları eklendi.

## Hesapların sınırları ve sonraki geliştirmeler

`/portfoy` hesap değeri gerçekleşmiş parasal sonuçları kullanır; açık
pozisyonların anlık değerlemesini içermez. Eski miktarsız kayıtların parasal
kazancı tahmin edilmez. Miktarsız açık kayıtlar toplam risk kontrolünü
engelleyebilir; bu kayıtların gerçek miktarı bilinmeden otomatik miktar atanmamalıdır.

Varsayılan otomatik tarama BIST/TL kapsamındadır. Elle başka piyasalar kullanılırsa
TL, USD ve USDT için ortak para birimine dönüşüm henüz yoktur. Korelasyon kontrolü
yalnız mevcut fiyat geçmişleri yeterliyse uygulanır. BIST günlük verisi İstanbul
saatine göre tamamlanır; önceki işlem günleri korunur. Güncel günün verisi 18:20
sonrasında kullanılır; bu, normal seans kapanışına veri gecikmesi payı ekler.
Yarım günler ve olağanüstü seans değişiklikleri için ayrıntılı takvim yoktur;
yarım gün kapanışı da muhafazakâr olarak aynı saati bekler.
[Borsa İstanbul işlem saatleri](https://www.borsaistanbul.com/piyasalar/pay-piyasasi/islem-saatleri).

Performans raporundaki normalize getiri eğrisi ile backtest işlem eğrisi,
eşzamanlı pozisyonları ve nakit kullanımını simüle eden gerçek bir portföy
backtest'i değildir. Yıllıklaştırılmış Sharpe/Sortino metrikleri günlük
örnekleme varsayımına dayanır; farklı sıklıktaki işlem sonuçları için ayrı
kalibrasyon gerekir.

BIST modeli 34 hissenin mevcut geçmişinden yeniden eğitildi; eski karışık piyasa
modelleri korunur fakat yüklenmez. Eğitim hedefleri botun gerçek ATR/stop/TP
kurallarını kullanır; 60 barlık sonuç penceresiyle etiketlenir. MACD model girdisi
hisse fiyatının yüzdesidir. En az 201 tamamlanmış bar olmadan ML uygulanmaz.
`ML_FILTER_SIGNALS=False` varsayılanı, modelin puanını yardımcı bilgi olarak
gösterir. Bağımsız test güçlü bir ayırt etme başarısı göstermediği için bu puan
teknik sinyalleri engellemez. Standart backtest kayıtlı modeli geçmişe uygulamaz;
ML doğrulaması ayrı, zaman sıralı ve örtüşen etiketleri dışlayan testle yapılır.

Başarılı yeni eğitimler ayrı bir model paketi oluşturur; `models/bist_v2/active.json`
tamamlanmış paketi atomik olarak etkinleştirir. Yeniden başlatmada yalnız bu
paket yüklenir; eski rejim dosyaları kendiliğinden devreye girmez. Henüz manifest
bulunmuyorsa mevcut BIST modelleriyle geriye uyumlu yükleme yapılır.

Sonraki model geliştirmeleri henüz görülmemiş dönemlerde ölçülmeli; mevcut test
dönemine göre eşik ayarlamak yeni başarı kanıtı sayılmaz. `.env`, loglar ve
`backtest_out/` çalışma zamanı çıktıları Git dışında tutulur.
