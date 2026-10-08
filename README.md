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
`CHAT_ID` boşsa korunan komutlar çalışmaz; `/chatid` ile kendi sohbet kimliğinizi
öğrenebilirsiniz. Grup kimliği seçilirse yetki o gruptaki komutlara verilir.
Ek analiz kullanıcıları `BOT_DATA_DIR/authorized_chats.json` içindeki JSON
sohbet kimliği listesiyle tanımlanır (yerelde varsayılan proje klasörü,
Docker'da kalıcı `/data` klasörü). Bu dosya Git'e veya imaja eklenmez.
Ek kullanıcılar hisse/kripto analizleri ve tarama komutlarını kullanabilir;
portföy, performans, grafik ve model eğitimi yalnız `CHAT_ID` sahibine açıktır.
KAP bildirimleri `CHAT_ID` ve ek yetkili sohbetlere gider; diğer otomatik
bildirimler yalnız mevcut `CHAT_ID` adresine gitmeye devam eder.

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
| `/haber ASELS`, `/haber` | Hisseye özel veya genel son KAP açıklamaları |
| `/haberogren ASELS`, `/haberogren` | Haber kategorilerinin geçmiş fiyat gözlemleri |
| `/kapdurum` | KAP veri ve ölçüm durumu (yalnız bot sahibi) |
| `/izle`, `/help`, `/chatid` | İzleme listeleri ve yardım |

`/hisse` BIST raporu, analizde kullanılan kapanış fiyatını ve mum tarihini
gösterir. RSI/MACD değerleri kısa açıklamalarla sunulur; teknik kurulum yoksa
neden sağlanmadığı, varsa hangi koşulların oluştuğu ve stop/hedef seviyeleri
yazılır. Anlık fiyat ile analizin dayandığı kapanış birbirine karıştırılmaz.
Eksik veya tutarsız OHLCV verisinde analiz üretilmez. Eski mumların takvim yaşı
açıkça gösterilir; son günlük mum varsayılan 7 takvim günü sınırını aşmışsa
güncel gösterge ve sinyal üretilmez. Tatil ile veri gecikmesi otomatik olarak
ayırt edilmez. Tek hisselik `/sinyal THYAO.IS` komutu da aynı açıklamalı raporu verir.
Bu rapor şirket haberlerini ve finansal tabloları değerlendirmez.

## KAP takibi ve haberlerden öğrenme

`KAP_ENABLED=true` (varsayılan) ile bot, KAP'ın herkese açık bildirim listesini
30 dakikada bir kontrol eder. `SCAN_STOCKS` içindeki hisseler, bildirimin hisse
ve ilgili hisse kodlarıyla tam eşleştirilir. İlk çalışmada son 7 takvim günü
sessizce arşivlenir; sonraki yeni açıklamalar bot sahibine ve
`authorized_chats.json` içindeki ek sohbetlere bildirilir. Alıcı listesi her
döngüde yeniden okunur; yeni eklenen kişilere eski arşiv topluca gönderilmez.
Her alıcıya bir döngüde en fazla 10 bildirim gönderilir; kalanlar sıradaki döngüye kalır.
Teslim durumu alıcı bazında saklanır; bir alıcının hatası diğerlerini engellemez,
başarılı alıcılara tekrar gönderilmez. Listeden çıkarılan alıcının kuyruğu silinir.
Alıcının önce Telegram'da botu başlatması ve engellememiş olması gerekir.
Başarısız gönderimler tekrar denenir; gönderim ile kayıt arasındaki ani kapanma
tek bir bildirimin tekrarına yol açabilir. Ek yetkili kullanıcılar haber ve
öğrenme komutlarını kullanabilir, otomatik haber alıcısını değiştiremez.

Başlık ve KAP'ın resmi kısa özeti, kaynak bağlantısıyla gösterilir. Tam metin
ve PDF ekleri okunmaz; başlık/özetteki anahtar kelimelerle konu sınıflandırması
yapılır. Bu sınıflandırma duygu analizi veya olumlu/olumsuz yatırım kararı değildir.
Bu erişim lisanslı API değildir ve sitenin değişmesiyle kesilebilir. Sürekli
kapsamlı arşiv erişimi için [KAP'ın resmi veri servisi koşulları](https://www.kap.org.tr/tr/api/about/content-file/8a019492945fbe080194b26d8bed4873)
ayrıca değerlendirilmelidir. Bu özellik ücretli API veya LLM çağrısı yapmaz.

"Öğrenme" ilk sürümde tahmin modeli değil, kalıcı tarihsel gözlem arşividir:

- Her 6 saatte fiyatlar kontrol edilir; yayından **sonraki seansın açılışından**
  1, 5 ve 20. seans kapanışına kadar düzeltilmiş fiyat değişimi ölçülür.
- Tamamlanmamış seanslar, eksik sonuç pencereleri, düzeltme/ilişkili bildirimler
  ve sonradan değişen metinler istatistiklere alınmaz.
- Aynı hisse ve ufukta örtüşen pencereler, kategoriler farklı olsa bile tekrar
  sayılmaz. Kategori başına en az 10 örtüşmeyen örnek olduğunda medyan değişim
  ve pozitif sonuç oranı gösterilir. Bu eşik istatistiksel güven garantisi değildir.
- Tarihler eşleşiyorsa BIST 100'e göre fark da hesaplanır. Veri hatasında
  eski kayıtlar korunur ve eksiklik raporda belirtilir.

İlk kurulum geçmiş yılları öğrenmiş sayılmaz: son 7 günle başlar ve zamanla
birikir. Uzun kesintilerde 7 günden eski eksik bildirimler tamamlanmaz; arşiv
boşluğu görünür tutulur. Mevcut takip listesi seçilim yanlılığı taşır; sonuçlar
nedensellik, yatırım getirisi veya gelecekteki başarı kanıtı değildir. İşlem
maliyeti/kayma ve yayın günündeki ilk fiyat tepkisi bu ölçümde yoktur.
Bu özellik teknik AL/SAT sinyallerini ve mevcut ML modelini değiştirmez.

Kayıtlar `BOT_DATA_DIR/kap_news.sqlite3` dosyasında tutulur (yerelde varsayılan
`backtest_out`, Docker'da kalıcı `/data`). Güncellemelerde korunur; yedeklere
bu dosyayı da dahil edin. `KAP_ENABLED=false` takibi ve otomatik ölçümü durdurur,
kayıtlı arşiv komutlarla okunmaya devam eder.

## Veri ve aşırı hareket kontrolü

Sinyal motoru, manuel tarama, otomatik tarama ve tarihsel testlerde ortak veri
kontrolünü kullanır. Tamamlanmış mumlarda OHLC fiyatları sonlu ve pozitif,
açılış/kapanış düşük-yüksek aralığında, hacim sonlu ve negatif olmayan bir sayı
olmalıdır. Eksik alanlar doldurularak sinyal üretilmez. Son mumda sıfır hacim veya
günlük hissede önceki 20 mumun medyan hacminin sıfır olması da yeni sinyali durdurur.

Günlük hisse verisinde son hacim **önceki 20 mumun medyanının en az 5 katı** ve
kapanış değişiminin mutlak değeri **en az %6** veya mumun yüksek-düşük aralığı
önceki kapanışa göre **en az %8** ise sinyal engellenir. Fiyat/aralık koşulu,
hacim koşuluyla birlikte aranır. Yalnız hacim sıçraması uyarı olarak gösterilir.
Bu günlük eşikler saatlik kripto verisine uygulanmaz. Ayarlar `.env.example`
içindedir; geçersiz eşik girilirse kontrol sessizce kapanmak yerine sinyali durdurur.

Bu başlangıç eşikleri ölçülmüş bir başarı iddiası veya manipülasyon tespiti
değildir. Meşru haber kaynaklı hareketleri de durdurabilir; gerçek manipülasyonu
kaçırabilir. Kontroller [Borsa İstanbul'un resmi tedbirlerinin](https://www.borsaistanbul.com/piyasalar/pay-piyasasi/piyasa-isleyisi)
yerine geçmez. KAP haber takibi ayrı bir bilgi katmanıdır; resmi tedbir listesi
henüz işlem engelleme kurallarına bağlanmamıştır.

`/hisse` ve tek hisselik `/sinyal` raporları engellenme nedenini gösterir.
Otomatik taramada değerlendirmeye gelen engellenmiş hisse için Telegram koruma
uyarısı gönderilir; yeni sanal işlem açılmaz. Aynı sembol, mum ve neden için
24 saat tekrar uyarı gönderilmez. Gönderim başarısızsa sonraki tarama yeniden dener.
Güncel olmayan günlük BIST mumları otomatik taramada zaten işlem adayı yapılmaz.

Canlı komutlarda saatlik kripto mumları için varsayılan güncellik sınırı 3 saat,
günlük hisseler için 7 takvim günüdür. Bunlar tam bir seans/tatil takvimi değildir.
Tarihsel testler duvar saatine göre eskilik kontrolü yapmaz; yalnız o karar anına
kadar olan mumları kullanır. Sonraki mumları değiştirmenin eski kararı etkilememesi
regresyon testleriyle kontrol edilir.

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
| `signal_safety.py` | Ortak OHLCV doğrulama, güncellik ve aşırı fiyat-hacim kontrolü |
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

Yeni güvenlik regresyonları bozuk OHLCV, eksik hacim, eski/gelecek tarih, tamamlanmamış
mum, fiyat-hacim sıçraması, uyarı tekrarı ve eksik sohbet yetkisini kapsar.
Log çıktılarında Telegram tokenleri maskelenir; ana log dosyası yaklaşık 5 MiB
olunca döndürülür ve üç yedek tutulur. Önceden yazılmış loglar geriye dönük silinmez
veya temizlenmez; döndürülen log dosyaları da Git dışında tutulur.

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

Stop seviyesinin altına açılış boşluğu oluşursa sanal işlem, backtest ve yeni
eğitim örneklerinde çıkış fiyatı açılışa düşürülür. Örneğin stop 90 iken sonraki
mum 80'den açılırsa hesap 90 yerine 80 üzerinden yapılır. Bu hâlâ bir simülasyondur;
gerçek emir defteri, tabanda alıcı bulunamaması ve likidite kayması modellenmez.
Eski ve açılış sütunu bulunmayan veri için stop seviyesindeki eski davranış korunur.
Tutarsız sonuç mumlarıyla kâr/zarar yazılmaz; canlı kayıtlarda pozisyon korunarak
yeniden değerlendirme beklenir. Mevcut işlem kayıtları yeniden yazılmamıştır.
Yeni sinyal koruması ve açılış boşluğu hesabı strateji sonuçlarını değiştirebilir;
eski model puanları ve eski başarı ölçümleri bu sürümün doğrulanması sayılmaz.
Kaydedilmiş model otomatik yeniden eğitilmez.

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
