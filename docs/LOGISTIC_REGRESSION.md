# Logistic Regression: yöntem, deney ve hata analizi

Bu not 3. kişinin model bölümüdür. Ortak veri/özellik deneylerinin ve
modeller arası karşılaştırmanın yerine geçmez; karşılaştırmayı 5. kişi hazırlar.
Tüm sayılar validation bölümündendir. Resmî test henüz çalıştırılmadı.

## Nasıl çalışıyor?

`Mesaj → TF-IDF özellikleri → Logistic Regression → 77 kategoriden biri`

Özellikler Naive Bayes ile birebir aynıdır: `TfidfVectorizer(ngram_range=(1, 2),
sublinear_tf=True)`, yalnızca train üzerinde öğrenilir. Böylece modeller arasındaki
fark özelliklerden değil sınıflandırıcıdan gelir. Mevcut veri sürümünde
vocabulary 21.595 unigram/bigram içerir.

Logistic Regression her kategori `c` için bir ağırlık vektörü `w_c` ve sabit `b_c`
öğrenir. Mesajın TF-IDF vektörü `x` için:

`score(c) = w_c · x + b_c`

İki çok sınıf yöntemi denendi:

- **Multinomial (softmax)** — `lbfgs` ve `saga`: 77 skor birlikte softmax ile
  olasılığa çevrilir, `P(c|x) = exp(score(c)) / Σ_k exp(score(k))`. Tek bir ortak
  kayıp fonksiyonu tüm sınıflar için birlikte en aza indirilir.
- **One-vs-rest (OvR)** — `liblinear-ovr`: her kategori için "bu kategori mi,
  değil mi?" diye ayrı bir ikili Logistic Regression eğitilir. En yüksek skoru
  veren kategori seçilir. liblinear çok sınıfı doğrudan desteklemediği için
  `OneVsRestClassifier` ile sarılmıştır.

Naive Bayes'ten farkı: NB kelimelerin sınıf içi sıklığını sayarak ağırlık
çıkarır ve özellikleri bağımsız varsayar. LR ise ağırlıkları doğrudan doğru
sınıfı ayırmak için optimize eder; aynı anlamı taşıyan bigram ve unigram'ların
birlikte bulunmasını kendisi dengeler.

**C parametresi** L2 düzenlileştirmenin tersidir. Kayıp
`C × (sınıflandırma hatası) + ½‖w‖²` biçimindedir. Küçük C ağırlıkları
sıfıra doğru güçlü biçimde iter (daha basit model, underfitting riski); büyük C
eğitim verisine daha sıkı uyar (overfitting riski). Doğru değer validation ile seçilir.

## Deney tasarımı

| Ayar | Değerler |
| --- | --- |
| C | 0.1, 1, 10, 100, 1000 |
| Solver | lbfgs (varsayılan), saga, liblinear-ovr |
| Sabit | L2, `max_iter=2000`, `random_state=42`, NB ile aynı TF-IDF |
| Seçim ölçütü | En yüksek validation macro F1; eşitlikte çizelgedeki ilk ayar |

Hocanın istediği C değerleri 0.1, 1 ve 10'dur. 100 ve 1000, en iyi değerin
ızgaranın kenarında kalıp kalmadığını kontrol etmek için eklendi. Toplam 15
çalıştırma, 15 çalıştırmanın tamamı yakınsadı (ConvergenceWarning yok).

Tek komutla aynı 15 ayarı çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -m banking77.data --offline
.\.venv\Scripts\python.exe -m banking77.benchmark_logistic_regression
```

Script yalnızca validation kullanır ve [C/solver tablosunu](../results/LOGISTIC_REGRESSION_C.md)
ile [JSON kaydını](../results/logistic_regression_validation.json) yeniden üretir.
Tam tahminler ve confusion matrix'ler `results/runs/`, modeller `artifacts/` altında yerelde kalır.
Bu bilgisayarda (2 çekirdek) 15 çalıştırmanın eğitim + tahmin süresi toplam ~2 dakikadır.

Tek bir ayarı çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -m banking77.train_logistic_regression --solver liblinear-ovr --C 100
.\.venv\Scripts\python.exe -m banking77.train_logistic_regression --solver lbfgs --C 10
```

## Sonuçlar

Mevcut veri sürümünde 8.499 train ve 1.500 validation kaydı vardır;
`dataset_summary_sha256` Naive Bayes deneyiyle aynıdır
(`468f5502…`). Validation ile train arasında metin örtüşmesi yoktur.

| Ayar | Accuracy | Macro F1 | Eğitim (s) |
| --- | ---: | ---: | ---: |
| lbfgs, C=0.1 | 66.67% | 0.6126 | 3.8 |
| lbfgs, C=1 (başlangıç) | 85.80% | 0.8555 | 13.7 |
| lbfgs, C=10 | 88.60% | 0.8872 | 18.8 |
| lbfgs, C=100 | 88.33% | 0.8838 | 17.6 |
| lbfgs, C=1000 | 87.47% | 0.8757 | 12.9 |
| saga, C=100 | 88.93% | 0.8912 | 14.5 |
| liblinear-ovr, C=10 | 89.00% | 0.8916 | 2.7 |
| **liblinear-ovr, C=100 (seçilen)** | **89.07%** | **0.8920** | **3.1** |
| liblinear-ovr, C=1000 | 88.93% | 0.8904 | 3.6 |

Tüm 15 satır [tabloda](../results/LOGISTIC_REGRESSION_C.md). Süreler tek
bilgisayarda tek ölçümdür; tahmin süresi her ayarda mesaj başına 0.1 ms altındadır.

### C'nin etkisi

C en belirleyici parametredir. C=0.1 açıkça underfitting yapar (macro F1 0.61):
77 sınıf ve kısa mesajlar için ağırlıklar fazla küçük tutuluyor. C=1'den C=10'a
geçiş macro F1'i 0.856'dan 0.887'ye çıkarır. Üç solver'da da 10–100 arası bir
plato vardır ve C=1000'de skor düşer. Yani en iyi bölge ızgaranın içindedir.

### Solver'ın etkisi

Aynı C'de solver'lar arasındaki fark küçüktür (en iyi ayarlarda ±0.005 macro F1).
Multinomial çözücüler (lbfgs, saga) aynı kayıp fonksiyonunu çözdüğü için
C=0.1–10 arasında neredeyse aynı skoru verir. C=100'de lbfgs (0.8838) ile saga
(0.8912) ayrışıyor; zayıf düzenlileştirmede optimum düzleştiği için iki çözücünün
farklı noktalarda durması olası bir açıklamadır, ayrıca ölçülmedi. Pratikte belirgin fark süredir:
lbfgs C=10 ile eğitim ~19 sn, liblinear-ovr ~3 sn sürer.

## Seçilen ayar ve gerekçesi

**Seçilen: `liblinear-ovr`, `C=100`** — validation accuracy %89.07, macro F1 0.8920.

1. Önceden belirlenen kural (en yüksek validation macro F1) bu ayarı seçer.
   Macro F1 kullanıyoruz çünkü 77 kategorinin her birindeki başarıyı eşit
   önemsiyoruz; accuracy de destekleyici olarak raporlanır.
2. C seçimi güçlü bir kanıta dayanır. Başlangıç ayarına (lbfgs, C=1) göre
   yanlış sayısı 213 → 164; 66 hata düzeldi, 17 doğru tahmin bozuldu.
   Exact McNemar testinde p = 5.6×10⁻⁸; iyileşme şansla açıklanamaz.
3. Solver seçimi zayıf bir kanıta dayanır. En iyi lbfgs ayarına (C=10) göre
   171 → 164 yanlış; 20 düzeldi, 13 bozuldu, McNemar p = 0.30. Bu fark
   istatistiksel olarak anlamlı değildir. OvR'nin multinomial'den daha iyi
   olduğu iddia edilmez; seçimi belirleyen kural ve ek olarak ~6 kat kısa eğitim süresidir.
4. C=10 ile C=100 (OvR) arasındaki fark 0.0004 macro F1'dir; ikisi de aynı platodadır.
   Kayıt dışı bir ön kontrolde ara değer C=30 (OvR) 0.8929 verdi; bu da aynı platodadır. Izgarayı
   daha da inceltmek validation'a aşırı uyum riski taşıdığı için yapılmadı.

Ayar bu haliyle dondurulmuştur. Nihai test, README'deki kurala göre 2. kişinin
veri/özellik çalışması kesinleştikten sonra çalıştırılacaktır:

```powershell
.\.venv\Scripts\python.exe -m banking77.train_logistic_regression --solver liblinear-ovr --C 100 --split test
```

Test script'i hazırlanmış train bölümüyle eğitir, validation eğitime eklenmez.
Resmî test ile train arasında 7 metin örtüşmesi vardır; script train ile
örtüşmeyen altkümenin skorlarını `nonoverlapping_subset` olarak ayrıca kaydeder.

## Hata analizi

Seçilen ayar 1.500 validation mesajının 164'ünü yanlış sınıflandırdı.
Validation satır kimlikleri `train-…` ile başlar, çünkü validation resmî
eğitim kümesinden ayrılmıştır.

**Hatalar çoğunlukla aynı konu ailesindedir.** Gerçek ve tahmin edilen etiketin
ortak bir konu kelimesi taşıdığı hatalar (`top_up`, `card`, `transfer`,
`cash_withdrawal`, `payment`, `fee`, `pending`, `declined` vb.) 164 hatanın
106'sıdır (%65). Bu, etiket adlarına dayalı kaba bir sayımdır. Model genelde
konuyu doğru bulup aynı konudaki alt kategoriyi (yöntem, durum veya ücret türü) karıştırıyor.

**Per-class:** 77 kategoriden 4'ünde F1 = 1.0 (`age_limit`, `passcode_forgotten`,
`verify_top_up`, `virtual_card_not_working`); 7'sinde F1 < 0.80. En düşük:
`top_up_failed` 0.750, `card_acceptance` 0.762, `pending_top_up` 0.762,
`pending_cash_withdrawal` 0.765, `compromised_card` 0.786.

### En çok karışan kategori çiftleri

| Gerçek kategori | Tahmin | Sayı |
| --- | --- | ---: |
| direct_debit_payment_not_recognised | card_payment_not_recognised | 5 |
| top_up_by_bank_transfer_charge | top_up_by_card_charge | 4 |
| transfer_fee_charged | card_payment_fee_charged | 3 |
| top_up_failed | top_up_reverted | 3 |
| pending_cash_withdrawal | declined_cash_withdrawal | 3 |

Yön ayırmadan sayıldığında `top_up_failed ↔ top_up_reverted` ve
`pending_top_up ↔ top_up_failed` çiftleri de 4'er hatayla ilk sıralardadır.

### Örnek hatalar ve kelime katkıları

Linear modelde her kelimenin karara katkısı doğrudan ölçülebilir. Aşağıdaki
"katkı" değeri `(w_tahmin − w_gerçek) × tfidf` farkıdır: pozitif değer mesajı yanlış
kategoriye, negatif değer doğru kategoriye iter. Bu, kaydedilmiş modelin
katsayılarından hesaplanmıştır; NB notundaki gibi yalnızca bir tahmin değildir.

| Mesaj | Gerçek | Tahmin | Ölçülen açıklama |
| --- | --- | --- | --- |
| There is a payment showing on my app that I didn't do. Will you please cancel this payment and refund my money? (`train-04812`) | direct_debit_payment_not_recognised | card_payment_not_recognised | Mesajda "direct debit" geçmiyor. En büyük katkı `payment` (+2.89) ve `this payment` (+1.63) kelimelerinden, bunlar eğitimde kart ödemesiyle daha sık geçiyor. Bu çiftin 5 hatasının hiçbirinde "direct debit" kelimesi yok; insan için de etiket metinden çıkarılamaz. |
| Are there topping up fees if I have to transfer? (`train-01901`) | top_up_by_bank_transfer_charge | top_up_by_card_charge | Model doğru ipucunu görüyor: `transfer` doğru kategoriye −2.58 iter. Ama `up fees` (+2.32), `topping` (+1.58) ve `fees` (+1.39) toplamda daha ağır basıyor. Bu çiftin 4 hatasının 3'ünde aynı örüntü var: "top up + ücret" kalıbı tek başına `transfer` kelimesini geçiyor. Dördüncüsünde (`train-01856`) kelime `trasfer` diye yanlış yazıldığı için doğru ipucu hiç görülmüyor. |
| My atm withdraw is stillpending (`train-00863`) | pending_cash_withdrawal | declined_cash_withdrawal | Yazım hatası: "stillpending" tek token olduğu için vocabulary'de yok ve kararı belirleyecek `pending` kelimesi hiç görülmüyor. Kalan `my atm` (+2.80), `atm`, `withdraw` iki kategoride de ortak. Bu, kelime tabanlı TF-IDF'in karakter düzeyindeki hatalara karşı kırılganlığını gösterir. |

Ek gözlemler:

- `train-00867` ("…did not go through due to a declined card…") `pending_cash_withdrawal`
  olarak etiketlenmiş ama mesajda `declined` (+1.68) açıkça geçiyor. Bu örnek
  bir model hatasından çok etiket belirsizliğine/gürültüsüne benziyor.
- `top_up_failed → top_up_reverted` hatalarında doğru kelimeler (`rejected` −3.16,
  `worked` −3.03, `work` −2.59) doğru yöne itiyor, ama `up was`, `my top`, `did my`
  gibi bigram'lar reverted kategorisine daha güçlü bağlanmış.

### Sonuç ve sınırlamalar

- C'nin doğru seçilmesi büyük ve anlamlı bir kazanç sağladı; solver seçimi sağlamadı.
- Kalan hataların çoğu, mesajın alt kategoriyi ayırt edecek bilgiyi içermediği
  veya bu bilginin tek bir kelimede olduğu durumlar. Kelime tabanlı TF-IDF, yazım hatalı
  biçimleri (`trasfer`, `stillpending`) doğru kelimeyle birleştiremez; karakter n-gram veya
  embedding tabanlı özellikler bu durumlar için denenebilir (2. kişinin özellik
  çalışmasının kapsamıdır, burada denenmedi).
- Sonuçlar tek bir 1.500 satırlık validation bölümüne dayanır. Macro F1'de
  ±0.005 içindeki farklar (yaklaşık 5–10 mesaj) bu boyutta güvenilir ayrım sayılmamalıdır.
- Model skorları kalibrasyon kontrolü yapılmadan doğruluk olasılığı gibi yorumlanmamalıdır.
