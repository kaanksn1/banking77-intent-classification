# Linear SVM: yöntem, deney ve hata analizi

Bu not 4. kişinin model bölümüdür. Ortak veri/özellik deneylerinin ve
modeller arası karşılaştırmanın yerine geçmez; karşılaştırmayı 5. kişi hazırlar.
Tüm sayılar validation bölümündendir. Resmî test henüz çalıştırılmadı.

## Nasıl çalışıyor?

`Mesaj → TF-IDF özellikleri → Linear SVM → 77 kategoriden biri`

Özellikler Naive Bayes ve Logistic Regression ile birebir aynıdır:
`TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)`, yalnızca train üzerinde
öğrenilir. Böylece modeller arasındaki fark özelliklerden değil sınıflandırıcıdan
gelir. Mevcut veri sürümünde vocabulary 21.595 unigram/bigram içerir.

**Mimari ve somut model.** Mimari, doğrusal maksimum-margin sınıflandırıcıdır
(Linear SVM). Somut model scikit-learn `LinearSVC`'dir: liblinear'ın dual coordinate
descent çözücüsü, L2 düzenlileştirme ve one-vs-rest (OvR) çok sınıf stratejisi.
Kernel kullanılmaz; 21.595 boyutlu seyrek TF-IDF uzayında sınıflar zaten büyük ölçüde
doğrusal ayrılabildiği için doğrusal SVM metin sınıflandırmada standart baseline'dır.

Model her kategori `c` için bir ağırlık vektörü `w_c` ve sabit `b_c` öğrenir:

`score(c) = w_c · x + b_c`

OvR'de her kategori için "bu kategori mi, değil mi?" diye ayrı bir ikili SVM
eğitilir (77 model). Tahmin, en yüksek skoru veren kategoridir. SVM skorları
olasılık değildir; kalibrasyon yapılmadan güven değeri gibi yorumlanmamalıdır.

Her ikili SVM şu amacı en aza indirir (`y = +1` bu kategori, `−1` diğerleri):

`½‖w‖² + C × Σ_i loss(y_i × f(x_i))`

- `½‖w‖²` margin'i genişletir: `‖w‖` küçüldükçe iki sınıf arasındaki boşluk büyür.
- `loss` margin ihlallerini cezalandırır. Denenen iki loss:
  - **hinge**: `max(0, 1 − y·f(x))`, klasik soft-margin SVM.
  - **squared_hinge**: `max(0, 1 − y·f(x))²`, `LinearSVC` varsayılanı. Büyük ihlalleri
    daha ağır, küçük ihlalleri daha hafif cezalandırır ve türevlenebilirdir.

**Logistic Regression'dan farkı:** LR log-loss kullanır; doğru sınıflandırılmış
örnekler bile kayba küçük de olsa katkı verir ve çıktı olasılık olarak yorumlanabilir.
Hinge kaybında margin'in doğru tarafında kalan örneklerin kaybı **sıfırdır**; karar
sınırını yalnızca margin'e yakın veya yanlış taraftaki örnekler (support vector'ler)
belirler. Naive Bayes ise ağırlıkları sınıf içi sıklıklardan sayarak çıkarır,
SVM ve LR ağırlıkları doğrudan ayırma hatasını azaltacak şekilde optimize eder.

**C parametresi** hata cezasının ağırlığıdır. Küçük C geniş margin ve güçlü
düzenlileştirme demektir (daha basit model, underfitting riski). Büyük C eğitim
hatalarına daha ağır ceza verir ve eğitim verisine daha sıkı uyar (overfitting riski).
Doğru değer validation ile seçilir.

## Deney tasarımı

| Ayar | Değerler |
| --- | --- |
| C | 0.01, 0.1, 1, 10, 100 |
| Loss | squared_hinge (varsayılan), hinge |
| Sabit | L2, OvR, `dual=True`, `max_iter=10000`, `random_state=42`, NB/LR ile aynı TF-IDF |
| Seçim ölçütü | En yüksek validation macro F1; eşitlikte çizelgedeki ilk ayar |

Ekip planındaki C değerleri 0.1, 1 ve 10'dur. 0.01 ve 100, en iyi değerin ızgaranın
kenarında kalıp kalmadığını kontrol etmek için eklendi. `dual=True` seçildi, çünkü
eğitim mesajı sayısı (8.499) özellik sayısından (21.595) azdır ve hinge loss
liblinear'da yalnızca dual çözücüyle çalışır. Toplam 10 çalıştırmanın tamamı yakınsadı
(ConvergenceWarning yok).

Tek komutla aynı 10 ayarı çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -m banking77.data --offline
.\.venv\Scripts\python.exe -m banking77.benchmark_linear_svm
```

Script yalnızca validation kullanır ve [C/loss tablosunu](../results/LINEAR_SVM_C.md)
ile [JSON kaydını](../results/linear_svm_validation.json) yeniden üretir.
Tam tahminler ve confusion matrix'ler `results/runs/`, modeller `artifacts/` altında yerelde kalır.
Bu bilgisayarda 10 çalıştırmanın eğitim + tahmin süresi toplam ~10 saniyedir.

Tek bir ayarı çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -m banking77.train_linear_svm --C 1
.\.venv\Scripts\python.exe -m banking77.train_linear_svm --loss hinge --C 1
```

## Sonuçlar

Mevcut veri sürümünde 8.499 train ve 1.500 validation kaydı vardır;
`dataset_summary_sha256` Naive Bayes ve Logistic Regression deneyleriyle aynıdır
(`468f5502…`). Validation ile train arasında normalize metin örtüşmesi yoktur.

| Loss | C | Accuracy | Macro F1 | Eğitim (s) | İterasyon |
| --- | ---: | ---: | ---: | ---: | ---: |
| squared_hinge | 0.01 | 80.47% | 0.7907 | 0.39 | 10 |
| squared_hinge | 0.1 | 86.00% | 0.8553 | 0.33 | 12 |
| **squared_hinge** | **1 (seçilen)** | **89.20%** | **0.8935** | **0.38** | **41** |
| squared_hinge | 10 | 87.93% | 0.8803 | 0.70 | 354 |
| squared_hinge | 100 | 87.80% | 0.8781 | 1.55 | 3593 |
| hinge | 0.01 | 83.80% | 0.8244 | 0.45 | 163 |
| hinge | 0.1 | 84.67% | 0.8335 | 0.44 | 229 |
| hinge | 1 | 88.00% | 0.8813 | 0.66 | 1421 |
| hinge | 10 | 87.67% | 0.8775 | 0.87 | 2562 |
| hinge | 100 | 87.67% | 0.8769 | 1.10 | 2922 |

Süreler tek bilgisayarda tek ölçümdür; tahmin süresi her ayarda 1.500 mesaj için
~0.013 saniyedir (mesaj başına 0.01 ms civarı, TF-IDF dönüşümü dahil).

### C'nin etkisi

C en belirleyici parametredir ve en iyi değer ızgaranın **ortasındadır**.
C=0.01 açıkça underfitting yapar (macro F1 0.79). C=0.1'den C=1'e geçiş macro F1'i
0.855'ten 0.894'e çıkarır; C=10 ve C=100'de skor tekrar düşer (0.880, 0.878), yani
büyük C eğitim verisine fazla uyar. Eşleştirilmiş karşılaştırmalar (aynı 1.500 mesaj):

- C=0.1 → C=1: yanlış sayısı 210 → 162; 59 hata düzeldi, 11 doğru tahmin bozuldu
  (exact McNemar p = 4.5×10⁻⁹).
- C=10 → C=1: 181 → 162; 29 düzeldi, 10 bozuldu (p = 0.0034).

C büyüdükçe çözücünün iterasyon sayısı da artar (41 → 3593): düzenlileştirme
zayıfladıkça optimizasyon problemi zorlaşır ve eğitim yavaşlar.

C değerleri modeller arasında doğrudan karşılaştırılamaz: LR'nin en iyi bölgesi
C=10–100, SVM'inki C=1'dir. İki model farklı kayıp fonksiyonlarını aynı C ile
ölçeklediği için aynı C sayısı aynı düzenlileştirme gücü anlamına gelmez.

### Loss'un etkisi

C=0.1 ve üzerindeki her değerde squared_hinge, hinge'den daha yüksek macro F1 verir;
yalnızca en güçlü düzenlileştirmede (C=0.01) hinge öndedir. En iyi ayarlarda (ikisi de C=1)
fark anlamlıdır: hinge → squared_hinge ile yanlış sayısı 180 → 162; 25 hata düzeldi,
7 doğru tahmin bozuldu (exact McNemar p = 0.0021). Squared hinge'in büyük margin
ihlallerini daha ağır cezalandırması olası bir açıklamadır; ayrıca ölçülmedi.
Hinge ayrıca daha çok iterasyon gerektirir (C=1'de 1421 ↔ 41).

## Seçilen ayar ve gerekçesi

**Seçilen: `squared_hinge`, `C=1`** — validation accuracy %89.20, macro F1 0.8935.

1. Önceden belirlenen kural (en yüksek validation macro F1) bu ayarı seçer.
   Macro F1 kullanıyoruz çünkü 77 kategorinin her birindeki başarıyı eşit
   önemsiyoruz; accuracy de destekleyici olarak raporlanır.
2. Bu ayar aynı zamanda `LinearSVC` varsayılanı ve ekip planındaki başlangıç ayarıdır.
   Ayar deneyi varsayılanı iyileştirmedi, ama varsayılanın bu veri için uygun olduğunu
   ve komşu C değerlerinden ve hinge loss'tan anlamlı biçimde iyi olduğunu gösterdi.
3. Izgara, kayıt dışı ara değerlerle inceltilmedi; tek bir 1.500 satırlık validation
   bölümünde ince ayar yapmak validation'a aşırı uyum riski taşır.

Ayar bu haliyle dondurulmuştur. Nihai test, README'deki kurala göre ortak veri/özellik
ayarı kesinleştikten sonra çalıştırılacaktır:

```powershell
.\.venv\Scripts\python.exe -m banking77.train_linear_svm --loss squared_hinge --C 1 --split test
```

Test script'i hazırlanmış train bölümüyle eğitir, validation eğitime eklenmez.
Resmî test ile train arasında 7 metin örtüşmesi vardır; script train ile
örtüşmeyen altkümenin skorlarını `nonoverlapping_subset` olarak ayrıca kaydeder.

## Hata analizi

Seçilen ayar 1.500 validation mesajının 162'sini yanlış sınıflandırdı.
Validation satır kimlikleri `train-…` ile başlar, çünkü validation resmî
eğitim kümesinden ayrılmıştır.

**Hatalar çoğunlukla aynı konu ailesindedir.** Gerçek ve tahmin edilen etiketin
adında ortak bir içerik kelimesi bulunan hatalar (`top_up`, `card`, `transfer`,
`cash_withdrawal`, `payment`, `identity` vb.; `my`, `not`, `by` gibi bağlaçlar hariç)
162 hatanın 109'udur (%67). Bu, etiket adlarına dayalı kaba bir sayımdır (tek seferlik
betik, repoya eklenmedi). Model genelde konuyu doğru bulup aynı konudaki alt kategoriyi
(yöntem, durum veya ücret türü) karıştırıyor.

**Per-class:** 77 kategoriden 5'inde F1 = 1.0 (`age_limit`, `atm_support`,
`card_about_to_expire`, `verify_top_up`, `virtual_card_not_working`); 6'sında F1 < 0.80.

| Kategori | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| pending_top_up | 0.762 | 0.727 | 0.744 | 22 |
| card_acceptance | 0.667 | 0.889 | 0.762 | 9 |
| balance_not_updated_after_bank_transfer | 0.826 | 0.731 | 0.776 | 26 |
| card_delivery_estimate | 0.737 | 0.824 | 0.778 | 17 |
| card_payment_not_recognised | 0.724 | 0.840 | 0.778 | 25 |

`card_payment_not_recognised`'ın düşük precision'ı (0.724) büyük ölçüde bir sonraki
tablodaki en sık hatadan gelir: 8 yanlış pozitifin 6'sı `direct_debit_payment_not_recognised`
mesajıdır. Model belirsiz "tanımadığım ödeme" mesajlarını bu sınıfa topluyor.

### En çok karışan kategori çiftleri

| Gerçek kategori | Tahmin | Sayı |
| --- | --- | ---: |
| direct_debit_payment_not_recognised | card_payment_not_recognised | 6 |
| top_up_by_bank_transfer_charge | top_up_by_card_charge | 4 |
| top_up_failed | top_up_reverted | 3 |
| pending_cash_withdrawal | declined_cash_withdrawal | 3 |
| get_disposable_virtual_card | disposable_card_limits | 3 |

### Örnek hatalar ve kelime katkıları

Linear modelde her özelliğin karara katkısı doğrudan ölçülebilir. "Katkı"
`(w_tahmin − w_gerçek) × tfidf` farkıdır: pozitif değer mesajı yanlış kategoriye,
negatif değer doğru kategoriye iter. Tüm katkıların toplamı ile sabitler farkı,
iki kategorinin skor farkına tam olarak eşittir (`tests/test_linear_svm.py` bunu doğrular).
Değerler kaydedilmiş modelin katsayılarından benchmark script'i tarafından hesaplanır.
SVM ağırlıkları LR ağırlıklarından farklı ölçektedir; katkı büyüklükleri modeller arasında karşılaştırılmamalıdır.

| Mesaj | Gerçek | Tahmin | Ölçülen açıklama |
| --- | --- | --- | --- |
| There is a payment in my app that I did not make.  I have not used that card all day  Please reimburse my money. (`train-04774`) | direct_debit_payment_not_recognised | card_payment_not_recognised | Mesajda "direct debit" geçmiyor. En büyük katkı `payment in` (+0.41) ve `payment` (+0.29) özelliklerinden geliyor. `not used` (−0.20) ve `all day` (−0.16) doğru yöne itiyor, çünkü kart kullanılmadıysa ödeme karttan olamaz; ama bu çıkarım model için birkaç zayıf bigram'a dağılmış durumda. Bu çiftin 6 hatası, etiketin çoğu zaman metinden açıkça çıkarılamadığını gösteriyor. |
| Are there topping up fees if I have to transfer? (`train-01901`) | top_up_by_bank_transfer_charge | top_up_by_card_charge | Model doğru ipucunu görüyor: `transfer` doğru kategoriye −0.40 iter. Ama `up fees` (+0.44), `fees` (+0.25) ve `topping` (+0.24) toplamda daha ağır basıyor: "top up + ücret" kalıbı eğitimde kart ücretiyle daha sık geçiyor. Aynı çiftteki `train-01856` mesajında kelime `trasfer` diye yanlış yazıldığı için doğru ipucu hiç görülmüyor. |
| My top-up was rejected (`train-08546`) | top_up_failed | top_up_reverted | En güçlü tek kelime `rejected` doğru kategoriye −0.56 iter. Ama `up was` (+0.45), `was rejected` (+0.24) ve `my top` (+0.15) bigram'ları toplamda reverted'a daha güçlü bağlanmış. "Reddedildi" ile "geri alındı" anlamca yakın; çok kısa mesajda birkaç bigram kararı belirliyor. |

Ek gözlemler:

- `train-00863` ("My atm withdraw is stillpending") `declined_cash_withdrawal` olarak
  tahmin edildi. Yazım hatası yüzünden "stillpending" tek token ve vocabulary'de yok;
  kararı belirleyecek `pending` kelimesi hiç görülmüyor. Kalan `my atm` (+0.46), `atm`,
  `withdraw` özellikleri iki kategoride de ortak. Bu, kelime tabanlı TF-IDF'in karakter
  düzeyindeki hatalara karşı kırılganlığını gösterir.
- `train-00867` ("…did not go through due to a declined card…") `pending_cash_withdrawal`
  olarak etiketlenmiş ama mesajda `declined` (+0.27) açıkça geçiyor. Bu örnek bir model
  hatasından çok etiket belirsizliğine/gürültüsüne benziyor.

### Sonuç ve sınırlamalar

- C'nin doğru seçilmesi ve squared hinge loss, istatistiksel olarak anlamlı farklar
  yarattı; seçilen ayar `LinearSVC` varsayılanıyla aynıdır.
- Kalan hataların çoğu, mesajın alt kategoriyi ayırt edecek bilgiyi içermediği
  veya bu bilginin tek bir kelimede olduğu durumlar. Kelime tabanlı TF-IDF, yazım hatalı
  biçimleri (`trasfer`, `stillpending`) doğru kelimeyle birleştiremez; karakter n-gram veya
  embedding tabanlı özellikler bu durumlar için denenebilir (özellik çalışmasının
  kapsamıdır, burada denenmedi).
- Sonuçlar tek bir 1.500 satırlık validation bölümüne dayanır. Macro F1'de
  ±0.005 içindeki farklar (yaklaşık 5–10 mesaj) bu boyutta güvenilir ayrım sayılmamalıdır.
- SVM skorları olasılık değildir. Olasılık gerekiyorsa kalibrasyon (ör. Platt scaling)
  ayrıca yapılmalıdır; bu çalışmada yapılmadı.
- Modeller arası karşılaştırma ve anlamlılık testleri 5. kişinin ortak değerlendirmesindedir.
