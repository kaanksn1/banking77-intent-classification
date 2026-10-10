# Naive Bayes: nihai test sonucu

Bu rapor repo sahibinin NB bölümüdür. LR/SVM'nin nihai testlerini veya
5. kişinin modeller arası test karşılaştırmasını içermez.

## Testten önce sabitlenen ayar

`TF-IDF (unigram + bigram, sublinear TF) + MultinomialNB(alpha=0.05)`

Alpha, beş aday arasındaki en yüksek **validation macro F1** ile seçildi.
Mevcut ortak TF-IDF temsili korundu. [Protokol kaydı](naive_bayes_final_protocol.json)
ilk test çalıştırmasından önce
[29312bb commit'i](https://github.com/kaanksn1/banking77-intent-classification/commit/29312bbcf591aca0073a88cc4121dae1d172ac0d)
ile sabitlendi. Teste bakılarak özellik veya parametre değiştirilmedi.

- Eğitim: hazırlanmış **8.499 train** mesajı; validation eğitime eklenmedi.
- Seçim: **1.500 validation** mesajı, stratified ayırma ve `seed=42`.
- Nihai değerlendirme: resmî **3.080 test** mesajı; 77 sınıfın her birinde 40 mesaj.
- Veri özeti SHA-256: `468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`.
- Çalıştırma kimliği: `naive_bayes_test_20261010T134914397317Z`.

## Sonuçlar

| Bölüm | Mesaj sayısı | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Validation — ayar seçimi | 1.500 | %86.20 | 0.8529 |
| Resmî test — sabit ayar | 3.080 | **%84.74** | **0.8458** |
| Test — train ile birebir örtüşmeyen altküme | 3.073 | %84.71 | 0.8454 |

Tam testte **2.610 doğru, 470 yanlış** tahmin vardır. Validation ve test farklı
mesajlardan oluştuğu için skorlarının aynı olması beklenmez. Test skoru yeni
ayar seçmek için kullanılmaz. Macro F1 her niyeti eşit ağırlıkla değerlendirir;
accuracy destekleyici metriktir.

7 test mesajı, NFKC/casefold/boşluk normalizasyonuna göre train ile birebir
örtüşür. Bu nedenle tam testin tamamen bağımsız ve tekrarsız olduğu iddia edilmez.
Alt küme yalnızca bu birebir örtüşmeleri dışlar; yakın tekrarları tamamen
gidermiş bir test değildir. Veri dosyaları değiştirilmedi.

Bu çalıştırmada TF-IDF dahil eğitim **0.2152 s**, TF-IDF dönüşümü dahil tahmin
**0.0575 s** (mesaj başına yaklaşık **0.0187 ms**) sürdü. Bunlar tek bilgisayarda
tek ölçümdür; farklı bilgisayarlardaki LR/SVM süreleriyle doğrudan sıralama yapılmaz.
Ortam: Python 3.12.0, scikit-learn 1.9.1, NumPy 2.5.3.

## En sık karışan üç yönlü kategori çifti

| Gerçek etiket | Tahmin | Hata sayısı |
| --- | --- | ---: |
| contactless_not_working | card_not_working | 10 |
| card_payment_not_recognised | direct_debit_payment_not_recognised | 7 |
| card_swallowed | declined_cash_withdrawal | 6 |

## Üç gerçek test hatası

Aşağıdaki yorumlar mesaj içeriğine dayalı olası açıklamalardır; özellik
katkısı veya ablation ile neden-sonuç kanıtı üretilmedi.

| Kimlik ve mesaj | Gerçek → tahmin | Yorum |
| --- | --- | --- |
| `test-00561`: My contanctless has stopped working | contactless_not_working → card_not_working | Temassız ödeme ile genel kart arızası yakın niyetlerdir; mesajda ayrıca yazım hatası vardır. Hatanın tek nedeninin yazım olduğu iddia edilmez. |
| `test-01084`: An unauthorized payment is in my app | card_payment_not_recognised → direct_debit_payment_not_recognised | Mesaj tanınmayan bir ödemeyi söylüyor; kart ödemesi ile direct debit ayrımını açıkça belirtmiyor. |
| `test-01926`: Is there a way I can get my ATM card back from the machine? | card_swallowed → declined_cash_withdrawal | Mesaj ATM'nin kartı geri vermemesiyle ilgili olduğu hâlde model para çekmenin reddedilmesini seçmiş. Bu örnekte ATM bağlamı içindeki alt niyet ayrımı başarısız. |

## Çıktılar ve yeniden çalıştırma

- [metrics.json](naive_bayes_test/metrics.json): ayar, veri hash'leri, ortam, skorlar ve süreler.
- [classification_report.json](naive_bayes_test/classification_report.json): sınıf başına precision, recall, F1 ve support.
- [predictions.csv](naive_bayes_test/predictions.csv): 3.080 mesajın gerçek/tahmin etiketi, doğruluğu ve train örtüşmesi.
- [confusion_matrix.csv](naive_bayes_test/confusion_matrix.csv): satırlar gerçek, sütunlar tahmin edilen sınıflar.

```powershell
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes --split test --ngram-max 2 --alpha 0.05
```

Yeni çalıştırma `results/runs/<run_id>/` altında aynı dört dosyayı ve `artifacts/`
altında model dosyasını üretir. Yayımlanan klasör seçilen tek nihai çalıştırmanın
kopyasıdır; tekrarların ve model dosyalarının tamamı Git'e eklenmez. Çalıştırma
kimliği/süre değişebilir; veri, kod ve ortam aynı olduğunda sınıflandırma skorları
tekrar üretilebilir. İlk NB alpha raporunun CRLF hash'leri ile güncel LF hash'lerinin
farkı [sonuçlar dizininde](README.md) açıklanmıştır.

5. kişi karşılaştırma için bu klasördeki standart çıktıları kullanabilir.
LR/SVM testleri aynı hazırlanmış train/test, kategori listesi ve ortak TF-IDF
temsiliyle kendi model sahiplerince üretilmelidir. Mesaj metinleri BANKING77'nin
CC BY 4.0 lisansına tabidir; kaynak ve atıf [ana README'dedir](../README.md#veri-kaynağı-ve-atıf).
