# Naive Bayes: yöntem, deney ve hata analizi

Bu not repo sahibinin model bölümüdür. Ortak veri/özellik deneylerinin ve
modeller arası karşılaştırmanın yerine geçmez.

## Nasıl çalışıyor?

`Mesaj → TF-IDF özellikleri → Multinomial Naive Bayes → 77 kategoriden biri`

TF-IDF, kelime ve kelime çiftlerini sayısal ağırlıklara dönüştürür. Mevcut ayar
`ngram_range=(1, 2)`, `sublinear_tf=True` şeklindedir. TF için pozitif sayımlarda
`1 + log(tf)` kullanılır. Scikit-learn'ün varsayılan smooth IDF, L2 normalizasyon,
küçük harfe çevirme ve en az iki karakterli kelime tokenizasyonu korunur.
Vocabulary ve IDF yalnızca train üzerinde öğrenilir; validation/test bu
öğrenilmiş dönüşümle işlenir. TF-IDF ayarı bu alpha deneyinde sabittir.

MultinomialNB her kategori için özellik ağırlıklarını ve sınıf öncüllerini
train verisinden öğrenir. Sınıf verildiğinde özelliklerin koşullu bağımsız
olduğunu varsayar; gerçek dilde bu varsayım tam sağlanmaz.

Bir mesaj için kategori puanı kabaca şöyle hesaplanır:

`score(c) = log P(c) + Σ_j TFIDF_j(message) × log θ(c,j)`

En yüksek puanlı kategori seçilir. `P(c)` eğitimdeki sınıf sıklığından öğrenilir.
`θ(c,j)`, kategori içindeki j özelliğinin yumuşatılmış ağırlığıdır:

`θ(c,j) = (N(c,j) + alpha) / (Σ_j N(c,j) + alpha × özellik_sayısı)`

Buradaki `N` değerleri bu uygulamada TF-IDF toplamlarıdır; tam sayı kelime
sayımları değildir. MultinomialNB'nin TF-IDF ile kullanımı pratik bir tercihtir.
Model çıktıları ayrıca kalibrasyon yapılmadan gerçek doğruluk olasılığı gibi
yorumlanmamalıdır.

## Alpha neden denendi?

Alpha sıfır özellik ağırlıklarını önleyen additive smoothing miktarıdır.
Alpha=1.0 başlangıç ayarıdır. Daha küçük değerler daha az yumuşatma uygular;
hangi değerin iyi çalışacağı validation ile ölçülür. Daha küçük alpha'nın
her zaman daha iyi olduğu söylenemez: burada 0.01, 0.05'ten daha düşük macro F1 verdi.

Tek komutla aynı beş değeri sırayla çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -m banking77.data --offline
.\.venv\Scripts\python.exe -m banking77.benchmark_naive_bayes
```

Script yalnızca `1.0, 0.5, 0.1, 0.05, 0.01` alpha değerlerini dener; ortak
özellik ayarını değiştirmez ve resmî testi değerlendirmez. Seçim en yüksek
validation macro F1 ile yapılır; eşitlikte listedeki ilk değer korunur.
Tam çıktı dosyaları `results/runs/`, modeller `artifacts/` altında yerelde kalır.
Paylaşılan raporlar [alpha tablosu](../results/NAIVE_BAYES_ALPHA.md) ve
[JSON kaydıdır](../results/naive_bayes_alpha_validation.json).

Mevcut veri sürümünde 8.499 train ve 1.500 validation kaydı vardır. Alpha=1.0
ile accuracy %81.60, macro F1 0.7872; alpha=0.05 ile accuracy %86.20, macro F1
0.8529 ölçüldü. 276 hatanın 92'si düzeldi, 23 yeni hata oluştu: net 69 daha az hata.
Bu sonuçlar nihai test başarısı değildir. Süreler tek bilgisayarda tek ölçümdür.

## NB için ölçüt seçimi

Accuracy, doğru tahminlerin tüm örneklere oranıdır. Macro F1, her kategori için
hesaplanan F1'in eşit ağırlıklı ortalamasıdır. Biz 77 kategorinin her birindeki
başarıya önem verdiğimiz için seçim metriği olarak macro F1 kullanıyoruz;
accuracy de destekleyici olarak kaydedilir. BANKING77'nin ağır dengesiz olduğu
iddia edilmez. Per-class support/F1 değerleri her run'ın classification report'undadır.
Ekip çapındaki metrik gerekçesi ve validation karşılaştırması
[ortak raporda](../results/MODEL_COMPARISON.md) 5. kişi tarafından hazırlanmıştır.

## Üç gerçek validation hatasının yorumu

Bu örnekler mevcut veri kimliğiyle alpha=0.05 çıktısından alınmıştır.
Yorumlar olası açıklamalardır; özellik katkısı/ablation ile nedensellik ispatlanmadı.

| Mesaj | Gerçek etiket | Tahmin | Olası açıklama |
| --- | --- | --- | --- |
| The balance has not been updated. | balance_not_updated_after_bank_transfer | balance_not_updated_after_cheque_or_cash_deposit | Mesaj hangi para yatırma/aktarma yöntemini kastettiğini belirtmiyor; iki yakın kategori için bağlam eksik. |
| What does the €1 fee mean? | extra_charge_on_statement | transfer_fee_charged | `fee` birden fazla ücret kategorisinde kullanılabilir; kısa mesaj işlemin türünü söylemiyor. |
| What is the procedure for an expired card? | card_about_to_expire | request_refund | TF-IDF kelime/kelime çifti örüntülerini kullanır; bu örnekte kartın sona erme anlamı doğru ayrıştırılamamış. Hangi özelliğin hataya yol açtığı ayrıca ölçülmedi. |

Bu hatalar, yakın niyetleri kısa ve belirsiz metinlerden ayırmanın güçlüğünü
gösterir. Sadece toplam accuracy ile bu kategori farkları görülemez.

## Devir ve nihai test

Mevcut veri ve unigram + bigram temsili korunmuştur. NB'nin `alpha=0.05`
ayarı, [protokol kaydı](../results/naive_bayes_final_protocol.json) ile
resmî test çalıştırılmadan önce ayrı commit'e alınmıştır. Nihai model yalnızca
8.499 train mesajından öğrenir; validation eğitime eklenmez.

3.080 resmî test mesajında accuracy **%84.74**, macro F1 **0.8458** ölçülmüştür.
Normalize anahtara göre train ile örtüşen 7 mesaj çıkarılınca 3.073 mesajda
accuracy **%84.71**, macro F1 **0.8454** olur. Teste bakarak ayar değiştirilmemiştir.
Tam kayıt, confusion matrix ve gerçek test hataları
[nihai test raporunda](../results/NAIVE_BAYES_TEST.md) bulunur.

## Teknik kaynaklar

- [Scikit-learn Naive Bayes açıklaması](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [MultinomialNB parametreleri](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
- [TfidfVectorizer parametreleri](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
