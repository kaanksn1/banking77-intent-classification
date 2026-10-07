# Deney planı

Problem: BANKING77'de kısa İngilizce müşteri mesajını 77 talepten birine sınıflandırma.

## Ortak protokol

1. Resmî eğitim kümesindeki boş kayıtları ve normalize edilmiş metin tekrarlarını kontrol et.
2. Aynı normalize metinde farklı etiket varsa işlemi durdur ve incele.
3. Eğitim kümesinden stratified %15 validation ayır (`seed=42`).
4. Resmî 3.080 satırlık test kümesini koru; ilk geliştirme aşamasında test çalıştırma.
5. Hazırlama raporundaki train/test ve validation/test metin örtüşmelerini raporla.
   Bunlar sıfır değilse resmî testin tamamen temiz bir holdout olduğu iddia edilmemeli.
   Naive Bayes çalıştırıcısı train ile örtüşmeyen altküme sonuçlarını ayrıca verir.
   Diğer model sahipleri de aynı raporlama kuralını kendi kodlarına uygulamalıdır.

Bu yerel protokol eğitim tekrarlarını kaldırıp validation ayırdığı için,
resmî 10.003 eğitim örneğinin tamamıyla eğitilen yayınlarla koşullar birebir aynı değildir.

## Küçük deney bütçesi

Aşağıdaki tablo ekip planıdır. Şu anda yalnızca Naive Bayes kodu repo sahibinin
bölümünde uygulanmıştır; diğer yöntem ve deneyleri ilgili sorumlular geliştirecektir.

| Yöntem | Başlangıç | Validation üzerinde denenecek |
| --- | --- | --- |
| TF-IDF | unigram + bigram, sublinear TF | `(1,1)` ile `(1,2)` karşılaştırması |
| Multinomial Naive Bayes | alpha=1.0 | alpha: 0.1, 0.5, 1.0 |
| Logistic Regression | C=1.0 | C: 0.1, 1.0, 10.0 |
| Linear SVM | C=1.0 | C: 0.1, 1.0, 10.0 |

Önce ortak özellik ayarıyla modelleri karşılaştırın. Özellik deneylerini ayrı
tabloda gösterin. Ana model seçme metriği macro F1; accuracy de raporlanır.
Sonuç uydurmayın: başlangıç değerleri veya literature skorları bizim sonuçlarımız değildir.

## Son değerlendirme

Her model için validation ile ayarları dondurun. Naive Bayes test çalıştırıcısı
hazırlanmış train bölümüyle eğitir; validation eğitim verisine eklenmez.
Testi ayarları dondurduktan sonra çalıştırın ve final raporunda protokolü yazın.
Her model için accuracy, macro F1, eğitim süresi ve tahmin süresi; en çok karışan
üç kategori çifti ve örnek yanlış tahminler raporlanmalıdır.
