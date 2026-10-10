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

Aşağıdaki tablo ekip deney kapsamını gösterir. Veri/özellik çalışması, Naive Bayes,
Logistic Regression ve Linear SVM kendi sorumlularının PR'larıyla `main` içine alınmıştır.
Modellerin ortak validation karşılaştırması [MODEL_COMPARISON.md](../results/MODEL_COMPARISON.md)
içindedir (5. kişi).

| Yöntem | Başlangıç | Validation deney kapsamı |
| --- | --- | --- |
| TF-IDF | unigram + bigram, sublinear TF | `(1,1)` ile `(1,2)` karşılaştırması |
| Multinomial Naive Bayes | alpha=1.0 | alpha: 0.01, 0.05, 0.1, 0.5, 1.0 |
| Logistic Regression | lbfgs, C=1.0 | C: 0.1, 1.0, 10.0, 100.0, 1000.0; lbfgs, saga, liblinear-ovr |
| Linear SVM | squared_hinge, C=1.0 | C: 0.01, 0.1, 1.0, 10.0, 100.0; squared_hinge, hinge |

Tamamlanan çalışmalar: [özellik deneyi](../results/FEATURE_EXPERIMENTS.md),
[NB alpha deneyi](../results/NAIVE_BAYES_ALPHA.md),
[LR C/solver deneyi](../results/LOGISTIC_REGRESSION_C.md),
[SVM C/loss deneyi](../results/LINEAR_SVM_C.md).
Üç modelde yeniden ayarlanarak yapılan unigram / bigram karşılaştırması
[FEATURE_COMPARISON.md](../results/FEATURE_COMPARISON.md) içindedir.
Ortak özellik ayarı henüz ekipçe kesinleştirilmedi; mevcut NB/LR/SVM deneyleri
unigram + bigram kullanır. Özellik raporundaki unigram önerisi alpha=1.0'lı NB
deneyine aittir ve diğer modellerin yapılandırmasını kendiliğinden değiştirmez.

Önce ortak özellik ayarıyla modelleri karşılaştırın. Özellik deneylerini ayrı
tabloda gösterin. Ana model seçme metriği macro F1; accuracy de raporlanır.
Sonuç uydurmayın: başlangıç değerleri veya literature skorları bizim sonuçlarımız değildir.

## Son değerlendirme

Her model için validation ile ayarları dondurun. Naive Bayes test çalıştırıcısı
hazırlanmış train bölümüyle eğitir; validation eğitim verisine eklenmez.
Testi ayarları dondurduktan sonra çalıştırın ve final raporunda protokolü yazın.
Her model için accuracy, macro F1, eğitim süresi ve tahmin süresi; en çok karışan
üç kategori çifti ve örnek yanlış tahminler raporlanmalıdır.
