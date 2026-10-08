# Deney sonuçları

Henüz nihai test sonucu yok. `runs/` klasöründeki yerel çıktılar Git'e eklenmez.
Naive Bayes çalıştırıcısı `metrics.json`, `classification_report.json`, `predictions.csv` ve
`confusion_matrix.csv` üretir. Diğer model sorumluları da aynı çıktı sözleşmesini kullanır.
Seçilen deneylerin küçük özetlerini, ayarları ve
veri kimliğiyle birlikte bu klasöre ekleyin; büyük model dosyalarını eklemeyin.

Validation sonuçları model seçimi içindir. Test sonuçlarını ayrı etiketleyin.
Eğitim/tahmin sürelerini aynı bilgisayarda, modelleri sırayla çalıştırarak ölçün;
tahmin süresi TF-IDF dönüşümünü de içerir.

Mevcut alpha benchmark raporu yalnızca repo sahibinin Naive Bayes sonuçlarını içerir.
Modellerin ortak karşılaştırmasını 5. kişi hazırlayacaktır.

`python -m banking77.benchmark_naive_bayes` yalnızca repo sahibinin alpha
deneyini tekrar çalıştırıp `NAIVE_BAYES_ALPHA.md` ve
`naive_bayes_alpha_validation.json` dosyalarını üretir. JSON'daki seçilen run
kimliği tam tahmin ve confusion matrix dosyalarını yerel `runs/` altında gösterir.
Resmî test bu scriptte kullanılmaz. Hata yorumları `docs/NAIVE_BAYES.md` içindedir.
