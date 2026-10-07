# Deney sonuçları

Henüz nihai test sonucu yok. `runs/` klasöründeki yerel çıktılar Git'e eklenmez.
Naive Bayes çalıştırıcısı `metrics.json`, `classification_report.json`, `predictions.csv` ve
`confusion_matrix.csv` üretir. Diğer model sorumluları da aynı çıktı sözleşmesini kullanır.
Seçilen deneylerin küçük özetlerini, ayarları ve
veri kimliğiyle birlikte bu klasöre ekleyin; büyük model dosyalarını eklemeyin.

Validation sonuçları model seçimi içindir. Test sonuçlarını ayrı etiketleyin.
Eğitim/tahmin sürelerini aynı bilgisayarda, modelleri sırayla çalıştırarak ölçün;
tahmin süresi TF-IDF dönüşümünü de içerir.

Mevcut başlangıç raporu yalnızca repo sahibinin Naive Bayes sonucunu içerir.
Modellerin ortak karşılaştırmasını 5. kişi hazırlayacaktır.
