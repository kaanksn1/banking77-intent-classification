# Deney sonuçları

NB'nin nihai test sonucu [NAIVE_BAYES_TEST.md](NAIVE_BAYES_TEST.md) ve
`naive_bayes_test/` içinde yayımlanmıştır. LR ve SVM dahil klasik nihai testler
[MODEL_COMPARISON_TEST.md](MODEL_COMPARISON_TEST.md) içinde teslim edilmiştir.
Yeni neural/embedding baseline'ları ve repo sahibinin NB + CNN katkısı için
[NEURAL_VALIDATION.md](NEURAL_VALIDATION.md) ve `neural_validation.json` bulunur.
Sabitlenen sekiz baseline ve NB + CNN'nin nihai testleri [NEURAL_TEST.md](NEURAL_TEST.md),
`neural_test.json` ve `neural_test/<yöntem>/` altında dört standart çıktı ile teslim edilir.
Dört Transformer'ın tam eğitimi beklemektedir. `runs/` klasöründeki tekrarlı
yerel çıktılar Git'e eklenmez.
Naive Bayes, Logistic Regression ve Linear SVM çalıştırıcıları `metrics.json`,
`classification_report.json`, `predictions.csv` ve `confusion_matrix.csv` üretir.
Diğer model sorumluları da aynı çıktı sözleşmesini kullanır.
Seçilen deneylerin küçük özetlerini, ayarları ve
veri kimliğiyle birlikte bu klasöre ekleyin; büyük model dosyalarını eklemeyin.

Validation sonuçları model seçimi içindir. Test sonuçlarını ayrı etiketleyin.
Eğitim/tahmin sürelerini aynı bilgisayarda, modelleri sırayla çalıştırarak ölçün;
tahmin süresi TF-IDF dönüşümünü de içerir.

Mevcut raporlar:

- [Naive Bayes alpha deneyi](NAIVE_BAYES_ALPHA.md): repo sahibinin çalışması.
- [Naive Bayes nihai test raporu](NAIVE_BAYES_TEST.md): repo sahibinin çalışması.
  `naive_bayes_final_protocol.json` testten önceki ayar/veri kaydıdır;
  `naive_bayes_test/` yalnızca seçilen nihai çalıştırmanın `metrics.json`,
  `classification_report.json`, `predictions.csv` ve `confusion_matrix.csv` dosyalarını içerir.
- [Yeni baseline ve NB + CNN validation deneyleri](NEURAL_VALIDATION.md): repo sahibine sonradan atanan kapsam.
- [Sabitlenen yeni modellerin nihai testi](NEURAL_TEST.md): `2425ccc` ile testten
  önce dondurulmuş checkpoint'ler. Dört Transformer'ın tam deneyleri henüz dahil değildir.
- [Unigram/bigram özellik deneyleri](FEATURE_EXPERIMENTS.md): 2. kişinin çalışması.
- [Logistic Regression C/solver deneyi](LOGISTIC_REGRESSION_C.md): 3. kişinin çalışması.
- [Linear SVM C/loss deneyi](LINEAR_SVM_C.md): 4. kişinin çalışması.

- [Üç modelde özellik karşılaştırması](FEATURE_COMPARISON.md): 5. kişinin çalışması;
  `python -m banking77.benchmark_features`. Unigram, unigram + bigram ve yalnızca bigram,
  her model için kendi ızgarasıyla yeniden ayarlanarak karşılaştırılır.
- [Nihai test karşılaştırması](MODEL_COMPARISON_TEST.md): 5. kişinin çalışması;
  `python -m banking77.benchmark_models --split test`. Ayarlar validation'da seçilip
  dondurulduktan sonra resmî testte bir kez değerlendirilir; hiçbir seçim test sonucuna bağlı değildir.
- [Ortak model karşılaştırması](MODEL_COMPARISON.md): 5. kişinin çalışması;
  `python -m banking77.benchmark_models` ile üretilir, grafikler `figures/` altındadır.

`python -m banking77.benchmark_naive_bayes` yalnızca repo sahibinin alpha
deneyini tekrar çalıştırıp `NAIVE_BAYES_ALPHA.md` ve
`naive_bayes_alpha_validation.json` dosyalarını üretir. JSON'daki seçilen run
kimliği tam tahmin ve confusion matrix dosyalarını yerel `runs/` altında gösterir.
Resmî test bu scriptte kullanılmaz. Hata yorumları `docs/NAIVE_BAYES.md` içindedir.

`python -m banking77.benchmark_logistic_regression` 15 validation ayarını tekrar
çalıştırıp `LOGISTIC_REGRESSION_C.md` ve `logistic_regression_validation.json`
dosyalarını üretir. Hata yorumları `docs/LOGISTIC_REGRESSION.md` içindedir.
Özellik deneyinin komutları `FEATURE_EXPERIMENTS.md` içinde yer alır.

`python -m banking77.benchmark_linear_svm` 10 validation ayarını tekrar çalıştırıp
`LINEAR_SVM_C.md` ve `linear_svm_validation.json` dosyalarını üretir.
Örnek hataların kelime katkıları aynı scriptte hesaplanır; yorumları
`docs/LINEAR_SVM.md` içindedir. Nihai test bu benchmark'ta kullanılmaz.

PR #2 hazırlanmış CSV yazımını LF satır sonuna sabitledi. İlk NB ve LR raporlarındaki
train/validation dosya hash'leri aynı kayıtların CRLF ile yazılmış sürümüne aittir.
Entegrasyonda farkın yalnızca satır sonlarından geldiği, LR'nin 15 deneyinin ve
NB'nin seçilen ayarının accuracy/macro F1 skorlarının tekrar üretildiği doğrulandı;
veri özeti kimliği değişmedi.
Benchmark yeniden çalıştırıldığında güncel LF dosyalarının hash'leri kaydedilir.
