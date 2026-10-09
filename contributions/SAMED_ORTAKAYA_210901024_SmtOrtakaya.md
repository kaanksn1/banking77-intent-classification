# Samed Ortakaya — 210901024

GitHub kullanıcı adı: SmtOrtakaya
Sorumluluk: 3. kişi — Logistic Regression modeli, C/solver deneyleri ve hata analizi

## Tamamladığım işler

- `src/banking77/logistic_regression.py`: TF-IDF + Logistic Regression pipeline'ı.
  TF-IDF ayarı Naive Bayes ile aynı tutuldu (unigram + bigram, sublinear TF), böylece
  modeller arasındaki fark yalnızca sınıflandırıcıdan gelir. Üç solver desteklenir:
  `lbfgs` ve `saga` (multinomial), `liblinear-ovr` (one-vs-rest).
- `src/banking77/train_logistic_regression.py`: tek bir ayarla eğitim, metrik ve tahmin
  kaydı. Naive Bayes ile aynı çıktı sözleşmesini (`metrics.json`, `predictions.csv`,
  `confusion_matrix.csv`, `classification_report.json`) kullanır; ek olarak iterasyon
  sayısını ve yakınsama durumunu kaydeder.
- `src/banking77/benchmark_logistic_regression.py`: C (0.1, 1, 10, 100, 1000) × solver
  ızgarasını yalnızca validation üzerinde çalıştırır, seçimi önceden belirlenen kuralla
  (en yüksek macro F1) yapar ve McNemar testiyle eşleştirilmiş karşılaştırma üretir.
- `tests/test_logistic_regression.py`: geçersiz ayarların reddi, tüm solver'ların
  çalışması, TF-IDF'in yalnızca train'de öğrenilmesi ve test setinin seçimde
  kullanılamaması için birim testleri.
- `docs/LOGISTIC_REGRESSION.md`: yöntem açıklaması, seçim gerekçesi ve hata analizi.
- `results/LOGISTIC_REGRESSION_C.md` ve `results/logistic_regression_validation.json`:
  script tarafından üretilen deney raporu.

Ortak veri bölümleri ve diğer üyelerin dosyaları değiştirilmedi.

## Doğrulama ve deneyler

- Çalıştırma komutu: `python -m banking77.benchmark_logistic_regression`
- Kullanılan veri bölümü: validation (8.499 train, 1.500 validation);
  `dataset_summary_sha256 = 468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`
  (Naive Bayes deneyiyle aynı).
- Sonuç ve üretilen çıktı dosyası: seçilen ayar `liblinear-ovr`, `C=100`;
  validation accuracy %89.07, macro F1 0.8920. Rapor: `results/LOGISTIC_REGRESSION_C.md`.
- Bulgular:
  - C en belirleyici parametre. Başlangıç ayarına (lbfgs, C=1) göre 66 hata düzeldi,
    17 doğru tahmin bozuldu (exact McNemar p = 5.6×10⁻⁸).
  - Solver'lar arasındaki fark anlamlı değil (en iyi lbfgs'e göre p = 0.30);
    liblinear-ovr ~6 kat daha kısa sürede eğitiliyor.
  - 164 hatanın %65'i aynı konu ailesindeki alt kategoriler arasında
    (ör. `direct_debit_payment_not_recognised` → `card_payment_not_recognised`).
  - Kelime katkısı analizi: yazım hataları (`stillpending`, `trasfer`) belirleyici
    kelimenin modelde hiç görülmemesine yol açıyor.
- Sınırlamalar: tek bir 1.500 satırlık validation bölümü; süreler tek bilgisayarda
  tek ölçüm. Resmî test, ekip kuralına göre veri/özellik çalışması kesinleştikten
  sonra dondurulan ayarla çalıştırılacak.

## GitHub kanıtı

- Kendi commit bağlantılarım (`feature/logistic-regression`):
  - [Model, eğitim ve benchmark scriptleri](https://github.com/kaanksn1/banking77-intent-classification/commit/8c62d27a16866b95f5ce49e6e52c54f578da074d)
  - [Birim testleri](https://github.com/kaanksn1/banking77-intent-classification/commit/527bdbc40fd6c30025650ce87b64c4aab617152d)
  - [Teknik not ve hata analizi](https://github.com/kaanksn1/banking77-intent-classification/commit/fcafff333ed68e76361c26ffef66076ed71ec995)
  - [Validation sonuçları](https://github.com/kaanksn1/banking77-intent-classification/commit/b5b2dc0d92007c20e1cfddb90efa89e2ff0666f5)
- Kendi PR bağlantılarım: [#3 Logistic Regression](https://github.com/kaanksn1/banking77-intent-classification/pull/3)

## Sunuma katkım

- Logistic Regression bölümü için sonuç tablosu, C'nin etkisi, seçim gerekçesi ve
  üç örnek hata `docs/LOGISTIC_REGRESSION.md` içinde hazır; slayt metni bu nottan hazırlanacak.
