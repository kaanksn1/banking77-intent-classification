# İlk validation sonuçları — 7 Ekim 2026

Bunlar başlangıç ayarlarıyla elde edilen geliştirme sonuçlarıdır; nihai test sonucu değildir.
Test bölümünde model değerlendirmesi henüz yapılmadı.

- Eğitim: 8.499; validation: 1.500; resmî test: 3.080 kayıt.
- TF-IDF: unigram + bigram, sublinear TF; seed=42.
- Naive Bayes alpha=1.0; Logistic Regression ve Linear SVM C=1.0.
- Modeller aynı Windows bilgisayarda sırayla çalıştırıldı. Süreler tek ölçümdür, tekrarlı benchmark değildir.

| Model | Validation accuracy | Validation macro F1 | Eğitim (sn) |
| --- | ---: | ---: | ---: |
| Naive Bayes | 81.60% | 0.7872 | 0.157 |
| Logistic Regression | 85.80% | 0.8555 | 5.776 |
| Linear SVM | 89.20% | 0.8935 | 0.519 |

## Veri kontrolü

Resmî eğitim verisindeki 4 normalize metin tekrarı kaldırıldı. Train/validation örtüşmesi 0;
train/resmî test örtüşmesi 7 metin. Resmî test değiştirilmedi. Son test raporunda bu sınırlama
ve train ile örtüşmeyen altkümenin sonuçları birlikte gösterilmelidir.

Tüm ayarlar, ortam sürümleri ve veri kimliği: [initial_validation.json](initial_validation.json).
Veri ayrımının ayrıntıları: [data_summary.json](data_summary.json).

Komutlar: `python -m banking77.train --model naive_bayes`,
`python -m banking77.train --model logistic_regression`, `python -m banking77.train --model svm`.
