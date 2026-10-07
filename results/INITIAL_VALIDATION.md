# Naive Bayes ilk validation sonucu — 7 Ekim 2026

Bu rapor repo sahibinin Naive Bayes başlangıcına aittir. Nihai test sonucu değildir.
Logistic Regression ve Linear SVM ilgili ekip üyelerinin kendi geliştireceği bölümlerdir.
Ortak model karşılaştırması, grafikler ve sunum 5. kişiye aittir.

- Eğitim: 8,499; validation: 1,500; resmî test: 3.080 kayıt.
- TF-IDF: unigram + bigram, sublinear TF; veri ayrımı seed=42.
- Multinomial Naive Bayes: alpha=1.0.
- Validation accuracy: **81.60%**.
- Validation macro F1: **0.7872**.

## Veri protokolü

Ortak başlangıçta resmî eğitim verisindeki 4 normalize metin tekrarı kaldırıldı.
Train/validation örtüşmesi 0; train/resmî test örtüşmesi 7 metin.
Resmî test değiştirilmedi. Bu sınırlama final raporunda belirtilmelidir.
Veri analizi ve özellik deneylerini 2. kişi geliştirecektir.

Ayarlar ve ortam: [initial_validation.json](initial_validation.json).
Ortak veri özeti: [data_summary.json](data_summary.json).

Komut: `python -m banking77.train_naive_bayes`.
