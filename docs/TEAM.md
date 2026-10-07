# Görev paylaşımı

| Sorumlu | Kapsam | Önerilen branch |
| --- | --- | --- |
| Repo sahibi (sen) | Naive Bayes, alpha deneyleri, hata analizi; GitHub ve son entegrasyon | `feature/naive-bayes` |
| 2. kişi | Veri analizi, kalite kontrolü, ortak bölümler; unigram/bigram deneyi | `feature/data-features` |
| 3. kişi | Logistic Regression, C deneyleri, hata analizi | `feature/logistic-regression` |
| 4. kişi | Linear SVM, C deneyleri, hata analizi | `feature/svm` |
| 5. kişi | Ortak değerlendirmeyi geliştirme, grafikler, karşılaştırma, slaytları birleştirme ve sunum | `feature/evaluation` |

Başlangıç modelleri ortak bir iskelet olarak hazırdır. Her sorumlu kendi yöntemini
anlamalı, deneylerini yapmalı ve bulgularını yazmalıdır. Herkes kendi bölümünün
slayt taslağını ve README açıklamasını hazırlar.

## Teslim sözleşmesi

- Aynı hazırlanmış veri ve `seed=42` kullanılır. Veri değişirse herkes yeniden çalıştırır.
- TF-IDF yalnızca eğitim verisinde öğrenilir; validation/test üzerinde `fit` yapılmaz.
- Model/özellik/parametre seçimi validation verisinde yapılır.
- Her model sahibi komutunu, metriklerini ve en az üç hata örneğini teslim eder.
- Tahmin dosyası sütunları: `id,text,true_label,predicted_label,correct,overlaps_training`.
- Karşılaştırılan deneylerin `dataset_summary_sha256` değerleri aynı olmalıdır.
- İlk sürümde BERT ve arayüz kapsam dışıdır. Sunum tarihi: 12 Ekim 2026.

## Git akışı

```powershell
git switch main
git pull --ff-only
git switch -c feature/naive-bayes
# Kendi dosyalarını düzenle ve kontrol et.
git add src/banking77/models.py docs
git commit -m "Add Naive Bayes validation experiments"
git push -u origin feature/naive-bayes
```

GitHub'da pull request açın. Repo sahibi inceleyip birleştirir. Veri, `.venv`,
kimlik bilgileri ve büyük model dosyalarını commit etmeyin. Diğer kişiler kendi
branch adlarını ve değişen dosyalarını kullanmalıdır.
