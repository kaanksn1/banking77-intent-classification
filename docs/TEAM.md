# Görev paylaşımı

| Sorumlu | Kapsam | Önerilen branch |
| --- | --- | --- |
| Repo sahibi (sen) | Naive Bayes, alpha deneyleri, hata analizi; GitHub ve son entegrasyon | `feature/naive-bayes` |
| 2. kişi | Veri analizi, kalite kontrolü, ortak bölümler; unigram/bigram deneyi | `feature/data-features` |
| 3. kişi | Logistic Regression, C deneyleri, hata analizi | `feature/logistic-regression` |
| 4. kişi | Linear SVM, C deneyleri, hata analizi | `feature/svm` |
| 5. kişi | Ortak değerlendirme kodunu yazma, grafikler, karşılaştırma, slaytları birleştirme ve sunum | `feature/evaluation` |

Repo sahibinin Naive Bayes başlangıcı hazırdır. Diğer model sorumluları kendi
kodlarını kendi branch'lerinde yazacaktır. Ortak veri kodu 2. kişinin geliştireceği
bir başlangıçtır; veri analizi ve özellik deneyleri ona aittir. Herkes kendi
bölümünün deneylerini, bulgularını, slayt taslağını ve README açıklamasını hazırlar.

## Kod sahipliği

- Repo sahibi: `src/banking77/naive_bayes.py`, `src/banking77/train_naive_bayes.py`,
  `src/banking77/benchmark_naive_bayes.py`, kendi modelinin sonuçları; GitHub ve son entegrasyon.
- 2. kişi: `src/banking77/data.py`, veri inceleme ve özellik deneyleri.
- 3. kişi: Logistic Regression için kendi model/eğitim dosyalarını ekler.
- 4. kişi: Linear SVM için kendi model/eğitim dosyalarını ekler.
- 5. kişi: ortak değerlendirme ve görselleştirme dosyalarını ekler; sunumu hazırlar.

Bir başkasının bölümüne değişiklik gerekiyorsa ilgili sorumluya bildirip PR üzerinden
anlaşın. Başlangıç altyapısı o kişinin deneylerinin veya analizinin tamamlandığı anlamına gelmez.

## Teslim sözleşmesi

- Aynı hazırlanmış veri ve `seed=42` kullanılır. Veri değişirse herkes yeniden çalıştırır.
- TF-IDF yalnızca eğitim verisinde öğrenilir; validation/test üzerinde `fit` yapılmaz.
- Model/özellik/parametre seçimi validation verisinde yapılır.
- Her model sahibi komutunu, metriklerini ve en az üç hata örneğini teslim eder.
- Her üye `contributions/AD_SOYAD_OGRENCI_NUMARASI_GITHUB_KULLANICI_ADI.md`
  dosyasında tamamladığı işleri ve kendi commit/PR bağlantılarını açıklar.
  [Katkı şablonu](../contributions/README.md) kullanılabilir.
- Tahmin dosyası sütunları: `id,text,true_label,predicted_label,correct,overlaps_training`.
- Karşılaştırılan deneylerin `dataset_summary_sha256` değerleri aynı olmalıdır.
- Arayüz planlanmıyor. Ders slaytları incelenerek gerekli baseline listesi
  doğrulanmalıdır; ek yöntemlerin kapsamı ve sorumlusu ayrıca belirlenir.
  Sunum tarihi: 12 Ekim 2026.

## Git akışı

```powershell
git switch main
git pull --ff-only
git switch -c feature/naive-bayes
# Kendi dosyalarını düzenle ve kontrol et.
git add src/banking77/naive_bayes.py src/banking77/train_naive_bayes.py
git commit -m "Add Naive Bayes validation experiments"
git push -u origin feature/naive-bayes
```

GitHub'da pull request açın. Repo sahibi inceleyip birleştirir. Hocanın istediği
ham ve hazırlanmış veri dosyaları, kaynak lisansı ve atıfla teslim edilecektir;
veri dosyalarının son entegrasyonunu repo sahibi yapar. `.venv`, kimlik bilgileri,
geçici araçlar ve büyük model dosyalarını commit etmeyin. Diğer kişiler kendi
branch adlarını ve değişen dosyalarını kullanmalıdır.

2. kişi için adımlar ve teslim dosyaları: [veri/özellik devir talimatı](HANDOFF_DATA.md).
