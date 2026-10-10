# Görev paylaşımı

| Sorumlu | Kapsam | Önerilen branch |
| --- | --- | --- |
| Repo sahibi (sen) | Naive Bayes, alpha deneyleri, hata analizi; GitHub/entegrasyon; sonradan açıkça atanan neural baseline ve NB + CNN çalışması | `feature/neural-baselines` |
| 2. kişi | Veri analizi, kalite kontrolü, ortak bölümler; unigram/bigram deneyi | `feature/data-features` |
| 3. kişi | Logistic Regression, C deneyleri, hata analizi | `feature/logistic-regression` |
| 4. kişi | Linear SVM, C deneyleri, hata analizi | `feature/svm` |
| 5. kişi | Ortak değerlendirme kodunu yazma, grafikler, karşılaştırma, slaytları birleştirme ve sunum | `feature/evaluation` |

Repo sahibinin Naive Bayes çalışması, 2. kişinin veri/özellik çalışması,
3. kişinin Logistic Regression ve 4. kişinin Linear SVM çalışması kendi PR'larıyla
`main` içine alınmıştır. 5. kişinin ortak validation değerlendirmesi ve
grafikleri PR #8, klasik modellerin nihai test karşılaştırması PR #11 ile
birleştirilmiştir. Yeni modelleri kapsayan karşılaştırma, PPTX ve sunum provası
beklemektedir. Hocanın son e-postasındaki ek baseline işi repo sahibine açıkça
atanmıştır; [kapsam ve durum](NEURAL_BASELINES.md) ayrı takip edilir.
Herkes kendi bölümünün deneylerini, bulgularını, slayt taslağını ve açıklamasını
hazırlar. Repo sahibi ortak README ve GitHub entegrasyonunu günceller.

## Kod sahipliği

- Repo sahibi: `src/banking77/naive_bayes.py`, `src/banking77/train_naive_bayes.py`,
  `src/banking77/benchmark_naive_bayes.py`, kendi modelinin sonuçları; GitHub ve son entegrasyon.
  Ek kapsam: `neural_models.py`, `train_neural.py`, `train_embeddings.py`,
  `prepare_embeddings.py`, `train_ensemble.py`, GPU hazırlığı ve bunların deneyleri.
- 2. kişi: `src/banking77/data.py`, veri inceleme ve özellik deneyleri.
- 3. kişi: Logistic Regression için kendi model/eğitim dosyalarını ekler.
- 4. kişi: Linear SVM için kendi model/eğitim dosyalarını ekler.
- 5. kişi: ortak değerlendirme ve görselleştirme dosyalarını ekler; sunumu hazırlar.

Bir başkasının bölümüne değişiklik gerekiyorsa ilgili sorumluya bildirip PR üzerinden
anlaşın. Başlangıç altyapısı o kişinin deneylerinin veya analizinin tamamlandığı anlamına gelmez.

## Teslim sözleşmesi

- Aynı hazırlanmış veri ve `seed=42` kullanılır. Veri değişirse herkes yeniden çalıştırır.
- Klasik modellerde mevcut unigram + bigram, sublinear TF temsili korunur;
  [NB test öncesi kayıt ve deney protokolü](EXPERIMENTS.md) uygulanır.
  Yeni modeller aynı mesaj bölümlerinde kendi token/embedding temsilini kullanır.
- TF-IDF yalnızca eğitim verisinde öğrenilir; validation/test üzerinde `fit` yapılmaz.
- Model/özellik/parametre seçimi validation verisinde yapılır.
- Her model sahibi komutunu, metriklerini ve en az üç hata örneğini teslim eder.
- Her üye `contributions/AD_SOYAD_OGRENCI_NUMARASI_GITHUB_KULLANICI_ADI.md`
  dosyasında tamamladığı işleri ve kendi commit/PR bağlantılarını açıklar.
  [Katkı şablonu](../contributions/README.md) kullanılabilir.
- Tahmin dosyası sütunları: `id,text,true_label,predicted_label,correct,overlaps_training`.
- Karşılaştırılan deneylerin `dataset_summary_sha256` değerleri aynı olmalıdır.
- Arayüz planlanmıyor. Ders slaytlarının metin ve resimleri incelendi;
  [baseline eşleştirmesi](NEURAL_BASELINES.md) hazır. Repo sahibi yeni veri ve
  JEV/Laya/AnyJev ekleme planını iptal etmiştir.
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
