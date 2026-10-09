# 2. kişiye devir: veri ve özellikler

Bu dosya ilk veri/özellik devir talimatını saklar. 2. kişinin çalışması
[PR #2](https://github.com/kaanksn1/banking77-intent-classification/pull/2) ile
`main` içine alınmıştır: [veri incelemesi](DATA.md),
[özellik deneyleri](../results/FEATURE_EXPERIMENTS.md).
Aşağıdaki adımlar ilk teslim kapsamıdır; nihai ortak özellik kararı ekipçe verilecektir.

## Hazır başlangıç

- Public repo: https://github.com/kaanksn1/banking77-intent-classification
- Ortak kurulum, sabit kaynak/hash kontrolü ve başlangıç veri hazırlama kodu var.
- Naive Bayes modeli, alpha-only benchmark ve kendi hata analizi var.
- Mevcut bölümler: train 8.499, validation 1.500, resmî test 3.080.
- Başlangıç kontrolü 4 eğitim tekrarını kaldırdı; boş eğitim kaydı bulmadı.
- Normalize metin örtüşmeleri: train/validation 0, train/test 7, validation/test 0.
- Ham ve hazırlanmış veri, lisans ve atıf repoda teslim edilir.

## Başlangıç komutları

```powershell
git clone https://github.com/kaanksn1/banking77-intent-classification.git
cd banking77-intent-classification
git switch -c feature/data-features
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m banking77.data --offline
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Repo zaten klonlandıysa güncel `main` için `git switch main` ve
`git pull --ff-only` kullanın; sonra kendi branch'inizi açın.
macOS/Linux'ta `.venv/bin/python` kullanılabilir. Python 3.12 önerilir.

## Yapılacak işler

1. Sınıf dağılımını, mesaj uzunluklarını, boş/tekrar kayıtları ve etiket
   tutarlılığını inceleyin. Sonuçları sayılarla kaydedin; hazır veride gerçekten
   bulunan sorunlar üzerinden ilerleyin.
2. `src/banking77/data.py` başlangıcını inceleyin. Uyguladığınız ön işleme
   kararlarını ve gerekçelerini `docs/DATA.md` içinde anlatın. Gerekli gördüğünüz
   veri hazırlama değişikliklerini kendi PR'ınızda yapın. Ham kaynağı değiştirmeyin.
3. Unigram/bigram özellik deneyini validation üzerinde, aynı alpha ve aynı veri
   bölümleriyle yapın. Örneğin mevcut NB çalıştırıcısını model kodunu değiştirmeden kullanabilirsiniz:

   ```powershell
   .\.venv\Scripts\python.exe -m banking77.train_naive_bayes --ngram-max 1 --alpha 1.0
   .\.venv\Scripts\python.exe -m banking77.train_naive_bayes --ngram-max 2 --alpha 1.0
   ```

   Çıktıdaki `run_id` ile `results/runs/` klasörlerini eşleştirin. Accuracy,
   macro F1, veri kimliği ve komutları `results/FEATURE_EXPERIMENTS.md` içinde
   kaydedin. Bu sizin özellik deneyinizdir; repo sahibinin alpha deneyi ayrıdır.
4. Veriniz/özellik ayarınız hazır olduğunda hangi ortak ayarın kullanılmasını
   önerdiğinizi ve nedenini yazın. Veri veya özellik değişirse tüm model sahipleri
   kendi deneylerini yeniden çalıştırır. NB model dosyalarını repo sahibi geliştirir.
5. Kendi `contributions/AD_SOYAD_OGRENCI_NUMARASI_GITHUB_KULLANICI_ADI.md`
   dosyanızı ekleyin; kendi commit ve PR bağlantılarınızla gerçek katkınızı açıklayın.

## Korunacak sözleşme

- Hazırlanmış CSV sütunları `id,text,category`; 77 etiketin adları korunur.
- Kaynak commit/hash ve lisans korunur. `--offline` ağ olmadan aynı veri bölümlerini üretmelidir.
- `seed=42`, stratified %15 validation ve train/validation ayrıklığı korunur.
- Vocabulary/IDF yalnızca train üzerinde öğrenilir; validation/test üzerinde fit yapılmaz.
- Resmî testin metin/etiketleri değiştirilmez. Test üzerinde ayar/özellik seçilmez.
- Testle mevcut örtüşmeler gizlenmez; [deney protokolü](EXPERIMENTS.md) uygulanır.
- `summary.json` gerçek veri değişikliklerini yansıtmalıdır; yalnızca eski özeti
  bırakıp CSV'leri değiştirmeyin. Karşılaştırılacak modeller aynı veri kimliğini kullanmalıdır.

## PR teslimi

PR'da kendi veri kodu değişikliklerinizi, `docs/DATA.md`, özellik deney raporunu,
kişisel katkı dosyanızı ve ilgili yeniden üretilmiş veri/özet dosyalarını ekleyin.
Model dosyaları, `.venv`, geçici araçlar ve büyük yerel run klasörlerini eklemeyin.
Çalıştırdığınız komutları ve sonuçları PR açıklamasına yazın. Repo sahibi
inceleyip birleştirecek; diğer model sahipleri güncellenen ortak ayarla devam edecek.
