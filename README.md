# BANKING77 — Banka Müşteri Taleplerinin Sınıflandırılması

COE025 NLP Project 1 için kısa İngilizce müşteri mesajlarını 77 talep kategorisine
ayıran metin sınıflandırma projesi. Ekip planında TF-IDF + Naive Bayes, Logistic
Regression ve Linear SVM aynı veri bölümlerinde karşılaştırılacaktır.

Örnek beklenen etiket: `I am still waiting on my card` → `card_arrival`.
Bu, görevi anlatan bir örnektir; her mesajın model tarafından doğru tahmin edildiği iddia edilmez.

## Durum

Ortak kurulum/veri altyapısı, repo sahibinin **Naive Bayes** çalışması,
2. kişinin veri analizi ve özellik deneyleri, 3. kişinin **Logistic Regression**
ve 4. kişinin **Linear SVM** çalışması `main` içinde hazırdır.
Ortak karşılaştırma ve grafikler 5. kişi tarafından `feature/evaluation`
branch'inde hazırlanmıştır.
Geliştirme için varsayılan bölüm **validation**. Nihai ortak özellik ayarı ekip
tarafından kesinleştirilecek; her model sahibi kendi nihai test çıktısını üretecektir.
Resmî testte henüz model değerlendirmesi yapılmadı.

Repo sahibinin mevcut veri sürümü için Naive Bayes devir paketi:
[teknik açıklama ve hata analizi](docs/NAIVE_BAYES.md),
[tekrar çalıştırılabilir alpha benchmark'ı](results/NAIVE_BAYES_ALPHA.md).
Tamamlanan veri çalışması: [ön işleme ve veri analizi](docs/DATA.md),
[unigram/bigram deneyleri](results/FEATURE_EXPERIMENTS.md).
Logistic Regression: [teknik açıklama ve hata analizi](docs/LOGISTIC_REGRESSION.md),
[C/solver benchmark'ı](results/LOGISTIC_REGRESSION_C.md).
Linear SVM: [teknik açıklama ve hata analizi](docs/LINEAR_SVM.md),
[C/loss benchmark'ı](results/LINEAR_SVM_C.md).

Üç modelin ortak validation karşılaştırması:
[tablo, Macro F1 gerekçesi ve örnek hatalar](results/MODEL_COMPARISON.md).

Hocanın üç e-postasına göre [teslim takibi](docs/SUBMISSION.md) tutulur.
Her üye [kişisel katkı dosyasını](contributions/README.md) kendi commit/PR'larıyla
hazırlar. Baseline kapsamına ilişkin açık nokta teslim takibinde açıklanır.

## Kurulum

Python 3.10+ gerekir; geliştirme ortamı Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m banking77.data --offline
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Bu bilgisayarda `python` komutu Microsoft Store'a yönlenirse ilk komut yerine:

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
```

macOS/Linux'ta `python3 -m venv .venv` ve `.venv/bin/python` kullanın.
`requirements-lock.txt` Python 3.12'de doğrulanan paket sürümlerini sabitler.
Farklı bir Python sürümünde gerekirse `requirements.txt` ile uyumlu sürümleri çözün.
Ham veri PolyAI'nin resmî GitHub deposundaki sabitlenmiş commit'ten indirilir;
hesap, Hugging Face token'ı veya API anahtarı gerekmez.
Teslimde ham verinin kopyası repodadır; `--offline` mevcut kaynak dosyalarının
hash'lerini doğrular ve bölümleri yeniden üretir. Ham veriyi kaynaktan tekrar
indirmek için `--offline` seçeneğini kaldırın.

## Çalıştığını nasıl kontrol ederim?

Kurulumdan sonra proje klasöründe PowerShell terminalini açın.

1. Kod ve veri protokolü kontrollerini çalıştırın:

   ```powershell
   .\.venv\Scripts\python.exe -m unittest discover -s tests -v
   ```

   Sonunda `OK` görünmesi kontrollerin geçtiğini gösterir. Bu kontroller veri ayrımı,
   TF-IDF'in eğitim sınırı, model ayarları ve validation ile seçim kurallarını doğrular.

2. Naive Bayes'i eğitip 1.500 validation mesajında değerlendirin:

   ```powershell
   .\.venv\Scripts\python.exe -m banking77.train_naive_bayes --split validation --alpha 0.05
   ```

   Mevcut veri ve ayarla beklenen `accuracy` 0.862, `macro_f1` yaklaşık 0.8529'dur.
   Ekrandaki `run_id`, `results/runs/` altındaki o çalıştırmanın sonuç klasörünü ve
   `artifacts/` altındaki model dosyasını belirtir. `predictions.csv` içinde gerçek
   etiket, model tahmini ve tahminin doğru olup olmadığı satır satır görülebilir.

3. Az önce kaydedilen Naive Bayes modeliyle kendi İngilizce mesajınızı deneyin:

   ```powershell
   $nbModel = Get-ChildItem -LiteralPath artifacts -Filter 'naive_bayes_validation_*.joblib' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
   .\.venv\Scripts\python.exe -m banking77.predict --model-path $nbModel.FullName --text "I lost my card"
   ```

   Doğrulanan çıktı: `lost_or_stolen_card` (kayıp veya çalıntı kart).
   `--text` içindeki mesajı değiştirerek farklı talepler deneyebilirsiniz.
   Tek bir doğru örnek genel başarıyı göstermez; bunun için validation metriklerine
   ve yanlış tahminlere birlikte bakılır. `predict` aynı biçimde kaydedilmiş LR ve
   SVM modelleriyle de çalışır; ilgili `.joblib` yolunu kullanın.

Bu adımlar validation ve tek mesaj tahmini kullanır; resmî testte model değerlendirmesi yapmaz.

## Veri analizi

```powershell
.\.venv\Scripts\python.exe -m banking77.analyze_data
```

Bu komut sınıf dağılımı, mesaj uzunluğu, boş/tekrar kayıt ve metin örtüşmelerini
raporlar; veri dosyalarını değiştirmez. Ön işleme kararları [DATA.md](docs/DATA.md),
özellik karşılaştırması [FEATURE_EXPERIMENTS.md](results/FEATURE_EXPERIMENTS.md) içindedir.
Özellik deneyindeki öneri ortak model ayarlarını otomatik değiştirmez.

## Naive Bayes başlangıcı

```powershell
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes --ngram-max 1 --alpha 0.5
```

Bu komutlar yalnızca Naive Bayes modelini çalıştırır. Her çalıştırma `results/runs/` altında ayrı bir klasör ve `artifacts/` altında
eğitilmiş model üretir. Metrikler accuracy, macro F1, eğitim ve tahmin sürelerini
içerir. `predictions.csv` ve `confusion_matrix.csv` hata analizinde kullanılır.

[Naive Bayes alpha deneyi](results/NAIVE_BAYES_ALPHA.md) yalnızca validation
üzerinde yapılmıştır. Denenen değerlerde en yüksek macro F1 alpha=0.05 ile
0.8529, accuracy %86.20'dir. Aynı deneyi çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes --alpha 0.05
```

Bu değerler nihai test sonucu değildir; varsayılan alpha=1.0 başlangıç ayarıdır.

Beş alpha denemesini sırayla çalıştırıp JSON ve Markdown raporlarını yeniden üretmek için:

```powershell
.\.venv\Scripts\python.exe -m banking77.benchmark_naive_bayes
```

Bu komut yalnızca validation kullanır. Tekrarlanan çalıştırmaların süreleri ve
run kimlikleri değişebilir; sabit veri ve ortamla sınıflandırma skorları tekrar üretilebilir.

Parametreleri validation ile seçip dondurduktan sonra, örneğin Naive Bayes için:

```powershell
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes --alpha 0.05 --split test
```

Buradaki alpha mevcut validation deneyinden gelir. 2. kişinin veri/özellik
çalışması sonuçlanmadan bu nihai test adımını çalıştırmayın; ayarlar değişirse
önce validation deneyini yeniden yapıp kesinleştirin.

Kaydedilmiş bir modelle tek mesaj tahmini:

```powershell
.\.venv\Scripts\python.exe -m banking77.predict --model-path artifacts/MODEL_DOSYASI.joblib --text "I am still waiting on my card"
```

`MODEL_DOSYASI` yerine çalıştırmanın ürettiği gerçek dosya adını yazın.

## Logistic Regression

3. kişinin başlangıç ayarını ve validation ile seçtiği ayarı çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -m banking77.train_logistic_regression --split validation
.\.venv\Scripts\python.exe -m banking77.train_logistic_regression --split validation --solver liblinear-ovr --C 100
```

Seçilen ayar `liblinear-ovr`, `C=100`: validation accuracy %89.07,
macro F1 0.8920. Bunlar nihai test sonuçları değildir.
Naive Bayes ile aynı `metrics.json`, `classification_report.json`,
`predictions.csv` ve `confusion_matrix.csv` çıktı sözleşmesi kullanılır.

Üç solver ve beş C değerinden oluşan 15 validation deneyini yeniden üretmek için:

```powershell
.\.venv\Scripts\python.exe -m banking77.benchmark_logistic_regression
```

Komut [C/solver raporunu](results/LOGISTIC_REGRESSION_C.md) ve
[JSON kaydını](results/logistic_regression_validation.json) yeniden üretir.
Yöntem, ayar seçimi ve hata analizi [teknik notta](docs/LOGISTIC_REGRESSION.md) açıklanır.
Nihai test, ortak veri/özellik ayarı kesinleştikten sonra model sorumlusu tarafından çalıştırılacaktır.

## Linear SVM

4. kişinin validation ile doğruladığı ayarı çalıştırmak için:

```powershell
.\.venv\Scripts\python.exe -m banking77.train_linear_svm --split validation --loss squared_hinge --C 1
```

Seçilen ayar `squared_hinge`, `C=1`: validation accuracy %89.20, macro F1 0.8935.
Bu ayar başlangıç ayarıyla aynıdır; C/loss deneyi başlangıç skorunu yükseltmemiştir.
TF-IDF ve `metrics.json`, `classification_report.json`, `predictions.csv`,
`confusion_matrix.csv` çıktı sözleşmesi NB/LR ile aynıdır.

İki loss ve beş C değerinden oluşan 10 validation deneyini yeniden üretmek için:

```powershell
.\.venv\Scripts\python.exe -m banking77.benchmark_linear_svm
```

Komut [C/loss raporunu](results/LINEAR_SVM_C.md) ve
[JSON kaydını](results/linear_svm_validation.json) yeniden üretir.
Yöntem, seçim gerekçesi ve örnek hataların kelime katkıları
[teknik notta](docs/LINEAR_SVM.md) açıklanır.
Bu skorlar nihai test sonucu değildir; nihai testi ortak özellik ayarı
kesinleşince model sorumlusu çalıştıracaktır.

## Ortak model karşılaştırması

NB, LR ve SVM'in başlangıç ve seçilen ayarlarını aynı bilgisayarda, sırayla ve
yalnızca validation üzerinde çalıştırmak için (tek komut):

```powershell
.\.venv\Scripts\python.exe -m banking77.benchmark_models
```

Komut mevcut eğitim fonksiyonlarını kullanır; her ayarı 3 kez çalıştırıp süreler için
medyan alır (`--repeats` ile değişir) ve [results/MODEL_COMPARISON.md](results/MODEL_COMPARISON.md),
`results/model_comparison_validation.json` ile `results/figures/*.png` dosyalarını üretir.
Grafik için matplotlib gerekir (`requirements-lock.txt` içinde).
Resmî test bu komutta kullanılmaz.

Özellik ayarını (unigram, unigram + bigram, yalnızca bigram) üç modelde, her modelin
kendi hiperparametre ızgarasıyla yeniden ayarlayarak karşılaştırmak için:

```powershell
.\.venv\Scripts\python.exe -m banking77.benchmark_features
```

Komut yalnızca validation kullanır (~2 dakika) ve [results/FEATURE_COMPARISON.md](results/FEATURE_COMPARISON.md)
ile `results/feature_comparison_validation.json` dosyalarını üretir.

## Yapı

```text
configs/          Veri kaynağı, sabit commit ve dosya hash'leri
src/banking77/    Veri hazırlama/analiz, Naive Bayes, Logistic Regression ve Linear SVM
tests/            Veri, TF-IDF eğitim sınırı ve model/seçim kontrolleri
docs/             Veri/model açıklamaları, görev dağılımı ve deney protokolü
contributions/    Kişisel katkı dosyaları ve katkı şablonu
data/raw/         Resmî train/test, kategori listesi ve kaynak veri lisansı
data/processed/   Hazırlanmış train/validation/test, kategoriler ve özet
results/          Deney sonuçları; runs/ yerelde tutulur
artifacts/        Eğitilmiş modeller (Git dışında)
.github/          Otomatik kontroller ve PR şablonu
LICENSE           Proje kodunun MIT lisansı
```

Mevcut scriptlerin görevleri:

| Dosya | Görevi |
| --- | --- |
| `src/banking77/data.py` | Sabit kaynağı indirme, hash kontrolü, başlangıç temizliği ve veri ayrımı |
| `src/banking77/analyze_data.py` | Veri dağılımı, mesaj uzunluğu, tekrar ve örtüşme analizi |
| `src/banking77/naive_bayes.py` | TF-IDF + Multinomial Naive Bayes pipeline'ını oluşturma |
| `src/banking77/train_naive_bayes.py` | Naive Bayes eğitimi, metrik/tahmin kaydı ve model kaydetme |
| `src/banking77/benchmark_naive_bayes.py` | Beş alpha değerini validation üzerinde karşılaştırma ve NB raporlarını üretme |
| `src/banking77/logistic_regression.py` | Aynı TF-IDF ile multinomial veya one-vs-rest Logistic Regression pipeline'ı |
| `src/banking77/train_logistic_regression.py` | LR eğitimi, ortak metrik/tahmin çıktıları, model ve yakınsama kaydı |
| `src/banking77/benchmark_logistic_regression.py` | 15 C/solver ayarını validation üzerinde karşılaştırma ve LR raporlarını üretme |
| `src/banking77/linear_svm.py` | Aynı TF-IDF ile L2 düzenlileştirmeli one-vs-rest Linear SVM pipeline'ı |
| `src/banking77/train_linear_svm.py` | SVM eğitimi, ortak metrik/tahmin çıktıları, model ve yakınsama kaydı |
| `src/banking77/benchmark_linear_svm.py` | 10 C/loss ayarını validation üzerinde karşılaştırma, SVM raporu ve kelime katkılarını üretme |
| `src/banking77/benchmark_models.py` | NB/LR/SVM başlangıç ve seçilen ayarlarını sırayla çalıştırma, karşılaştırma raporu ve eşleştirilmiş testler |
| `src/banking77/benchmark_features.py` | Unigram, unigram + bigram ve yalnızca bigram'ı üç modelde yeniden ayarlayarak karşılaştırma |
| `src/banking77/plot_comparison.py` | Karşılaştırma grafikleri (skor, süre, karışan çiftler) |
| `src/banking77/predict.py` | Kaydedilmiş modelle tek mesajın kategorisini tahmin etme |
| `tests/test_protocol.py` | Tekrar/etiket kontrollerini, veri ayrımını ve TF-IDF eğitim sınırını doğrulama |
| `tests/test_data_files.py` | CSV'nin LF satır sonuyla yazılmasını ve metinlerin korunmasını doğrulama |
| `tests/test_logistic_regression.py` | LR ayarları, solver'lar, TF-IDF eğitim sınırı ve validation seçim kuralı |
| `tests/test_benchmark_models.py` | Benchmark ayarları, yalnızca validation kullanımı, medyan süre, McNemar ve bootstrap kontrolleri |
| `tests/test_benchmark_features.py` | Özellik karşılaştırmasının seçim kuralı, validation-only kullanımı ve ızgaraları |
| `tests/test_linear_svm.py` | SVM ayarları, loss'lar, TF-IDF eğitim sınırı, seçim kuralı ve kelime katkıları |

Ham CSV'ler değiştirilmeden korunur. Hazırlanmış train/validation dosyalarında
tekrarlardan arındırılmış resmî eğitim verisi ve satır kimlikleri bulunur;
resmî testin metin ve etiketleri korunur. Mevcut başlangıç temizliği 4 eğitim
tekrarını kaldırır, boş eğitim kaydı bulmaz. NFKC/casefold/boşluk normalizasyonu
yalnızca tekrar kontrolü içindir; model orijinal mesaj metnini alır.
Ön işleme pipeline'ının 2. kişi tarafından hazırlanan ayrıntılı incelemesi
[docs/DATA.md](docs/DATA.md) içindedir.

## Ekip ve deneyler

- [Görev dağılımı ve Git akışı](docs/TEAM.md)
- [Deney planı ve değerlendirme protokolü](docs/EXPERIMENTS.md)
- [Sonuç dosyaları](results/README.md)
- [Naive Bayes alpha deneyi](results/NAIVE_BAYES_ALPHA.md)
- [Naive Bayes teknik notu ve hata analizi](docs/NAIVE_BAYES.md)
- [Veri analizi ve ön işleme kararları](docs/DATA.md)
- [Unigram/bigram özellik deneyleri](results/FEATURE_EXPERIMENTS.md)
- [Logistic Regression C/solver deneyi](results/LOGISTIC_REGRESSION_C.md)
- [Logistic Regression teknik notu ve hata analizi](docs/LOGISTIC_REGRESSION.md)
- [Linear SVM C/loss deneyi](results/LINEAR_SVM_C.md)
- [Linear SVM teknik notu ve hata analizi](docs/LINEAR_SVM.md)
- [Ortak model karşılaştırması](results/MODEL_COMPARISON.md)
- [Unigram / bigram özellik karşılaştırması (üç model)](results/FEATURE_COMPARISON.md)
- [2. kişiye veri/özellik devir talimatı](docs/HANDOFF_DATA.md)

Veri daha önce indirilmişse ağ bağlantısı olmadan yeniden hazırlamak için
`python -m banking77.data --offline` kullanılabilir; bu adım kaynak hash'lerini de kontrol eder.

## Veri kaynağı ve atıf

BANKING77: 10.003 resmî eğitim ve 3.080 resmî test örneği, 77 kategori.
Yerel train/validation sayıları tekrar kontrolünden sonra `data/processed/summary.json`
içinde bulunur. Resmî test korunur; mevcut metin örtüşmeleri açıkça raporlanır.

- Veri: https://github.com/PolyAI-LDN/task-specific-datasets
- Makale: https://arxiv.org/abs/2003.04807
- Veri lisansı: **CC BY 4.0**; ham ve hazırlanmış veriye uygulanır.
  Kaynak lisansın tam metni [data/raw/LICENSE](data/raw/LICENSE) dosyasındadır.
- Projenin kendi kodu ve dokümantasyonu: [MIT](LICENSE). Veri dosyalarının
  kaynak lisansı ayrıca korunur. Yukarıda hazırlama sırasında yapılan değişiklikler açıklanmıştır.

```bibtex
@inproceedings{Casanueva2020,
  author = {Iñigo Casanueva and Tadas Temcinas and Daniela Gerz and Matthew Henderson and Ivan Vulić},
  title = {Efficient Intent Detection with Dual Sentence Encoders},
  year = {2020},
  booktitle = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020},
  url = {https://arxiv.org/abs/2003.04807}
}
```
