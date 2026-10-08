# BANKING77 — Banka Müşteri Taleplerinin Sınıflandırılması

COE025 NLP Project 1 için kısa İngilizce müşteri mesajlarını 77 talep kategorisine
ayıran metin sınıflandırma projesi. Ekip planında TF-IDF + Naive Bayes, Logistic
Regression ve Linear SVM aynı veri bölümlerinde karşılaştırılacaktır.

Örnek beklenen etiket: `I am still waiting on my card` → `card_arrival`.
Bu, görevi anlatan bir örnektir; her mesajın model tarafından doğru tahmin edildiği iddia edilmez.

## Durum

Bu aşamada ortak kurulum/veri altyapısı ve repo sahibinin **Naive Bayes** kodu
hazırdır. Logistic Regression 3. kişinin, Linear SVM 4. kişinin kendi branch'inde
geliştireceği bölümlerdir. Ortak karşılaştırma, grafikler ve sunum 5. kişinin görevidir.
Veri analizi ve özellik deneyleri 2. kişiye aittir; mevcut veri kodu ortak başlangıçtır.
Geliştirme için varsayılan bölüm **validation**. Nihai model seçimi, hata analizi ve
test raporu ekip çalışmasıyla tamamlanacaktır.

Repo sahibinin mevcut veri sürümü için Naive Bayes devir paketi:
[teknik açıklama ve hata analizi](docs/NAIVE_BAYES.md),
[tekrar çalıştırılabilir alpha benchmark'ı](results/NAIVE_BAYES_ALPHA.md).
2. kişi [veri/özellik devir talimatıyla](docs/HANDOFF_DATA.md) kendi çalışmasına başlayabilir.

Hocanın üç e-postasına göre [teslim takibi](docs/SUBMISSION.md) tutulur.
Her üye [kişisel katkı dosyasını](contributions/README.md) kendi commit/PR'larıyla
hazırlar. Önceki paylaşılan konuşmadaki slayt metin dökümü yeniden bulunmuştur;
model başlıkları ve baseline kapsamına ilişkin açık nokta teslim takibinde açıklanır.

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

## Yapı

```text
configs/          Veri kaynağı, sabit commit ve dosya hash'leri
src/banking77/    Ortak veri başlangıcı, Naive Bayes eğitimi ve tahmin
tests/            Veri ayrımı ve TF-IDF sızıntısı kontrolleri
docs/             Görev dağılımı ve deney protokolü
contributions/    Kişisel katkı dosyaları için biçim ve şablon
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
| `src/banking77/naive_bayes.py` | TF-IDF + Multinomial Naive Bayes pipeline'ını oluşturma |
| `src/banking77/train_naive_bayes.py` | Naive Bayes eğitimi, metrik/tahmin kaydı ve model kaydetme |
| `src/banking77/benchmark_naive_bayes.py` | Beş alpha değerini validation üzerinde karşılaştırma ve NB raporlarını üretme |
| `src/banking77/predict.py` | Kaydedilmiş modelle tek mesajın kategorisini tahmin etme |
| `tests/test_protocol.py` | Tekrar/etiket kontrollerini, veri ayrımını ve TF-IDF eğitim sınırını doğrulama |

Ham CSV'ler değiştirilmeden korunur. Hazırlanmış train/validation dosyalarında
tekrarlardan arındırılmış resmî eğitim verisi ve satır kimlikleri bulunur;
resmî testin metin ve etiketleri korunur. Mevcut başlangıç temizliği 4 eğitim
tekrarını kaldırır, boş eğitim kaydı bulmaz. NFKC/casefold/boşluk normalizasyonu
yalnızca tekrar kontrolü içindir; model orijinal mesaj metnini alır.
Ön işleme pipeline'ının ayrıntılı incelemesi ve dokümantasyonu 2. kişinin görevidir.

## Ekip ve deneyler

- [Görev dağılımı ve Git akışı](docs/TEAM.md)
- [Deney planı ve değerlendirme protokolü](docs/EXPERIMENTS.md)
- [Sonuç dosyaları](results/README.md)
- [Naive Bayes alpha deneyi](results/NAIVE_BAYES_ALPHA.md)
- [Naive Bayes teknik notu ve hata analizi](docs/NAIVE_BAYES.md)
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
