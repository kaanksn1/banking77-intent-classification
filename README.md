# BANKING77 — Banka Müşteri Taleplerinin Sınıflandırılması

COE025 NLP Project 1 için kısa İngilizce müşteri mesajlarını 77 talep kategorisine
ayıran metin sınıflandırma projesi. TF-IDF + Naive Bayes, Logistic Regression ve
Linear SVM aynı veri bölümlerinde karşılaştırılır.

Örnek: `I am still waiting on my card` → `card_arrival`.

## Durum

Başlangıç altyapısı ve üç klasik model tanımlı. Geliştirme için varsayılan bölüm
**validation**. Nihai model seçimi, hata analizi ve test raporu ekip çalışmasıyla
tamamlanacaktır. Henüz nihai başarı iddiası yoktur.

## Kurulum

Python 3.10+ gerekir; geliştirme ortamı Python 3.12.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m banking77.data
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

## İlk deneyler

```powershell
.\.venv\Scripts\python.exe -m banking77.train --model naive_bayes
.\.venv\Scripts\python.exe -m banking77.train --model naive_bayes --ngram-max 1 --alpha 0.5
.\.venv\Scripts\python.exe -m banking77.train --model logistic_regression
.\.venv\Scripts\python.exe -m banking77.train --model svm
```

Her çalıştırma `results/runs/` altında ayrı bir klasör ve `artifacts/` altında
eğitilmiş model üretir. Metrikler accuracy, macro F1, eğitim ve tahmin sürelerini
içerir. `predictions.csv` ve `confusion_matrix.csv` hata analizinde kullanılır.

Parametreleri validation ile seçip dondurduktan sonra, örneğin Naive Bayes için:

```powershell
.\.venv\Scripts\python.exe -m banking77.train --model naive_bayes --alpha 0.5 --split test
```

Buradaki alpha yalnızca komut örneğidir; en iyi ayar olduğu iddia edilmez.

Kaydedilmiş bir modelle tek mesaj tahmini:

```powershell
.\.venv\Scripts\python.exe -m banking77.predict --model-path artifacts/MODEL_DOSYASI.joblib --text "I am still waiting on my card"
```

`MODEL_DOSYASI` yerine çalıştırmanın ürettiği gerçek dosya adını yazın.

## Yapı

```text
configs/          Veri kaynağı, sabit commit ve dosya hash'leri
src/banking77/    Veri hazırlama, modeller, eğitim ve tahmin
tests/            Veri ayrımı ve TF-IDF sızıntısı kontrolleri
docs/             Görev dağılımı ve deney protokolü
data/             İndirilen ve hazırlanan veriler (Git dışında)
results/          Deney sonuçları; runs/ yerelde tutulur
artifacts/        Eğitilmiş modeller (Git dışında)
.github/          Otomatik kontroller ve PR şablonu
```

## Ekip ve deneyler

- [Görev dağılımı ve Git akışı](docs/TEAM.md)
- [Deney planı ve değerlendirme protokolü](docs/EXPERIMENTS.md)
- [Sonuç dosyaları](results/README.md)
- [İlk doğrulama sonuçları](results/INITIAL_VALIDATION.md)

Veri daha önce indirilmişse ağ bağlantısı olmadan yeniden hazırlamak için
`python -m banking77.data --offline` kullanılabilir; bu adım kaynak hash'lerini de kontrol eder.

## Veri kaynağı ve atıf

BANKING77: 10.003 resmî eğitim ve 3.080 resmî test örneği, 77 kategori.
Yerel train/validation sayıları tekrar kontrolünden sonra `data/processed/summary.json`
içinde bulunur. Resmî test korunur; mevcut metin örtüşmeleri açıkça raporlanır.

- Veri: https://github.com/PolyAI-LDN/task-specific-datasets
- Makale: https://arxiv.org/abs/2003.04807
- Veri lisansı: **CC BY 4.0**; indirme adımında kaynak lisans dosyası da alınır.

```bibtex
@inproceedings{Casanueva2020,
  author = {Iñigo Casanueva and Tadas Temcinas and Daniela Gerz and Matthew Henderson and Ivan Vulić},
  title = {Efficient Intent Detection with Dual Sentence Encoders},
  year = {2020},
  booktitle = {Proceedings of the 2nd Workshop on NLP for ConvAI - ACL 2020},
  url = {https://arxiv.org/abs/2003.04807}
}
```
