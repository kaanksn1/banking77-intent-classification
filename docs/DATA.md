# Veri analizi ve ön işleme incelemesi

Bu belge 2. kişinin (veri analizi ve özellik mühendisliği) çalışmasıdır. Tablolardaki
sayılar repodaki veriden `python -m banking77.analyze_data` ile üretilmiştir. Örnek
cümleler, ASCII dışı karakter dökümü ve "boşluk normalizasyonu sözlüğü değiştirmez"
kontrolü ise tek seferlik inceleme betikleriyle yapılmıştır (repoya eklenmedi). Elle
yazılmış veya literatürden alınmış değer yoktur. Ham kaynak dosyalar değiştirilmemiştir.

Veri kimliği: `data/processed/summary.json` SHA-256 =
`468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`
(bu çalışmada **değişmedi**; model sahiplerinin deneyleri yeniden çalıştırması gerekmez).

## 1. Veri seti özeti

| Küme | Satır | Sınıf | Sınıf başına min / ort / maks | Dengesizlik (maks/min) |
| --- | ---: | ---: | --- | ---: |
| Ham train | 10.003 | 77 | 35 / 129,9 / 187 | 5,34 |
| Hazır train | 8.499 | 77 | 30 / 110,4 / 159 | 5,30 |
| Hazır validation | 1.500 | 77 | 5 / 19,5 / 28 | 5,60 |
| Resmî test | 3.080 | 77 | 40 / 40,0 / 40 | 1,00 |

- Train ve validation aynı oranlarla (stratified, `seed=42`) bölünmüştür; test ise
  **tam dengelidir** (her sınıftan 40). Eğitim dağılımı dengesiz, test dağılımı dengeli
  olduğu için test accuracy'si ile macro F1 arasında fark beklenebilir.
- En küçük sınıflar: `contactless_not_working` (train 30 / validation 5),
  `virtual_card_not_working` (35 / 6), `card_acceptance` (50 / 9). Validation'daki 5–9 örnekli
  sınıflarda sınıf bazlı F1 çok oynaktır; macro F1 bu yüzden gürültülüdür.
- En büyük sınıflar: `card_payment_fee_charged` (159 / 28),
  `direct_debit_payment_not_recognised` (155 / 27),
  `balance_not_updated_after_cheque_or_cash_deposit` (154 / 27).
- Her kümede 77 sınıfın tamamı bulunur; satır kimlikleri üç kümede de benzersizdir.

### Mesaj uzunlukları

| Küme | Karakter ort. / medyan / min / maks | Kelime ort. / medyan / min / maks |
| --- | --- | --- |
| Hazır train | 59,74 / 47 / 13 / 433 | 12,01 / 10 / 2 / 79 |
| Hazır validation | 57,97 / 46 / 13 / 409 | 11,61 / 10 / 2 / 78 |
| Resmî test | 54,23 / 45 / 13 / 368 | 10,95 / 9 / 2 / 69 |

Mesajlar kısadır (medyan 9–10 kelime). Bu, TF-IDF'te belge başına yalnızca ~10 unigram
ve ~20 unigram+bigram özelliği demektir (bkz. `results/FEATURE_EXPERIMENTS.md`).
Test mesajları ortalama olarak eğitim mesajlarından biraz kısadır.

### Boş/eksik kayıtlar, tekrarlar, etiket çakışmaları

| Kontrol | Ham train | Hazır train | Hazır validation | Resmî test |
| --- | ---: | ---: | ---: | ---: |
| Boş / yalnızca boşluk metin | 0 | 0 | 0 | 0 |
| Eksik etiket | 0 | 0 | 0 | 0 |
| Birebir aynı metin | 0 | 0 | 0 | 0 |
| Normalize edilmiş (NFKC + casefold + boşluk) tekrar | 4 | 0 | 0 | 1 |
| Aynı metinde farklı etiket | 0 | 0 | 0 | 0 |
| Baş/son boşluk veya satır sonu içeren metin | 9 | 6 | 0 | 3 |
| Ardışık iç boşluk içeren metin | 455 | 392 | 63 | 100 |
| ASCII dışı karakter içeren metin | 52 | 44 | 8 | 9 |

- Ham train'deki 4 tekrar, yalnızca baştaki/sondaki satır sonu ile ayrışan aynı cümlelerdir
  ve etiketleri aynıdır (örn. `"\nWhere can I withdraw money from?"` ve
  `"Where can I withdraw money from?"`, `atm_support`). Hazırlama bunları kaldırmıştır.
- Resmî testte 1 normalize tekrar vardır (`test-01441` ve `test-01461`, `atm_support`).
  Resmî test değiştirilmediği için bırakılmıştır.
- Etiket çakışması hiçbir kümede ve kümeler arasında yoktur.
- ASCII dışı karakterler `£` (34), `€` (26), kesintisiz boşluk (2) ve `…` (1) olarak sayılır;
  bozuk kodlama (`U+FFFD`) bulunmamıştır.

### Küme bütünlüğü ve sızıntı

| Çift | Birebir (normalize) ortak metin | Etiket çakışması | Noktalamasız ortak metin |
| --- | ---: | ---: | ---: |
| train / validation | 0 | 0 | **4** |
| train / test | **7** | 0 | **21** |
| validation / test | 0 | 0 | **4** |

- Normalize anahtarla train/validation ayrıktır; `data.py` bunu ayırma sırasında zorunlu kılar.
- **Resmî test tamamen temiz bir holdout değildir:** 7 test mesajı eğitim kümesinde birebir
  (normalize) bulunur ve etiketleri aynıdır. Bu, `docs/EXPERIMENTS.md` kuralı gereği
  gizlenmez; NB çalıştırıcısı bu satırları `overlaps_training` ile işaretler.
- Noktalama/tire/para birimi farkıyla ayrışan **yakın tekrarlar** mevcut anahtarla
  yakalanmaz. Train/validation arasındaki 4 örnek (hepsi aynı etiketle):
  `"Why is there a $1 charge on my statement?"` ↔ `"Why is there a £1 charge on my statement?"`,
  `"Why hasn't my top up gone through?"` ↔ `"... top-up ..."`,
  `"The disposable cards, what are they for?"` ↔ `"... cards - what are they for?"`,
  `"My top-up is still pending"` ↔ `"My top up is still pending"`.
  Bu 4 satır validation'ın %0,27'sidir; skorları iyimser yapabilir.
  Yakın tekrarların model karşılaştırmalarına (A/B) etkisi ölçülmemiştir;
  aynı veriyi kullanmak tüm modellerin aynı ölçüde etkileneceğini garanti etmez.

## 2. Mevcut ön işleme hattı (`src/banking77/data.py`)

1. **Kaynak:** PolyAI `task-specific-datasets` deposundan, `configs/data.json` içinde sabitlenmiş
   40 karakterlik commit'ten indirilir; SHA-256 değerleri doğrulanır. `--offline` modunda
   yerel ham dosyaların hash'i doğrulanır, ağ gerekmez.
2. **Okuma:** CSV `utf-8-sig` ile okunur; `text` ve `category` sütunları zorunludur.
   77 benzersiz kategori beklenir.
3. **Eğitim temizliği (`clean_training_records`):** bilinmeyen etiket → hata; boş metin →
   atılır; `normalize_text` (NFKC, casefold, boşlukları tek boşluğa indirme) anahtarıyla
   tekrarlar → ilk görülen kalır; aynı anahtarda farklı etiket → hata. Satır kimliği
   `train-<ham satır indeksi>`. Model **özgün** metni alır, normalize metin yalnızca tekrar
   denetiminde kullanılır.
4. **Bölme (`split_training_records`):** temizlenmiş eğitimden stratified %15 validation,
   `seed=42`; sonrasında normalize anahtarla ayrıklık kontrol edilir.
5. **Test:** resmî test satırları değiştirilmeden alınır (`test-<indeks>`); yalnızca etiketin
   geçerli ve metnin boş olmadığı doğrulanır.
6. **Çıktı:** `train.csv`, `validation.csv`, `test.csv` (`id,text,category`), `categories.json`,
   `summary.json` (sayılar, sınıf dağılımları, örtüşmeler, ham dosya hash'leri).
7. **Özellik çıkarımı (model tarafı, `naive_bayes.py`):** `TfidfVectorizer(sublinear_tf=True)`;
   küçük harfe çevirme, `\b\w\w+\b` token deseni, stop-word listesi yok. `Pipeline` içinde
   yalnızca eğitim verisine `fit` edilir; validation/test yalnızca `transform` görür
   (`tests/test_protocol.py` bunu doğrular).

## 3. Saptanan sorunlar ve alınan kararlar

| # | Bulgu | Karar | Gerekçe |
| --- | --- | --- | --- |
| 1 | `write_records` Python `csv` varsayılanı olan CRLF ile yazıyordu; `.gitattributes` ise `data/processed/*.csv` için LF istiyor. `--offline` yeniden üretimi Windows'ta CSV'leri "değişmiş" gösteriyor ve `dataset_files_sha256` değerleri platforma göre farklılaşabiliyordu. | **Düzeltildi:** `lineterminator="\n"`. Test eklendi (`tests/test_data_files.py`). | Yeniden üretim artık commit'teki dosyalarla **bayt bayt aynı** (SHA-256 doğrulandı); içerik, `summary.json` ve veri kimliği değişmedi. Model arayüzü etkilenmez. |
| 2 | Noktalama/tire/para birimi farklı yakın tekrarlar (train/val 4, train/test 21, val/test 4). | **Değiştirilmedi**, raporlandı. | Anahtarı genişletmek satır sayılarını ve bölmeyi (dolayısıyla veri kimliğini) değiştirir; handoff sözleşmesi (`seed=42`, sabit bölme) ve diğer üç modelin sonuçlarını geçersiz kılar. Test için bu zaten yapılamaz (resmî test sabit). Yakın tekrarlar validation'ın %0,27'sidir; skorlara etkisi ölçülmedi. Gerekirse ekip kararıyla tek seferde yapılmalıdır. |
| 3 | Baş/son boşluk ve satır sonu içeren metinler (train 6, test 3); bir tekrar çiftinde satır sonlu varyant tutulmuş. | **Değiştirilmedi.** | `TfidfVectorizer` boşlukları token ayırıcı sayar: baş/son/iç boşlukları normalize etmek unigram (2.176) ve unigram+bigram (21.595) sözlüğünü **değiştirmez** (doğrulandı, sözlükler birebir aynı). Test metni değiştirilemez. Gereksiz değişiklik veri kimliğini bozardı. |
| 4 | Varsayılan token deseni tek karakterli token'ları ve `£ € $ ? !` gibi simgeleri atar; `$1` ile `£1` aynı hale gelir. | **Değiştirilmedi**, öneri olarak not edildi. | Vectorizer model dosyasında (NB sahibi) durur ve bu görev kapsamı dışındadır. Ücret/para birimi sınıfları için denemeye değer bir özellik fikridir; ölçülmedi, etkisi bilinmiyor. |
| 5 | Test dengeli, train/validation dengesiz; validation'da sınıf başına 5–28 örnek. | Bilgi olarak raporlandı. | Macro F1 yorumlanırken sınıf başına az sayıdaki validation örneğinin yarattığı belirsizlik dikkate alınmalı; Macro F1 için bir belirsizlik aralığı hesaplanmadı. |
| 6 | Resmî test: 7 birebir train örtüşmesi, 1 iç tekrar. | Değiştirilmedi. | Test sabit tutulur ve `docs/EXPERIMENTS.md` gereği örtüşme raporlanır. |

Sızıntı güvencesi: TF-IDF yalnızca eğitimde öğrenilir; model/özellik seçiminde resmî test
**hiç kullanılmamıştır**.

### Durum özeti

| Durum | Maddeler |
| --- | --- |
| **Saptanan** | Ham train'de 4 normalize tekrar; train/test'te 7 birebir ve 21 noktalamasız ortak metin; train/validation'da 4 ve validation/test'te 4 noktalamasız yakın tekrar; testte 1 iç tekrar; baş/son boşluklu metinler; sınıf dengesizliği; validation'da küçük sınıflar; `data.py`'de CRLF/LF tutarsızlığı |
| **Gerçekten düzeltilen** | (a) Ham train'deki 4 tekrar — bu temizliği bu çalışma değil, **mevcut ortak başlangıç kodu** yapmıştı; (b) CSV satır sonu (LF) — bu çalışmada yapılan **tek** kod değişikliği |
| **Bilerek değiştirilmeyen** | Yakın tekrarlar (veri kimliğini ve bölmeyi bozar), resmî test (sabit), baş/son boşluklar (özelliklere etkisi yok), token deseni/simge kaybı (model tarafı), sınıf dengesizliği (veri gerçeği) |

Bu çalışmada veri satırları, bölme, `summary.json` ve model arayüzleri **değişmemiştir**.

## 4. Değerlendirme metriklerinin gerekçesi

Projede **accuracy** ve **macro F1** birlikte raporlanır; model seçimi macro F1 ile yapılır
(`docs/EXPERIMENTS.md`).

- **Accuracy**, doğru tahmin edilen mesajların oranıdır; yorumlaması kolaydır ve literatürle
  karşılaştırmada yaygındır. Ancak her mesaja eşit ağırlık verdiği için büyük sınıflar
  sonucu daha çok belirler. Eğitim/validation dağılımı dengesizdir (sınıf başına 30–159 /
  5–28 örnek; en büyük sınıf validation'ın yalnızca %1,87'si, 28/1.500). Dengesizlik orta
  düzeydedir (≈5,6 kat), bu yüzden accuracy yanıltıcı derecede şişmez, fakat küçük
  sınıflardaki zayıflığı örtebilir.
- **Macro F1**, 77 sınıfın F1 değerinin ağırlıksız ortalamasıdır; her niyet, örnek sayısından
  bağımsız olarak eşit sayılır. Bir bankacılık sisteminde nadir bir talebin (örn.
  `contactless_not_working`, train 30 örnek) sık bir talep kadar önemli olabileceği ve
  precision ile recall'un birlikte ölçülmesi gerektiği için model seçimine uygun metriktir.
- Resmî test dengeli (her sınıf 40) olduğundan test accuracy'si makro ortalama recall'a
  eşittir; validation dengesiz olduğundan iki metrik validation'da ayrışır. Bu yüzden ikisini
  birlikte okumak gerekir.
- **Sınırlama:** validation'da bazı sınıflarda yalnızca 5–9 örnek vardır
  (`contactless_not_working` 5, `virtual_card_not_working` 6, `card_acceptance` 9). Bu
  sınıflarda tek bir hata sınıf recall'unu 11–20 puan oynatır; macro F1 bu yüzden
  validation bölmesine ve tohuma duyarlıdır. Küçük macro F1 farkları (örn. 0,007) tek başına
  güçlü kanıt sayılmamalı, yalnızca tek bölme/tek tohum kanıtı olarak okunmalıdır.

## 5. Kapsam, yenilik sınırı ve lisans

- **Yenilik iddiası yoktur.** BANKING77 daha önce yayımlanmış herkese açık bir
  karşılaştırma veri setidir. Bu çalışmanın katkısı veriyi doğrulamak, temizliği ve
  sızıntı risklerini ölçmek, ön işlemeyi belgelemek ve özellik karşılaştırmasıdır.
  TF-IDF unigram/bigram + Naive Bayes **baseline** yöntemlerdir; yeni bir yöntem
  önerilmemiştir.
- **Ham veri:** `data/raw/` altında değiştirilmeden, sabit commit ve SHA-256 değerleriyle durur.
  **Hazırlanmış veri:** `data/processed/` ham veriden `data.py` ile üretilir; yapılan
  değişiklikler (4 tekrarın çıkarılması, stratified validation ayrımı, `id` sütunu) bu
  belgede ve README'de açıklanmıştır.
- **Lisans/atıf:** veri CC BY 4.0 (`data/raw/LICENSE`), kod MIT (`LICENSE`); atıf README'dedir.
  Bu çalışmada lisans dosyaları veya README değiştirilmemiştir.

## 6. Yeniden üretim

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m banking77.data --offline
.\.venv\Scripts\python.exe -m banking77.analyze_data
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`banking77.analyze_data` yalnızca okur; hiçbir dosyayı değiştirmez. Özellik deneyi için
bkz. [results/FEATURE_EXPERIMENTS.md](../results/FEATURE_EXPERIMENTS.md).
