# Unigram ve unigram+bigram özellik deneyi

Tüm değerler `banking77.train_naive_bayes` çalıştırıcısının ürettiği `metrics.json`
dosyalarından alınmıştır (yerel `results/runs/` klasörleri Git'e eklenmez), dosyalarla
yeniden karşılaştırılmış ve sklearn ile bağımsız olarak yeniden hesaplanıp aynı çıkmıştır.
Değerlendirme yalnızca **validation** üzerinde yapılmıştır; resmî test bu çalışmada hiç
çalıştırılmadı. Metriklerin gerekçesi için bkz. [docs/DATA.md](../docs/DATA.md) (bölüm 4).

## Protokol

- Veri: `data/processed/train.csv` (8.499) ile eğitim, `validation.csv` (1.500) ile değerlendirme.
- Veri özeti SHA-256: `468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`
  (tüm çalıştırmalarda aynı; `train.csv` `1e627600…a775`, `validation.csv` `3d7ea088…1025`).
- TF-IDF yalnızca eğitim verisine `fit` edilir (`Pipeline`); `sublinear_tf=True`, diğer
  vectorizer parametreleri varsayılan. Sınıflandırıcı: `MultinomialNB`. `seed=42`.
- Ortam: Python 3.12.10, scikit-learn 1.9.1, numpy 2.5.3.
- Deney A ve B arasında **yalnızca** `ngram_range` değişir. Model kodu değiştirilmedi.

## Ana deney (alpha = 1.0)

| Deney | TF-IDF `ngram_range` | Alpha | Accuracy | Macro F1 | Özellik sayısı | Run kimliği |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A | (1,1) | 1,0 | **82,40 %** | **0,7942** | 2.176 | `naive_bayes_validation_20261008T214115144244Z` |
| B | (1,2) | 1,0 | 81,60 % | 0,7872 | 21.595 | `naive_bayes_validation_20261008T214116150430Z` |

Fark (A − B): accuracy +0,80 puan (1.500 mesajda 12 mesaj), macro F1 +0,0070.

Komutlar (repo kökünden, PowerShell):

```powershell
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes --split validation --ngram-max 1 --alpha 1.0
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes --split validation --ngram-max 2 --alpha 1.0
```

Her çalıştırma yeni bir `run_id` üretir; sınıflandırma skorları sabit veri ve ortamla
tekrarlanır, süreler ve run kimlikleri değişir.

## Özellik boyutu ve maliyet (yalnızca eğitim verisine fit)

| | (1,1) | (1,2) | Oran |
| --- | ---: | ---: | ---: |
| Sözlük (özellik) sayısı | 2.176 | 21.595 | 9,9× |
| Eğitim mesajı başına sıfırdan farklı özellik | 10,3 | 20,1 | 2,0× |
| Validation sözlüğünün eğitim sözlüğünde bulunma oranı | %87,4 | %72,7 | — |
| Eğitim süresi, tek ölçüm (α=1) | 0,061 s | 0,129 s | ~2,1× |
| Tahmin süresi, 1.500 mesaj, tek ölçüm (α=1) | 0,0087 s | 0,0185 s | ~2,1× |

Süreler bu bilgisayardaki tek ölçümdür; genel bir hız iddiası değildir. Bigram'lar
sözlüğün %89,9'unu (19.419 özellik) oluşturur. Özellik sayıları ve oranlar, çalıştırıcıdan
bağımsız, eğitim verisine fit edilen `TfidfVectorizer` ile tek seferlik betikle hesaplandı.

## Belirsizlik (alpha = 1.0)

1.500 örnekte accuracy'nin %95 binom aralığı yaklaşık ±1,9 puandır. Aynı örnekler üzerinde
eşleştirilmiş karşılaştırmada yalnızca A doğru 64, yalnızca B doğru 52 mesaj vardır
(161 mesajda tahmin farklı); iki yönlü tam McNemar p = 0,31 (`math.comb` ile hesaplanan
binom p; tek seferlik betik, repoya eklenmedi). Yani A−B farkı bu bölmede istatistiksel
olarak ayırt edilemez. Çalışma tek validation bölmesi ve tek tohumla yapılmıştır.

## Öneri: bu kontrollü deney için tercih edilen özellik ayarı

**Alpha=1,0'lı kontrollü karşılaştırmada tercih edilen ayar: unigram, TF-IDF `ngram_range=(1,1)`
(`sublinear_tf=True`).**

Gerekçe:

- Aynı veri, aynı alpha ve aynı vectorizer ayarlarıyla unigram hem accuracy'de (82,40 % ↔ 81,60 %)
  hem macro F1'de (0,7942 ↔ 0,7872) öndedir.
- Çok daha az özellik kullanır (2.176 ↔ 21.595; yaklaşık 10 kat az), eğitim/tahmin yaklaşık 2 kat
  daha hızlıdır ve doğrulama sözlüğünün daha büyük bölümü eğitimde görülmüştür (%87,4 ↔ %72,7);
  bigram'lar seyrek olduğu için daha az görülür.

Sınırlamalar ve dikkat edilmesi gerekenler:

- **Fark küçüktür** (0,80 puan; p = 0,31) ve tek bölme/tek tohum sonucudur; bir "bigram işe
  yaramaz" hükmü değildir.
- **Bigram, farklı model hiperparametreleriyle daha iyi sonuç verebilir.** Yalnızca alpha=1,0
  denendi; sonuç Naive Bayes'in bu düzgünleştirme değerine bağlıdır. Aşağıdaki ek gözlem bunun
  gerçekten olabileceğini gösteriyor.
- Bu bulgu yalnızca Naive Bayes içindir. Logistic Regression ve Linear SVM için sıralama
  farklı olabilir; bunları kendi sahipleri kendi C değerleriyle değerlendirmelidir.
- Yalnızca `ngram_range` karşılaştırıldı; `min_df`, `max_df`, stop-word, `sublinear_tf`
  ve simge/sayı özellikleri denenmedi.
- **Nihai ortak özellik ayarı ekip tarafından birlikte kararlaştırılmalıdır.** Bu öneri,
  diğer modellerin veya `build_naive_bayes` varsayılanının (şu an `ngram_max=2`) yapılandırmasını
  **otomatik olarak değiştirmez**; hiçbir ortak yapılandırma dosyası değiştirilmedi.

## Ek duyarlılık gözlemi (alpha = 0.05) — öneriye temel değildir

Hiperparametre ayarı Üye 1'in (Naive Bayes) sorumluluğundadır; bu bölüm bir ayar çalışması
değil, ana deneydeki sıralamanın alpha'ya duyarlı olup olmadığını göstermek için yapılmış
**tek seferlik bir ek gözlemdir**. Başka alpha değeri denenmedi ve hata analizi yapılmadı.
Alpha=0,05, Üye 1'in `results/NAIVE_BAYES_ALPHA.md` içindeki validation sonucundan alınmıştır;
bu bölüm yukarıdaki öneriyi belirlemek için kullanılmamıştır.

| Deney | `ngram_range` | Alpha | Accuracy | Macro F1 | Run kimliği |
| --- | --- | ---: | ---: | ---: | --- |
| A | (1,1) | 0,05 | 84,87 % | 0,8422 | `naive_bayes_validation_20261008T214145640456Z` |
| B | (1,2) | 0,05 | 86,20 % | 0,8529 | `naive_bayes_validation_20261008T214146651166Z` |

```powershell
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes --split validation --ngram-max 1 --alpha 0.05
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes --split validation --ngram-max 2 --alpha 0.05
```

Bu alpha'da sıralama tersine dönmüş (bigram +1,33 puan, macro F1 +0,0107; McNemar p = 0,08,
%5 düzeyinde anlamlı değil). B'nin değerleri `results/NAIVE_BAYES_ALPHA.md`'deki
alpha=0.05 (86,20 %, 0,8529) ve alpha=1 (81,60 %, 0,7872) satırlarıyla aynıdır; A'nın
alpha=0.05 değeri yeni bir ölçümdür. Sonuç: "hangi n-gram daha iyi" sorusunun yanıtı alpha'ya
bağlıdır, bu yüzden ortak ayar kararı Üye 1'in ayar sonuçları ve diğer modellerin sonuçlarıyla
birlikte verilmelidir.
