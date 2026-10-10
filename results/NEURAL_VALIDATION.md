# Yeni baseline ve NB + CNN validation sonuçları

Yalnızca **8.499 train / 1.500 validation** kullanıldı; bu tabloda test skoru yoktur.
Tüm modeller seed=42 ile bir kez çalıştırıldı. Ana seçim metriği 77 sınıfa eşit ağırlık veren macro F1'dır.
Veri ve özellik sahiplerinin mevcut çalışmaları değişmedi.

| Yöntem | Accuracy (%) | Macro F1 | Seçilen epoch / bütçe | Fit (sn) | Tahmin (sn) |
| --- | ---: | ---: | --- | ---: | ---: |
| RNN | 67.67 | 0.6745 | 15 / 15 | 17.62 | 0.050 |
| CNN | 88.13 | 0.8816 | 10 / 15 | 43.59 | 0.191 |
| LSTM | 84.00 | 0.8379 | 14 / 15 | 38.53 | 0.094 |
| BiLSTM | 84.33 | 0.8409 | 15 / 15 | 74.77 | 0.173 |
| GloVe 100d + ortalama + doğrusal başlık | 72.73 | 0.7261 | 16 / 50 | 3.62 | 0.016 |
| Word2Vec CBOW + ortalama + LR | 74.13 | 0.7337 | sabit 20 | 1.75 | 0.028 |
| Word2Vec Skip-gram + ortalama + LR | 79.13 | 0.7817 | sabit 20 | 3.25 | 0.026 |
| FastText + ortalama + LR | 71.53 | 0.7092 | sabit 20 | 4.22 | 0.029 |
| NB + CNN soft voting (ekip katkısı) | 90.07 | 0.8997 | bileşenler sabit | 43.72 | 0.213 |

Tam ayarlar, epoch geçmişleri, veri/kod hash'leri, süreler ve çalıştırma kimlikleri
[neural_validation.json](neural_validation.json) içinde. Her seçilmiş çalışma için
`<yöntem>_final_protocol.json` testten önce commit edilir.

## NB + CNN katkısı ve ablation

`p = (1-w) p_NB + w p_CNN`. NB alpha=0.05 ve unigram + bigram ayarını korur;
CNN checkpoint'i validation'da seçilir. Aşağıdaki 11 ağırlık yalnız validation
macro F1 ile karşılaştırıldı; eşitlikte ilk aday korunur.

| CNN ağırlığı | NB ağırlığı | Accuracy (%) | Macro F1 |
| ---: | ---: | ---: | ---: |
| 0.0 | 1.0 | 86.20 | 0.8529 |
| 0.1 | 0.9 | 87.33 | 0.8689 |
| 0.2 | 0.8 | 88.40 | 0.8795 |
| 0.3 | 0.7 | 89.67 | 0.8959 |
| 0.4 | 0.6 | 90.07 | 0.8997 |
| 0.5 | 0.5 | 89.87 | 0.8985 |
| 0.6 | 0.4 | 89.80 | 0.8978 |
| 0.7 | 0.3 | 89.33 | 0.8931 |
| 0.8 | 0.2 | 89.00 | 0.8898 |
| 0.9 | 0.1 | 88.53 | 0.8855 |
| 1.0 | 0.0 | 88.13 | 0.8816 |

Seçim: **NB=0.6, CNN=0.4**. Yalnız NB macro F1 0.8529, yalnız CNN 0.8816,
birleştirme 0.8997. Bu validation iyileşmesi test başarısı veya istatistiksel
üstünlük kanıtı değildir. Olasılıklar ayrıca kalibre edilmedi; ağırlık bunların
ölçeğine de bağlıdır. Bu proje katkısı yeni bir araştırma algoritması iddiası taşımaz.

## Tekrarlama

Önce [kurulum ve GloVe hazırlığı](../docs/NEURAL_BASELINES.md) tamamlanır.

```powershell
python -m banking77.train_neural train --model rnn --epochs 15 --device cpu
python -m banking77.train_neural train --model cnn --epochs 15 --device cpu
python -m banking77.train_neural train --model lstm --epochs 15 --device cpu
python -m banking77.train_neural train --model bilstm --epochs 15 --device cpu
python -m banking77.train_neural train --model mean --embedding-file data/embeddings/glove.6B.100d.train.txt --freeze-embeddings --epochs 50 --patience 5 --learning-rate 0.01 --device cpu
python -m banking77.train_embeddings train --model word2vec_cbow --epochs 20
python -m banking77.train_embeddings train --model word2vec_skipgram --epochs 20
python -m banking77.train_embeddings train --model fasttext --epochs 20
python -m banking77.train_ensemble train --cnn-validation-run cnn_validation_20261010T145345688617Z
```

## Sınırlar ve bekleyen işler

- Bu deneyler Windows CPU'da çalıştı. Fit süreleri neural modellerde epoch validation ve
  checkpoint kaydını içerir; indirme/ilk tokenizasyonu içermez. Word2Vec/FastText fit
  süresi embedding ve LR başlığını içerir. Ensemble süreleri bileşenlerin toplamıdır;
  CNN yeniden eğitilmedi. Klasik tabloyla sürelerin kapsamı birebir aynı değildir.
- GloVe 100d dış ön eğitimlidir ve donduruldu; 2.216 gerçek train kelimesinden 2.154'ü
  eşleşti. Kaynak arşiv ve dosya hash'leri `configs/glove.json` içindedir.
  Word2Vec/FastText sadece bu projenin train mesajlarından öğrenildi; temsil protokolleri eşdeğer değildir.
- RNN/CNN/LSTM/BiLSTM aynı 100d öğrenilen embedding, hidden=128, dropout=0.3,
  max_length=64 ve 15 epoch/patience=3 bütçesini kullanır. Ayrıntılı ayarlar JSON'dadır.
- Sabit bütçe ve tek seed mimariler arasında kapsamlı tuning veya varyans tahmini değildir.
  Checkpoint ve ağırlık aynı validation'da seçildiği için bu metrikler iyimser olabilir.
- Slayt 37'deki BERT-base, DistilBERT, RoBERTa-base, ALBERT-base-v2'nin **tam eğitimleri bekliyor**.
  Kısa DistilBERT geliştirme kontrolü bu tabloya dahil edilmedi. GPU preflight henüz geçmedi.
- Klasik test skorları kapsam genişlemesinden önce görüldü; tüm araştırma tamamen kör diye sunulmaz.
  Yeni testler protokoller commit edildikten sonra sabit checkpoint'lerle çalıştırılır.
- Genişletilmiş ortak kıyas, eşleştirilmiş istatistikler, grafikler ve PPTX 5. kişinin görevidir.
