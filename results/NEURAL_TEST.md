# Sabitlenen yeni modellerin nihai test sonuçları

8.499 train ile öğrenilmiş checkpoint'ler kullanıldı; validation eğitime eklenmedi.
Ayarlar ve ağırlıklar ayrı commit'te sabitlendikten sonra 3.080 resmî test mesajında
değerlendirildi. Test sonucu yeni ayar seçmek için kullanılmadı.

| Yöntem | Accuracy (%) | Macro F1 | Örtüşmeyen 3.073: accuracy (%) | Örtüşmeyen: macro F1 |
| --- | ---: | ---: | ---: | ---: |
| RNN | 68.99 | 0.6852 | 68.99 | 0.6849 |
| CNN | 89.16 | 0.8917 | 89.13 | 0.8914 |
| LSTM | 82.18 | 0.8219 | 82.13 | 0.8214 |
| BiLSTM | 83.15 | 0.8314 | 83.11 | 0.8310 |
| GloVe 100d + ortalama + doğrusal başlık | 71.01 | 0.7093 | 70.97 | 0.7090 |
| Word2Vec CBOW + ortalama + LR | 75.45 | 0.7524 | 75.43 | 0.7521 |
| Word2Vec Skip-gram + ortalama + LR | 79.55 | 0.7926 | 79.50 | 0.7921 |
| FastText + ortalama + LR | 72.53 | 0.7233 | 72.47 | 0.7227 |
| NB + CNN soft voting (ekip katkısı) | 90.65 | 0.9061 | 90.63 | 0.9059 |

Tam metrikler, süreler, protokol kimlikleri ve çıktı hash'leri [neural_test.json](neural_test.json)
içindedir. Her modelin dört standart çıktısı `neural_test/<yöntem>/` altında bulunur.
Tahminler resmî kimlik/metin/etiketlerle, accuracy/macro F1 ve confusion matrix yeniden hesaplanarak doğrulandı.

## Tekrarlama

[NEURAL_BASELINES.md](../docs/NEURAL_BASELINES.md) kurulumunu ve validation komutlarını uygulayın.
Checkpoint'ler Git'e eklenmez. Başka makinede yeniden eğitim bit düzeyinde aynı ağırlığı
garanti etmez; yeni checkpoint için yeni protokol oluşturup testten önce commit edin.
Mevcut protokolle farklı ağırlıklar sessizce kullanılmaz. Bu bilgisayardaki dondurulmuş
checkpoint'lerin test komutları:

```powershell
python -m banking77.train_ensemble test --protocol results/nb_cnn_final_protocol.json --device cpu
python -m banking77.train_neural test --protocol results/rnn_final_protocol.json --device cpu
python -m banking77.train_neural test --protocol results/lstm_final_protocol.json --device cpu
python -m banking77.train_neural test --protocol results/bilstm_final_protocol.json --device cpu
python -m banking77.train_neural test --protocol results/glove_final_protocol.json --device cpu
python -m banking77.train_embeddings test --protocol results/word2vec_cbow_final_protocol.json
python -m banking77.train_embeddings test --protocol results/word2vec_skipgram_final_protocol.json
python -m banking77.train_embeddings test --protocol results/fasttext_final_protocol.json
```

Ensemble test komutu CNN'i bir kez değerlendirir ve onun çıktısını da üretir.

## Gerçek hata örnekleri

Her yöntem için dosya sırasındaki ilk üç hata; örnekler nedensel model açıklaması değildir.

| Yöntem | Mesaj | Gerçek etiket | Tahmin |
| --- | --- | --- | --- |
| RNN | How do I locate my card? | card_arrival | card_delivery_estimate |
| RNN | My card has not arrived yet. | card_arrival | request_refund |
| RNN | Do you know if there is a tracking number for the new card you sent me? | card_arrival | card_about_to_expire |
| CNN | How do I locate my card? | card_arrival | card_delivery_estimate |
| CNN | Is there a way to know when my card will arrive? | card_arrival | card_delivery_estimate |
| CNN | When will I get my card? | card_arrival | card_delivery_estimate |
| LSTM | How do I locate my card? | card_arrival | order_physical_card |
| LSTM | When will I get my card? | card_arrival | card_delivery_estimate |
| LSTM | still waiting on that card | card_arrival | pending_card_payment |
| BiLSTM | still waiting on that card | card_arrival | pending_card_payment |
| BiLSTM | How long does a card delivery take? | card_arrival | card_delivery_estimate |
| BiLSTM | I am still waiting for my card after 1 week.  Is this ok? | card_arrival | pending_card_payment |
| GloVe 100d + ortalama + doğrusal başlık | How do I locate my card? | card_arrival | activate_my_card |
| GloVe 100d + ortalama + doğrusal başlık | I still have not received my new card, I ordered over a week ago. | card_arrival | transaction_charged_twice |
| GloVe 100d + ortalama + doğrusal başlık | Is there a way to know when my card will arrive? | card_arrival | card_delivery_estimate |
| Word2Vec CBOW + ortalama + LR | How do I locate my card? | card_arrival | activate_my_card |
| Word2Vec CBOW + ortalama + LR | I ordered a card but it has not arrived. Help please! | card_arrival | reverted_card_payment? |
| Word2Vec CBOW + ortalama + LR | Is there a way to know when my card will arrive? | card_arrival | card_about_to_expire |
| Word2Vec Skip-gram + ortalama + LR | How do I locate my card? | card_arrival | order_physical_card |
| Word2Vec Skip-gram + ortalama + LR | I ordered a card but it has not arrived. Help please! | card_arrival | reverted_card_payment? |
| Word2Vec Skip-gram + ortalama + LR | When will I get my card? | card_arrival | card_delivery_estimate |
| FastText + ortalama + LR | How do I locate my card? | card_arrival | activate_my_card |
| FastText + ortalama + LR | I ordered a card but it has not arrived. Help please! | card_arrival | reverted_card_payment? |
| FastText + ortalama + LR | Is there a way to know when my card will arrive? | card_arrival | order_physical_card |
| NB + CNN soft voting (ekip katkısı) | How do I locate my card? | card_arrival | get_physical_card |
| NB + CNN soft voting (ekip katkısı) | When will I get my card? | card_arrival | card_delivery_estimate |
| NB + CNN soft voting (ekip katkısı) | How long does a card delivery take? | card_arrival | card_delivery_estimate |

## Sınırlılıklar

- Yedi normalize birebir train örtüşmesi için ayrı sonuç verildi. Yakın tekrarların tamamen giderildiği iddia edilmez.
- Tek seed ve sabit bütçe kullanıldı. Bu rapor eşleştirilmiş anlamlılık testi veya kapsamlı tuning içermez.
- Klasik test sonuçları yeni kapsamdan önce görülmüştür; tüm araştırmanın tamamen kör olduğu iddia edilmez.
- Dört Transformer'ın tam deneyleri halen bekliyor; CPU baseline'larının bitmesi tüm projeyi tamamlamaz.
- Ortak karşılaştırma, eşleştirilmiş istatistikler, grafikler ve PPTX 5. kişinin görevidir.
