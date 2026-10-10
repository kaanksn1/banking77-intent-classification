# Transformer baseline nihai test sonuçları

Ayar ve checkpoint'ler [testten önceki commit](https://github.com/kaanksn1/banking77-intent-classification/commit/1e4ff66d49e7d184d3dbe1548fa7b038b79833c2) ile sabitlendi.
8.499 train ile öğrenilmiş checkpoint'ler yeniden eğitim yapılmadan 3.080 resmî test mesajında değerlendirildi.
Validation eğitime eklenmedi; test sonucuyla ayar veya model seçilmedi.

| Belirli model | Accuracy (%) | Macro F1 | Örtüşmeyen 3.073: accuracy (%) | Örtüşmeyen: macro F1 |
| --- | ---: | ---: | ---: | ---: |
| BERT-base-uncased | 90.58 | 0.9016 | 90.56 | 0.9013 |
| DistilBERT-base-uncased | 89.61 | 0.8961 | 89.59 | 0.8957 |
| RoBERTa-base | 93.02 | 0.9301 | 93.00 | 0.9300 |
| ALBERT-base-v2 | 90.29 | 0.9027 | 90.27 | 0.9024 |

[transformer_test.json](transformer_test.json) metrik/süre özeti ve çıktı hash'lerini içerir.
Her modelin dört standart çıktısı `transformer_test/<model>/` altındadır. Tahminler resmî kimlik/metin/etiketlerle
eşleştirildi; accuracy, macro F1 ve confusion matrix dosyalardan bağımsız olarak yeniden hesaplanıp doğrulandı.

## Gerçek hata örnekleri

Her modelin dosya sırasındaki ilk üç hatası; bu örnekler nedensel açıklama değildir.

| Model | Mesaj | Gerçek etiket | Tahmin |
| --- | --- | --- | --- |
| BERT-base-uncased | Is there a way to know when my card will arrive? | card_arrival | card_delivery_estimate |
| BERT-base-uncased | When will I get my card? | card_arrival | card_delivery_estimate |
| BERT-base-uncased | How long does a card delivery take? | card_arrival | card_delivery_estimate |
| DistilBERT-base-uncased | How do I locate my card? | card_arrival | card_linking |
| DistilBERT-base-uncased | When will I get my card? | card_arrival | card_delivery_estimate |
| DistilBERT-base-uncased | How long does a card delivery take? | card_arrival | card_delivery_estimate |
| RoBERTa-base | How do I locate my card? | card_arrival | lost_or_stolen_card |
| RoBERTa-base | When will I get my card? | card_arrival | card_delivery_estimate |
| RoBERTa-base | How long does a card delivery take? | card_arrival | card_delivery_estimate |
| ALBERT-base-v2 | Is there a way to know when my card will arrive? | card_arrival | card_delivery_estimate |
| ALBERT-base-v2 | When will I get my card? | card_arrival | card_delivery_estimate |
| ALBERT-base-v2 | How long does a card delivery take? | card_arrival | card_delivery_estimate |

## Tekrarlama ve sınırlılıklar

[Validation raporu](TRANSFORMER_VALIDATION.md) ve [GPU kurulum protokolü](../docs/NEURAL_BASELINES.md) uygulanır.
Ağırlıklar Git dışında tutulur; yeniden eğitim bit düzeyinde aynı checkpoint'i garanti etmez. Başka ağırlık için yeni
protokol oluşturup testten önce commit etmek gerekir; kayıtlı protokol altında farklı ağırlıklar sessizce kullanılmaz.
Mevcut checkpoint'ler için GPU venv'inde, repo kökünde:

```bash
export HSA_ENABLE_DXG_DETECTION=1
python -m banking77.train_neural test --protocol results/bert_final_protocol.json --device cuda
python -m banking77.train_neural test --protocol results/distilbert_final_protocol.json --device cuda
python -m banking77.train_neural test --protocol results/roberta_final_protocol.json --device cuda
python -m banking77.train_neural test --protocol results/albert_final_protocol.json --device cuda
```

Yedi normalize birebir train örtüşmesi için ayrı sonuç verilir; yakın tekrarların tamamen temizlendiği iddia edilmez.
Tek seed ve ortak beş epoch bütçesi kullanıldı. Eşleştirilmiş anlamlılık testi veya çoklu seed araştırması bu raporda yoktur.
[Validation bütçe kararı](transformer_budget_decision.json), yalnız DistilBERT ve BERT'in ilk üç epoch validation
eğrilerine dayanarak resmî Transformer testlerinden önce alındı. Dört model için beş epoch ortak tutuldu;
model başına üç/beş epoch sonuçları arasında seçim yapılmadı. Bu bütçe yakınsama veya optimum tuning kanıtı değildir.
Önceki test sonuçları bu kapsamdan önce görüldü; bütün araştırmanın kör olduğu iddia edilmez.
Bunlar ders baseline'larıdır. NB + CNN katkısının tüm Transformer'ları geçtiği iddia edilmez.
Ortak genişletilmiş karşılaştırma/grafikler ve PPTX 5. kişinin sorumluluğundadır.
