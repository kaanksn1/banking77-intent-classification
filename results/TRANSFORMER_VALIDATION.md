# Transformer baseline validation deneyleri

Ders slaytındaki dört belirli model, **8.499 train / 1.500 validation** ayrımında ince ayarlandı.
Bu tabloda resmî test sonucu yoktur. Sabit kaynaklar `configs/neural_models.json` içindedir.
Mimari ailesi (BERT gibi) ile kullanılan belirli ön eğitimli model birbirinden ayrılır.

| Belirli model | Accuracy (%) | Macro F1 | Seçilen epoch / bütçe | Fit (sn) | Tahmin (sn) |
| --- | ---: | ---: | --- | ---: | ---: |
| BERT-base-uncased | 90.60 | 0.9010 | 5 / 5 | 471.98 | 7.946 |
| DistilBERT-base-uncased | 89.87 | 0.8961 | 5 / 5 | 250.19 | 3.904 |
| RoBERTa-base | 92.33 | 0.9264 | 5 / 5 | 484.56 | 7.753 |
| ALBERT-base-v2 | 89.87 | 0.8989 | 5 / 5 | 410.43 | 7.885 |

## Protokol ve kaynaklar

- Tüm encoder parametreleri ve 77 sınıflı yeni başlık train üzerinde öğrenilir. Validation eğitime eklenmez.
- Ortak bütçe: FP32, seed=42, batch=16, max_length=64, AdamW lr=2e-5, weight_decay=0.01,
  gradient_clip=1, 5 epoch, patience=3, doğrusal %10 warmup/decay. En iyi checkpoint validation macro F1 ile seçilir.
- [Bütçe kararı](transformer_budget_decision.json): başlangıçtaki üç epoch bütçesiyle yalnız DistilBERT ve BERT'in
  tam validation deneyleri tamamlandı. Bu iki validation eğrisine dayanarak, dört Transformer'ın resmî testinden
  önce nihai ortak bütçe beş epoch olarak belirlendi. Dört model de ön eğitimli ağırlıklardan yeniden başlatıldı;
  model başına üç/beş epoch sonuçları arasında seçim yapılmadı. İlk iki üç epoch deneyi karar kaydında korunur.
- Farklı ön eğitim/tokenizer ve model boyutları korunur. Aynı epoch sayısı aynı hesaplama maliyeti veya optimum tuning demek değildir.
- Tam ayarlar, epoch kayıpları/skorları, kategori listesi, kaynak/veri/ağırlık/tokenizer hash'leri her modelin
  `<model>_final_protocol.json` dosyasındadır. Protokoller resmî testten önce ayrı commit'e alınır.
- [transformer_validation.json](transformer_validation.json) metrik/süre özeti, komutlar, veri kimliği, ortam ve çıktı hash'lerini içerir.
- GPU: RX 7800 XT. GPU ortam kaydı gerçek ROCm forward/backward kontrolünün geçtiğini gösterir;
  her tam deneyin PyTorch/HIP sürümü bu kayıtla eşleştirildi. Sistem ve paket sürümleri JSON ortam kaydındadır.
- Ön eğitimli kaynakların lisansları registry'de; kendi kodumuz MIT, BANKING77 CC BY 4.0. Ağırlıklar Git dışında.
- Fit süresi epoch validation ve checkpoint yazımını içerir, indirme/ilk tokenizasyonu içermez.
  GPU süreleri CPU klasik/kelime tablolarıyla donanım ve ölçüm kapsamı eşitmiş gibi karşılaştırılmaz.

## Tekrarlama

[GPU ortamı ve WSL komutları](../docs/NEURAL_BASELINES.md) kullanılır.
Aşağıdaki komutlar repo kökünde, GPU venv'inde ve gerekli HSA ortam değişkeniyle çalıştırılır.

```bash
export HSA_ENABLE_DXG_DETECTION=1
python -m banking77.train_neural train --model bert --epochs 5 --patience 3 --batch-size 16 --learning-rate 0.00002 --max-length 64 --seed 42 --device cuda
python -m banking77.train_neural train --model distilbert --epochs 5 --patience 3 --batch-size 16 --learning-rate 0.00002 --max-length 64 --seed 42 --device cuda
python -m banking77.train_neural train --model roberta --epochs 5 --patience 3 --batch-size 16 --learning-rate 0.00002 --max-length 64 --seed 42 --device cuda
python -m banking77.train_neural train --model albert --epochs 5 --patience 3 --batch-size 16 --learning-rate 0.00002 --max-length 64 --seed 42 --device cuda
```

## Sınırlılıklar

Tek seed ve ortak beş epoch bütçesi kullanıldı; çoklu seed veya kapsamlı öğrenme oranı taraması yapılmadı.
Beş epoch bütçesi her modelin yakınsadığını veya optimum ayarlarının bulunduğunu göstermez.
Validation checkpoint seçimi iyimserlik oluşturabilir. Kısa geliştirme kontrolü benchmark sayılmadı.
Önceki klasik ve kelime modeli testleri bu deneylerden önce görülmüştür; tüm araştırmanın kör test kullandığı iddia edilmez.
Bu dört yöntem ders baseline'larıdır. Ekip katkısı olan NB + CNN'nin [kendi ablation ve sonuçları](NEURAL_VALIDATION.md)
ayrıdır; Transformer'lardan üstün olduğu iddia edilmez. Ortak kıyas/istatistik/grafikler ve PPTX 5. kişinin sorumluluğundadır.
