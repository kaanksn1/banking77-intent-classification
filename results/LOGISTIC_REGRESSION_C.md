# Logistic Regression C ve solver deneyi

Aynı 8,499 eğitim ve 1,500 validation kaydı kullanıldı.
TF-IDF unigram + bigram ve sublinear TF sabit tutuldu (Naive Bayes ile aynı özellikler).
Yalnızca C ve solver değiştirildi. L2 düzenlileştirme, max_iter=2000, random_state=42.
Resmî testte değerlendirme yapılmadı. Bu rapor script tarafından üretilir.

Komut: `python -m banking77.benchmark_logistic_regression`.

Veri özeti SHA-256: `468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`.

C, düzenlileştirmenin tersidir: küçük C daha güçlü düzenlileştirme (daha basit model),
büyük C eğitim verisine daha sıkı uyum demektir.

## C karşılaştırması (lbfgs, multinomial)

| C | Validation accuracy | Validation macro F1 | Eğitim (s) | Tahmin (s) | İterasyon |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 66.67% | 0.6126 | 3.79 | 0.0329 | 12 |
| 1 | 85.80% | 0.8555 | 13.74 | 0.0486 | 39 |
| 10 | 88.60% | 0.8872 | 18.81 | 0.0329 | 56 |
| 100 | 88.33% | 0.8838 | 17.63 | 0.0478 | 52 |
| 1000 | 87.47% | 0.8757 | 12.89 | 0.0454 | 38 |

## Solver karşılaştırması (tüm C değerleri)

| Solver | Çok sınıf yöntemi | C | Accuracy | Macro F1 | Eğitim (s) | Yakınsadı |
| --- | --- | ---: | ---: | ---: | ---: | :---: |
| lbfgs | multinomial | 0.1 | 66.67% | 0.6126 | 3.79 | evet |
| lbfgs | multinomial | 1 | 85.80% | 0.8555 | 13.74 | evet |
| lbfgs | multinomial | 10 | 88.60% | 0.8872 | 18.81 | evet |
| lbfgs | multinomial | 100 | 88.33% | 0.8838 | 17.63 | evet |
| lbfgs | multinomial | 1000 | 87.47% | 0.8757 | 12.89 | evet |
| saga | multinomial | 0.1 | 66.73% | 0.6136 | 1.27 | evet |
| saga | multinomial | 1 | 85.73% | 0.8549 | 1.36 | evet |
| saga | multinomial | 10 | 88.60% | 0.8868 | 3.44 | evet |
| saga | multinomial | 100 | 88.93% | 0.8912 | 14.46 | evet |
| saga | multinomial | 1000 | 88.93% | 0.8907 | 30.29 | evet |
| liblinear-ovr | one-vs-rest | 0.1 | 70.67% | 0.6583 | 1.49 | evet |
| liblinear-ovr | one-vs-rest | 1 | 85.20% | 0.8476 | 2.03 | evet |
| liblinear-ovr | one-vs-rest | 10 | 89.00% | 0.8916 | 2.72 | evet |
| liblinear-ovr | one-vs-rest | 100 | 89.07% | 0.8920 | 3.09 | evet |
| liblinear-ovr | one-vs-rest | 1000 | 88.93% | 0.8904 | 3.59 | evet |

## Seçim

En yüksek validation macro F1: **liblinear-ovr, C=100** (accuracy 89.07%, macro F1 0.8920).
Eşit macro F1 durumunda çizelgedeki ilk ayar seçilir.

- Başlangıç ayarına göre (lbfgs, C=1): yanlış sayısı 213 → 164; 66 hata düzeldi, 17 doğru tahmin bozuldu (McNemar exact p = 5.6e-08).
- En iyi lbfgs ayarına göre (lbfgs, C=10): yanlış sayısı 171 → 164; 20 düzeldi, 13 bozuldu (McNemar exact p = 0.3).
Süreler bu bilgisayardaki tek ölçümdür; genel bir hız üstünlüğü göstermez.

## Sık karışan kategoriler (seçilen ayar)

| Gerçek kategori | Tahmin | Sayı |
| --- | --- | ---: |
| direct_debit_payment_not_recognised | card_payment_not_recognised | 5 |
| top_up_by_bank_transfer_charge | top_up_by_card_charge | 4 |
| transfer_fee_charged | card_payment_fee_charged | 3 |
| top_up_failed | top_up_reverted | 3 |
| pending_cash_withdrawal | declined_cash_withdrawal | 3 |

## En sık üç karışmadan birer örnek

- `train-04774`: There is a payment in my app that I did not make.  I have not used that card all day  Please reimburse my money.
  Gerçek: `direct_debit_payment_not_recognised`; tahmin: `card_payment_not_recognised`.
- `train-01901`: Are there topping up fees if I have to transfer?
  Gerçek: `top_up_by_bank_transfer_charge`; tahmin: `top_up_by_card_charge`.
- `train-07166`: Is there a fee that comes with transfers?
  Gerçek: `transfer_fee_charged`; tahmin: `card_payment_fee_charged`.

## En düşük F1'li beş kategori (seçilen ayar)

| Kategori | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| top_up_failed | 0.833 | 0.682 | 0.750 | 22 |
| card_acceptance | 0.667 | 0.889 | 0.762 | 9 |
| pending_top_up | 0.800 | 0.727 | 0.762 | 22 |
| pending_cash_withdrawal | 1.000 | 0.619 | 0.765 | 21 |
| compromised_card | 0.733 | 0.846 | 0.786 | 13 |

Ayarlar, ortam ve veri dosyası hash'leri: [JSON raporu](logistic_regression_validation.json).
Seçilen çalıştırmanın tam tahminleri ve confusion matrix'i yerel `results/runs/` altındadır.
Bunlar aynı komutla yeniden üretilir; eğitilmiş modeller Git'e eklenmez.
Teknik açıklama ve hata yorumları: [Logistic Regression notu](../docs/LOGISTIC_REGRESSION.md).
