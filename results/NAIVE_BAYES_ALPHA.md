# Naive Bayes alpha deneyi

Aynı 8,499 eğitim ve 1,500 validation kaydı kullanıldı.
TF-IDF unigram + bigram ve sublinear TF sabit tutuldu. Yalnızca alpha değiştirildi.
Resmî testte değerlendirme yapılmadı. Bu rapor script tarafından üretilir.

Komut: `python -m banking77.benchmark_naive_bayes`.

Veri özeti SHA-256: `468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`.

| Alpha | Validation accuracy | Validation macro F1 | Eğitim (s) | Tahmin (s) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 81.60% | 0.7872 | 0.1769 | 0.0229 |
| 0.5 | 83.53% | 0.8130 | 0.1641 | 0.0261 |
| 0.1 | 85.93% | 0.8495 | 0.1615 | 0.0242 |
| 0.05 | 86.20% | 0.8529 | 0.1642 | 0.0242 |
| 0.01 | 85.20% | 0.8460 | 0.1603 | 0.0228 |

Denenen değerlerde en yüksek validation macro F1: alpha=0.05.
Eşit macro F1 durumunda listedeki ilk alpha seçilir. Varsayılan alpha=1.0 korunur.
Yanlış sayısı 276 → 207; 92 hata düzeldi, 23 doğru tahmin bozuldu.
Süreler bu bilgisayardaki tek ölçümdür; genel bir hız üstünlüğü göstermez.

## Sık karışan kategoriler

| Gerçek kategori | Tahmin | Sayı |
| --- | --- | ---: |
| card_payment_not_recognised | direct_debit_payment_not_recognised | 5 |
| balance_not_updated_after_bank_transfer | balance_not_updated_after_cheque_or_cash_deposit | 4 |
| top_up_by_bank_transfer_charge | top_up_by_card_charge | 4 |
| top_up_failed | pending_top_up | 4 |
| why_verify_identity | verify_my_identity | 3 |

## Yanlış tahmin örnekleri

- `train-08703`: The balance has not been updated.
  Gerçek: `balance_not_updated_after_bank_transfer`; tahmin: `balance_not_updated_after_cheque_or_cash_deposit`.
- `train-00688`: What does the €1 fee mean?
  Gerçek: `extra_charge_on_statement`; tahmin: `transfer_fee_charged`.
- `train-09546`: What is the procedure for an expired card?
  Gerçek: `card_about_to_expire`; tahmin: `request_refund`.

Ayarlar, ortam ve veri dosyası hash'leri: [JSON raporu](naive_bayes_alpha_validation.json).
Seçilen çalıştırmanın tam tahminleri ve confusion matrix'i yerel `results/runs/` altındadır.
Bunlar aynı komutla yeniden üretilir; eğitilmiş modeller Git'e eklenmez.
Teknik açıklama ve hata yorumları: [Naive Bayes notu](../docs/NAIVE_BAYES.md).
