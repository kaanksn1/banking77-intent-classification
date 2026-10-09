# Linear SVM C ve loss deneyi

Aynı 8,499 eğitim ve 1,500 validation kaydı kullanıldı.
TF-IDF unigram + bigram ve sublinear TF sabit tutuldu (Naive Bayes ve Logistic Regression ile aynı özellikler).
Yalnızca C ve loss değiştirildi. L2 düzenlileştirme, one-vs-rest, dual=True, max_iter=10000, random_state=42.
Resmî testte değerlendirme yapılmadı. Bu rapor script tarafından üretilir.

Komut: `python -m banking77.benchmark_linear_svm`.

Veri özeti SHA-256: `468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`.

C, hata cezasının ağırlığıdır: küçük C geniş margin ve güçlü düzenlileştirme (daha basit model),
büyük C eğitim hatalarına daha ağır ceza ve eğitim verisine daha sıkı uyum demektir.

- `squared_hinge`: max(0, 1 − y·f(x))², scikit-learn varsayılanı
- `hinge`: max(0, 1 − y·f(x)), klasik soft-margin SVM

| Loss | C | Validation accuracy | Validation macro F1 | Eğitim (s) | Tahmin (s) | İterasyon | Yakınsadı |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| squared_hinge | 0.01 | 80.47% | 0.7907 | 0.39 | 0.0129 | 10 | evet |
| squared_hinge | 0.1 | 86.00% | 0.8553 | 0.33 | 0.0126 | 12 | evet |
| squared_hinge | 1 | 89.20% | 0.8935 | 0.38 | 0.0127 | 41 | evet |
| squared_hinge | 10 | 87.93% | 0.8803 | 0.70 | 0.0126 | 354 | evet |
| squared_hinge | 100 | 87.80% | 0.8781 | 1.55 | 0.0126 | 3593 | evet |
| hinge | 0.01 | 83.80% | 0.8244 | 0.45 | 0.0127 | 163 | evet |
| hinge | 0.1 | 84.67% | 0.8335 | 0.44 | 0.0169 | 229 | evet |
| hinge | 1 | 88.00% | 0.8813 | 0.66 | 0.0149 | 1421 | evet |
| hinge | 10 | 87.67% | 0.8775 | 0.87 | 0.0124 | 2562 | evet |
| hinge | 100 | 87.67% | 0.8769 | 1.10 | 0.0127 | 2922 | evet |

## Seçim

En yüksek validation macro F1: **squared_hinge, C=1** (accuracy 89.20%, macro F1 0.8935).
Eşit macro F1 durumunda çizelgedeki ilk ayar seçilir.

- Başlangıç ayarına göre (squared_hinge, C=1): seçilen ayarla aynı çalıştırma.
- Diğer loss'un en iyi ayarına göre (hinge, C=1): yanlış sayısı 180 → 162; 25 hata düzeldi, 7 doğru tahmin bozuldu (McNemar exact p = 0.0021).
Süreler bu bilgisayardaki tek ölçümdür; genel bir hız üstünlüğü göstermez.

## Sık karışan kategoriler (seçilen ayar)

Seçilen ayarda yanlış sınıflandırılan validation mesajı: 162 / 1,500.

| Gerçek kategori | Tahmin | Sayı |
| --- | --- | ---: |
| direct_debit_payment_not_recognised | card_payment_not_recognised | 6 |
| top_up_by_bank_transfer_charge | top_up_by_card_charge | 4 |
| top_up_failed | top_up_reverted | 3 |
| pending_cash_withdrawal | declined_cash_withdrawal | 3 |
| get_disposable_virtual_card | disposable_card_limits | 3 |

## En sık üç karışmadan birer örnek

Katkı = `(w_tahmin − w_gerçek) × tfidf`; pozitif değer yanlış kategoriye, negatif değer doğru kategoriye iter.

- `train-04774`: There is a payment in my app that I did not make.  I have not used that card all day  Please reimburse my money.
  Gerçek: `direct_debit_payment_not_recognised`; tahmin: `card_payment_not_recognised`.
  Yanlışa iten: `payment in` (+0.41), `payment` (+0.29), `app that` (+0.11). Doğruya iten: `not used` (-0.20), `all day` (-0.16), `card all` (-0.15).
- `train-01901`: Are there topping up fees if I have to transfer?
  Gerçek: `top_up_by_bank_transfer_charge`; tahmin: `top_up_by_card_charge`.
  Yanlışa iten: `up fees` (+0.44), `fees` (+0.25), `topping` (+0.24). Doğruya iten: `transfer` (-0.40), `there` (-0.04), `if` (-0.04).
- `train-08546`: My top-up was rejected
  Gerçek: `top_up_failed`; tahmin: `top_up_reverted`.
  Yanlışa iten: `up was` (+0.45), `was rejected` (+0.24), `my top` (+0.15). Doğruya iten: `rejected` (-0.56), `my` (-0.09), `up` (-0.08).

## En düşük F1'li beş kategori (seçilen ayar)

| Kategori | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| pending_top_up | 0.762 | 0.727 | 0.744 | 22 |
| card_acceptance | 0.667 | 0.889 | 0.762 | 9 |
| balance_not_updated_after_bank_transfer | 0.826 | 0.731 | 0.776 | 26 |
| card_delivery_estimate | 0.737 | 0.824 | 0.778 | 17 |
| card_payment_not_recognised | 0.724 | 0.840 | 0.778 | 25 |

Ayarlar, ortam ve veri dosyası hash'leri: [JSON raporu](linear_svm_validation.json).
Seçilen çalıştırmanın tam tahminleri ve confusion matrix'i yerel `results/runs/` altındadır.
Bunlar aynı komutla yeniden üretilir; eğitilmiş modeller Git'e eklenmez.
Teknik açıklama ve hata yorumları: [Linear SVM notu](../docs/LINEAR_SVM.md).
