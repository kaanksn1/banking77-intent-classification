# Kemal Arif Karabacıoğlu — 220911802

GitHub kullanıcı adı: kemalarif
Sorumluluk: 4. kişi — Linear SVM modeli, C/loss deneyleri ve hata analizi

## Tamamladığım işler

- `src/banking77/linear_svm.py`: TF-IDF + `LinearSVC` pipeline'ı. TF-IDF ayarı Naive Bayes
  ve Logistic Regression ile aynı tutuldu (unigram + bigram, sublinear TF), böylece modeller
  arasındaki fark yalnızca sınıflandırıcıdan gelir. İki loss desteklenir: `squared_hinge`
  (varsayılan) ve klasik `hinge`; ikisi de L2 düzenlileştirmeli one-vs-rest SVM'dir.
- `src/banking77/train_linear_svm.py`: tek bir ayarla eğitim, metrik ve tahmin kaydı.
  NB/LR ile aynı çıktı sözleşmesini (`metrics.json`, `predictions.csv`,
  `confusion_matrix.csv`, `classification_report.json`) kullanır; ek olarak iterasyon
  sayısını ve yakınsama durumunu kaydeder.
- `src/banking77/benchmark_linear_svm.py`: C (0.01, 0.1, 1, 10, 100) × loss ızgarasını
  yalnızca validation üzerinde çalıştırır, seçimi önceden belirlenen kuralla (en yüksek
  macro F1) yapar, McNemar testiyle eşleştirilmiş karşılaştırma üretir ve örnek hatalar
  için kelime katkılarını kaydedilmiş modelin katsayılarından hesaplar.
- `tests/test_linear_svm.py`: geçersiz ayarların reddi, iki loss'un çalışması, TF-IDF'in
  yalnızca train'de öğrenilmesi, test setinin seçimde kullanılamaması ve kelime
  katkılarının toplamının skor farkına eşit olması için birim testleri.
- `docs/LINEAR_SVM.md`: yöntem açıklaması (margin, hinge/squared hinge, C), seçim gerekçesi
  ve hata analizi.
- `results/LINEAR_SVM_C.md` ve `results/linear_svm_validation.json`: script tarafından
  üretilen deney raporu.

Ortak veri bölümleri ve diğer üyelerin dosyaları değiştirilmedi.

## Doğrulama ve deneyler

- Çalıştırma komutu: `python -m banking77.benchmark_linear_svm`
- Kullanılan veri bölümü: validation (8.499 train, 1.500 validation);
  `dataset_summary_sha256 = 468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`
  (Naive Bayes ve Logistic Regression deneyleriyle aynı).
- Sonuç ve üretilen çıktı dosyası: seçilen ayar `squared_hinge`, `C=1`;
  validation accuracy %89.20, macro F1 0.8935. Rapor: `results/LINEAR_SVM_C.md`.
- Bulgular:
  - En iyi C ızgaranın ortasında. C=0.1'e göre 59 hata düzeldi, 11 doğru tahmin bozuldu
    (exact McNemar p = 4.5×10⁻⁹); C=10'a göre 29 düzeldi, 10 bozuldu (p = 0.0034).
  - squared_hinge, en iyi hinge ayarından anlamlı biçimde iyi (180 → 162 yanlış, p = 0.0021).
  - 162 hatanın %67'si aynı konu ailesindeki alt kategoriler arasında
    (ör. `direct_debit_payment_not_recognised` → `card_payment_not_recognised`).
  - Kelime katkısı analizi: yazım hataları (`stillpending`, `trasfer`) belirleyici
    kelimenin modelde hiç görülmemesine yol açıyor.
- Sınırlamalar: tek bir 1.500 satırlık validation bölümü; süreler tek bilgisayarda
  tek ölçüm. Resmî test, ekip kuralına göre veri/özellik çalışması kesinleştikten
  sonra dondurulan ayarla çalıştırılacak.

## GitHub kanıtı

- Kendi commit bağlantılarım (`feature/svm`):
  - [Model, eğitim ve benchmark scriptleri](https://github.com/kaanksn1/banking77-intent-classification/commit/eca896f40da761e5c3fece897eaeb1993c724aa9)
  - [Birim testleri](https://github.com/kaanksn1/banking77-intent-classification/commit/9314f6fb756e7a0fb0b4d09fe4175e3be166c9d0)
  - [Teknik not ve hata analizi](https://github.com/kaanksn1/banking77-intent-classification/commit/816712021c812660224b8794db77e6f43c31ff40)
  - [Validation sonuçları](https://github.com/kaanksn1/banking77-intent-classification/commit/a3947f59547bae2d8213bec2b24129ed4949f9cd)

## Sunuma katkım

- Linear SVM bölümü için sonuç tablosu, C'nin ve loss'un etkisi, seçim gerekçesi ve
  üç örnek hata `docs/LINEAR_SVM.md` içinde hazır; slayt metni bu nottan hazırlanacak.
