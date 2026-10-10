# Yeni baseline sonuçlarının 5. kişiye devri

Bu belge ortak karşılaştırmayı ve PPTX'i hazırlayacak ekip arkadaşına yöneliktir.
Repo sahibinin ek model çalışması ile ortak değerlendirme sorumluluğunu ayırır.
Veri, mevcut LR/SVM uygulamaları ve ortak benchmark kodları değiştirilmemiştir.
Üç klasik, sekiz kelime/sinir ağı ve dört Transformer baseline'ı ile NB + CNN
katkısının sonuçları hazırdır: toplam 15 baseline ve bir katkı yöntemi.

## Kullanılacak çıktılar

| Kapsam | Sonuç ve protokol | Satır bazlı çıktılar |
| --- | --- | --- |
| Seçilmiş klasik NB/LR/SVM | [Klasik test raporu](../results/MODEL_COMPARISON_TEST.md) | Mevcut klasik test çıktıları; NB için `results/naive_bayes_test/` |
| RNN, CNN, LSTM, BiLSTM, GloVe, Word2Vec CBOW/Skip-gram, FastText | [Validation](../results/NEURAL_VALIDATION.md), [test](../results/NEURAL_TEST.md) | `results/neural_test/<model>/` |
| Ekip katkısı: NB + CNN soft voting | Aynı neural raporlar; `results/nb_cnn_final_protocol.json` | `results/neural_test/nb_cnn/` |
| BERT-base-uncased, DistilBERT-base-uncased, RoBERTa-base, ALBERT-base-v2 | [Validation](../results/TRANSFORMER_VALIDATION.md), [nihai test](../results/TRANSFORMER_TEST.md); protokoller testten önce `1e4ff66` ile sabitlendi | `results/transformer_test/<model>/` |

Her yayımlanan yeni model klasöründe `metrics.json`, `classification_report.json`,
`predictions.csv` ve `confusion_matrix.csv` bulunur. Büyük checkpoint'ler ve yerel
`results/runs/` çıktıları Git'e eklenmez. Tekrarlama ve ağırlık doğrulama adımları
[neural protokolde](NEURAL_BASELINES.md) açıklanır.

## Ortak karşılaştırmada yapılacaklar

1. Seçilmiş üç klasik model, sekiz kelime/sinir ağı baseline'ı, dört Transformer
   ve NB + CNN katkısını nihai test tablosunda birleştir. Eğitim tamamlanmamış
   modeli sonuç varmış gibi gösterme. Validation ve test tablolarını ayrı tut.
2. Tahminleri `id` ile eşleştir; metin ve gerçek etiketleri de doğrula. Aynı veri
   özeti SHA-256 değeri `468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`
   olmalıdır. Her model 3.080 resmî test mesajını kapsamalıdır.
3. Ana metrik macro F1, destekleyici metrik accuracy olsun. Resmî test her sınıfta
   40 mesaj içerir; macro F1 gerekçesini her talep kategorisine eşit önem vermekle açıkla.
4. Mevcut eşleştirilmiş istatistik yaklaşımını yeni tahminlere uygula; kullanılan
   yöntem, seed ve varsa çoklu karşılaştırma düzeltmesini belirt. Tek seed sonucu
   veya küçük skor farkını doğrudan anlamlı üstünlük diye sunma.
5. Yedi normalize birebir train örtüşmesinin dışarıda bırakıldığı 3.073 mesajlık
   sonuçları ayrı göster. Yakın tekrarların tamamen temizlendiğini iddia etme.
6. Süre tablosunda CPU ve AMD GPU ölçümlerini, dış ön eğitimi ve ölçüm kapsamını
   belirt. Neural fit süresi epoch validation/checkpoint yazımını içerir;
   ilk indirme/tokenizasyonu içermez. Tahmin süresinin kapsamı da yönteme göre değişir.
7. Sunum için anlaşılır bir sonuç grafiği, NB + CNN ablation özeti ve birkaç gerçek
   hata örneği seç. Standart yöntemler baseline; NB + CNN projenin katkısıdır.
   Yeni bir mimari veya Transformer'ların tümünden üstünlük iddiası kurma.

Test sonuçlarına göre yeniden ağırlık, öğrenme oranı veya checkpoint seçme.
Önceki klasik/kelime modeli testleri genişleyen kapsamdan önce görüldü;
tüm araştırmanın tamamen kör test kullandığı söylenmez. Yeni protokoller kendi
resmî değerlendirmelerinden önce commit edilir.

## Teslim

Ortak değerlendirme, grafikler ve PPTX değişikliklerini kendi branch/PR'ında yap;
kendi katkı dosyanı gerçek commit/PR bağlantılarınla güncelle. Klasik model raporu
`results/MODEL_COMPARISON_TEST.md` ve `results/model_comparison_test.json`
içindeki komut alanının `--split test` içerdiğini de kontrol et. Neural ve
Transformer çalıştırıcıları ise `test --protocol ...` alt komutunu kullanır.
Hocaya teslimde hem PPTX hem public GitHub URL'si gerekir. Sunumu iki dakikanın
altına sığdırmak için prova yap; model listesini okumak yerine problem, deney
protokolü, sonuç ve katkıyı teknik olarak açıkla.
