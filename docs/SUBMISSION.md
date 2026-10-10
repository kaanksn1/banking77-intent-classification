# Project 1 teslim takibi

Kaynak: hocanın Project Instructions, About benchmarks, About JEV as a baseline
e-postaları ve 10 Ekim'de paylaşılan son baseline açıklaması.
Sunum: 12 Ekim 2026 Pazartesi. Notun %50'si sunum, %50'si implementation repo.

## Kesinleşen kapsam

- BANKING77, 77 bankacılık talep kategorisi. Veri ve mevcut bölümler değişmez.
- Repo sahibi, BANKING77'yi başka grubun seçmediğini bildirdi. Yeni veri/CLINC
  ve isteğe bağlı JEV/Laya/AnyJev ekleme planları iptal edilmiştir.
- Hoca son yanıtında sunumlarda adı geçen tüm yöntemlerin baseline olmasını,
  RNN/CNN/LSTM/Transformer dahil, ve bunların üzerine ekip katkısı istedi.
  Önceki üç klasik baseline'ın tüm zorunlu kapsamı karşıladığı varsayılmaz.
- Slaytların metinleri ve gömülü resimleri incelendi:
  [yöntem ve model eşleştirmesi](NEURAL_BASELINES.md).
- NB/LR/SVM, Word2Vec CBOW/Skip-gram, GloVe, FastText, CNN/RNN/LSTM/BiLSTM
  ve BERT-base/DistilBERT/RoBERTa-base/ALBERT-base-v2 baseline listesindedir.
  Standart veya ayarlanmış baseline yeni mimari diye adlandırılmaz.
- Repo sahibinin ek katkısı NB + CNN soft voting deneyidir; validation'da
  ağırlık seçimi ve tek model ablation'ları vardır. Yeni bir araştırma algoritması
  veya ek puan garantisi iddia edilmez.
- Hazır veri ve dört mükerrer kaydın temizlenmesi veri özgünlüğünden tam puan
  alındığı anlamına gelmez. İşlem miktarı olduğu gibi raporlanır.

## Tamamlanan işler

- [x] Public GitHub repo, README, script/klasör açıklamaları, kurulum ve CI.
- [x] Ham ve hazırlanmış veri, yeniden üretilebilir hazırlama scripti, kaynak hash'leri.
- [x] Kod için MIT; BANKING77 için upstream CC BY 4.0 lisansı ve atıf.
- [x] Beş kişisel katkı dosyası ve GitHub katkı kanıtları.
- [x] 2. kişinin veri analizi/ön işleme ve özellik deneyleri (PR #2).
- [x] 3. kişinin LR kodu ve 15 validation ayarı (PR #3, açıklama düzeltmesi #7).
- [x] 4. kişinin SVM kodu ve 10 validation ayarı (PR #5).
- [x] Repo sahibinin NB alpha benchmark'ı, teknik notu ve hata analizi.
- [x] NB ayarlarının testten önce commit edilmesi ve nihai test teslimi (PR #10).
- [x] 5. kişinin ortak validation benchmark'ı, istatistikleri ve grafikleri (PR #8).
- [x] 5. kişinin üç modelde 90 özellik deneyi (PR #9), entegrasyonda doğrulandı.
- [x] 5. kişinin klasik nihai test karşılaştırması (PR #11). Sabit NB/LR/SVM
  skorları, eşleştirilmiş istatistikler ve örtüşme sonuçları entegrasyonda doğrulandı.
- [x] Yeni kapsamın neural/embedding eğitim, sabitleme ve test scriptleri.
- [x] Sekiz yeni baseline'ın tam train/validation deneyleri; NB + CNN validation
  ağırlık taraması. [Kaydedilmiş sonuçlar](../results/NEURAL_VALIDATION.md).
- [x] Sekiz baseline ve NB + CNN protokolleri testten önce `2425ccc` commit'inde
  sabitlendi; seçilen checkpoint'lerle resmî test çalıştırıldı. [Yeni test çıktıları](../results/NEURAL_TEST.md).
  NB + CNN: accuracy %90.65, macro F1 0.9061. Tahminler ve metrikler yeniden hesaplanarak doğrulandı.
- [x] GloVe kaynak arşivi/vektör hash'leri; dört Transformer'ın sabit model
  repository/commit/lisans kayıtları. Büyük ağırlıklar Git dışında tutulur.
- [x] CPU forward/backward, kısa Transformer geliştirme kontrolü; 52 yerel test
  ve NB validation geçti. Geliştirme kontrolü tam benchmark diye sunulmaz.
- [x] WSL2 üzerinde Ubuntu 24.04.5, ROCm 7.2.1, ROCDXG 1.2.0 ve AMD PyTorch kurulumu.
  RX 7800 XT ile gerçek forward/backward ve kısa DistilBERT GPU eğitimi geçti.
  Linux ortamında 52 test ve NB validation da geçti; geliştirme kontrolü benchmark sayılmadı.
- [x] BERT-base, DistilBERT, RoBERTa-base ve ALBERT-base-v2'nin ortak beş epoch
  bütçesiyle tam validation deneyleri tamamlandı. [Sonuç raporu](../results/TRANSFORMER_VALIDATION.md)
  ve [tam metrik/ortam kaydı](../results/transformer_validation.json) hazırdır;
  seçilmiş checkpoint'ler için dört nihai protokol dosyası oluşturuldu.
- [x] Dört Transformer'ın protokolleri
  [testten önceki `1e4ff66` commit'i](https://github.com/kaanksn1/banking77-intent-classification/commit/1e4ff66d49e7d184d3dbe1548fa7b038b79833c2)
  ile sabitlendi ve resmî test değerlendirmeleri tamamlandı.
  [Nihai test raporu](../results/TRANSFORMER_TEST.md) ve
  [tam metrik kaydı](../results/transformer_test.json) hazırdır. RoBERTa-base:
  accuracy %93.02, macro F1 0.9301. Standart çıktılar doğrulandı;
  eşleştirilmiş karşılaştırma [on altı yöntemlik raporda](../results/ALL_MODELS_TEST.md).

## Kalan işler ve sorumlular

- [ ] 4. kişi: teknik nottaki Macro F1 farkını sabit mesaj sayısına çeviren ifadeyi düzeltme.
- [x] 5. kişi: özellik raporunda bigram-only ablation'ın tanı amaçlı olduğu ve seçilmeyen
  unigram SVM (`hinge`, `C=10`) adayının 10.000 iterasyon sınırına ulaştığı açıklandı.
  Seçilen modeller yakınsamıştır; bu not nihai NB ayarını değiştirmez.
- [x] 5. kişi: test raporu/JSON'daki komut alanına `--split test` eklendi; benchmark kodu
  komutu artık split'e göre yazar (`command_line`) ve testle korunur.
- [x] 5. kişi: on altı yöntemin ortak test karşılaştırması, eşleştirilmiş istatistikler
  (McNemar + Holm, seed'li bootstrap) ve grafikler:
  [rapor](../results/ALL_MODELS_TEST.md), [tam kayıt](../results/all_models_test.json).
  RoBERTa-base en yüksek macro F1 (0.9301) ve diğer 15 yöntemden Holm düzeltmesiyle ayrışır;
  NB + CNN (0.9061) CNN, LR ve SVM'den ayrışır; BERT, DistilBERT ve ALBERT'ten ayrışmaz,
  RoBERTa-base'in altındadır. Yeni bir mimari veya Transformer üstünlüğü iddia edilmez.
- [ ] 5. kişi: PPTX, sunum provası ve süre kontrolü.
- [ ] Tüm üyeler: kendi katkı dosyalarını son gerçek işleri ve kendi commit/PR'larıyla güncelleme.
- [ ] Ödev sistemine hem PPTX hem public GitHub URL'sini yükleme.

Başka üyelerin uygulama, deney veya kişisel raporları repo sahibi tarafından
sahiplenilmez. Veri hazırlama, LR/SVM ve ortak grafik/rapor kodları bu ek kapsamda
değiştirilmedi. Klasik test sonuçları yeni kapsamdan önce görülmüştür; tüm
araştırma için tamamen kör test iddiası kurulmaz. Yeni ayarlar validation'da seçilir.

## Benchmark ve sunum

Ana metrik macro F1, destekleyici metrik accuracy. Macro F1 her talep kategorisine
eşit önem vermek için kullanılır; resmî testin dengeli olduğu açıkça belirtilir.
Her deney veri kimliği, bölüm, tam model/özellik ayarları, eğitim/tahmin süreleri,
tahminler ve örtüşme raporunu içerir. Ön eğitim ve donanım farkları açıklanır.
Tek seed veya validation iyileşmesi istatistiksel üstünlük kanıtı sayılmaz.

Sunumda teknik derinlik %50, akıcılık %25, süre %25. Hedef iki dakikanın altı;
mailde bu süre 6, 2–3 dakika 5, 3:00–3:15 arası 3, 3:30'da 0 ve kesilme olarak
belirtilmiştir. Okumadan anlatım gerekir. Soru cevap teknik derinliği etkileyebilir.
