# Serdar Kaan Kesen — 220911810

GitHub kullanıcı adı: [kaanksn1](https://github.com/kaanksn1)

Sorumluluk: Naive Bayes ve GitHub/entegrasyon; son hoca yanıtından sonra açıkça
atanan neural/embedding baseline'ları ve NB + CNN katkısı. Grup: Anonymous.

## Tamamladığım işler

- `src/banking77/naive_bayes.py`: TF-IDF + MultinomialNB pipeline'ını hazırladım.
- `src/banking77/train_naive_bayes.py`: eğitim, değerlendirme, veri/ortam kaydı,
  satır bazlı tahminler, classification report, confusion matrix ve model çıktısını
  hazırladım. Train ile birebir örtüşen değerlendirme mesajlarını ayrıca raporladım.
- `src/banking77/benchmark_naive_bayes.py`: beş alpha adayını validation macro F1
  ile karşılaştıran, resmî testi seçim dışında tutan benchmark'ı hazırladım.
- `docs/NAIVE_BAYES.md`, `results/NAIVE_BAYES_ALPHA.md`: yöntem, smoothing,
  metrik gerekçesi ve üç gerçek validation hatasının yorumunu yazdım.
- Kendi başlangıç commitlerimde public repo, kurulum, en küçük ortak indirme/veri
  sözleşmesi, lisans/atıf, çıktı sözleşmesi, CI, protokol kontrolleri ve tek mesaj
  tahmin komutunu hazırladım. `docs/HANDOFF_DATA.md` ile veri/özellik çalışmasını
  ilgili ekip arkadaşına devrettim.
- Ekip arkadaşlarının PR'larını inceleyip birleştirdim; README ve ortak teslim
  takibini güncelledim. Veri/özellik, LR ve SVM deneylerinin mevcut skorlarını
  entegrasyon amacıyla yeniden çalıştırarak doğruladım.
- PR #7'nin açıklama düzeltmelerini ve PR #8'in son benchmark revizyonunu
  inceledim. 15 validation çalıştırmasının skorlarını, istatistiklerini,
  veri hash'lerini ve yakın tekrar duyarlılık sonuçlarını doğrulayıp birleştirdim.
- PR #9'un unigram, unigram + bigram ve bigram-only karşılaştırmasındaki 90
  validation denemesini entegrasyon için yeniden çalıştırdım; raporlanan skorları
  ve istatistikleri doğruladım. Ortak deney belgesindeki birleştirme çakışmasını
  çözdüm. Deney tasarımı, uygulaması ve raporu ilgili ekip arkadaşının katkısıdır.
- `results/naive_bayes_final_protocol.json`: NB ayarını ilk resmî testten önce
  ayrı commit ile sabitledim. Mevcut veri ve ortak unigram + bigram temsili korundu.
- `results/NAIVE_BAYES_TEST.md`, `results/naive_bayes_test/`: sabit ayarla nihai
  NB testini çalıştırdım, standart çıktıları yayımladım ve gerçek test hatalarını
  raporladım. Skorları, confusion matrix'i ve 3.080 tahminin resmî kayıtlarla
  eşleşmesini kaydedilmiş çıktılardan ayrıca doğruladım.
- `.gitattributes`: yayımlanan nihai NB CSV çıktılarının satır sonlarının Git
  tarafından dönüştürülmesini önledim; mesaj içindeki yeni satırlar ve çıktı
  dosyalarının özgün baytları korunur.

2. kişinin veri analizi/özellik deneyleri, 3. kişinin LR uygulaması/deneyleri,
4. kişinin SVM uygulaması/deneyleri ve 5. kişinin ortak benchmark/grafikleri
kendi katkılarıdır. Bunları kendi uygulamam veya deney tasarımım olarak sahiplenmiyorum.
PPTX hazırlama ve sunma görevi 5. kişidedir.

## Doğrulama ve deneyler

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes --split validation --ngram-max 2 --alpha 0.05
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes --split test --ngram-max 2 --alpha 0.05
```

- Yeni kapsam dahil birleşik kodda **52 test geçti**; bu testlerin tamamını ben yazmadım.
- Alpha benchmark'ımın komutu: `python -m banking77.benchmark_naive_bayes`.
- Veri özeti SHA-256: `468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`.
- 8.499 train, 1.500 validation, 3.080 resmî test mesajı. Validation eğitime eklenmedi.
- Başlangıç NB (`alpha=1`): validation accuracy %81.60, macro F1 0.7872.
- Seçilen NB (`alpha=0.05`): validation accuracy %86.20, macro F1 0.8529.
- Nihai test: accuracy %84.74, macro F1 0.8458; 2.610 doğru, 470 yanlış.
- Train ile birebir örtüşen 7 mesaj çıkarılınca: 3.073 mesajda accuracy %84.71,
  macro F1 0.8454. Yakın tekrarların tamamen giderildiği iddia edilmez.
- Test, ayar seçiminde kullanılmadı; protokol önce commit edildi.
- Süreler donanıma bağlıdır. Hata yorumları özellik katkısıyla kanıtlanmış nedensel
  açıklamalar değildir. Hazır veri kullanımı için özgünlük puanı garantisi verilmez.

## GitHub kanıtı

- [Başlangıç kurulumu ve NB](https://github.com/kaanksn1/banking77-intent-classification/commit/9f0b3e255130652b4c49c0173aa4edcff0957986)
- [Kapsamı NB ve entegrasyonla sınırlama](https://github.com/kaanksn1/banking77-intent-classification/commit/618acb2050d02e233f04f081d40d2ff9f5862e4c)
- [NB devir PR'ım #1](https://github.com/kaanksn1/banking77-intent-classification/pull/1)
- [Veri/LR entegrasyon PR'ım #4](https://github.com/kaanksn1/banking77-intent-classification/pull/4)
- [SVM entegrasyon PR'ım #6](https://github.com/kaanksn1/banking77-intent-classification/pull/6)
- [Testten önce NB ayarlarını sabitleyen commit'im](https://github.com/kaanksn1/banking77-intent-classification/commit/29312bbcf591aca0073a88cc4121dae1d172ac0d)
- [Nihai NB teslim PR'ım #10](https://github.com/kaanksn1/banking77-intent-classification/pull/10)
- [Özellik karşılaştırması entegrasyon incelemesi #9](https://github.com/kaanksn1/banking77-intent-classification/pull/9)
- [Yeni baseline ve NB + CNN ayarlarının test öncesi commit'i](https://github.com/kaanksn1/banking77-intent-classification/commit/2425ccc)
- [AMD GPU kurulumu ve validation bütçesi kararı](https://github.com/kaanksn1/banking77-intent-classification/commit/214b1799ef35ac9e170c909d901842e4e6300e3c)

## 10 Ekim'de açıkça atanan ek kapsam

- PR #11'in klasik nihai test karşılaştırmasını çektim. Beş sabit model ayarının
  skorlarını ve raporun istatistik/örtüşme hesaplarını yeniden üreterek doğruladım.
  Karşılaştırma uygulaması ve raporu 5. kişinin katkısıdır; onun kodunu değiştirmedim.
- Ders slaytlarının gömülü resimlerini de inceleyip zorunlu baseline listesini
  `docs/NEURAL_BASELINES.md` içinde eşleştirdim.
- `neural_models.py`, `train_neural.py`: maskeli CNN, vanilla RNN, LSTM, BiLSTM,
  GloVe ortalama başlığı ve sabit kaynak sürümlü Transformer ince ayar akışını
  hazırladım. Validation checkpoint seçimi, veri/kod/ağırlık hash'leri ve testten
  önce commit zorunluluğu ekledim. Mevcut veri, LR/SVM ve ortak benchmark kodları değişmedi.
- `prepare_embeddings.py`: Stanford GloVe indirme ve hash kontrolünü, yalnız
  train sözlüğüne uyan vektörlerin ayrılmasını hazırladım. Kaynak lisanslarını kaydettim.
- `train_embeddings.py`: train'den Word2Vec CBOW/Skip-gram ve FastText + ortalama
  vektör + sabit LR başlığı baseline'larını hazırlayıp tam validation'da çalıştırdım.
- Sekiz yeni baseline ve BERT-base-uncased, DistilBERT-base-uncased, RoBERTa-base,
  ALBERT-base-v2'nin tam validation deneylerini kaydettim; yöntem/ön eğitim
  farklarını ve süre ölçümünün sınırlarını açıkladım. Kısa DistilBERT kontrolünü
  benchmark saymadım. İlk üç epoch validation eğrilerine dayanarak tüm dört
  Transformer'ın nihai bütçesini beş epoch olarak belirledim; ön eğitimli
  ağırlıklardan yeniden eğittim, her modelde validation macro F1 ile checkpoint seçtim.
  Kararı `results/transformer_budget_decision.json` içinde testten önce kaydettim.
- `docs/NEURAL_BASELINES.md` içinde üç birincil BANKING77 makalesinin test accuracy
  sonuçlarını, model/veri ayrımı farklarını ve doğrudan kıyas yapılamayacağını açıkladım.
- `train_ensemble.py`: kendi NB modelim ve CNN olasılıklarını birleştirdim.
  11 ağırlığı yalnız validation macro F1 ile seçtim; yalnız NB/CNN ablation'larını
  korudum. Seçilen ağırlık NB=0.6, CNN=0.4; validation accuracy %90.07,
  macro F1 0.8997. Bu değerin test sonucu veya yeni araştırma algoritması olduğu iddia edilmez.
- Neural/embedding/ensemble için 12 anlamlı kontrol ekledim: padding, heldout
  kelimeler, FastText OOV, öğrenme, 77 sınıflı başlık, kayıt/yükleme, protokol ve
  olasılık hizası. Opsiyonel neural CI işi ve GPU preflight hazırladım.
- WSL2 üzerinde Ubuntu 24.04.5, ROCm 7.2.1, ROCDXG 1.2.0 ve AMD PyTorch
  ortamını kurdum; `scripts/setup_rocm_wsl.sh` ile adımları kaydettim. RX 7800 XT
  ile gerçek forward/backward ve kısa DistilBERT GPU eğitimi geçti.
  Linux ortamında 52 test ve NB validation sonucunu da doğruladım.
- `docs/HANDOFF_NEURAL.md`: yeni model çıktılarının ortak karşılaştırma ve PPTX
  sorumlusuna devrini, veri/etiket eşleşmesini ve süre karşılaştırmasının sınırlarını açıkladım.
- Sekiz baseline ve NB + CNN checkpoint'lerini testten önce ayrı commit'le sabitledim.
  Resmî testte yeniden eğitim yapmadan değerlendirdim; 3.080 tahminin kimlik/metin/etiketlerini,
  accuracy/macro F1 ve confusion matrix'i çıktı dosyalarından yeniden hesaplayarak doğruladım.
  NB + CNN: accuracy %90.65, macro F1 0.9061. Standart çıktılar ve her yöntem için
  üç gerçek hata örneği `results/NEURAL_TEST.md` ile `results/neural_test/` altında.

## Sunuma katkım

Naive Bayes'in teknik açıklaması, validation alpha tablosu, hata örnekleri ve
nihai test raporunu hazırladım. Bu içerik 5. kişinin sunum hazırlığında kullanabileceği
NB devir paketidir; PPTX'in tamamlandığı veya sunum yaptığım iddia edilmez.
