# Project 1 teslim takibi

Kaynak: hocanın paylaşılan Project Instructions, About benchmarks ve
About JEV as a baseline e-postaları. Sunum tarihi: 12 Ekim 2026 Pazartesi.
Notun %50'si sunum, %50'si implementation repo.

## Konu ve baseline kapsamı

- Proje: BANKING77 bankacılık talep sınıflandırması, 77 sınıf.
- 8 Ekim'de repo sahibi BANKING77'yi başka grubun seçmediğini bildirdi.
- Hazır veri kullanımı ve mevcut temizlik miktarı veri özgünlüğünden tam puan
  alındığı anlamına gelmez. Gerçek temizlik işlemleri ve sayıları raporlanmalıdır.
- Ders slaytlarının mevcut metin dökümünde Week 2 için Naive Bayes, Logistic
  Regression ve SVM; Week 3 için GloVe embeddings, CNN, BiLSTM ve BERT başlıkları
  vardır. GloVe bir temsil yöntemidir.
- Hocanın yeni benchmark mailine göre ders yöntemlerinin standart biçimleri
  baseline'dır; standart BERT'i yeni yöntem olarak adlandırmamalıyız. Mailin
  ders yöntemlerinin tamamının uygulanmasını zorunlu kılıp kılmadığı açık
  değildir; NB/LR/SVM planının tüm zorunlu kapsamı tamamladığı varsayılmaz.
- JEV/Laya/AnyJev gibi yöntemler hocanın son mailinde isteğe bağlı yeni yöntem
  olarak kabul edilmiştir. Repo sahibi 10 Ekim'de bu ekleri ve yeni veri ekleme
  planını iptal etmiştir; mevcut BANKING77 ve üç baseline ile devam edilir.
  Bu yöntemlere ilişkin ek puan iddiasında bulunulmaz.

## Repo kontrolü

- [x] Public repo: https://github.com/kaanksn1/banking77-intent-classification
- [x] Kurulum, klasörler ve mevcut script açıklamaları README'de.
- [x] Ortak indirme/hazırlama başlangıcı ve Naive Bayes eğitim scripti mevcut.
- [x] Ham ve hazırlanmış veri dosyaları, kaynak lisansı ve atıfla teslim paketinde.
- [x] Proje kodu için MIT LICENSE, veri için upstream CC BY 4.0 lisansı ve atıf mevcut.
- [x] Kişisel katkı dosyası biçimi ve şablonu contributions/README.md'de.
- [x] Naive Bayes alpha benchmark'ı tek komutla tekrar çalıştırılabilir.
- [x] Naive Bayes teknik açıklaması ve üç gerçek validation hatasının yorumu mevcut.
- [x] 2. kişinin kendi çalışması için devir adımları docs/HANDOFF_DATA.md'de.
- [x] PR #2, #3 ve #5 main'e alındı; ortak README ve çalıştırma bağlantıları güncellendi.
- [x] PR #7 ve #8 ana dala alındı; son benchmark revizyonundaki 15 validation çalıştırmasının skorları ve istatistikleri doğrulandı.
- [x] Birleşik kodda 33 yerel kontrol geçti; Naive Bayes validation çalışması doğrulandı.
- [x] Logistic Regression'ın 15 validation deneyi ve raporlanan skorları entegrasyonda doğrulandı.
- [x] Linear SVM'nin 10 validation deneyi, veri hash'leri ve hata katkıları incelemede doğrulandı.
- [x] Üç modelin seçilen ayarları aynı veriyle validation üzerinde çalıştırıldı; tek mesaj tahmin komutu NB ve SVM ile doğrulandı.
- [x] Beş üyenin kendi gerçek katkı dosyaları ve GitHub kanıtları mevcut; repo sahibinin dosyası bu teslim PR'ında eklendi.
- [x] 2. kişi: veri analizi, veri ön işleme pipeline dokümantasyonu, özellik deneyleri ve katkı dosyası.
- [x] 3. kişi: Logistic Regression kodu, validation deneyleri ve katkı dosyası.
- [x] Repo sahibi: mevcut veri ve ortak unigram + bigram temsilini koruma kararını deney protokolünde kaydetme.
- [x] 3. kişi: Macro F1 ve train/validation örtüşme açıklamalarını PR #7 ile düzeltme.
- [x] 4. kişi: Linear SVM kodu, validation deneyleri ve katkı dosyası.
- [ ] 4. kişi: teknik notta Macro F1 farkını sabit mesaj sayısına çeviren ifadeyi düzeltme.
- [x] Repo sahibi: NB ayarlarını testten önce ayrı commit ile sabitleme ve nihai test çıktısını yayımlama (`results/NAIVE_BAYES_TEST.md`).
- [ ] 3. ve 4. kişi: kendi nihai ayar kayıtlarını ve test çıktılarını teslim etme.
- [x] 5. kişi: ortak benchmark, metrik gerekçesi, validation karşılaştırması ve grafikler (`results/MODEL_COMPARISON.md`).
- [ ] 5. kişi: nihai test karşılaştırması ve PPTX sunum.
- [x] Bu entegrasyonun repo kontrolü: testler ve NB validation geçti; nihai NB tahminleri/metrikleri doğrulandı, geçici araçlar ve modeller Git dışında.

Kişisel raporların ve başkalarının deneylerinin ilgili üye tarafından hazırlanması
gerekir. Test seti ayar seçiminde kullanılmaz; örtüşme raporlama sözleşmesi
[EXPERIMENTS.md](EXPERIMENTS.md) içinde tanımlıdır.

## Benchmark teslim sözleşmesi

Her model sahibi aynı veri kimliğiyle değerlendirme bölümü, model/özellik ayarları,
komut, accuracy, macro F1, eğitim/tahmin süreleri ve tahmin dosyasını teslim eder.
Baseline ve ayarlanmış sürümler ayrı adlandırılır. Ortak değerlendirme sorumlusu
metrik seçimini sınıf dağılımı ve görev amacıyla gerekçelendirir; ana sıralama
metriğimiz macro F1, destekleyici metriğimiz accuracy'dir.

## Sunum ve ödev yükleme

- [ ] PPTX sunum hazır ve prova edilmiş.
- [ ] Sunucu okumadan teknik açıklama yapabiliyor; hedef süre 2 dakikanın altı.
- [ ] Ödev sistemine hem PPTX hem public GitHub URL'si yüklenmiş.

Sunum puanında teknik derinlik %50, akıcılık %25, süre %25 ağırlıktadır.
Mailde 2 dakikanın altı 6; 2–3 dakika 5; 3:00–3:15 arası 3;
3:30'da 0 puan ve sunumun kesilmesi belirtilmiştir. Soru cevap ayrı ağırlık
taşımaz; teknik derinlik puanını artırabilir veya azaltabilir.
