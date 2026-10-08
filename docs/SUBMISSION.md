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
  olarak kabul edilmiştir. Eklenirse sorumlusu ayrıca belirlenmeli; kullanılan
  model/sürüm, girdi tanımları, değerlendirme örnekleri ve çalıştırma koşulları
  kaydedilmelidir. Ek puan garanti değildir.

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
- [x] Yedi yerel kontrol geçti; Naive Bayes validation benchmark'ı çalıştırıldı.
- [ ] Beş üyenin kendi gerçek katkı dosyalarını, commit ve PR bağlantılarını eklemesi.
- [ ] 2. kişi: veri analizi, veri ön işleme pipeline dokümantasyonu ve özellik deneyleri.
- [ ] 3. kişi: Logistic Regression kodu, deneyleri ve katkı dosyası.
- [ ] 4. kişi: Linear SVM kodu, deneyleri ve katkı dosyası.
- [ ] Her model sahibi: validation ile ayarlarını dondurma, sonra nihai test çıktısı.
- [ ] 5. kişi: ortak benchmark, metrik gerekçesi, karşılaştırma ve sunum.
- [ ] Son repo kontrolü: komutlar çalışıyor; geçici ve gereksiz dosyalar dışarıda.

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
