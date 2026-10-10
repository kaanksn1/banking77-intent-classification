# Deney planı

Problem: BANKING77'de kısa İngilizce müşteri mesajını 77 talepten birine sınıflandırma.

## Ortak protokol

1. Resmî eğitim kümesindeki boş kayıtları ve normalize edilmiş metin tekrarlarını kontrol et.
2. Aynı normalize metinde farklı etiket varsa işlemi durdur ve incele.
3. Eğitim kümesinden stratified %15 validation ayır (`seed=42`).
4. Resmî 3.080 satırlık test kümesini koru; ilk geliştirme aşamasında test çalıştırma.
5. Hazırlama raporundaki train/test ve validation/test metin örtüşmelerini raporla.
   Bunlar sıfır değilse resmî testin tamamen temiz bir holdout olduğu iddia edilmemeli.
   Naive Bayes çalıştırıcısı train ile örtüşmeyen altküme sonuçlarını ayrıca verir.
   Diğer model sahipleri de aynı raporlama kuralını kendi kodlarına uygulamalıdır.

Bu yerel protokol eğitim tekrarlarını kaldırıp validation ayırdığı için,
resmî 10.003 eğitim örneğinin tamamıyla eğitilen yayınlarla koşullar birebir aynı değildir.

## Küçük deney bütçesi

Aşağıdaki tablo ekip deney kapsamını gösterir. Veri/özellik çalışması, Naive Bayes,
Logistic Regression ve Linear SVM kendi sorumlularının PR'larıyla `main` içine alınmıştır.
Modellerin ortak validation karşılaştırması [MODEL_COMPARISON.md](../results/MODEL_COMPARISON.md)
içindedir (5. kişi).

| Yöntem | Başlangıç | Validation deney kapsamı |
| --- | --- | --- |
| TF-IDF | unigram + bigram, sublinear TF | `(1,1)` ile `(1,2)` karşılaştırması |
| Multinomial Naive Bayes | alpha=1.0 | alpha: 0.01, 0.05, 0.1, 0.5, 1.0 |
| Logistic Regression | lbfgs, C=1.0 | C: 0.1, 1.0, 10.0, 100.0, 1000.0; lbfgs, saga, liblinear-ovr |
| Linear SVM | squared_hinge, C=1.0 | C: 0.01, 0.1, 1.0, 10.0, 100.0; squared_hinge, hinge |

Tamamlanan çalışmalar: [özellik deneyi](../results/FEATURE_EXPERIMENTS.md),
[NB alpha deneyi](../results/NAIVE_BAYES_ALPHA.md),
[LR C/solver deneyi](../results/LOGISTIC_REGRESSION_C.md),
[SVM C/loss deneyi](../results/LINEAR_SVM_C.md).
Üç modelde yeniden ayarlanarak yapılan unigram / bigram karşılaştırması
[FEATURE_COMPARISON.md](../results/FEATURE_COMPARISON.md) içindedir.

10 Ekim teslim entegrasyonunda mevcut ortak TF-IDF temsili korunur:
unigram + bigram (`ngram_range=(1, 2)`), `sublinear_tf=True`; veri bölümleri
ve `seed=42` değişmez. Bu, mevcut üç modelin validation karşılaştırmasıyla
aynı temsildir. Özellik raporundaki unigram önerisi alpha=1.0'lı NB deneyine
aittir; tüm modeller için en iyi temsil bulunduğu iddia edilmez. Sonradan gelen
özellik karşılaştırması NB'nin test öncesinde sabitlenen ayarını değiştirmez.

Önce ortak özellik ayarıyla modelleri karşılaştırın. Özellik deneylerini ayrı
tabloda gösterin. Ana model seçme metriği macro F1; accuracy de raporlanır.
Sonuç uydurmayın: başlangıç değerleri veya literature skorları bizim sonuçlarımız değildir.

## Son değerlendirme

Her model için validation ile ayarları dondurun. Naive Bayes test çalıştırıcısı
hazırlanmış train bölümüyle eğitir; validation eğitim verisine eklenmez.
Testi ayarları dondurduktan sonra çalıştırın ve final raporunda protokolü yazın.
Her model için accuracy, macro F1, eğitim süresi ve tahmin süresi; en çok karışan
üç kategori çifti ve örnek yanlış tahminler raporlanmalıdır.

### Naive Bayes için test öncesi kayıt

Repo sahibi NB ayarını `alpha=0.05`, `ngram_max=2`, `sublinear_tf=True`
olarak sabitlemiştir. Alpha, beş aday arasındaki en yüksek validation macro F1
ile seçilmiştir; test sonucu seçimde kullanılmaz.
[Sabitlenen protokol](../results/naive_bayes_final_protocol.json), test
çalıştırılmadan önce ayrı commit'e alınır. Veri dosyalarının hash'leri,
eğitim kodunun kimliği ve validation seçim kanıtı bu kayıttadır.

NB yalnızca 8.499 train mesajıyla eğitilir; 1.500 validation mesajı eğitime
eklenmez. Tam resmî test ve normalize anahtara göre train ile örtüşmeyen
altküme ayrı raporlanır. Bu altküme, yakın tekrarların tamamen giderildiği
anlamına gelmez. Test sonucu görüldükten sonra ayarlar değiştirilmez.

LR ve SVM'nin sabitlenmiş ayarlarla nihai testleri 5. kişinin ortak benchmark'ında
çalıştırılmış ve PR #11 ile teslim edilmiştir. Model uygulamaları 3. ve 4. kişinin;
karşılaştırma kodu ve raporu 5. kişinin katkısıdır. Repo sahibi yalnız entegrasyon
için bu çıktıları yeniden üretip doğrulamıştır.
Yeni veri veya JEV/Laya/AnyJev ekleme planı, repo sahibinin kararıyla iptal edilmiştir.

## Son e-postayla genişleyen baseline kapsamı

Hocanın 10 Ekim'de paylaşılan açıklaması tüm ders yöntemlerini baseline olarak
istiyor. [Slayt eşleştirmesi ve ayrıntılı neural protokol](NEURAL_BASELINES.md)
Word2Vec CBOW/Skip-gram, GloVe, FastText, CNN, RNN, LSTM, BiLSTM ve slayt 37'deki
dört Transformer modelini kapsar. Sekiz kelime/sinir ağı baseline'ının
[validation sonuçları](../results/NEURAL_VALIDATION.md) ve dört Transformer'ın
[tam validation sonuçları](../results/TRANSFORMER_VALIDATION.md) tamamlandı.
Vanilla ve ayarlanmış modeller yeni mimari diye sunulmaz.

Veri ve kategori listesi değişmez. Kelime sözlüğü ve train'den öğrenilen embedding'ler
heldout mesajları kullanmaz; dış ön eğitimli GloVe/Transformer kaynakları ayrıca
kaydedilir. Transformer ince ayarı 8.499 train mesajında yapılır.
Kelime modellerinde checkpoint validation macro F1 ile seçilir; Word2Vec/FastText
sabit 20 epoch bütçesi kullanır. Yeni model protokolleri testten önce ayrı commit'e
alınır; seçilmiş checkpoint'ler yeniden eğitilmeden test edilir.
Transformer'larda ilk DistilBERT/BERT validation eğrileri üç epoch'ta hâlâ
iyileştiği için tüm dört modelin nihai bütçesi beş epoch olarak belirlendi.
[Karar kaydı](../results/transformer_budget_decision.json) testten önce oluşturuldu;
nihai dört deney ön eğitimli ağırlıklardan yeniden başlatıldı. Tek seed ve sınırlı
eğitim bütçesi optimum performans veya yakınsama garantisi değildir.
Dört Transformer'ın seçilmiş checkpoint protokolleri resmî testten önce
`1e4ff66` commit'iyle sabitlendi ve GitHub'a pushlandı. Yeniden eğitim veya testle
ayar seçimi yapılmadan [nihai test değerlendirmesi](../results/TRANSFORMER_TEST.md)
tamamlandı; tüm tahminler ve standart çıktılar `results/transformer_test/` içindedir.

Repo sahibinin proje katkısı NB + CNN soft voting'dir. 11 ağırlık yalnız validation
macro F1 ile karşılaştırılır; yalnız NB ve yalnız CNN uçları ablation olarak korunur.
Bu katkı yeni bir araştırma algoritması iddiası taşımaz. Tek seed ve aynı validation'da
birden fazla seçim yapılması, sonuçların belirsizliğidir. Klasik test sonuçları kapsam
genişlemesinden önce görülmüştür; tüm araştırmanın kör test kullandığı iddia edilmez.
Yeni modellerin ortak karşılaştırması ve sunum 5. kişide kalır.
