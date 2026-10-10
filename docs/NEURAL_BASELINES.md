# Neural baseline kapsamı ve çalıştırma

10 Ekim'de paylaşılan yeni hoca yanıtı, slaytlarda adı geçen tüm yöntemlerin
baseline olmasını ve bunların üzerine ekip katkısı eklenmesini istiyor.
Bu yanıt, önceki kapsam belirsizliğini giderir. Yeni çalışma repo sahibinin
açık kullanıcı talebiyle `feature/neural-baselines` dalında başlatıldı;
mevcut veri hazırlama, LR/SVM ve 5. kişinin rapor kodları değiştirilmedi.

## Slayt eşleştirmesi

Yerel kaynaklar: `COE025 - Natural Language Processing - W02.pptx` ve
`NLP_Week3_Slides.pptx`. Slayt resimleri de incelendi; yalnızca PPTX metin
çıkarımı yapılmadı. Ders dosyaları repoda yeniden yayımlanmaz.

| Kaynak | Yöntem / temsil | Repo karşılığı |
| --- | --- | --- |
| W02, 15–23 | Naive Bayes, Logistic Regression, SVM | Mevcut üç klasik pipeline ve nihai test karşılaştırması |
| W03, 6–11 | GloVe, Word2Vec (Skip-gram / CBOW), FastText | GloVe: `mean` + sabit vektörler; `train_embeddings`: `word2vec_cbow`, `word2vec_skipgram`, `fasttext` |
| W03, 13–17 | CNN | `--model cnn`, 3/4/5 kelimelik filtreler ve maskeli max pooling |
| W03, 19–22 | RNN | `--model rnn`, ayrı vanilla RNN çalıştırması |
| W03, 23–29 | LSTM ve BiLSTM | `--model lstm` ve `--model bilstm`, ayrı çalıştırmalar |
| W03, 31–37 | BERT / Transformer; belirli model varyantları | `bert`, `distilbert`, `roberta`, `albert` |

Slayt 37'deki belirli modeller `bert-base-uncased`, `distilbert-base-uncased`,
`roberta-base`, `albert-base-v2` olarak yazılıdır. Hepsi
`configs/neural_models.json` içinde kaynak repository, sabit commit ve lisansla
kaydedilmiştir. Eğitim komutunun bir modeli desteklemesi, onun tam deneyinin
tamamlandığı anlamına gelmez; durum ve gerçek skorlar
[NEURAL_VALIDATION.md](../results/NEURAL_VALIDATION.md) içinde tutulur.

## Ortak deney protokolü

- Hazırlanmış 8.499 train / 1.500 validation / 3.080 test ayrımı korunur.
- Klasik modellerde TF-IDF unigram + bigram kalır. Sinir ağları token dizileri
  kullanır; aynı TF-IDF temsilini kullanmak zorunda değildir. Karşılaştırma aynı
  mesaj kimlikleri, kategori listesi ve metriklerle yapılır.
- Kelime sözlüğü sadece train'den oluşturulur. RNN/CNN/LSTM'de yeni kelimeler
  UNK olur; Word2Vec'te atlanır; FastText'te öğrenilmiş karakter n-gramlarıyla
  vektör üretilir ve heldout kelime sözlüğe eklenmez. Transformer tokenizer'ı
  sabit ön eğitim kaynağından gelir.
- RNN/LSTM'de gerçek dizi uzunlukları kullanılır; CNN ve ortalamada PAD konumları
  sınıflandırma temsilini değiştirmez.
- `seed=42`, AdamW, gradient clipping=1, weight decay=0.01. Varsayılan kelime
  modellerinde 100 boyutlu öğrenilen embedding, hidden=128, dropout=0.3,
  batch=64, lr=0.001; en fazla 15 epoch, patience=3. En iyi checkpoint sadece
  validation macro F1 ile seçilir; eşitlikte önceki checkpoint korunur.
- Transformer'larda varsayılan batch=16, lr=0.00002, lineer warmup/decay vardır.
  Epoch bütçesi komutta açıkça yazılır. Ön eğitim ağırlıkları ince ayarlanır;
  77 sınıflı başlık BANKING77 train'den öğrenilir.
- Süreler açıkça ayrılır: `fit_seconds`, epoch validation kontrollerini ve
  checkpoint kaydını da içerir; indirme ve başlangıç tokenizasyonunu içermez.
  `tokenization_seconds` ayrı yazılır. GPU tahmin zamanlaması senkronize edilir.
  Bu süreler mevcut TF-IDF tablosundaki sürelerle doğrudan eşdeğer değildir.
- `--max-train-rows` / `--max-validation-rows` yalnızca geliştirme kontrolüdür;
  bu çıktılar final seçim protokolü olarak sabitlenemez.
- Resmî test için önce tam validation çalıştırması sabitlenip protokol commit
  edilir. Test alt komutu veri/kod/checkpoint/tokenizer hash'lerini denetler ve
  yeniden eğitmeden seçilmiş checkpoint'i değerlendirir. Train ile birebir
  örtüşmeyen altküme ayrıca raporlanır; yakın tekrarların tamamen giderildiği
  iddia edilmez.
- Klasik modellerin test sonuçları bu kapsam genişlemesinden önce görülmüştür.
  Yeni modellerin ayarları validation'da seçilir; tüm proje için testin hiç
  görülmediği veya tamamen kör bir araştırma yapıldığı iddia edilmez.

## Kurulum

Mevcut klasik kurulum korunur. Windows CPU doğrulaması için:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-neural.txt
.\.venv\Scripts\python.exe -m banking77.neural_preflight
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

PyTorch CPU paketi RX 7800 XT'yi kullanmaz. 10 Ekim'de kullanıcının onayıyla
WSL 3.0.1 kuruldu ve VirtualMachinePlatform etkinleştirildi. Kurulum çıktısı
Windows'un yeniden başlatılmasını istiyor; Ubuntu başlangıcı ve GPU ortamı
henüz doğrulanmadı. Mevcut preflight CPU forward/backward işlemini doğruladı.
AMD'nin [ROCm 7.2.1 WSL matrisi](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/wsl/wsl_compatibility.html)
RX 7800 XT'yi desteklenen GPU'lar arasında gösterir. Windows üzerindeki GPU
kurulumu için önce WSL2 + Ubuntu gerekir.

Yönetici PowerShell'de, açık işler kaydedildikten sonra:

```powershell
.\scripts\Install-NeuralWSL.ps1
```

Bu script `wsl --install --distribution Ubuntu-22.04 --no-launch` çalıştırır;
bilgisayarı kendisi yeniden başlatmaz. Windows özelliklerini etkinleştirme,
yeniden başlatma ve ilk Ubuntu kullanıcı kurulumu gerekebilir.
[Microsoft kurulum açıklaması](https://learn.microsoft.com/en-us/windows/wsl/install),
[komut seçenekleri](https://learn.microsoft.com/en-us/windows/wsl/basic-commands#install).

Ubuntu hazır olduktan sonra, sürücü/runtime ve PyTorch paketi birlikte
[AMD'nin WSL kurulum yönergesine](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/wsl/install-pytorch.html)
göre kurulmalıdır. ROCm paketleri normal Windows CPU sanal ortamına kurulmaz.
Linux'ta ayrı Python 3.12 sanal ortamı oluşturulur, klasik bağımlılıklar ve
`requirements-neural.txt` yüklenir. Donanıma uygun AMD PyTorch paketi önce
kurulur; pip'in bunu CPU paketiyle değiştirmediği kontrol edilir.

GPU eğitimi başlamadan önce Linux ortamında:

```bash
python -m banking77.neural_preflight --require-gpu
```

GPU adı, ROCm/HIP sürümü ve gerçek forward/backward doğrulaması görülmeden
GPU eğitiminin çalıştığı iddia edilmez. ROCm, PyTorch'ta `cuda` aygıt arayüzünü
kullandığından eğitim komutunda `--device cuda` seçilir.

## Eğitim ve nihai değerlendirme

GloVe'nin kaynak arşivi ve 100d dosyası `configs/glove.json` içindeki SHA-256
değerleriyle doğrulanır; yalnızca train sözlüğüne uyan vektörler ayrılır:

```powershell
.\.venv\Scripts\python.exe -m banking77.prepare_embeddings
.\.venv\Scripts\python.exe -m banking77.train_neural train --model mean --embedding-file data/embeddings/glove.6B.100d.train.txt --freeze-embeddings --epochs 50 --patience 5 --learning-rate 0.01 --device cpu
```

GloVe çalışmasında yalnızca sınıflandırma başlığı öğrenilir. Kaynak embedding
arşivi yaklaşık 822 MB'dır; arşiv ve vektörler Git dışında tutulur.

Word2Vec ve FastText burada dışarıdan ön eğitimli değildir. Temsil 8.499 train
mesajından öğrenilir: 100 boyut, window=5, min_count=1, negative=5, 20 epoch,
tek worker, seed=42 ve sabit hash fonksiyonu. Mesaj vektörü kelime vektörlerinin
ortalamasıdır; başlık sabit `LogisticRegression(C=1, solver=lbfgs, max_iter=2000)`.
Bu başlık, 3. kişinin TF-IDF + LR çalışmasından ayrı bir temsil baseline'ıdır.
FastText CBOW kullanır; karakter uzunlukları 3–6 ve bucket=20.000'dir.
Tamamen bilinmeyen Word2Vec mesajı sıfır vektör alır; FastText altkelimeleri
kullanabilir. GloVe ve Transformer dış ön eğitim kullanırken bu üç temsilin
train'den öğrenildiği, sonuç yorumunda açıkça belirtilmelidir.

```powershell
.\.venv\Scripts\python.exe -m banking77.train_embeddings train --model word2vec_cbow
.\.venv\Scripts\python.exe -m banking77.train_embeddings train --model word2vec_skipgram
.\.venv\Scripts\python.exe -m banking77.train_embeddings train --model fasttext
```

Bu üç yöntemde sabit bütçe kullanılır; validation'da embedding ayarları taranmaz.
`train_embeddings freeze` / `test`, neural komutlarıyla aynı parametre adlarını
kullanır; joblib checkpoint'i ve veri/kod kimliği testten önce sabitlenir.

```powershell
.\.venv\Scripts\python.exe -m banking77.train_neural train --model cnn --epochs 15 --device cpu
.\.venv\Scripts\python.exe -m banking77.train_neural train --model distilbert --epochs 3 --device cuda
```

Komutlar `results/runs/<run_id>/` altında standart `metrics.json`,
`predictions.csv`, `classification_report.json`, `confusion_matrix.csv` ve
gelecekteki ensemble için kategori/mesaj sırası belli `probabilities.npz` üretir.
Checkpoint, sözlük/tokenizer ve metadata `artifacts/<run_id>/` altındadır.
İndirme önbelleği, model ağırlıkları ve tekrarlı çıktılar Git'e eklenmez.

Validation çalıştırması seçilip tamamlandıktan sonra:

```powershell
.\.venv\Scripts\python.exe -m banking77.train_neural freeze --validation-run RUN_ID --output results/cnn_final_protocol.json
git add results/cnn_final_protocol.json
git commit -m "Freeze CNN before official test"
.\.venv\Scripts\python.exe -m banking77.train_neural test --protocol results/cnn_final_protocol.json --device cpu
```

Başka bilgisayarda checkpoint yeniden üretilecekse seçimin yeniden yapıldığı
ve donanım farkının bit düzeyinde eşitliği garanti etmediği açıkça yazılır.
Bu protokol dosyası, belirli checkpoint'in hash'ini sabitler; farklı ağırlıkla
sessizce aynı protokol altında test çalıştırılmaz.

## Ekip katkısı: NB + CNN soft voting

`p(y|x) = (1-w) p_NB(y|x) + w p_CNN(y|x)`; ardından en büyük olasılığa sahip
etiket seçilir. NB, önceki protokoldeki alpha=0.05 ve unigram + bigram ayarını;
CNN, kendi validation checkpoint'ini korur. `w=0,0.1,...,1` arasından validation
macro F1 en yüksek olan seçilir; eşitlikte ilk ağırlık korunur. `w=0` yalnız NB,
`w=1` yalnız CNN ablation'ıdır. Olasılıklar ayrıca kalibre edilmediği için ağırlık
yalnız model önemini ölçen bir katsayı olarak yorumlanmamalıdır.

```powershell
.\.venv\Scripts\python.exe -m banking77.train_ensemble train --cnn-validation-run CNN_RUN_ID
.\.venv\Scripts\python.exe -m banking77.train_ensemble freeze --validation-run ENSEMBLE_RUN_ID --cnn-protocol results/cnn_final_protocol.json --output results/nb_cnn_final_protocol.json
# Protokolleri önce commit edin.
.\.venv\Scripts\python.exe -m banking77.train_ensemble test --protocol results/nb_cnn_final_protocol.json --device cpu
```

Son komut dondurulmuş CNN checkpoint'ini bir kez test eder ve onun sonucunu
ayrıca kaydeder; aynı checkpoint'i CNN için tekrar çalıştırmak gerekmez.
Olasılıkların satır kimlikleri ve kategori sırası birleştirilmeden önce doğrulanır.
Validation ağırlık tablosu [raporda](../results/NEURAL_VALIDATION.md) bulunur.
Bu, proje için geliştirilmiş bir birleştirme deneyidir; yeni bir mimari veya
literatürde ilk kez önerilen bir algoritma olduğu iddia edilmez.

## Lisanslar ve kalan işler

Kod projenin MIT lisansı altındadır. BANKING77 verisinin CC BY 4.0 lisansı ve
atıfı geçerlidir. Transformer kaynak lisansları registry'de, kaynak model
kartlarında tam koşullarıyla yer alır. GloVe vektörleri Stanford'un
[PDDL 1.0 açıklaması](https://nlp.stanford.edu/projects/glove/) kapsamında
kullanılır; GloVe kodunun Apache lisansıyla vektörlerin lisansı karıştırılmaz.

Bu dosyadaki vanilla RNN/CNN/LSTM/Transformer'lar ders baseline'larıdır.
NB + CNN katkısının validation deneyi ve iki uç ablation'ı tamamlandı.
Dört Transformer'ın tam eğitimleri bekliyor; kısa geliştirme kontrolü benchmark
sonucu olarak sunulmaz. Tek seed sonuçları genelleme veya istatistiksel üstünlük
kanıtı değildir. Yeni modellerin dahil olduğu eşleştirilmiş karşılaştırma henüz yapılmadı.
Ortak grafikler, genişletilmiş karşılaştırma ve PPTX 5. kişinin sorumluluğundadır.
