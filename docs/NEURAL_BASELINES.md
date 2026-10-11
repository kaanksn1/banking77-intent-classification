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
kaydedilmiştir. Kelime/embedding baseline'larının gerçek skorları
[NEURAL_VALIDATION.md](../results/NEURAL_VALIDATION.md), dört Transformer'ın
tam validation sonuçları [TRANSFORMER_VALIDATION.md](../results/TRANSFORMER_VALIDATION.md)
içinde tutulur. Dört Transformer'ın resmî testleri de tamamlandı;
[nihai test raporu](../results/TRANSFORMER_TEST.md) ayrı tutulur.

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

PyTorch CPU paketi RX 7800 XT'yi kullanmaz. 10 Ekim'de Windows yeniden
başlatıldıktan sonra WSL2 üzerinde Ubuntu 24.04.5 kuruldu; Python 3.12.3,
`6.18.40.1` WSL2 çekirdeği ve `/dev/dxg` aygıtı doğrulandı. WSL uygulamasının
sürümü 3.0.1'dir; dağıtımın çalışma modu WSL2'dir. Mevcut Windows sürücüsü
`32.0.32015.2008`, AMD'nin [Adrenalin 26.9.2 sürümüdür](https://www.amd.com/en/resources/support-articles/release-notes/RN-RAD-WIN-26-9-2.html)
ve korundu. Linux GPU kurulumu tamamlandı: ROCm paketi
`7.2.1.70201-81~24.04`, ROCDXG `1.2.0`, PyTorch
`2.9.1+rocm7.2.1.gitff65f5bc`, HIP `7.2.53211-e1a6bc5663`.
RX 7800 XT (`gfx1101`, yaklaşık 16 GB) üzerinde GPU matris forward/backward
kontrolü hem root hem normal `serda` kullanıcısıyla geçti. Linux'ta 52 test ve
Naive Bayes validation kontrolü geçti: accuracy `0.862`, macro F1
`0.8528894646`. Dört Transformer'ın ortak beş epoch bütçesiyle tam validation
deneyleri tamamlandı; [sonuç raporu](../results/TRANSFORMER_VALIDATION.md) ve
[JSON kaydı](../results/transformer_validation.json) hazırdır. Dört modelin
[resmî test değerlendirmesi](../results/TRANSFORMER_TEST.md) de tamamlandı.
Küçük GPU geliştirme kontrolü tam deney yerine geçmez.

[AMD'nin ROCm 7.2.1 WSL yönergesi](https://rocm.docs.amd.com/projects/radeon-ryzen/en/docs-7.2.1/docs/install/installrad/wsl/howto_wsl.html)
ROCDXG köprüsünü kullanır. Bu yöntem Adrenalin 26.2.2 ile sunulmuştur;
ROCDXG, Windows sürücüsü güncellemelerinden bağımsız geliştirilir.
[ROCDXG 1.2.0 uyumluluk tablosu](https://github.com/ROCm/librocdxg/blob/v1.2.0/README.md#wsl-compatiblity-matrix)
ROCm 7.2.x, Ubuntu 24.04/22.04 ve RX 7800 XT desteğini listeler.
Python 3.12 ortamı için burada Ubuntu 24.04 kullanılır. WSL, Windows ekran
sürücüsünü kullanır; Linux çekirdeği veya `amdgpu-dkms` kurulmaz.

`scripts/setup_rocm_wsl.sh`, root olarak Ubuntu paketlerini kurmak içindir.
Script, [sürümü sabit ROCm 7.2.1 deposunu](https://rocm.docs.amd.com/projects/install-on-linux/en/docs-7.2.1/install/quick-start.html)
`amdgpu-install_7.2.1.70201-1_all.deb` ile ekler; aday paketin `7.2.1.*`
olduğunu denetleyip bu sürümün yalnız ROCm kullanıcı alanını kurar.
`umask 022`, root'un oluşturduğu Linux ortamının normal kullanıcı tarafından
okunup çalıştırılmasını sağlar. Eski ROCm 7.2 WSL kurulumundaki
`--usecase=wsl,rocm` yolu bu 7.2.1 kurulumu için kullanılmaz.
[ROCDXG çalışma zamanı paketi](https://github.com/ROCm/librocdxg/releases/tag/v1.2.0)
`rocdxg-roct_1.2.0_amd64.deb`, [yayımlanmış SHA-256](https://github.com/ROCm/librocdxg/releases/expanded_assets/v1.2.0)
`3ed9526719290cd8f590150dad8ea0f234fa779bea6a4c9a8449d7ae6b8cfb6e`
ile doğrulanır. Script, `requirements-lock.txt` içindeki klasik bağımlılıkları,
[AMD'nin Python 3.12 PyTorch 2.9.1 + ROCm 7.2.1 ve Triton wheel'lerini](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/native_linux/install-pytorch.html)
ve `requirements-neural.txt` bağımlılıklarını ayrı `/opt/banking77-venv`
ortamına yükler; sonunda `pip check` ve GPU preflight çalıştırır.
Wheel'lerin yerel SHA-256 kayıtları indirme kimliğidir; bağımsız yayımlanmış
üretici checksum'ı olarak sunulmaz. Windows `.venv` ortamı CPU içindir.

Kurulum komutu, repo kökünü açıkça seçerek PowerShell'den çalıştırılır:

```powershell
wsl.exe --distribution Ubuntu-24.04 --user root --cd '/mnt/c/Users/serda/OneDrive/Masaüstü/nlp' -- bash scripts/setup_rocm_wsl.sh
```

ROCm 7.2.1'de **her etkileşimsiz preflight, eğitim ve test süreci**
`HSA_ENABLE_DXG_DETECTION=1` almalıdır. Kurulum script'indeki `export`, daha
sonra açılan ayrı WSL süreçlerine aktarılmaz; yalnız `.bashrc` içine eklemek
de etkileşimsiz komutlar için yeterli değildir. Aşağıdaki komutlarda `env` bu
değeri doğrudan ilgili sürece verir. Eğitim için Ubuntu'da oluşturulmuş
normal `serda` kullanıcısı kullanılır; paket kurulumu root ile yapılır.

```powershell
wsl.exe --distribution Ubuntu-24.04 --user serda --cd '/mnt/c/Users/serda/OneDrive/Masaüstü/nlp' -- env HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo
wsl.exe --distribution Ubuntu-24.04 --user serda --cd '/mnt/c/Users/serda/OneDrive/Masaüstü/nlp' -- env HSA_ENABLE_DXG_DETECTION=1 /opt/banking77-venv/bin/python -m banking77.neural_preflight --require-gpu
```

Bu ortamda `rocminfo`, RX 7800 XT'yi (`gfx1101`) gösterdi; preflight GPU adı,
ROCm/HIP sürümü ve gerçek matris forward/backward kontrolünü doğruladı.
Yeni bir kurulumda aynı kontrol yeniden yapılır. Bu kontrol, Transformer'ın
tam eğitim/validation işleminin tamamlandığını göstermez. ROCm, PyTorch'ta
`cuda` aygıt arayüzünü kullandığından eğitimde `--device cuda` seçilir.

Dört Transformer için nihai ortak validation bütçesi FP32 (`--amp`
verilmez), 5 epoch, patience=3, batch=16, lr=0.00002, max_length=64 ve
seed=42'dir. Aşağıdaki tam validation komutu, GPU altyapısı doğrulanmış Linux
ortamı içindir; diğer modellerde yalnız `--model` değeri `bert`, `roberta` veya
`albert` olur:

```powershell
wsl.exe --distribution Ubuntu-24.04 --user serda --cd '/mnt/c/Users/serda/OneDrive/Masaüstü/nlp' -- env HSA_ENABLE_DXG_DETECTION=1 /opt/banking77-venv/bin/python -m banking77.train_neural train --model distilbert --epochs 5 --patience 3 --batch-size 16 --learning-rate 0.00002 --max-length 64 --seed 42 --device cuda
```

İlk tam DistilBERT ve BERT denemeleri üç epoch'ta hâlâ iyileşiyordu.
[Validation bütçesi kararı](../results/transformer_budget_decision.json) bu nedenle
dört model için ortak beş epoch bütçesine geçişi kaydeder. Tüm modeller ön
eğitimli ağırlıklardan yeniden başlatıldı; her model için ayrı üç/beş epoch
seçimi yapılmadı. Beş epoch yakınsama garantisi değildir.

Tam validation deneyleri dört model için de tamamlandı; seçilen checkpoint'ler
validation macro F1'e göre belirlendi ve her modelde beşinci epoch seçildi:

| Model | Validation accuracy (%) | Validation macro F1 |
| --- | ---: | ---: |
| BERT-base-uncased | 90.60 | 0.9010 |
| DistilBERT-base-uncased | 89.87 | 0.8961 |
| RoBERTa-base | 92.33 | 0.9264 |
| ALBERT-base-v2 | 89.87 | 0.8989 |

[Tam rapor](../results/TRANSFORMER_VALIDATION.md) ayarları, süreleri ve
sınırlılıkları; [JSON kaydı](../results/transformer_validation.json) tam metrikleri,
komutları, ortam ve çıktı kimliklerini içerir. Dört seçilmiş checkpoint'in
`<model>_final_protocol.json` dosyaları
[testten önceki `1e4ff66` commit'i](https://github.com/kaanksn1/banking77-intent-classification/commit/1e4ff66d49e7d184d3dbe1548fa7b038b79833c2)
ile sabitlendi. Ardından checkpoint'ler yeniden eğitilmeden 3.080 resmî test
mesajında değerlendirildi; validation eğitime eklenmedi:

| Model | Test accuracy (%) | Test macro F1 |
| --- | ---: | ---: |
| BERT-base-uncased | 90.58 | 0.9016 |
| DistilBERT-base-uncased | 89.61 | 0.8961 |
| RoBERTa-base | 93.02 | 0.9301 |
| ALBERT-base-v2 | 90.29 | 0.9027 |

[Nihai test raporu](../results/TRANSFORMER_TEST.md) gerçek hata örneklerini ve
train ile birebir örtüşmeyen altküme sonuçlarını;
[test JSON kaydı](../results/transformer_test.json) metrikleri, süreleri ve çıktı
hash'lerini içerir. Her modelin dört standart çıktısı `results/transformer_test/<model>/`
altındadır. Test sonuçları ayar veya checkpoint seçmek için kullanılmadı.

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
```

Transformer GPU komutu yukarıdaki WSL/Linux ortamında çalıştırılır;
Windows CPU `.venv` ortamına `--device cuda` vermek GPU desteği eklemez.

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

## Literatür bağlamı

Aşağıdaki değerler BANKING77 için yayımlanmış **test accuracy** sonuçlarıdır.
Validation accuracy veya macro F1 ile aynı sütunda karşılaştırılmaz.

| Birincil kaynak | Belirli yöntem | Test accuracy (%) | Karşılaştırma sınırı |
| --- | --- | ---: | --- |
| [Casanueva vd., 2020, tablo 3](https://aclanthology.org/2020.nlp4convai-1.5.pdf#page=4) | BERT-TUNED: cased BERT-Large | 93.66 | 10.003 eğitim örneğinin tamamı; ayrı validation yok. Bizim BERT-base modelimizden daha büyük. |
| [Mehri vd., 2020, DialoGLUE tablo 1](https://arxiv.org/pdf/2009.13570#page=6) | BERT-base | 93.02 | Train'den validation ayırıyor; kullanılan ayrım bizim sabit ayrımımızla aynı değil. |
| [Mehri ve Eric, 2021, tablo 1](https://aclanthology.org/2021.naacl-main.237.pdf#page=6) | ConvBERT + MLM + Example | 94.06 | Ek konuşma ön eğitimi, MLM ve örnek tabanlı karar yöntemi içeriyor; sıradan sınıflandırma başlığından farklı. |

Bu projede 8.499 train / 1.500 validation / 3.080 resmî test mesajı kullanılır.
Ortak veri seti adı, eşit eğitim verisi veya protokol anlamına gelmez. Model boyutu,
ön eğitim, ayrım, checkpoint seçimi ve eğitim bütçesi farkları nedeniyle bu tablo
kontrollü bir üstünlük kıyası veya geçilmesi gereken sabit bir accuracy eşiği değildir.
Proje içindeki yöntemler kendi ortak bölümlerimizde değerlendirilir; literatür
değerleri beklenen performansın bağlamını ve farklı deney koşullarını açıklar.

## Lisanslar ve kalan işler

Kod projenin MIT lisansı altındadır. BANKING77 verisinin CC BY 4.0 lisansı ve
atıfı geçerlidir. Transformer kaynak lisansları registry'de, kaynak model
kartlarında tam koşullarıyla yer alır. GloVe vektörleri Stanford'un
[PDDL 1.0 açıklaması](https://nlp.stanford.edu/projects/glove/) kapsamında
kullanılır; GloVe kodunun Apache lisansıyla vektörlerin lisansı karıştırılmaz.

Bu dosyadaki vanilla RNN/CNN/LSTM/Transformer'lar ders baseline'larıdır.
NB + CNN katkısının validation deneyi ve iki uç ablation'ı tamamlandı.
Sekiz baseline ve NB + CNN ayarları `2425ccc` ile testten önce sabitlendi;
[nihai test ve gerçek hata örnekleri](../results/NEURAL_TEST.md) teslim edildi.
Dört Transformer'ın [tam validation deneyleri](../results/TRANSFORMER_VALIDATION.md)
ve [resmî test değerlendirmeleri](../results/TRANSFORMER_TEST.md) tamamlandı;
seçilmiş checkpoint protokolleri testten önce `1e4ff66` commit'iyle sabitlendi.
Kısa geliştirme kontrolü benchmark sonucu olarak sunulmaz.
Tek seed sonuçları genelleme veya istatistiksel üstünlük
kanıtı değildir. Yeni modellerin dahil olduğu eşleştirilmiş karşılaştırma
[results/ALL_MODELS_TEST.md](../results/ALL_MODELS_TEST.md) içindedir; PPTX 5. kişinin sorumluluğundadır.
