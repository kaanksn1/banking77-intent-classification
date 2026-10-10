# All-method comparison (official test, final)

Sixteen methods on the same 3080 official test messages (77 categories, 40 messages each). Every setting was frozen on validation before the official test; nothing in this report selects a setting, a checkpoint or a weight.
Command: `python -m banking77.compare_all_models`. Dataset summary SHA-256: `468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`.
The script trains nothing: it reads the committed `predictions.csv` of each method, checks that all files contain the same message ids, texts and true labels, recomputes accuracy and macro F1, and compares the file hashes with the records written when the predictions were produced.

## Results

Sorted by macro F1. The interval is a paired bootstrap 95% interval (1000 resamples of test messages, seed 42, the same resamples for every method). The last column is the exact McNemar test against the best method with Holm correction over all 120 method pairs.

| # | Method | Family | Accuracy | Macro F1 [95% interval] | Macro F1 without 7 overlapping messages | only method right / only best right | Holm p |
| ---: | --- | --- | ---: | --- | ---: | ---: | ---: |
| 1 | RoBERTa-base | Pretrained Transformers (fine-tuned) | 93.02% | 0.9301 [0.9202, 0.9382] | 0.9300 | best | - |
| 2 | NB + CNN soft voting | Team contribution | 90.65% | 0.9061 [0.8949, 0.9158] | 0.9059 | 65 / 138 | 1.0e-05 |
| 3 | ALBERT-base-v2 | Pretrained Transformers (fine-tuned) | 90.29% | 0.9027 [0.8910, 0.9115] | 0.9024 | 43 / 127 | 2.8e-09 |
| 4 | BERT-base | Pretrained Transformers (fine-tuned) | 90.58% | 0.9016 [0.8909, 0.9110] | 0.9013 | 46 / 121 | 1.9e-07 |
| 5 | DistilBERT-base | Pretrained Transformers (fine-tuned) | 89.61% | 0.8961 [0.8840, 0.9052] | 0.8957 | 35 / 140 | 2.1e-14 |
| 6 | CNN | Neural networks trained from scratch | 89.16% | 0.8917 [0.8796, 0.9012] | 0.8914 | 50 / 169 | 1.3e-14 |
| 7 | Logistic Regression | Classical TF-IDF baselines | 88.77% | 0.8878 [0.8762, 0.8976] | 0.8875 | 73 / 204 | 7.1e-14 |
| 8 | Linear SVM | Classical TF-IDF baselines | 88.64% | 0.8865 [0.8748, 0.8960] | 0.8862 | 78 / 213 | 5.3e-14 |
| 9 | Naive Bayes | Classical TF-IDF baselines | 84.74% | 0.8458 [0.8331, 0.8575] | 0.8454 | 80 / 335 | 2.5e-36 |
| 10 | BiLSTM | Neural networks trained from scratch | 83.15% | 0.8314 [0.8173, 0.8422] | 0.8310 | 51 / 355 | 3.1e-55 |
| 11 | LSTM | Neural networks trained from scratch | 82.18% | 0.8219 [0.8064, 0.8337] | 0.8214 | 49 / 383 | 2.4e-63 |
| 12 | Word2Vec Skip-gram + mean + LR | Embeddings + linear head | 79.55% | 0.7926 [0.7788, 0.8055] | 0.7921 | 57 / 472 | 2.3e-80 |
| 13 | Word2Vec CBOW + mean + LR | Embeddings + linear head | 75.45% | 0.7524 [0.7357, 0.7654] | 0.7521 | 61 / 602 | 9.0e-111 |
| 14 | FastText + mean + LR | Embeddings + linear head | 72.53% | 0.7233 [0.7061, 0.7370] | 0.7227 | 54 / 685 | 4.2e-138 |
| 15 | GloVe (frozen) + mean + linear | Embeddings + linear head | 71.01% | 0.7093 [0.6917, 0.7221] | 0.7090 | 39 / 717 | 2.2e-160 |
| 16 | RNN | Neural networks trained from scratch | 68.99% | 0.6852 [0.6666, 0.6990] | 0.6849 | 38 / 778 | 2.0e-178 |

Reading the table: RoBERTa-base has the highest macro F1 (0.9301). 15 of the other 15 methods are separable from it (Holm-adjusted p < 0.05).
Methods that are close to each other in macro F1 are compared pair by pair below; a single test split does not give a firm ranking when the intervals overlap.

## Pre-specified paired comparisons

Chosen before looking at the pairwise results; they answer the questions the presentation asks. "Fixed" counts messages the candidate gets right and the reference gets wrong.

| Reference | Candidate | Why this pair | Fixed / broken | Macro F1 difference [95% interval] | p | Holm p |
| --- | --- | --- | ---: | --- | ---: | ---: |
| Linear SVM | RoBERTa-base | best Transformer vs best classical TF-IDF model | 213 / 78 | +0.0436 [+0.0330, +0.0544] (excludes 0) | 1.2e-15 | 5.3e-14 |
| NB + CNN soft voting | RoBERTa-base | best Transformer vs team contribution | 138 / 65 | +0.0240 [+0.0147, +0.0336] (excludes 0) | 3.3e-07 | 1.0e-05 |
| CNN | RoBERTa-base | best Transformer vs best network trained from scratch | 169 / 50 | +0.0384 [+0.0295, +0.0483] (excludes 0) | 2.7e-16 | 1.3e-14 |
| Logistic Regression | Linear SVM | the two linear classical models | 24 / 28 | -0.0013 [-0.0057, +0.0028] (includes 0) | 0.678 | 1.000 |
| Linear SVM | CNN | best classical model vs CNN | 146 / 130 | +0.0052 [-0.0054, +0.0153] (includes 0) | 0.367 | 1.000 |
| CNN | NB + CNN soft voting | team contribution vs its stronger component | 94 / 48 | +0.0144 [+0.0074, +0.0221] (excludes 0) | 1.4e-04 | 0.004 |
| Naive Bayes | NB + CNN soft voting | team contribution vs its weaker component | 223 / 41 | +0.0603 [+0.0511, +0.0706] (excludes 0) | 1.8e-31 | 1.1e-29 |
| DistilBERT-base | RoBERTa-base | distilled vs robustly pre-trained Transformer | 140 / 35 | +0.0340 [+0.0264, +0.0429] (excludes 0) | 4.5e-16 | 2.1e-14 |
| Word2Vec CBOW + mean + LR | Word2Vec Skip-gram + mean + LR | CBOW vs Skip-gram on the same data | 255 / 129 | +0.0402 [+0.0287, +0.0534] (excludes 0) | 1.2e-10 | 4.2e-09 |
| LSTM | BiLSTM | one vs two reading directions | 219 / 189 | +0.0095 [-0.0035, +0.0226] (includes 0) | 0.151 | 1.000 |

## Team contribution: NB + CNN soft voting

The CNN weight (0.4; Naive Bayes 0.6) was chosen on validation macro F1 from a grid of 0.0 to 1.0 in steps of 0.1; the two end points are the single models.

| CNN weight | Validation accuracy | Validation macro F1 |
| ---: | ---: | ---: |
| 0.0 | 86.20% | 0.8529 |
| 0.1 | 87.33% | 0.8689 |
| 0.2 | 88.40% | 0.8795 |
| 0.3 | 89.67% | 0.8959 |
| 0.4 (selected) | 90.07% | 0.8997 |
| 0.5 | 89.87% | 0.8985 |
| 0.6 | 89.80% | 0.8978 |
| 0.7 | 89.33% | 0.8931 |
| 0.8 | 89.00% | 0.8898 |
| 0.9 | 88.53% | 0.8855 |
| 1.0 | 88.13% | 0.8816 |

| Official test | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Naive Bayes alone | 84.74% | 0.8458 |
| CNN alone | 89.16% | 0.8917 |
| NB + CNN soft voting | 90.65% | 0.9061 |

The two components make different mistakes: both correct 2477, only Naive Bayes correct 133, only CNN correct 269, both wrong 201. An oracle that always picks the right component would reach 93.47% accuracy; the ensemble recovers part of that gap (2 messages that both components got wrong are right in the ensemble, 89 messages that at least one component had right are wrong in the ensemble). The ensemble is a team-built combination of two standard baselines; it is not a new architecture, and it does not reach the best Transformer.

Against every other method (exact McNemar, Holm-adjusted over all 120 pairs, p < 0.05): NB + CNN is better than Naive Bayes, Logistic Regression, Linear SVM, GloVe (frozen) + mean + linear, Word2Vec CBOW + mean + LR, Word2Vec Skip-gram + mean + LR, FastText + mean + LR, RNN, CNN, LSTM, BiLSTM; it is not separable from BERT-base, DistilBERT-base, ALBERT-base-v2; it is worse than RoBERTa-base.

## Cost

Training and prediction time are not comparable across rows without the device and the scope below. Classical models were timed as medians of three sequential repeats; neural models and embeddings on a 4-thread CPU; Transformers on one AMD GPU. Neural fit time includes validation and checkpoint writing per epoch and excludes first downloads and tokenisation; Transformer fit time is the recorded training run whose checkpoint was evaluated here. External pre-training cost is not included.

| Method | Fit (s) | Predict (ms/message) | Device | Pre-training |
| --- | ---: | ---: | --- | --- |
| RoBERTa-base | 484.6 | 5.212 | AMD Radeon RX 7800 XT GPU (ROCm, WSL2) | large external pre-training (public checkpoint) |
| NB + CNN soft voting | 43.7 | 0.113 | CPU, Windows (4 threads) | none (components learned from our training split) |
| ALBERT-base-v2 | 410.4 | 5.324 | AMD Radeon RX 7800 XT GPU (ROCm, WSL2) | large external pre-training (public checkpoint) |
| BERT-base | 472.0 | 5.277 | AMD Radeon RX 7800 XT GPU (ROCm, WSL2) | large external pre-training (public checkpoint) |
| DistilBERT-base | 250.2 | 2.565 | AMD Radeon RX 7800 XT GPU (ROCm, WSL2) | large external pre-training (public checkpoint) |
| CNN | 43.6 | 0.101 | CPU, Windows (4 threads) | none (random embeddings) |
| Logistic Regression | 1.50 | 0.008 | CPU (benchmark machine, sequential runs) | none (learned from our training split) |
| Linear SVM | 0.36 | 0.007 | CPU (benchmark machine, sequential runs) | none (learned from our training split) |
| Naive Bayes | 0.09 | 0.006 | CPU (benchmark machine, sequential runs) | none (learned from our training split) |
| BiLSTM | 74.8 | 0.089 | CPU, Windows (4 threads) | none (random embeddings) |
| LSTM | 38.5 | 0.045 | CPU, Windows (4 threads) | none (random embeddings) |
| Word2Vec Skip-gram + mean + LR | 3.25 | 0.016 | CPU, Windows (4 threads) | embeddings learned from our training split only |
| Word2Vec CBOW + mean + LR | 1.75 | 0.017 | CPU, Windows (4 threads) | embeddings learned from our training split only |
| FastText + mean + LR | 4.22 | 0.017 | CPU, Windows (4 threads) | embeddings learned from our training split only |
| GloVe (frozen) + mean + linear | 3.62 | 0.008 | CPU, Windows (4 threads) | external GloVe 6B 100d vectors, frozen |
| RNN | 17.6 | 0.022 | CPU, Windows (4 threads) | none (random embeddings) |

## Where the methods fail

- 58 of 3080 test messages are wrong for all 16 methods; 145 are wrong for all four Transformers.
- Hardest categories (share of the 40 test messages that a method gets wrong, averaged over the 16 methods; the best method's own error rate in brackets):
  - `contactless_not_working`: 38.9% (best method 10.0%)
  - `virtual_card_not_working`: 38.1% (best method 10.0%)
  - `compromised_card`: 33.6% (best method 10.0%)
  - `pending_transfer`: 33.0% (best method 17.5%)
  - `balance_not_updated_after_bank_transfer`: 31.4% (best method 17.5%)
  - `topping_up_by_card`: 31.2% (best method 17.5%)
  - `why_verify_identity`: 30.2% (best method 12.5%)
  - `transfer_not_received_by_recipient`: 27.0% (best method 10.0%)
  - `declined_transfer`: 26.6% (best method 15.0%)
  - `card_acceptance`: 26.1% (best method 10.0%)

Most frequent confusions of RoBERTa-base:

| True category | Predicted category | Messages |
| --- | --- | ---: |
| `why_verify_identity` | `verify_my_identity` | 6 |
| `card_delivery_estimate` | `card_arrival` | 5 |
| `transfer_into_account` | `topping_up_by_card` | 5 |
| `pending_transfer` | `transfer_not_received_by_recipient` | 4 |
| `get_disposable_virtual_card` | `getting_virtual_card` | 4 |

Real messages that every method gets wrong (first six in file order; examples, not causal explanations):

| Message | True category | Linear SVM | CNN | NB + CNN | RoBERTa-base |
| --- | --- | --- | --- | --- | --- |
| How long does a card delivery take? | `card_arrival` | `card_delivery_estimate` | `card_delivery_estimate` | `card_delivery_estimate` | `card_delivery_estimate` |
| Why am I being charged more ? | `card_payment_wrong_exchange_rate` | `card_payment_fee_charged` | `card_payment_fee_charged` | `card_payment_fee_charged` | `extra_charge_on_statement` |
| My account was charged for a withdraw I tried to make that was decline. | `pending_cash_withdrawal` | `declined_card_payment` | `cash_withdrawal_charge` | `cash_withdrawal_charge` | `declined_cash_withdrawal` |
| There is an incoming payment into my account, but it is deactivated. Will they still be processed? | `fiat_currency_support` | `balance_not_updated_after_cheque_or_cash_deposit` | `direct_debit_payment_not_recognised` | `balance_not_updated_after_cheque_or_cash_deposit` | `reverted_card_payment?` |
| Am I able to exchange currencies? | `fiat_currency_support` | `exchange_via_app` | `exchange_via_app` | `exchange_via_app` | `exchange_via_app` |
| my card was not in the mail again can you advise? | `card_delivery_estimate` | `declined_card_payment` | `card_arrival` | `card_arrival` | `card_arrival` |

## Sensitivity to overlap with the training data

7 test messages are exact normalised copies of training messages; the macro F1 column in the results table excludes them. A looser check removes the 197 test messages (6.4%) whose character 3-5-gram TF-IDF cosine similarity to a training message is at least 0.95.

| Method | Macro F1 (all) | Macro F1 (without near-duplicates) | Change |
| --- | ---: | ---: | ---: |
| RoBERTa-base | 0.9301 | 0.9280 | -0.0022 |
| NB + CNN soft voting | 0.9061 | 0.9018 | -0.0043 |
| ALBERT-base-v2 | 0.9027 | 0.8988 | -0.0039 |
| BERT-base | 0.9016 | 0.8963 | -0.0053 |
| DistilBERT-base | 0.8961 | 0.8917 | -0.0044 |
| CNN | 0.8917 | 0.8859 | -0.0058 |
| Logistic Regression | 0.8878 | 0.8825 | -0.0053 |
| Linear SVM | 0.8865 | 0.8807 | -0.0058 |
| Naive Bayes | 0.8458 | 0.8386 | -0.0072 |
| BiLSTM | 0.8314 | 0.8221 | -0.0093 |
| LSTM | 0.8219 | 0.8126 | -0.0094 |
| Word2Vec Skip-gram + mean + LR | 0.7926 | 0.7849 | -0.0077 |
| Word2Vec CBOW + mean + LR | 0.7524 | 0.7427 | -0.0097 |
| FastText + mean + LR | 0.7233 | 0.7124 | -0.0109 |
| GloVe (frozen) + mean + linear | 0.7093 | 0.7012 | -0.0080 |
| RNN | 0.6852 | 0.6689 | -0.0163 |

## Limitations

- One official test split of 3,080 messages (40 per category) and one training run (one seed) per method; differences of about one point are not firm rankings.
- Tuning budgets differ by family: validation grids for the classical models, one configuration with the epoch chosen on validation macro F1 for each network, and a common five-epoch budget for the Transformers; the comparison is between protocols, not between optimal versions of each method.
- The classical test results were seen before the neural scope was added, so the whole study is not a fully blind test; the new settings were frozen on validation before their official test.
- Word2Vec and FastText vectors are trained on our 8,499 training messages only, GloVe is frozen and externally pre-trained, and the Transformers use large external pre-training; the families do not have the same information.
- Times come from different devices and measurement scopes (see Cost) and are not a speed ranking.
- Hyper-parameter tuning for the classical models and the NB + CNN weight use validation; the validation set was not added to training for any method.
