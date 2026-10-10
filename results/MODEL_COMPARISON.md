# Model comparison (validation)

All models use the same prepared data and the same TF-IDF features (unigram + bigram, sublinear TF).
They run on the validation split only. The official test set was not evaluated.
Command: `python -m banking77.benchmark_models`. Dataset summary SHA-256: `468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`.
Training rows: 8499, validation rows: 1500.

## Models

| Model | Implementation | Learned from |
| --- | --- | --- |
| Naive Bayes | TF-IDF + MultinomialNB (scikit-learn) | our training split, no pretrained weights |
| Logistic Regression | TF-IDF + LogisticRegression (scikit-learn) | our training split, no pretrained weights |
| Linear SVM | TF-IDF + LinearSVC (scikit-learn) | our training split, no pretrained weights |

All three are the vanilla course methods used as baselines. The selected settings are tuned versions of the same baselines, not new methods. We use scikit-learn implementations: the algorithms are not coded from scratch, but all parameters are learned from our own training data.

## Results

| Model | Stage | Settings | Accuracy | Macro F1 | Fit (s) | Predict (s) | Predict (ms/msg) | Converged |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |
| Naive Bayes | initial | alpha=1 | 81.60% | 0.7872 | 0.10 | 0.0129 | 0.009 | n/a |
| Naive Bayes | selected | alpha=0.05 | 86.20% | 0.8529 | 0.10 | 0.0122 | 0.008 | n/a |
| Logistic Regression | initial | lbfgs, C=1 | 85.80% | 0.8555 | 4.45 | 0.0164 | 0.011 | yes |
| Logistic Regression | selected | liblinear-ovr, C=100 | 89.07% | 0.8920 | 1.88 | 0.0188 | 0.013 | yes |
| Linear SVM | initial = selected | squared_hinge, C=1 | 89.20% | 0.8935 | 0.46 | 0.0132 | 0.009 | yes |

Times are medians of 3 repeats on one machine, models run sequentially. Fit includes TF-IDF fitting; predict includes the TF-IDF transform. Absolute times depend on the machine and are only comparable within this table.

Initial and selected settings are shown separately. The initial settings are the baselines defined in `docs/EXPERIMENTS.md`; the selected settings were chosen by each model owner on validation macro F1. Linear SVM's selected setting equals its initial one, so it has a single row.

## Why macro F1

The task has 77 categories and every one matters equally for a bank: a rare request type (for example a swallowed card) is as important to route correctly as a frequent one. Macro F1 averages the per-category F1 with equal weight, so a model cannot hide poor performance on small categories behind good performance on large ones. Accuracy counts every message equally, so it favours frequent categories; it is reported next to macro F1 as a supporting metric.

Category sizes are not equal: in training, from 30 messages (`contactless_not_working`) to 159 (`card_payment_fee_charged`); in validation, from 5 to 28. The official test set is balanced (40 messages in every category), so accuracy and macro F1 are expected to be closer there. We still rank by macro F1 because the reason for choosing it, equal importance of every category, does not depend on the split.

## Paired comparison of the selected settings

Score differences between models are small, so each pair is compared on the same validation messages. A single validation split does not allow a firm ranking when the interval includes 0.

- Logistic Regression vs Naive Bayes: 84 messages fixed, 41 broken (exact McNemar p = 0.00015); macro F1 difference +0.0391, paired bootstrap 95% interval [+0.0226, +0.0589] (the interval excludes 0).
- Linear SVM vs Naive Bayes: 87 messages fixed, 42 broken (exact McNemar p = 9.2e-05); macro F1 difference +0.0406, paired bootstrap 95% interval [+0.0235, +0.0607] (the interval excludes 0).
- Linear SVM vs Logistic Regression: 13 messages fixed, 11 broken (exact McNemar p = 0.84); macro F1 difference +0.0015, paired bootstrap 95% interval [-0.0060, +0.0089] (the interval includes 0).

## Sensitivity to near-duplicate messages

Some validation messages are near-copies of training messages (reordered sentences, one added word). They are mostly easy and keep the same label, so they can raise the scores slightly. We did not change the split; instead the scores are recomputed without those messages.

Method: max cosine similarity of character 3-5-gram TF-IDF (fitted on train) to any training message; evaluation messages at or above the threshold are removed. Macro F1 in the reduced sets uses only the categories that remain.

| Threshold | Removed | Model | Setting | Accuracy (all → reduced) | Macro F1 (all → reduced) |
| --- | ---: | --- | --- | ---: | ---: |
| ≥ 0.95 | 63 (4.2%) | Naive Bayes | alpha=1 | 81.60% → 81.14% | 0.7872 → 0.7831 |
|  |  | Naive Bayes | alpha=0.05 | 86.20% → 85.73% | 0.8529 → 0.8478 |
|  |  | Logistic Regression | lbfgs, C=1 | 85.80% → 85.32% | 0.8555 → 0.8510 |
|  |  | Logistic Regression | liblinear-ovr, C=100 | 89.07% → 88.66% | 0.8920 → 0.8883 |
|  |  | Linear SVM | squared_hinge, C=1 | 89.20% → 88.80% | 0.8935 → 0.8896 |
| ≥ 0.90 | 148 (9.9%) | Naive Bayes | alpha=1 | 81.60% → 80.33% | 0.7872 → 0.7760 |
|  |  | Naive Bayes | alpha=0.05 | 86.20% → 85.06% | 0.8529 → 0.8417 |
|  |  | Logistic Regression | lbfgs, C=1 | 85.80% → 84.62% | 0.8555 → 0.8442 |
|  |  | Logistic Regression | liblinear-ovr, C=100 | 89.07% → 87.94% | 0.8920 → 0.8827 |
|  |  | Linear SVM | squared_hinge, C=1 | 89.20% → 88.09% | 0.8935 → 0.8840 |

## Most confused category pairs (selected settings)

**Naive Bayes**

- `card_payment_not_recognised` → `direct_debit_payment_not_recognised`: 5
- `balance_not_updated_after_bank_transfer` → `balance_not_updated_after_cheque_or_cash_deposit`: 4
- `top_up_by_bank_transfer_charge` → `top_up_by_card_charge`: 4
- `top_up_failed` → `pending_top_up`: 4
- `why_verify_identity` → `verify_my_identity`: 3

**Logistic Regression**

- `direct_debit_payment_not_recognised` → `card_payment_not_recognised`: 5
- `top_up_by_bank_transfer_charge` → `top_up_by_card_charge`: 4
- `transfer_fee_charged` → `card_payment_fee_charged`: 3
- `top_up_failed` → `top_up_reverted`: 3
- `pending_cash_withdrawal` → `declined_cash_withdrawal`: 3

**Linear SVM**

- `direct_debit_payment_not_recognised` → `card_payment_not_recognised`: 6
- `top_up_by_bank_transfer_charge` → `top_up_by_card_charge`: 4
- `top_up_failed` → `top_up_reverted`: 3
- `pending_cash_withdrawal` → `declined_cash_withdrawal`: 3
- `get_disposable_virtual_card` → `disposable_card_limits`: 3

## Example errors (selected settings)

117 validation messages are wrong for all three models; 139 are right for some models and wrong for others.

Wrong for all three models:

| Message | True | Naive Bayes | Logistic Regression | Linear SVM |
| --- | --- | --- | --- | --- |
| Please help! I was looking at my account and saw a transaction with a seller I don't remember. Is it possible for you to track the sale, because I really don't think I made a payment to them. | `direct_debit_payment_not_recognised` | `card_payment_not_recognised` | `cancel_transfer` | `card_payment_not_recognised` |
| Purchases I did not make are appearing in my bank statement? | `direct_debit_payment_not_recognised` | `cash_withdrawal_not_recognised` | `card_payment_not_recognised` | `card_payment_not_recognised` |
| There is a payment showing on my app that I didn't do. Will you please cancel this payment and refund my money ? | `direct_debit_payment_not_recognised` | `card_payment_not_recognised` | `card_payment_not_recognised` | `card_payment_not_recognised` |
| Is there a charge for using transfers for top ups? | `top_up_by_bank_transfer_charge` | `top_up_by_card_charge` | `top_up_by_card_charge` | `top_up_by_card_charge` |
| Does it cost anything to transfer money for a top-up? | `top_up_by_bank_transfer_charge` | `top_up_by_card_charge` | `top_up_by_card_charge` | `top_up_by_card_charge` |

Models disagree:

| Message | True | Naive Bayes | Logistic Regression | Linear SVM |
| --- | --- | --- | --- | --- |
| Has my card been lost in delivery? | `card_arrival` | `card_arrival` | `card_arrival` | `card_delivery_estimate` |
| I don't have my card after 1 week. What are my next steps? | `card_arrival` | `lost_or_stolen_card` | `card_arrival` | `card_arrival` |
| Where can I find the exchange rate for my transfer? | `exchange_rate` | `exchange_rate` | `exchange_rate` | `wrong_exchange_rate_for_cash_withdrawal` |

## Figures

- `results/figures/scores.png`: macro F1 and accuracy, initial vs selected
- `results/figures/timing.png`: fit and prediction time
- `results/figures/confusions.png`: most confused category pairs

## Limitations

- One validation split of 1,500 messages; scores are not final test results.
- Final test numbers are reported separately after the shared feature setting is fixed.
- Timings come from one machine and are not a general speed claim.
- Near-duplicate detection is a similarity heuristic with an arbitrary threshold, not a proof of leakage.
