# Model comparison (official test, final)

All models use the same prepared data and the same TF-IDF features (unigram + bigram, sublinear TF).
The settings were frozen on validation before this run; the test set was not used to select anything.
Command: `python -m banking77.benchmark_models`. Dataset summary SHA-256: `468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`.
Training rows: 8499, test rows: 3080.

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
| Naive Bayes | initial | alpha=1 | 78.31% | 0.7679 | 0.09 | 0.0213 | 0.007 | n/a |
| Naive Bayes | selected | alpha=0.05 | 84.74% | 0.8458 | 0.09 | 0.0197 | 0.006 | n/a |
| Logistic Regression | initial | lbfgs, C=1 | 84.58% | 0.8438 | 3.64 | 0.0203 | 0.007 | yes |
| Logistic Regression | selected | liblinear-ovr, C=100 | 88.77% | 0.8878 | 1.50 | 0.0254 | 0.008 | yes |
| Linear SVM | initial = selected | squared_hinge, C=1 | 88.64% | 0.8865 | 0.36 | 0.0217 | 0.007 | yes |

Times are medians of 3 repeats on one machine, models run sequentially. Fit includes TF-IDF fitting; predict includes the TF-IDF transform. Absolute times depend on the machine and are only comparable within this table.

Initial and selected settings are shown separately. The initial settings are the baselines defined in `docs/EXPERIMENTS.md`; the selected settings were chosen by each model owner on validation macro F1. Linear SVM's selected setting equals its initial one, so it has a single row.

## Why macro F1

The task has 77 categories and every one matters equally for a bank: a rare request type (for example a swallowed card) is as important to route correctly as a frequent one. Macro F1 averages the per-category F1 with equal weight, so a model cannot hide poor performance on small categories behind good performance on large ones. Accuracy counts every message equally, so it favours frequent categories; it is reported next to macro F1 as a supporting metric.

Category sizes are not equal: in training, from 30 messages (`contactless_not_working`) to 159 (`card_payment_fee_charged`); in validation, from 5 to 28. The official test set is balanced (40 messages in every category), so accuracy and macro F1 are expected to be closer there. We still rank by macro F1 because the reason for choosing it, equal importance of every category, does not depend on the split.

## Paired comparison of the selected settings

Score differences between models are small, so each pair is compared on the same test messages. A single test split does not allow a firm ranking when the interval includes 0.

- Logistic Regression vs Naive Bayes: 197 messages fixed, 73 broken (exact McNemar p = 2.5e-14); macro F1 difference +0.0420, paired bootstrap 95% interval [+0.0317, +0.0515] (the interval excludes 0).
- Linear SVM vs Naive Bayes: 199 messages fixed, 79 broken (exact McNemar p = 4.1e-13); macro F1 difference +0.0407, paired bootstrap 95% interval [+0.0308, +0.0505] (the interval excludes 0).
- Linear SVM vs Logistic Regression: 24 messages fixed, 28 broken (exact McNemar p = 0.68); macro F1 difference -0.0013, paired bootstrap 95% interval [-0.0057, +0.0028] (the interval includes 0).

## Overlap with training data (official test)

Per `docs/EXPERIMENTS.md`, test messages whose normalized text also appears in training are reported separately; the test set is not claimed to be a perfectly clean holdout.

| Model | Setting | Overlapping rows | Non-overlapping rows | Accuracy (non-overlapping) | Macro F1 (non-overlapping) |
| --- | --- | ---: | ---: | ---: | ---: |
| Naive Bayes | alpha=1 | 7 | 3073 | 78.29% | 0.7668 |
| Naive Bayes | alpha=0.05 | 7 | 3073 | 84.71% | 0.8454 |
| Logistic Regression | lbfgs, C=1 | 7 | 3073 | 84.54% | 0.8433 |
| Logistic Regression | liblinear-ovr, C=100 | 7 | 3073 | 88.74% | 0.8875 |
| Linear SVM | squared_hinge, C=1 | 7 | 3073 | 88.61% | 0.8862 |

## Sensitivity to near-duplicate messages

Some test messages are near-copies of training messages (reordered sentences, one added word). They are mostly easy and keep the same label, so they can raise the scores slightly. We did not change the split; instead the scores are recomputed without those messages.

Method: max cosine similarity of character 3-5-gram TF-IDF (fitted on train) to any training message; evaluation messages at or above the threshold are removed. Macro F1 in the reduced sets uses only the categories that remain.

| Threshold | Removed | Model | Setting | Accuracy (all → reduced) | Macro F1 (all → reduced) |
| --- | ---: | --- | --- | ---: | ---: |
| ≥ 0.95 | 197 (6.4%) | Naive Bayes | alpha=1 | 78.31% → 77.11% | 0.7679 → 0.7566 |
|  |  | Naive Bayes | alpha=0.05 | 84.74% → 83.91% | 0.8458 → 0.8386 |
|  |  | Logistic Regression | lbfgs, C=1 | 84.58% → 83.70% | 0.8438 → 0.8362 |
|  |  | Logistic Regression | liblinear-ovr, C=100 | 88.77% → 88.17% | 0.8878 → 0.8825 |
|  |  | Linear SVM | squared_hinge, C=1 | 88.64% → 88.00% | 0.8865 → 0.8807 |
| ≥ 0.90 | 408 (13.2%) | Naive Bayes | alpha=1 | 78.31% → 76.38% | 0.7679 → 0.7498 |
|  |  | Naive Bayes | alpha=0.05 | 84.74% → 83.12% | 0.8458 → 0.8303 |
|  |  | Logistic Regression | lbfgs, C=1 | 84.58% → 82.97% | 0.8438 → 0.8278 |
|  |  | Logistic Regression | liblinear-ovr, C=100 | 88.77% → 87.57% | 0.8878 → 0.8761 |
|  |  | Linear SVM | squared_hinge, C=1 | 88.64% → 87.39% | 0.8865 → 0.8743 |

## Most confused category pairs (selected settings)

**Naive Bayes**

- `contactless_not_working` → `card_not_working`: 10
- `card_payment_not_recognised` → `direct_debit_payment_not_recognised`: 7
- `card_swallowed` → `declined_cash_withdrawal`: 6
- `compromised_card` → `cash_withdrawal_not_recognised`: 5
- `pending_transfer` → `transfer_timing`: 5

**Logistic Regression**

- `unable_to_verify_identity` → `verify_my_identity`: 5
- `pending_transfer` → `balance_not_updated_after_bank_transfer`: 5
- `top_up_failed` → `top_up_reverted`: 5
- `balance_not_updated_after_bank_transfer` → `transfer_not_received_by_recipient`: 5
- `verify_my_identity` → `why_verify_identity`: 5

**Linear SVM**

- `why_verify_identity` → `verify_my_identity`: 6
- `unable_to_verify_identity` → `verify_my_identity`: 5
- `top_up_failed` → `top_up_reverted`: 5
- `balance_not_updated_after_bank_transfer` → `transfer_not_received_by_recipient`: 5
- `top_up_reverted` → `top_up_failed`: 4

## Example errors (selected settings)

257 test messages are wrong for all three models; 300 are right for some models and wrong for others.

Wrong for all three models:

| Message | True | Naive Bayes | Logistic Regression | Linear SVM |
| --- | --- | --- | --- | --- |
| I'd rather not verify my identity. | `why_verify_identity` | `verify_my_identity` | `unable_to_verify_identity` | `verify_my_identity` |
| I won't verify my identity. | `why_verify_identity` | `unable_to_verify_identity` | `verify_my_identity` | `verify_my_identity` |
| What other methods are there to verify my identity? | `why_verify_identity` | `verify_my_identity` | `verify_my_identity` | `verify_my_identity` |
| Do I need to verify my identity? | `why_verify_identity` | `verify_my_identity` | `verify_my_identity` | `verify_my_identity` |
| I cannot verify my identity | `unable_to_verify_identity` | `verify_my_identity` | `verify_my_identity` | `verify_my_identity` |

Models disagree:

| Message | True | Naive Bayes | Logistic Regression | Linear SVM |
| --- | --- | --- | --- | --- |
| I ordered a card but it has not arrived. Help please! | `card_arrival` | `card_arrival` | `transfer_not_received_by_recipient` | `transfer_not_received_by_recipient` |
| How long should my new card take to arrive? | `card_arrival` | `card_delivery_estimate` | `card_arrival` | `card_arrival` |
| I'm starting to think my card is lost because it still hasn't arrived, can you help? | `card_arrival` | `card_arrival` | `lost_or_stolen_card` | `lost_or_stolen_card` |

## Figures

- `results/figures/test_scores.png`: macro F1 and accuracy, initial vs selected
- `results/figures/test_timing.png`: fit and prediction time
- `results/figures/test_confusions.png`: most confused category pairs

## Limitations

- One official test split; the settings were chosen on validation, no choice depends on test results (the 3 repeats only measure time; scores are identical).
- The official test set has a few normalized-text overlaps with training (see the overlap section), so it is not claimed to be a perfectly clean holdout.
- The Logistic Regression and Linear SVM final runs were executed centrally by Person 5 with the owners' frozen settings; Naive Bayes was frozen and run by its owner (`results/NAIVE_BAYES_TEST.md`) with the same result.
- Timings come from one machine and are not a general speed claim.
- Near-duplicate detection is a similarity heuristic with an arbitrary threshold, not a proof of leakage.
