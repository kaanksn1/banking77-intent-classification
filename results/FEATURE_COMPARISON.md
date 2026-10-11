# Unigram vs unigram+bigram (validation)

Same prepared data, TF-IDF with sublinear TF, validation split only; the official test set was not used.
Command: `python -m banking77.benchmark_features`. Dataset summary SHA-256: `468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`.
Selection rule for tuned settings: highest validation macro F1, ties keep the first scheduled setting.

Vocabulary size (fitted on train only): unigram 2,176, unigram+bigram 21,595, bigram only 19,419.

Two comparisons are shown. **Same hyperparameters** keeps the settings that the model owners selected with bigram features and only changes the features; this can disadvantage unigram. **Re-tuned** runs each model's full grid for each feature setting and compares the best of each, which is the fair comparison.

| Model | Comparison | Unigram setting | Bigram setting | Unigram macro F1 / acc | Bigram macro F1 / acc | Macro F1 difference (bigram − unigram) | Fixed / broken by bigram | McNemar p |
| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| Naive Bayes | same hyperparameters | alpha=0.05 | alpha=0.05 | 0.8422 / 84.87% | 0.8529 / 86.20% | +0.0107 [-0.0066, +0.0278] (includes 0) | 69 / 49 | 0.08 |
| Naive Bayes | re-tuned | alpha=0.05 | alpha=0.05 | 0.8422 / 84.87% | 0.8529 / 86.20% | +0.0107 [-0.0066, +0.0278] (includes 0) | 69 / 49 | 0.08 |
| Logistic Regression | same hyperparameters | solver=liblinear-ovr, C=100 | solver=liblinear-ovr, C=100 | 0.8847 / 88.60% | 0.8920 / 89.07% | +0.0074 [-0.0073, +0.0228] (includes 0) | 52 / 45 | 0.54 |
| Logistic Regression | re-tuned | solver=liblinear-ovr, C=10 | solver=liblinear-ovr, C=100 | 0.8933 / 89.33% | 0.8920 / 89.07% | -0.0012 [-0.0152, +0.0128] (includes 0) | 43 / 47 | 0.75 |
| Linear SVM | same hyperparameters | loss=squared_hinge, C=1 | loss=squared_hinge, C=1 | 0.8941 / 89.53% | 0.8935 / 89.20% | -0.0006 [-0.0144, +0.0139] (includes 0) | 44 / 49 | 0.68 |
| Linear SVM | re-tuned | loss=squared_hinge, C=1 | loss=squared_hinge, C=1 | 0.8941 / 89.53% | 0.8935 / 89.20% | -0.0006 [-0.0144, +0.0139] (includes 0) | 44 / 49 | 0.68 |

## Ablation: bigram only (`ngram_range=(2,2)`)

Not part of the owners' training functions; run with the same grids and selection rule by changing only the TF-IDF range of their pipeline builders.

| Model | Unigram best | Unigram + bigram best | Bigram-only best | Bigram-only − (unigram + bigram) | McNemar p |
| --- | ---: | ---: | ---: | --- | ---: |
| Naive Bayes | 0.8422 | 0.8529 | 0.8067 (alpha=0.1) | -0.0462 [-0.0605, -0.0353] (excludes 0) | 3.2e-14 |
| Logistic Regression | 0.8933 | 0.8920 | 0.8340 (solver=saga, C=100) | -0.0580 [-0.0767, -0.0442] (excludes 0) | 1.7e-16 |
| Linear SVM | 0.8941 | 0.8935 | 0.8305 (loss=squared_hinge, C=1) | -0.0630 [-0.0791, -0.0504] (excludes 0) | 2.2e-21 |

## Full grids (validation macro F1)

**Naive Bayes** (5 settings)

| Setting | Unigram | Unigram + bigram | Bigram only | Fit (s) uni / uni+bi / bi |
| --- | ---: | ---: | ---: | ---: |
| alpha=1 | 0.7942 | 0.7872 | 0.7225 | 0.04 / 0.09 / 0.06 |
| alpha=0.5 | 0.8197 | 0.8130 | 0.7617 | 0.03 / 0.08 / 0.06 |
| alpha=0.1 | 0.8345 | 0.8495 | 0.8067 | 0.04 / 0.08 / 0.06 |
| alpha=0.05 | 0.8422 | 0.8529 | 0.8049 | 0.03 / 0.08 / 0.06 |
| alpha=0.01 | 0.8397 | 0.8460 | 0.8054 | 0.04 / 0.08 / 0.06 |

**Logistic Regression** (15 settings)

| Setting | Unigram | Unigram + bigram | Bigram only | Fit (s) uni / uni+bi / bi |
| --- | ---: | ---: | ---: | ---: |
| solver=lbfgs, C=0.1 | 0.6822 | 0.6126 | 0.5036 | 0.16 / 0.86 / 0.79 |
| solver=lbfgs, C=1 | 0.8676 | 0.8555 | 0.7719 | 0.45 / 3.21 / 2.33 |
| solver=lbfgs, C=10 | 0.8898 | 0.8872 | 0.8297 | 0.63 / 4.63 / 4.14 |
| solver=lbfgs, C=100 | 0.8907 | 0.8838 | 0.8304 | 0.77 / 4.25 / 3.97 |
| solver=lbfgs, C=1000 | 0.8802 | 0.8757 | 0.8121 | 0.54 / 3.03 / 2.91 |
| solver=saga, C=0.1 | 0.6846 | 0.6136 | 0.5052 | 0.20 / 0.46 / 0.30 |
| solver=saga, C=1 | 0.8675 | 0.8549 | 0.7724 | 0.22 / 0.52 / 0.34 |
| solver=saga, C=10 | 0.8917 | 0.8868 | 0.8324 | 1.01 / 1.35 / 0.88 |
| solver=saga, C=100 | 0.8890 | 0.8912 | 0.8340 | 2.33 / 5.97 / 4.06 |
| solver=saga, C=1000 | 0.8817 | 0.8907 | 0.8324 | 6.53 / 12.24 / 9.37 |
| solver=liblinear-ovr, C=0.1 | 0.7107 | 0.6583 | 0.5585 | 0.33 / 0.56 / 0.37 |
| solver=liblinear-ovr, C=1 | 0.8573 | 0.8476 | 0.7712 | 0.44 / 0.79 / 0.52 |
| solver=liblinear-ovr, C=10 | 0.8933 | 0.8916 | 0.8314 | 0.58 / 1.03 / 0.67 |
| solver=liblinear-ovr, C=100 | 0.8847 | 0.8920 | 0.8306 | 0.74 / 1.23 / 0.80 |
| solver=liblinear-ovr, C=1000 | 0.8721 | 0.8904 | 0.8275 | 0.89 / 1.38 / 0.91 |

**Linear SVM** (10 settings)

| Setting | Unigram | Unigram + bigram | Bigram only | Fit (s) uni / uni+bi / bi |
| --- | ---: | ---: | ---: | ---: |
| loss=squared_hinge, C=0.01 | 0.7856 | 0.7907 | 0.7091 | 0.22 / 0.32 / 0.23 |
| loss=squared_hinge, C=0.1 | 0.8627 | 0.8553 | 0.7893 | 0.17 / 0.29 / 0.25 |
| loss=squared_hinge, C=1 | 0.8941 | 0.8935 | 0.8305 | 0.20 / 0.31 / 0.31 |
| loss=squared_hinge, C=10 | 0.8721 | 0.8803 | 0.8209 | 0.39 / 0.57 / 0.64 |
| loss=squared_hinge, C=100 | 0.8387 | 0.8781 | 0.8173 | 1.10 / 1.21 / 1.42 |
| loss=hinge, C=0.01 | 0.8089 | 0.8244 | 0.7731 | 0.25 / 0.38 / 0.44 |
| loss=hinge, C=0.1 | 0.8258 | 0.8335 | 0.7778 | 0.21 / 0.34 / 0.39 |
| loss=hinge, C=1 | 0.8814 | 0.8813 | 0.8201 | 0.28 / 0.50 / 0.56 |
| loss=hinge, C=10 | 0.8733 | 0.8775 | 0.8139 | 0.78 / 0.71 / 0.98 |
| loss=hinge, C=100 | 0.8338 | 0.8769 | 0.8135 | 1.23 / 0.88 / 1.08 |

## Limitations

- One validation split of 1,500 messages; differences of this size are within sampling noise when the interval includes 0.
- Times are single measurements from one machine, not medians of repeats.
- Only two feature settings (unigram, unigram+bigram) are compared; no other feature engineering.
- Convergence: of the 60 unigram and unigram+bigram runs, one did not converge. The unigram Linear SVM candidate `loss=hinge, C=10` stopped at the 10,000-iteration limit (run `linear_svm_validation_20261010T134605382109Z`, macro F1 0.8733). It is not a selected setting (the selected unigram SVM is `loss=squared_hinge, C=1`, 0.8941, which converged), so no conclusion of this report depends on it. The bigram-only ablation runs do not record convergence status.
- The bigram-only ablation is a diagnostic, not a candidate for the shared feature setting: dropping the unigrams costs 0.046 to 0.063 macro F1 for every model, so bigrams are only useful here as an addition to unigrams.
