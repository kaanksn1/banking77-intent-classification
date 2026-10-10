# Ozan Serhan Denizhan — 220911758

GitHub username: serhandenizhan
Responsibility: Person 5 — shared evaluation: benchmark script, comparison tables and figures, metric justification

## What I completed

- `src/banking77/benchmark_models.py`: one command (`python -m banking77.benchmark_models`) that calls
  the existing `run_naive_bayes`, `run_logistic_regression` and `run_linear_svm` functions unchanged and
  runs the initial and selected settings sequentially on the same machine, validation split only.
  Each setting is run in 3 rounds; scores must be identical across rounds and times are medians.
  It refuses to compare runs whose `dataset_summary_sha256` differs.
- Paired comparison of the selected settings on the same validation messages: exact McNemar test and a
  seeded paired bootstrap 95% interval for the macro F1 difference. Small score differences are therefore
  reported with an uncertainty statement instead of being read as a ranking.
- `src/banking77/plot_comparison.py`: figures for scores (initial vs selected), fit/prediction time and
  the most confused category pairs.
- Consistency checks in the benchmark: identical prepared data files (per-file SHA-256) across all runs,
  and the convergence status of LR and SVM is reported in the table.
- Near-duplicate sensitivity: validation scores are recomputed without validation messages that are
  near-copies of training messages (character n-gram TF-IDF cosine >= 0.95 and >= 0.90). The split was not
  changed; the model ordering is unchanged and the scores drop by about 0.4 to 0.5 points at 0.95.
- `src/banking77/benchmark_features.py` and `results/FEATURE_COMPARISON.md`: unigram vs unigram + bigram
  on all three models, each re-tuned with the owners' own hyperparameter grids and selection rule
  (validation only), with exact McNemar and paired bootstrap, plus a bigram-only ablation. Findings:
  unigram vs unigram + bigram is not separable for any model (p = 0.08 to 0.75, NB slightly favours
  bigrams); bigram only is clearly worse (macro F1 down 0.046 to 0.063, p < 1e-13).
- Final test comparison (`python -m banking77.benchmark_models --split test`): the same benchmark on the
  official test split with the settings frozen on validation and the agreed shared feature setting
  (unigram + bigram), producing `results/MODEL_COMPARISON_TEST.md`, `results/model_comparison_test.json`
  and `results/figures/test_*.png`, including the train/test overlap report required by
  `docs/EXPERIMENTS.md`. Test scores: Naive Bayes 0.8458, Logistic Regression 0.8878, Linear SVM 0.8865
  macro F1; Logistic Regression and Linear SVM are again not separable (p = 0.68). The LR and SVM final
  runs were executed centrally by me with the owners' frozen settings and their agreement.
- `tests/test_benchmark_models.py`: 14 unit tests (agreed settings, validation-only use, split passing, median timing,
  rejection of changed scores or datasets or data files, convergence reporting, McNemar counts, macro F1
  against scikit-learn, seeded bootstrap, near-duplicate similarity, subset scores, error example selection);
  `tests/test_benchmark_features.py` adds 5 more (selection rule, validation-only use, grids).
  The full suite has 40 passing tests.
- `results/MODEL_COMPARISON.md`, `results/model_comparison_validation.json`, `results/figures/*.png`:
  generated report with the comparison table, the macro F1 justification, most confused pairs and real
  example errors.
- README, `docs/TEAM.md`, `docs/EXPERIMENTS.md`, `docs/SUBMISSION.md`, `results/README.md`: updated to
  point to the comparison and to describe its status. `pyproject.toml` and `requirements-lock.txt` now list
  `scipy` and `matplotlib`.

I did not change the other members' model, training or data files.

## Verification and experiments

- Command: `python -m banking77.benchmark_models` (about 30 seconds).
- Data split: validation (8,499 train, 1,500 validation);
  `dataset_summary_sha256 = 468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`,
  identical to the runs of the other members.
- Output: `results/MODEL_COMPARISON.md` and `results/model_comparison_validation.json`.
- Results (validation macro F1 / accuracy): Naive Bayes alpha=1 0.7872 / 81.60%, alpha=0.05
  0.8529 / 86.20%; Logistic Regression lbfgs C=1 0.8555 / 85.80%, liblinear-ovr C=100 0.8920 / 89.07%;
  Linear SVM squared_hinge C=1 (initial = selected) 0.8935 / 89.20%. The selected-setting scores match
  the values reported by the model owners.
- Findings: Logistic Regression and Linear SVM are clearly better than Naive Bayes (McNemar p < 0.001,
  bootstrap interval excludes 0). Linear SVM and Logistic Regression are not separable on this split:
  13 messages fixed, 11 broken, p = 0.84, interval [-0.0060, +0.0089]. 117 of 1,500 validation messages
  are wrong for all three models, mostly between neighbouring categories
  (for example `direct_debit_payment_not_recognised` → `card_payment_not_recognised`).
- Limitations: one validation split of 1,500 messages with 5 to 28 messages per category; times come from
  one machine and are not a general speed claim; these are not final test results. Final test numbers will
  be produced by the model owners after the shared feature setting is fixed.

## GitHub evidence

- My commit links (`feature/evaluation`):
  - [Shared benchmark, plots and tests](https://github.com/kaanksn1/banking77-intent-classification/commit/cdb5d6f3238efd649d6a1908fb4320a9acfeea3f)
  - [Validation comparison report and figures](https://github.com/kaanksn1/banking77-intent-classification/commit/00132870e5e2bdfd5e43d6cb91dbe9e280021e53)
  - [Documentation and submission tracking](https://github.com/kaanksn1/banking77-intent-classification/commit/9733efcd63c9ca02450fe4e46eed1f6a681e006a)
  - [Data-file, convergence and near-duplicate checks](https://github.com/kaanksn1/banking77-intent-classification/commit/0f1358d5d561a71b820336b97821d8e0a5768492)
  - [Regenerated report and contribution file](https://github.com/kaanksn1/banking77-intent-classification/commit/1a019b6adcd9d0f4f612de9ea7d42f8170e96cf7)
  - [Three-model feature comparison with bigram-only ablation](https://github.com/kaanksn1/banking77-intent-classification/commit/f6e77b74a9e9b5aa601a2c07018050c7997dba49)
  - [Feature comparison report and documentation](https://github.com/kaanksn1/banking77-intent-classification/commit/fb7d2339566793431917df7941fccfd9ca8dd91e)
  - [`--split test` mode and overlap report](https://github.com/kaanksn1/banking77-intent-classification/commit/9474cb47c0dd038b7adcf245b72512050560cc23)
  - [Final test comparison of the three models](https://github.com/kaanksn1/banking77-intent-classification/commit/6053fd2e0f8c526e8dead5978b9803052f4dde61)
- My pull request links:
  - [#8 Shared model benchmark and validation comparison](https://github.com/kaanksn1/banking77-intent-classification/pull/8) (merged)
  - [#9 Three-model unigram/bigram feature comparison](https://github.com/kaanksn1/banking77-intent-classification/pull/9) (merged)
  - [#11 Final test comparison of NB, LR and Linear SVM](https://github.com/kaanksn1/banking77-intent-classification/pull/11)

## Contribution to the presentation

- Prepared the comparison table, figures, metric justification (why macro F1) and the real example
  errors that the presentation is built from.
