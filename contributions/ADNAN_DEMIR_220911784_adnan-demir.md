# Adnan Demir — 220911784

GitHub username: adnan-demir
Responsibility: Member 2 — Data Analysis & Feature Engineering
Branch: `feature/data-features`

## Scope and separation from other members

My assignment covers dataset analysis, preprocessing review, the unigram versus
unigram+bigram TF-IDF comparison, a shared feature recommendation, and this contribution
file. The following belong to other members and were **not** done by me:

- Member 1: Naive Bayes implementation (`naive_bayes.py`, `train_naive_bayes.py`,
  `benchmark_naive_bayes.py`), alpha tuning, error analysis, GitHub management, integration.
- Member 3: Logistic Regression. Member 4: Linear SVM.
- Member 5: shared evaluation, comparison charts, slides, presentation.

The repository's shared setup (pinned download, hash checks, initial cleaning and split in
`data.py`, the Naive Bayes runner) existed before my work and is not claimed as my
contribution. I used the existing Naive Bayes runner without changing it.

## Completed work

### 1. Dataset analysis

All figures below come from `python -m banking77.analyze_data` (a read-only script I added)
and are documented in [docs/DATA.md](../docs/DATA.md).

- **Classes:** 77 intents, all present in every split. Splits: train 8,499, validation
  1,500, official test 3,080 (raw train had 10,003 rows).
- **Class distribution and imbalance:** train 30–159 examples per class (max/min ≈ 5.3);
  validation 5–28 (≈ 5.6); the official test set is perfectly balanced (40 per class).
  Smallest classes: `contactless_not_working` (30 train / 5 validation),
  `virtual_card_not_working` (35 / 6), `card_acceptance` (50 / 9).
- **Message length (train):** characters mean 59.74, median 47, min 13, max 433; words mean
  12.01, median 10, min 2, max 79. Validation and test statistics are in `docs/DATA.md`.
- **Missing and empty records:** none in any split.
- **Duplicates:** 0 exact duplicates; 4 normalized duplicates in raw train (differing only by
  leading/trailing newlines, same labels), already removed by the shared cleaning step; 1
  normalized duplicate inside the official test set (left unchanged).
- **Label consistency:** no text has conflicting labels, within or across splits.
- **Split integrity:** train/validation share 0 normalized texts; ids are unique across splits.
- **Exact and near-duplicate overlap:** train/test share 7 normalized texts (same labels), so
  the official test set is not a perfectly clean holdout. Punctuation-insensitive near
  duplicates: train/validation 4, train/test 21, validation/test 4.

### 2. Preprocessing review

- Documented the existing pipeline in `docs/DATA.md`: pinned source and SHA-256 check, CSV
  reading, cleaning (unknown labels rejected, blank rows dropped, normalized duplicates
  removed, conflicting labels rejected), stratified 15 % validation split with `seed=42`,
  unchanged official test, and TF-IDF fitted on training data only.
- **CSV line-ending fix (the only code change):** `write_records` in `src/banking77/data.py`
  used the csv module's default CRLF line ending while `.gitattributes` requires LF for
  `data/processed/*.csv`. Regenerating the data on Windows made the files appear modified and
  made file hashes platform-dependent. I set `lineterminator="\n"`.
- **Reproducibility verification:** after the fix, `python -m banking77.data --offline`
  regenerates `train.csv`, `validation.csv` and `test.csv` byte-for-byte identical to the
  committed versions (SHA-256 compared) and `git status` stays clean for `data/`. The dataset
  summary hash is unchanged (`468f55025cf719656d2351996fd0eb5b36b1ae4666a1c57d28743d9c565cadb5`).
- **Why the shared splits were preserved:** the handoff contract fixes `seed=42`, the
  stratified 15 % validation split and the official test set. Changing rows, for example
  removing near-duplicates, would change the dataset identity and invalidate every member's
  results. I therefore reported the near-duplicates instead of removing them. Whitespace
  differences were also left unchanged because normalizing them does not change the TF-IDF
  vocabulary (checked for both n-gram ranges).
- Data detected, corrected and intentionally left unchanged are listed separately in
  `docs/DATA.md`.

### 3. Feature experiment: unigram versus unigram+bigram

- Used the existing runner (`banking77.train_naive_bayes`) without modification, on the
  validation split only; the official test set was not used. Same training and validation
  data, TF-IDF fitted on training data only, `sublinear_tf=True`, `MultinomialNB`,
  `alpha=1.0`, `seed=42`. Only `ngram_range` differed.

| Experiment | `ngram_range` | Accuracy | Macro F1 | Features |
| --- | --- | ---: | ---: | ---: |
| A | (1,1) | 82.40 % | 0.7942 | 2,176 |
| B | (1,2) | 81.60 % | 0.7872 | 21,595 |

- Results match the saved `metrics.json` files and an independent scikit-learn recomputation.
  Run ids and commands are in [results/FEATURE_EXPERIMENTS.md](../results/FEATURE_EXPERIMENTS.md).
- Unigram+bigram has about 9.9x more features and about 2x more non-zero values per message
  (20.1 vs 10.3); single-run fit and predict times were roughly 2x higher.

```powershell
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes --split validation --ngram-max 1 --alpha 1.0
.\.venv\Scripts\python.exe -m banking77.train_naive_bayes --split validation --ngram-max 2 --alpha 1.0
```

- **Supplementary observation (not the basis of my recommendation):** one extra run of each
  configuration at `alpha=0.05` (the value from Member 1's alpha report) gave 84.87 % / 0.8422
  for unigram and 86.20 % / 0.8529 for unigram+bigram, i.e. the ranking reverses. Alpha tuning
  belongs to Member 1; I did no further tuning or error analysis.

### 4. Feature recommendation

- In the controlled `alpha=1.0` experiment, **unigram is preferred**: slightly higher accuracy
  and macro F1, and about ten times fewer features.
- The difference is small (0.80 points, 12 of 1,500 messages; exact McNemar p = 0.31), so it is
  not statistically distinguishable on this single split, and bigrams may perform better with
  other hyperparameters (as the supplementary observation suggests).
- The result applies to Naive Bayes only. **The final shared configuration must be agreed by
  the team**; my recommendation does not change any other model or the default in
  `build_naive_bayes`.

### 5. Evaluation metric justification

Documented in `docs/DATA.md` (section 4). Accuracy is simple and weights every message equally,
so large classes dominate; macro F1 averages F1 over all 77 classes equally and is the
selection metric, which suits a validation set that is imbalanced (5–28 per class) and rare
intents that matter as much as frequent ones. Caveat: some validation classes have only 5–9
examples, so macro F1 is noisy and small differences are weak evidence. The official test set
is balanced, where accuracy equals macro-averaged recall.

## Files created and modified

| File | Change |
| --- | --- |
| `src/banking77/data.py` | Modified: LF line endings in `write_records` (2 lines) |
| `src/banking77/analyze_data.py` | Created: read-only dataset statistics and overlap report |
| `tests/test_data_files.py` | Created: checks LF output and round-trip of embedded newlines |
| `docs/DATA.md` | Created: analysis, preprocessing review, decisions, metrics, licensing note |
| `results/FEATURE_EXPERIMENTS.md` | Created: experiment protocol, results, recommendation |
| `contributions/ADNAN_DEMIR_220911784_adnan-demir.md` | Created: this file |

I did not modify the README, LICENSE, raw or processed data, or any Naive Bayes, Logistic
Regression or SVM code.

## Tests and reproducibility

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m banking77.data --offline
.\.venv\Scripts\python.exe -m banking77.analyze_data
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

- All 8 tests pass (7 existing + 1 added). Environment: Python 3.12.10, scikit-learn 1.9.1,
  numpy 2.5.3. Data identity (`dataset_summary_sha256`) is the same in all my runs.

## Limitations and team coordination

- Single validation split and seed; no cross-validation; only `ngram_range` was varied
  (no `min_df`, stop-words or symbol features).
- Near-duplicates and the 7 train/test overlaps remain; removing them is a team decision
  because it would change the data identity and require every member to rerun experiments.
- The shared n-gram configuration needs agreement with Members 1, 3 and 4 after their own
  validation results.
- Suggestions for Member 1: link `docs/DATA.md` and `results/FEATURE_EXPERIMENTS.md` from the
  README and update `AGENTS.md`, which still describes only the Naive Bayes scope.

## GitHub evidence

- My commit links: not yet created (pending).
- My pull request links: not yet created (pending).

## Contribution to the presentation

- Dataset statistics, split and leakage findings, the unigram/bigram table and the metric
  justification above can be used as source material. No slides have been prepared by me yet.
