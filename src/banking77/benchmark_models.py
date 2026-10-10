"""Ortak benchmark: NB, LR ve Linear SVM'i sırayla çalıştırıp karşılaştırır.

Takım arkadaşlarının mevcut eğitim fonksiyonları değiştirilmeden kullanılır.
Yalnızca validation bölümü değerlendirilir; resmî test kümesine dokunulmaz.
"""

import argparse
import csv
import json
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from banking77.data import ROOT, read_records
from banking77.train_linear_svm import run_linear_svm
from banking77.train_logistic_regression import run_logistic_regression
from banking77.train_naive_bayes import run_naive_bayes

REPEATS = 3
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
NEAR_DUPLICATE_THRESHOLDS = (0.95, 0.90)
RESULTS = ROOT / "results"


@dataclass(frozen=True)
class Config:
    key: str
    model: str
    stage: str  # "initial", "selected" or "initial = selected"
    runner: object
    kwargs: dict
    label: str
    reported_macro_f1: float = None  # model sahibinin raporladığı validation değeri


CONFIGS = (
    Config("nb_initial", "Naive Bayes", "initial", run_naive_bayes, {"alpha": 1.0}, "alpha=1"),
    Config("nb_selected", "Naive Bayes", "selected", run_naive_bayes, {"alpha": 0.05}, "alpha=0.05", 0.8529),
    Config("lr_initial", "Logistic Regression", "initial", run_logistic_regression,
           {"solver": "lbfgs", "C": 1.0}, "lbfgs, C=1"),
    Config("lr_selected", "Logistic Regression", "selected", run_logistic_regression,
           {"solver": "liblinear-ovr", "C": 100.0}, "liblinear-ovr, C=100", 0.8920),
    Config("svm", "Linear SVM", "initial = selected", run_linear_svm,
           {"loss": "squared_hinge", "C": 1.0}, "squared_hinge, C=1", 0.8935),
)
SELECTED_KEYS = ("nb_selected", "lr_selected", "svm")
IMPLEMENTATIONS = {
    "Naive Bayes": "TF-IDF + MultinomialNB (scikit-learn)",
    "Logistic Regression": "TF-IDF + LogisticRegression (scikit-learn)",
    "Linear SVM": "TF-IDF + LinearSVC (scikit-learn)",
}
MODEL_LABELS = {"nb_selected": "Naive Bayes", "lr_selected": "Logistic Regression", "svm": "Linear SVM"}


def run_all(repeats=REPEATS):
    """Her turda tüm ayarları bir kez çalıştırır; süreler için turlar tekrarlanır."""
    rounds = []
    for _ in range(repeats):
        rounds.append({config.key: config.runner("validation", **config.kwargs) for config in CONFIGS})
    return rounds


def aggregate(rounds):
    """Tekrarları birleştirir: skorlar birebir aynı olmalı, süreler için medyan alınır."""
    result = {}
    for config in CONFIGS:
        runs = [round_[config.key] for round_ in rounds]
        first = runs[0]
        for run in runs[1:]:
            if (run["accuracy"], run["macro_f1"]) != (first["accuracy"], first["macro_f1"]):
                raise ValueError(f"{config.key}: scores changed between repeats")
            if run["dataset_summary_sha256"] != first["dataset_summary_sha256"]:
                raise ValueError(f"{config.key}: data changed between repeats")
            if run["dataset_files_sha256"] != first["dataset_files_sha256"]:
                raise ValueError(f"{config.key}: data files changed between repeats")
        result[config.key] = {
            "model": config.model,
            "stage": config.stage,
            "label": config.label,
            "settings": first["settings"],
            "run_id": first["run_id"],
            "repeat_run_ids": [run["run_id"] for run in runs],
            "dataset_summary_sha256": first["dataset_summary_sha256"],
            "dataset_files_sha256": first["dataset_files_sha256"],
            "converged": first.get("converged"),
            "optimizer_iterations": first.get("optimizer_iterations"),
            "training_rows": first["training_rows"],
            "evaluation_rows": first["evaluation_rows"],
            "accuracy": first["accuracy"],
            "macro_f1": first["macro_f1"],
            "fit_seconds": statistics.median(run["fit_seconds"] for run in runs),
            "predict_seconds": statistics.median(run["predict_seconds"] for run in runs),
            "prediction_ms_per_message": statistics.median(run["prediction_ms_per_message"] for run in runs),
            "fit_seconds_all": [run["fit_seconds"] for run in runs],
            "predict_seconds_all": [run["predict_seconds"] for run in runs],
            "matches_reported_macro_f1": (
                None if config.reported_macro_f1 is None
                else round(first["macro_f1"], 4) == config.reported_macro_f1
            ),
        }
    hashes = {entry["dataset_summary_sha256"] for entry in result.values()}
    if len(hashes) != 1:
        raise ValueError("Compared runs use different dataset_summary_sha256 values")
    files = {json.dumps(entry["dataset_files_sha256"], sort_keys=True) for entry in result.values()}
    if len(files) != 1:
        raise ValueError("Compared runs use different prepared data files")
    return result


def read_predictions(run_id):
    path = ROOT / "results/runs" / run_id / "predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_comparison(reference_correct, candidate_correct):
    """Hizalı mesaj başı doğruluk listeleri üzerinde exact McNemar testi."""
    only_reference = sum(r and not c for r, c in zip(reference_correct, candidate_correct))
    only_candidate = sum(c and not r for r, c in zip(reference_correct, candidate_correct))
    discordant = only_reference + only_candidate
    p_value = binomtest(only_candidate, discordant, 0.5).pvalue if discordant else 1.0
    return {
        "reference_only_correct": only_reference,
        "candidate_only_correct": only_candidate,
        "mcnemar_exact_p": float(p_value),
    }


def macro_f1_codes(true, predicted, classes):
    """Tüm sınıflar üzerinden macro F1; sklearn gibi eksik sınıf 0 sayılır."""
    tp = np.bincount(true[true == predicted], minlength=classes)
    denominator = np.bincount(true, minlength=classes) + np.bincount(predicted, minlength=classes)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(classes), where=denominator > 0)
    return float(f1.mean())


def bootstrap_difference(true, reference, candidate, classes, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Macro F1 farkı (aday - referans) için eşleştirilmiş bootstrap %95 aralığı."""
    rng = np.random.default_rng(seed)
    size = len(true)
    differences = np.empty(resamples)
    for i in range(resamples):
        index = rng.integers(0, size, size)
        differences[i] = (
            macro_f1_codes(true[index], candidate[index], classes)
            - macro_f1_codes(true[index], reference[index], classes)
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    return {
        "observed": macro_f1_codes(true, candidate, classes) - macro_f1_codes(true, reference, classes),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "resamples": resamples,
        "seed": seed,
    }


def top_confusions(rows, limit=5):
    pairs = Counter((r["true_label"], r["predicted_label"]) for r in rows if r["correct"] == "False")
    return [{"true_label": t, "predicted_label": p, "count": n} for (t, p), n in pairs.most_common(limit)]


def pair_table(predictions, limit=5):
    """Her modelin en sık karışan çiftlerinin birleşimi için üç modelin sayıları."""
    counts = {
        key: Counter((r["true_label"], r["predicted_label"]) for r in rows if r["correct"] == "False")
        for key, rows in predictions.items()
    }
    union = []
    for key in predictions:
        for pair, _ in counts[key].most_common(limit):
            if pair not in union:
                union.append(pair)
    union.sort(key=lambda pair: (-sum(counts[k][pair] for k in counts), pair))
    return [
        {"true_label": t, "predicted_label": p, "counts": {k: counts[k][(t, p)] for k in counts}}
        for t, p in union
    ]


def error_examples(predictions, all_wrong_limit=5, disagreement_limit=3):
    """Gerçek mesajlardan örnek seçer: üç model de yanlış ve modellerin ayrıştığı."""
    keys = list(predictions)
    by_id = {key: {row["id"]: row for row in predictions[key]} for key in keys}
    ids = sorted(by_id[keys[0]])
    frequency = Counter(
        (r["true_label"], r["predicted_label"]) for r in predictions["svm"] if r["correct"] == "False"
    )

    def view(message_id):
        row = by_id[keys[0]][message_id]
        return {
            "id": message_id,
            "text": row["text"],
            "true_label": row["true_label"],
            "predictions": {key: by_id[key][message_id]["predicted_label"] for key in keys},
        }

    all_wrong = [i for i in ids if all(by_id[k][i]["correct"] == "False" for k in keys)]
    all_wrong.sort(key=lambda i: (-frequency[(by_id["svm"][i]["true_label"], by_id["svm"][i]["predicted_label"])], i))
    disagree = [
        i for i in ids
        if 0 < sum(by_id[k][i]["correct"] == "True" for k in keys) < len(keys)
    ]
    return {
        "all_models_wrong_count": len(all_wrong),
        "all_models_wrong": [view(i) for i in all_wrong[:all_wrong_limit]],
        "disagreement_count": len(disagree),
        "disagreement": [view(i) for i in disagree[:disagreement_limit]],
    }


def nearest_train_similarity(train_texts, eval_texts, chunk=500):
    """Her değerlendirme mesajının train'deki en benzer mesaja karakter n-gram kosinüs benzerliği."""
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    train_matrix = vectorizer.fit_transform(train_texts)  # yalnızca train üzerinde öğrenilir
    eval_matrix = vectorizer.transform(eval_texts)
    best = np.empty(len(eval_texts))
    for start in range(0, len(eval_texts), chunk):
        block = (eval_matrix[start:start + chunk] @ train_matrix.T).toarray()
        best[start:start + chunk] = block.max(axis=1)
    return best


def subset_scores(true_labels, predicted_labels):
    """Alt küme skorları; macro F1 yalnızca alt kümede bulunan gerçek etiketler üzerinden."""
    labels = sorted(set(true_labels))
    return {
        "rows": len(true_labels),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(f1_score(true_labels, predicted_labels, labels=labels, average="macro", zero_division=0)),
        "categories": len(labels),
    }


def near_duplicate_sensitivity(models, thresholds=NEAR_DUPLICATE_THRESHOLDS):
    """Train'e çok benzeyen validation mesajları çıkarılınca skorlar nasıl değişir."""
    processed = ROOT / "data/processed"
    train = read_records(processed / "train.csv")
    validation = read_records(processed / "validation.csv")
    similarity = {
        row["id"]: score
        for row, score in zip(validation, nearest_train_similarity(
            [r["text"] for r in train], [r["text"] for r in validation]))
    }
    predictions = {key: read_predictions(entry["run_id"]) for key, entry in models.items()}
    result = {}
    for threshold in thresholds:
        flagged = {message_id for message_id, score in similarity.items() if score >= threshold}
        per_model = {}
        for key, rows in predictions.items():
            kept = [row for row in rows if row["id"] not in flagged]
            per_model[key] = {
                "all": subset_scores([r["true_label"] for r in rows], [r["predicted_label"] for r in rows]),
                "without_near_duplicates": subset_scores(
                    [r["true_label"] for r in kept], [r["predicted_label"] for r in kept]),
            }
        result[f"{threshold:.2f}"] = {
            "flagged_messages": len(flagged),
            "flagged_share": len(flagged) / len(similarity),
            "models": per_model,
        }
    return {
        "method": "max cosine similarity of character 3-5-gram TF-IDF (fitted on train) to any training "
                  "message; validation messages at or above the threshold are removed",
        "thresholds": result,
    }


def class_balance():
    summary = json.loads((ROOT / "data/processed/summary.json").read_text(encoding="utf-8"))
    with (ROOT / "data/processed/validation.csv").open(encoding="utf-8", newline="") as handle:
        validation = Counter(row["category"] for row in csv.DictReader(handle))
    with (ROOT / "data/processed/test.csv").open(encoding="utf-8", newline="") as handle:
        test = Counter(row["category"] for row in csv.DictReader(handle))
    train = summary["class_counts"]["train"]
    return {
        "classes": len(train),
        "train_min": min(train.values()),
        "train_max": max(train.values()),
        "train_min_class": min(train, key=train.get),
        "train_max_class": max(train, key=train.get),
        "validation_min": min(validation.values()),
        "validation_max": max(validation.values()),
        "test_min": min(test.values()),
        "test_max": max(test.values()),
    }


def build_summary(rounds):
    models = aggregate(rounds)
    predictions = {key: read_predictions(models[key]["run_id"]) for key in SELECTED_KEYS}
    categories = json.loads((ROOT / "data/processed/categories.json").read_text(encoding="utf-8"))
    index = {name: i for i, name in enumerate(categories)}
    ids = [row["id"] for row in predictions["svm"]]
    for key in SELECTED_KEYS:
        if [row["id"] for row in predictions[key]] != ids:
            raise ValueError("Predictions must cover the same records in the same order")
    true = np.array([index[row["true_label"]] for row in predictions["svm"]])
    coded = {k: np.array([index[r["predicted_label"]] for r in predictions[k]]) for k in SELECTED_KEYS}
    correct = {k: [r["correct"] == "True" for r in predictions[k]] for k in SELECTED_KEYS}
    comparisons = []
    for reference, candidate in (("nb_selected", "lr_selected"), ("nb_selected", "svm"), ("lr_selected", "svm")):
        entry = {"reference": reference, "candidate": candidate}
        entry.update(paired_comparison(correct[reference], correct[candidate]))
        entry["macro_f1_difference"] = bootstrap_difference(true, coded[reference], coded[candidate], len(categories))
        comparisons.append(entry)
    return {
        "command": "python -m banking77.benchmark_models",
        "evaluation_split": "validation",
        "official_test_evaluated": False,
        "repeats": len(rounds),
        "timing": "median over repeats; fit includes TF-IDF, predict includes TF-IDF transform; "
                  "one machine, models run sequentially",
        "dataset_summary_sha256": next(iter(models.values()))["dataset_summary_sha256"],
        "models": models,
        "class_balance": class_balance(),
        "comparisons": comparisons,
        "near_duplicate_sensitivity": near_duplicate_sensitivity(models),
        "top_confusions": {key: top_confusions(predictions[key]) for key in SELECTED_KEYS},
        "confusion_pair_table": pair_table(predictions),
        "examples": error_examples(predictions),
    }


def _pct(value):
    return f"{100 * value:.2f}%"


def _comparison_sentence(item):
    diff = item["macro_f1_difference"]
    first, second = MODEL_LABELS[item["reference"]], MODEL_LABELS[item["candidate"]]
    verdict = "the interval includes 0" if diff["ci95_low"] <= 0 <= diff["ci95_high"] else "the interval excludes 0"
    return (
        f"- {second} vs {first}: {item['candidate_only_correct']} messages fixed, "
        f"{item['reference_only_correct']} broken (exact McNemar p = {item['mcnemar_exact_p']:.2g}); "
        f"macro F1 difference {diff['observed']:+.4f}, paired bootstrap 95% interval "
        f"[{diff['ci95_low']:+.4f}, {diff['ci95_high']:+.4f}] ({verdict})."
    )


def render_markdown(summary):
    models, balance = summary["models"], summary["class_balance"]
    lines = [
        "# Model comparison (validation)",
        "",
        "All models use the same prepared data, the same TF-IDF features (unigram + bigram, sublinear TF)",
        "and run on the validation split only. The official test set was not evaluated.",
        f"Command: `{summary['command']}`. Dataset summary SHA-256: `{summary['dataset_summary_sha256']}`.",
        f"Training rows: {next(iter(models.values()))['training_rows']}, "
        f"validation rows: {next(iter(models.values()))['evaluation_rows']}.",
        "",
        "## Models",
        "",
        "| Model | Implementation | Learned from |",
        "| --- | --- | --- |",
    ]
    for name, implementation in IMPLEMENTATIONS.items():
        lines.append(f"| {name} | {implementation} | our training split, no pretrained weights |")
    lines += [
        "",
        "All three are the vanilla course methods used as baselines. The selected settings are tuned "
        "versions of the same baselines, not new methods. We use scikit-learn implementations: the "
        "algorithms are not coded from scratch, but all parameters are learned from our own training data.",
        "",
        "## Results",
        "",
        "| Model | Stage | Settings | Accuracy | Macro F1 | Fit (s) | Predict (s) | Predict (ms/msg) | Converged |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for entry in models.values():
        converged = {True: "yes", False: "no", None: "n/a"}[entry["converged"]]
        lines.append(
            f"| {entry['model']} | {entry['stage']} | {entry['label']} | {_pct(entry['accuracy'])} "
            f"| {entry['macro_f1']:.4f} | {entry['fit_seconds']:.2f} | {entry['predict_seconds']:.4f} "
            f"| {entry['prediction_ms_per_message']:.3f} | {converged} |"
        )
    lines += [
        "",
        f"Times are medians of {summary['repeats']} repeats on one machine, models run sequentially. "
        "Fit includes TF-IDF fitting; predict includes the TF-IDF transform. "
        "Absolute times depend on the machine and are only comparable within this table.",
        "",
        "Initial and selected settings are shown separately. The initial settings are the baselines "
        "defined in `docs/EXPERIMENTS.md`; the selected settings were chosen by each model owner on "
        "validation macro F1. Linear SVM's selected setting equals its initial one, so it has a single row.",
        "",
        "## Why macro F1",
        "",
        f"The task has {balance['classes']} categories and every one matters equally for a bank: a rare "
        "request type (for example a swallowed card) is as important to route correctly as a frequent one. "
        "Macro F1 averages the per-category F1 with equal weight, so a model cannot hide poor performance "
        "on small categories behind good performance on large ones. Accuracy counts every message equally, "
        "so it favours frequent categories; it is reported next to macro F1 as a supporting metric.",
        "",
        f"Category sizes are not equal: in training, from {balance['train_min']} messages "
        f"(`{balance['train_min_class']}`) to {balance['train_max']} (`{balance['train_max_class']}`); "
        f"in validation, from {balance['validation_min']} to {balance['validation_max']}. "
        "The official test set is balanced ("
        + (f"{balance['test_min']} messages in every category" if balance["test_min"] == balance["test_max"]
           else f"{balance['test_min']} to {balance['test_max']} messages per category")
        + "), so accuracy and macro F1 are expected to be closer there. We still rank by macro F1 "
        "because the reason for choosing it, equal importance of every category, does not depend on the split.",
        "",
        "## Paired comparison of the selected settings",
        "",
        "Score differences between models are small, so each pair is compared on the same validation "
        "messages. A single validation split does not allow a firm ranking when the interval includes 0.",
        "",
    ]
    lines += [_comparison_sentence(item) for item in summary["comparisons"]]
    sensitivity = summary["near_duplicate_sensitivity"]
    lines += [
        "",
        "## Sensitivity to near-duplicate messages",
        "",
        "Some validation messages are near-copies of training messages (reordered sentences, one added word). "
        "They are mostly easy and keep the same label, so they can raise the scores slightly. We did not "
        "change the split; instead the scores are recomputed without those messages.",
        "",
        f"Method: {sensitivity['method']}. Macro F1 in the reduced sets uses only the categories that remain.",
        "",
        "| Threshold | Removed | Model | Setting | Accuracy (all → reduced) | Macro F1 (all → reduced) |",
        "| --- | ---: | --- | --- | ---: | ---: |",
    ]
    for threshold, block in sensitivity["thresholds"].items():
        first = True
        for key, entry in models.items():
            scores = block["models"][key]
            lines.append(
                f"| {'≥ ' + threshold if first else ''} "
                f"| {str(block['flagged_messages']) + ' (' + format(100 * block['flagged_share'], '.1f') + '%)' if first else ''} "
                f"| {entry['model']} | {entry['label']} "
                f"| {_pct(scores['all']['accuracy'])} → {_pct(scores['without_near_duplicates']['accuracy'])} "
                f"| {scores['all']['macro_f1']:.4f} → {scores['without_near_duplicates']['macro_f1']:.4f} |"
            )
            first = False
    lines += ["", "## Most confused category pairs (selected settings)", ""]
    for key, pairs in summary["top_confusions"].items():
        lines.append(f"**{MODEL_LABELS[key]}**")
        lines.append("")
        lines += [f"- `{p['true_label']}` → `{p['predicted_label']}`: {p['count']}" for p in pairs]
        lines.append("")
    examples = summary["examples"]
    lines += [
        "## Example errors (selected settings)",
        "",
        f"{examples['all_models_wrong_count']} validation messages are wrong for all three models; "
        f"{examples['disagreement_count']} are right for some models and wrong for others.",
        "",
        "Wrong for all three models:",
        "",
        "| Message | True | Naive Bayes | Logistic Regression | Linear SVM |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in examples["all_models_wrong"]:
        p = item["predictions"]
        lines.append(f"| {item['text']} | `{item['true_label']}` | `{p['nb_selected']}` "
                     f"| `{p['lr_selected']}` | `{p['svm']}` |")
    lines += [
        "",
        "Models disagree:",
        "",
        "| Message | True | Naive Bayes | Logistic Regression | Linear SVM |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in examples["disagreement"]:
        p = item["predictions"]
        lines.append(f"| {item['text']} | `{item['true_label']}` | `{p['nb_selected']}` "
                     f"| `{p['lr_selected']}` | `{p['svm']}` |")
    lines += [
        "",
        "## Figures",
        "",
        "- `results/figures/scores.png`: macro F1 and accuracy, initial vs selected",
        "- `results/figures/timing.png`: fit and prediction time",
        "- `results/figures/confusions.png`: most confused category pairs",
        "",
        "## Limitations",
        "",
        "- One validation split of 1,500 messages; scores are not final test results.",
        "- Final test numbers will be produced by the model owners after the shared feature setting is fixed.",
        "- Timings come from one machine and are not a general speed claim.",
        "- Near-duplicate detection is a similarity heuristic with an arbitrary threshold, not a proof of leakage.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(summary):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "model_comparison_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    (RESULTS / "MODEL_COMPARISON.md").write_text(render_markdown(summary), encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    summary = build_summary(run_all(args.repeats))
    write_outputs(summary)
    if not args.no_figures:
        from banking77.plot_comparison import make_figures
        make_figures(summary, Path(RESULTS / "figures"))
    for entry in summary["models"].values():
        print(f"{entry['model']:<20} {entry['label']:<22} acc={entry['accuracy']:.4f} "
              f"macro_f1={entry['macro_f1']:.4f} fit={entry['fit_seconds']:.2f}s "
              f"predict={entry['predict_seconds']:.4f}s")


if __name__ == "__main__":
    main()
