"""Unigram ve unigram+bigram özelliklerini üç modelde adil biçimde karşılaştırır.

Her özellik ayarı için her modelin kendi hiperparametre ızgarası (model sahiplerinin
ızgaraları) yalnızca validation üzerinde çalıştırılır ve aynı kuralla (en yüksek
macro F1, eşitlikte ilk ayar) en iyi ayar seçilir. Böylece bigram için seçilmiş
ayarların unigram'ı haksız yere dezavantajlı bırakması engellenir.
Ek olarak yalnızca bigram (ngram_range=(2,2)) ablasyonu aynı ızgaralarla çalışır;
bu ablasyon model sahiplerinin eğitim fonksiyonlarını değiştirmeden, onların
pipeline kurucuları üzerinde TF-IDF aralığını değiştirerek yapılır.
Resmî test kümesi kullanılmaz.
"""

import argparse
import json
from collections import OrderedDict

import hashlib
import warnings
from time import perf_counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from banking77.benchmark_linear_svm import schedule as svm_schedule
from banking77.benchmark_logistic_regression import schedule as lr_schedule
from banking77.benchmark_models import (
    CONFIGS, bootstrap_difference, paired_comparison, read_predictions,
)
from banking77.benchmark_naive_bayes import ALPHAS
from banking77.data import ROOT, read_records
from banking77.linear_svm import build_linear_svm
from banking77.logistic_regression import build_logistic_regression
from banking77.naive_bayes import build_naive_bayes
from banking77.train_linear_svm import run_linear_svm
from banking77.train_logistic_regression import run_logistic_regression
from banking77.train_naive_bayes import run_naive_bayes

NGRAMS = (1, 2)
MODELS = OrderedDict([
    ("Naive Bayes", (run_naive_bayes, [{"alpha": alpha} for alpha in ALPHAS], "nb_selected")),
    ("Logistic Regression", (run_logistic_regression, lr_schedule(), "lr_selected")),
    ("Linear SVM", (run_linear_svm, svm_schedule(), "svm")),
])


BUILDERS = {
    "Naive Bayes": lambda kw: build_naive_bayes(2, kw["alpha"]),
    "Logistic Regression": lambda kw: build_logistic_regression(2, kw["C"], kw["solver"]),
    "Linear SVM": lambda kw: build_linear_svm(2, kw["C"], kw["loss"]),
}


def run_bigram_only(model, kwargs):
    """Yalnızca bigram ablasyonu: aynı kurucu, TF-IDF aralığı (2,2); tahminler bellekte tutulur."""
    processed = ROOT / "data/processed"
    train = read_records(processed / "train.csv")
    validation = read_records(processed / "validation.csv")
    categories = json.loads((processed / "categories.json").read_text(encoding="utf-8"))
    pipeline = BUILDERS[model](kwargs)
    pipeline.set_params(tfidf__ngram_range=(2, 2))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        start = perf_counter()
        pipeline.fit([r["text"] for r in train], [r["category"] for r in train])
        fit_seconds = perf_counter() - start
    start = perf_counter()
    predicted = pipeline.predict([r["text"] for r in validation])
    predict_seconds = perf_counter() - start
    actual = [r["category"] for r in validation]
    key = json.dumps(kwargs, sort_keys=True)
    return {
        "run_id": f"ablation_bigram_only_{hashlib.sha256((model + key).encode()).hexdigest()[:10]}",
        "evaluation_split": "validation",
        "settings": {**kwargs, "ngram_range": [2, 2]},
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=categories, average="macro", zero_division=0)),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "dataset_summary_sha256": hashlib.sha256((processed / "summary.json").read_bytes()).hexdigest(),
        "predictions": [
            {"id": r["id"], "true_label": r["category"], "predicted_label": p, "correct": str(r["category"] == p)}
            for r, p in zip(validation, predicted)
        ],
    }


def select_best(runs):
    """Model sahipleriyle aynı kural: en yüksek validation macro F1, eşitlikte ilk ayar."""
    if not runs or any(run["evaluation_split"] != "validation" for run in runs):
        raise ValueError("Selection requires nonempty validation-only runs")
    return max(runs, key=lambda run: run["macro_f1"])


def compact(run):
    return {
        "run_id": run["run_id"],
        "settings": {k: v for k, v in run["settings"].items() if k not in ("seed", "random_state")},
        "accuracy": run["accuracy"],
        "macro_f1": run["macro_f1"],
        "fit_seconds": run["fit_seconds"],
        "predict_seconds": run["predict_seconds"],
        "dataset_summary_sha256": run["dataset_summary_sha256"],
    }


def same_setting(run, kwargs):
    return all(run["settings"].get(key) == value for key, value in kwargs.items())


def run_grids():
    """{ngram: {model: [metrics]}}; her koşu validation üzerinde, sırayla çalışır.

    "bigram_only" anahtarı (2,2) ablasyonudur.
    """
    grids = {}
    for ngram in NGRAMS:
        grids[ngram] = {}
        for model, (runner, schedule, _) in MODELS.items():
            grids[ngram][model] = [runner("validation", ngram_max=ngram, **kwargs) for kwargs in schedule]
    grids["bigram_only"] = {
        model: [run_bigram_only(model, kwargs) for kwargs in schedule]
        for model, (_, schedule, _) in MODELS.items()
    }
    return grids


def vocabulary_sizes():
    train = read_records(ROOT / "data/processed/train.csv")
    texts = [row["text"] for row in train]
    sizes = {
        ngram: len(TfidfVectorizer(ngram_range=(1, ngram), sublinear_tf=True).fit(texts).vocabulary_)
        for ngram in NGRAMS
    }
    sizes["bigram_only"] = len(TfidfVectorizer(ngram_range=(2, 2), sublinear_tf=True).fit(texts).vocabulary_)
    return sizes


def _aligned(first_run, second_run, categories):
    """İki koşunun aynı mesajlar üzerindeki doğruluk ve etiket kodları."""
    first = first_run.get("predictions") or read_predictions(first_run["run_id"])
    second = second_run.get("predictions") or read_predictions(second_run["run_id"])
    if [r["id"] for r in first] != [r["id"] for r in second]:
        raise ValueError("Compared runs must cover the same records in the same order")
    index = {name: i for i, name in enumerate(categories)}
    true = np.array([index[r["true_label"]] for r in first])
    return (
        [r["correct"] == "True" for r in first], [r["correct"] == "True" for r in second], true,
        np.array([index[r["predicted_label"]] for r in first]),
        np.array([index[r["predicted_label"]] for r in second]),
    )


def compare(unigram_run, bigram_run, categories, names=("unigram", "bigram")):
    correct_uni, correct_bi, true, predicted_uni, predicted_bi = _aligned(unigram_run, bigram_run, categories)
    result = {names[0]: compact(unigram_run), names[1]: compact(bigram_run)}
    result.update(paired_comparison(correct_uni, correct_bi))  # aday = ikinci koşu, referans = ilk koşu
    result["macro_f1_difference_candidate_minus_reference"] = bootstrap_difference(
        true, predicted_uni, predicted_bi, len(categories))
    return result


def build_summary(grids):
    categories = json.loads((ROOT / "data/processed/categories.json").read_text(encoding="utf-8"))
    hashes = {run["dataset_summary_sha256"] for g in grids.values() for runs in g.values() for run in runs}
    if len(hashes) != 1:
        raise ValueError("Runs use different dataset_summary_sha256 values")
    selected_kwargs = {c.key: c.kwargs for c in CONFIGS}
    models = OrderedDict()
    for model, (_, schedule, key) in MODELS.items():
        per_ngram = {}
        for ngram in NGRAMS:
            runs = grids[ngram][model]
            fixed = next(run for run in runs if same_setting(run, selected_kwargs[key]))
            per_ngram[ngram] = {"best": select_best(runs), "fixed": fixed, "grid": runs}
        models[model] = {
            "grid_size": len(schedule),
            "fixed_settings": {k: v for k, v in selected_kwargs[key].items()},
            "grids": {str(n): [compact(r) for r in per_ngram[n]["grid"]] for n in NGRAMS},
            "same_hyperparameters": compare(per_ngram[1]["fixed"], per_ngram[2]["fixed"], categories),
            "retuned_per_feature_setting": compare(per_ngram[1]["best"], per_ngram[2]["best"], categories),
        }
        bigram_only_runs = grids["bigram_only"][model]
        best_bigram_only = select_best(bigram_only_runs)
        models[model]["grids"]["bigram_only"] = [compact(r) for r in bigram_only_runs]
        models[model]["ablation_bigram_only_vs_unigram_bigram"] = compare(
            per_ngram[2]["best"], best_bigram_only, categories, names=("unigram_bigram", "bigram_only"))
        models[model]["ablation_bigram_only_vs_unigram"] = compare(
            per_ngram[1]["best"], best_bigram_only, categories, names=("unigram", "bigram_only"))
    return {
        "command": "python -m banking77.benchmark_features",
        "evaluation_split": "validation",
        "official_test_evaluated": False,
        "selection_rule": "highest validation macro F1, ties keep the first scheduled setting",
        "dataset_summary_sha256": hashes.pop(),
        "vocabulary_size": {str(k): v for k, v in vocabulary_sizes().items()},
        "models": models,
    }


def _setting(compact_run):
    """Tabloda yalnızca ayarlanan hiperparametreler gösterilir."""
    keys = ("alpha", "solver", "loss", "C")
    return ", ".join(f"{k}={compact_run['settings'][k]:g}" if isinstance(compact_run["settings"][k], float)
                     else f"{k}={compact_run['settings'][k]}" for k in keys if k in compact_run["settings"])


def _row(model, label, entry):
    uni, bi = entry["unigram"], entry["bigram"]
    diff = entry["macro_f1_difference_candidate_minus_reference"]
    interval = "includes 0" if diff["ci95_low"] <= 0 <= diff["ci95_high"] else "excludes 0"
    return (
        f"| {model} | {label} | {_setting(uni)} | {_setting(bi)} "
        f"| {uni['macro_f1']:.4f} / {100 * uni['accuracy']:.2f}% "
        f"| {bi['macro_f1']:.4f} / {100 * bi['accuracy']:.2f}% "
        f"| {diff['observed']:+.4f} [{diff['ci95_low']:+.4f}, {diff['ci95_high']:+.4f}] ({interval}) "
        f"| {entry['candidate_only_correct']} / {entry['reference_only_correct']} "
        f"| {entry['mcnemar_exact_p']:.2g} |"
    )


def render_markdown(summary):
    vocab = summary["vocabulary_size"]
    lines = [
        "# Unigram vs unigram+bigram (validation)",
        "",
        "Same prepared data, TF-IDF with sublinear TF, validation split only; the official test set was not used.",
        f"Command: `{summary['command']}`. Dataset summary SHA-256: `{summary['dataset_summary_sha256']}`.",
        f"Selection rule for tuned settings: {summary['selection_rule']}.",
        "",
        f"Vocabulary size (fitted on train only): unigram {vocab['1']:,}, unigram+bigram {vocab['2']:,}, "
        f"bigram only {vocab['bigram_only']:,}.",
        "",
        "Two comparisons are shown. **Same hyperparameters** keeps the settings that the model owners selected "
        "with bigram features and only changes the features; this can disadvantage unigram. "
        "**Re-tuned** runs each model's full grid for each feature setting and compares the best of each, "
        "which is the fair comparison.",
        "",
        "| Model | Comparison | Unigram setting | Bigram setting | Unigram macro F1 / acc | "
        "Bigram macro F1 / acc | Macro F1 difference (bigram − unigram) | Fixed / broken by bigram | McNemar p |",
        "| --- | --- | --- | --- | ---: | ---: | --- | ---: | ---: |",
    ]
    for model, entry in summary["models"].items():
        lines.append(_row(model, "same hyperparameters", entry["same_hyperparameters"]))
        lines.append(_row(model, "re-tuned", entry["retuned_per_feature_setting"]))
    lines += [
        "",
        "## Ablation: bigram only (`ngram_range=(2,2)`)",
        "",
        "Not part of the owners' training functions; run with the same grids and selection rule by "
        "changing only the TF-IDF range of their pipeline builders.",
        "",
        "| Model | Unigram best | Unigram + bigram best | Bigram-only best | Bigram-only − (unigram + bigram) | McNemar p |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for model, entry in summary["models"].items():
        both = entry["ablation_bigram_only_vs_unigram_bigram"]
        uni = entry["retuned_per_feature_setting"]["unigram"]
        diff = both["macro_f1_difference_candidate_minus_reference"]
        interval = "includes 0" if diff["ci95_low"] <= 0 <= diff["ci95_high"] else "excludes 0"
        lines.append(
            f"| {model} | {uni['macro_f1']:.4f} | {both['unigram_bigram']['macro_f1']:.4f} "
            f"| {both['bigram_only']['macro_f1']:.4f} ({_setting(both['bigram_only'])}) "
            f"| {diff['observed']:+.4f} [{diff['ci95_low']:+.4f}, {diff['ci95_high']:+.4f}] ({interval}) "
            f"| {both['mcnemar_exact_p']:.2g} |"
        )
    lines += ["", "## Full grids (validation macro F1)", ""]
    for model, entry in summary["models"].items():
        lines += [f"**{model}** ({entry['grid_size']} settings)", "",
                  "| Setting | Unigram | Unigram + bigram | Bigram only | Fit (s) uni / uni+bi / bi |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for uni, bi, only in zip(entry["grids"]["1"], entry["grids"]["2"], entry["grids"]["bigram_only"]):
            lines.append(f"| {_setting(uni)} | {uni['macro_f1']:.4f} | {bi['macro_f1']:.4f} | {only['macro_f1']:.4f} "
                         f"| {uni['fit_seconds']:.2f} / {bi['fit_seconds']:.2f} / {only['fit_seconds']:.2f} |")
        lines.append("")
    lines += [
        "## Limitations",
        "",
        "- One validation split of 1,500 messages; differences of this size are within sampling noise "
        "when the interval includes 0.",
        "- Times are single measurements from one machine, not medians of repeats.",
        "- Only two feature settings (unigram, unigram+bigram) are compared; no other feature engineering.",
        "",
    ]
    return "\n".join(lines)


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    summary = build_summary(run_grids())
    results = ROOT / "results"
    (results / "feature_comparison_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    (results / "FEATURE_COMPARISON.md").write_text(render_markdown(summary), encoding="utf-8", newline="\n")
    for model, entry in summary["models"].items():
        for label in ("same_hyperparameters", "retuned_per_feature_setting"):
            e = entry[label]
            print(f"{model:<20} {label:<28} uni={e['unigram']['macro_f1']:.4f} bi={e['bigram']['macro_f1']:.4f} "
                  f"p={e['mcnemar_exact_p']:.2g}")


if __name__ == "__main__":
    main()
