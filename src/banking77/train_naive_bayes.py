"""Train and inspect the repository owner's Naive Bayes model."""

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from time import perf_counter

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from banking77.data import ROOT, normalize_text, read_records
from banking77.naive_bayes import build_naive_bayes


def run_naive_bayes(split="validation", ngram_max=2, alpha=1.0):
    """Train on the prepared train split and save one auditable model run."""
    if split not in ("validation", "test"):
        raise ValueError("split must be validation or test")
    pipeline = build_naive_bayes(ngram_max, alpha)
    processed = ROOT / "data/processed"
    if not (processed / "summary.json").exists():
        raise ValueError("Run python -m banking77.data first")
    train = read_records(processed / "train.csv")
    evaluation = read_records(processed / f"{split}.csv")
    categories = json.loads((processed / "categories.json").read_text(encoding="utf-8"))
    data_summary = json.loads((processed / "summary.json").read_text(encoding="utf-8"))
    if not train or not evaluation:
        raise ValueError("Training and evaluation splits must not be empty")
    dataset_files_sha256 = {
        name: hashlib.sha256((processed / name).read_bytes()).hexdigest()
        for name in ("train.csv", f"{split}.csv", "categories.json", "summary.json")
    }
    start = perf_counter()
    pipeline.fit([row["text"] for row in train], [row["category"] for row in train])
    fit_seconds = perf_counter() - start
    start = perf_counter()
    predicted = pipeline.predict([row["text"] for row in evaluation])
    predict_seconds = perf_counter() - start
    actual = [row["category"] for row in evaluation]
    report = classification_report(actual, predicted, labels=categories, output_dict=True, zero_division=0)
    dataset_sha = hashlib.sha256((processed / "summary.json").read_bytes()).hexdigest()
    settings = {"model": "naive_bayes", "ngram_max": ngram_max, "alpha": alpha, "seed": data_summary["seed"], "tfidf_sublinear_tf": True}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"naive_bayes_{split}_{stamp}"
    output = ROOT / "results/runs" / run_id
    output.mkdir(parents=True, exist_ok=False)
    metrics = {
        "run_id": run_id,
        "evaluation_split": split,
        "settings": settings,
        "dataset_summary_sha256": dataset_sha,
        "dataset_files_sha256": dataset_files_sha256,
        "training_rows": len(train),
        "evaluation_rows": len(evaluation),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=categories, average="macro", zero_division=0),
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "prediction_ms_per_message": 1000 * predict_seconds / len(evaluation),
        "environment": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__},
    }
    train_keys = {normalize_text(row["text"]) for row in train}
    nonoverlapping = [i for i, row in enumerate(evaluation) if normalize_text(row["text"]) not in train_keys]
    metrics["overlapping_evaluation_rows"] = len(evaluation) - len(nonoverlapping)
    if len(nonoverlapping) < len(evaluation) and nonoverlapping:
        clean_actual = [actual[i] for i in nonoverlapping]
        clean_predicted = predicted[nonoverlapping]
        metrics["nonoverlapping_subset"] = {
            "rows": len(nonoverlapping),
            "accuracy": accuracy_score(clean_actual, clean_predicted),
            "macro_f1": f1_score(clean_actual, clean_predicted, labels=categories, average="macro", zero_division=0),
        }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "classification_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    fields = ["id", "text", "true_label", "predicted_label", "correct", "overlaps_training"]
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row, predicted_label in zip(evaluation, predicted):
            writer.writerow({"id": row["id"], "text": row["text"], "true_label": row["category"], "predicted_label": predicted_label, "correct": row["category"] == predicted_label, "overlaps_training": normalize_text(row["text"]) in train_keys})
    matrix = confusion_matrix(actual, predicted, labels=categories)
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label/predicted_label", *categories])
        for label, counts in zip(categories, matrix):
            writer.writerow([label, *counts.tolist()])
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    joblib.dump(pipeline, artifacts / f"{run_id}.joblib")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--ngram-max", type=int, choices=(1, 2), default=2)
    parser.add_argument("--alpha", type=float, default=1.0)
    args = parser.parse_args()
    try:
        metrics = run_naive_bayes(args.split, args.ngram_max, args.alpha)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
