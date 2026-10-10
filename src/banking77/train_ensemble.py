"""Owner's contribution: validation-selected soft voting of frozen NB and CNN."""

import argparse
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from time import perf_counter

import joblib
import numpy as np

from banking77.data import ROOT, read_records
from banking77.naive_bayes import build_naive_bayes
from banking77 import train_neural as neural

NB_PROTOCOL = "results/naive_bayes_final_protocol.json"


def source_hashes():
    names = ("src/banking77/train_ensemble.py", "src/banking77/naive_bayes.py", NB_PROTOCOL)
    return {**neural.source_hashes(), **{name: neural.sha256(ROOT / name) for name in names}}


def aligned_probabilities(path, rows, categories):
    with np.load(path, allow_pickle=False) as saved:
        if saved["ids"].tolist() != [row["id"] for row in rows] or saved["categories"].tolist() != categories:
            raise ValueError("Probability rows or category columns are misaligned")
        probabilities = saved["probabilities"].copy()
    if probabilities.shape != (len(rows), len(categories)) or not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("Invalid probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Probability rows must sum to one")
    return probabilities


def select_weight(actual, categories, nb, cnn):
    candidates = []
    for step in range(11):
        weight = step / 10
        predicted = np.asarray(categories)[((1 - weight) * nb + weight * cnn).argmax(axis=1)]
        candidates.append({"cnn_weight": weight, **neural.score(actual, predicted, categories)})
    # The endpoints are the two single-model ablations. Ties keep the first weight.
    selected = max(candidates, key=lambda row: row["macro_f1"])
    return selected, candidates


def nb_probabilities(model, rows, categories):
    probabilities = model.predict_proba([row["text"] for row in rows])
    return probabilities[:, [model.classes_.tolist().index(label) for label in categories]]


def train(cnn_run):
    cnn = json.loads((ROOT / "results/runs" / cnn_run / "metrics.json").read_text(encoding="utf-8"))
    if cnn["evaluation_split"] != "validation" or cnn["development_subset"] or cnn["settings"]["model"] != "cnn":
        raise ValueError("Expected a full CNN validation run")
    if cnn["source_sha256"] != neural.source_hashes() or cnn["dataset_files_sha256"] != neural.data_hashes():
        raise ValueError("CNN validation data or implementation changed")
    rows = read_records(ROOT / "data/processed/validation.csv")
    training = read_records(ROOT / "data/processed/train.csv")
    categories = json.loads((ROOT / "data/processed/categories.json").read_text(encoding="utf-8"))
    cnn_probabilities = aligned_probabilities(ROOT / "results/runs" / cnn_run / "probabilities.npz", rows, categories)
    nb_settings = json.loads((ROOT / NB_PROTOCOL).read_text(encoding="utf-8"))["settings"]
    model = build_naive_bayes(nb_settings["ngram_max"], nb_settings["alpha"])
    start = perf_counter()
    model.fit([row["text"] for row in training], [row["category"] for row in training])
    nb_fit = perf_counter() - start
    start = perf_counter()
    nb = nb_probabilities(model, rows, categories)
    nb_predict = perf_counter() - start
    actual = [row["category"] for row in rows]
    selected, candidates = select_weight(actual, categories, nb, cnn_probabilities)
    weight = selected["cnn_weight"]
    run_id = f"nb_cnn_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    artifact = ROOT / "artifacts" / run_id
    artifact.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, artifact / "naive_bayes.joblib")
    result = {"run_id": run_id, "evaluation_split": "validation", "development_subset": False,
              "settings": {"model": "nb_cnn_soft_voting", "cnn_weight": weight, "nb_weight": 1 - weight,
                           "weight_grid": [row["cnn_weight"] for row in candidates],
                           "selection_metric": "validation_macro_f1", "nb_settings": nb_settings},
              "weight_candidates": candidates, "categories": categories,
              "cnn_validation_run": cnn_run, "cnn_validation_metrics_sha256": neural.sha256(ROOT / "results/runs" / cnn_run / "metrics.json"),
              "artifact_directory": artifact.relative_to(ROOT).as_posix(),
              "dataset_summary_sha256": cnn["dataset_summary_sha256"],
              "dataset_files_sha256": neural.data_hashes(), "source_sha256": source_hashes(),
              "training_rows": len(training), "evaluation_rows": len(rows),
              "fit_seconds": nb_fit + cnn["fit_seconds"],
              "component_fit_seconds": {"nb": nb_fit, "cnn": cnn["fit_seconds"]},
              "predict_seconds": nb_predict + cnn["predict_seconds"],
              "prediction_ms_per_message": 1000 * (nb_predict + cnn["predict_seconds"]) / len(rows),
              "timing_policy": "sum of separately measured component times; excludes downloads, model loading and initial tokenization",
              "environment": cnn["environment"], "official_test_evaluated": False}
    result = neural.save_outputs(ROOT / "results/runs" / run_id, result, rows, categories,
                                 (1 - weight) * nb + weight * cnn_probabilities)
    neural.write_json(artifact / "metadata.json", result)
    return result


def freeze(run_id, cnn_protocol, output):
    metadata = json.loads((ROOT / "results/runs" / run_id / "metrics.json").read_text(encoding="utf-8"))
    if metadata["evaluation_split"] != "validation" or metadata["settings"]["model"] != "nb_cnn_soft_voting":
        raise ValueError("Expected an ensemble validation run")
    if metadata["source_sha256"] != source_hashes() or metadata["dataset_files_sha256"] != neural.data_hashes():
        raise ValueError("Ensemble implementation or data changed")
    cnn_protocol = cnn_protocol.resolve()
    cnn = json.loads(cnn_protocol.read_text(encoding="utf-8"))["validation_run"]
    if cnn["run_id"] != metadata["cnn_validation_run"] or neural.sha256(ROOT / "results/runs" / cnn["run_id"] / "metrics.json") != metadata["cnn_validation_metrics_sha256"]:
        raise ValueError("CNN protocol must freeze the validation checkpoint used in voting")
    artifact = neural.artifact_directory(metadata)
    protocol = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(), "official_test_evaluated_at_freeze": False,
                "validation_run": metadata, "dataset_files_sha256": neural.data_hashes("test"),
                "nb_checkpoint_sha256": neural.sha256(artifact / "naive_bayes.joblib"),
                "cnn_protocol": cnn_protocol.relative_to(ROOT).as_posix(), "cnn_protocol_sha256": neural.sha256(cnn_protocol)}
    output = output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()) or output.exists():
        raise ValueError("Use a new protocol path under results/")
    neural.write_json(output, protocol)
    return {"protocol": output.relative_to(ROOT).as_posix(), "next_step": "Commit before official test"}


def test(path, device="cpu"):
    path = path.resolve()
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True, check=True).stdout
    if committed != path.read_bytes():
        raise ValueError("Protocol must match its committed version")
    protocol = json.loads(committed)
    metadata = protocol["validation_run"]
    if metadata["source_sha256"] != source_hashes() or protocol["dataset_files_sha256"] != neural.data_hashes("test"):
        raise ValueError("Frozen ensemble data or implementation changed")
    artifact = neural.artifact_directory(metadata)
    if neural.sha256(artifact / "naive_bayes.joblib") != protocol["nb_checkpoint_sha256"]:
        raise ValueError("Frozen NB checkpoint changed")
    cnn_path = ROOT / protocol["cnn_protocol"]
    if neural.sha256(cnn_path) != protocol["cnn_protocol_sha256"]:
        raise ValueError("Frozen CNN protocol changed")
    cnn = neural.test(cnn_path, device)
    rows = read_records(ROOT / "data/processed/test.csv")
    categories = metadata["categories"]
    cnn_probabilities = aligned_probabilities(ROOT / "results/runs" / cnn["run_id"] / "probabilities.npz", rows, categories)
    model = joblib.load(artifact / "naive_bayes.joblib")
    start = perf_counter()
    nb = nb_probabilities(model, rows, categories)
    seconds = perf_counter() - start + cnn["predict_seconds"]
    weight = metadata["settings"]["cnn_weight"]
    run_id = f"nb_cnn_test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    result = {**metadata, "run_id": run_id, "evaluation_split": "test", "evaluation_rows": len(rows),
              "official_test_evaluated": True, "dataset_files_sha256": neural.data_hashes("test"),
              "cnn_test_run": cnn["run_id"], "protocol": relative, "protocol_sha256": neural.sha256(path),
              "predict_seconds": seconds, "prediction_ms_per_message": 1000 * seconds / len(rows)}
    return neural.save_outputs(ROOT / "results/runs" / run_id, result, rows, categories,
                               (1 - weight) * nb + weight * cnn_probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--cnn-validation-run", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--validation-run", required=True)
    freezing.add_argument("--cnn-protocol", type=Path, required=True)
    freezing.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("test")
    evaluation.add_argument("--protocol", type=Path, required=True)
    evaluation.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    try:
        if args.command == "train":
            result = train(args.cnn_validation_run)
        elif args.command == "freeze":
            result = freeze(args.validation_run, args.cnn_protocol, args.output)
        else:
            result = test(args.protocol, args.device)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps({key: result[key] for key in ("run_id", "accuracy", "macro_f1", "settings") if key in result} or result, indent=2))


if __name__ == "__main__":
    main()
