"""Train on train/validation; freeze a committed protocol before official test."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
from datetime import datetime, timezone
from time import perf_counter

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from banking77.data import ROOT, normalize_text, read_records
from banking77.neural_models import WORD_MODELS, WordClassifier, build_vocabulary, embedding_weights, encode

REGISTRY = ROOT / "configs/neural_models.json"
SOURCE_FILES = ("src/banking77/train_neural.py", "src/banking77/neural_models.py", "configs/neural_models.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def data_hashes(split="validation"):
    return {name: sha256(ROOT / "data/processed" / name)
            for name in dict.fromkeys(("train.csv", "validation.csv", f"{split}.csv", "categories.json", "summary.json"))}


def source_hashes():
    return {name: sha256(ROOT / name) for name in SOURCE_FILES}


def choose_device(name="auto"):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("No CUDA/ROCm GPU is visible. Run the preflight or explicitly choose --device cpu.")
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def seed_all(seed, threads=4):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def environment(device):
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "hip": torch.version.hip, "cuda": torch.version.cuda, "threads": torch.get_num_threads(),
            "deterministic_algorithms": "enabled, warning on unavailable deterministic kernels"}


class EncodedDataset(Dataset):
    def __init__(self, features, targets):
        self.features, self.targets = features, targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return {key: value[index] for key, value in self.features.items()}, self.targets[index]


def encode_rows(rows, categories, settings, vocabulary=None, tokenizer=None):
    if tokenizer is not None:
        features = dict(tokenizer([row["text"] for row in rows], padding="max_length", truncation=True,
                                  max_length=settings["max_length"], return_tensors="pt"))
    else:
        sequences = [encode(row["text"], vocabulary, settings["max_length"]) for row in rows]
        ids = torch.zeros((len(rows), settings["max_length"]), dtype=torch.long)
        for i, sequence in enumerate(sequences):
            ids[i, :len(sequence)] = torch.tensor(sequence)
        features = {"input_ids": ids, "attention_mask": ids.ne(0).long()}
    index = {name: i for i, name in enumerate(categories)}
    return EncodedDataset(features, torch.tensor([index[row["category"]] for row in rows], dtype=torch.long))


def get_logits(model, features):
    result = model(**features)
    return result if isinstance(result, torch.Tensor) else result.logits


def predict(model, dataset, device, batch_size):
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = perf_counter()
    probabilities = []
    with torch.inference_mode():
        for features, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            logits = get_logits(model, {key: value.to(device) for key, value in features.items()})
            probabilities.append(logits.softmax(dim=-1).float().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(probabilities), perf_counter() - start


def score(targets, predicted, labels):
    return {"accuracy": float(accuracy_score(targets, predicted)),
            "macro_f1": float(f1_score(targets, predicted, labels=labels, average="macro", zero_division=0))}


def save_outputs(output, metrics, rows, categories, probabilities):
    output.mkdir(parents=True, exist_ok=False)
    actual = [row["category"] for row in rows]
    predicted = np.asarray(categories)[probabilities.argmax(axis=1)].tolist()
    train_keys = {normalize_text(row["text"]) for row in read_records(ROOT / "data/processed/train.csv")}
    overlaps = [normalize_text(row["text"]) in train_keys for row in rows]
    metrics.update(score(actual, predicted, categories))
    metrics["overlapping_evaluation_rows"] = sum(overlaps)
    keep = [i for i, overlap in enumerate(overlaps) if not overlap]
    if 0 < len(keep) < len(rows):
        metrics["nonoverlapping_subset"] = {"rows": len(keep), **score(
            [actual[i] for i in keep], [predicted[i] for i in keep], categories)}
    write_json(output / "metrics.json", metrics)
    write_json(output / "classification_report.json", classification_report(
        actual, predicted, labels=categories, output_dict=True, zero_division=0))
    with (output / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("id", "text", "true_label", "predicted_label", "correct", "overlaps_training"))
        for row, prediction, overlap in zip(rows, predicted, overlaps):
            writer.writerow((row["id"], row["text"], row["category"], prediction,
                             row["category"] == prediction, overlap))
    with (output / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("true_label/predicted_label", *categories))
        for label, counts in zip(categories, confusion_matrix(actual, predicted, labels=categories)):
            writer.writerow((label, *counts.tolist()))
    # Needed for a later validation-selected ensemble; retained locally with each run.
    np.savez_compressed(output / "probabilities.npz", ids=[row["id"] for row in rows],
                        categories=categories, probabilities=probabilities)
    return metrics


def train(args):
    seed_all(args.seed, args.threads)
    device = choose_device(args.device)
    categories = json.loads((ROOT / "data/processed/categories.json").read_text(encoding="utf-8"))
    train_rows = read_records(ROOT / "data/processed/train.csv")
    validation = read_records(ROOT / "data/processed/validation.csv")
    if args.max_train_rows:
        train_rows = train_rows[:args.max_train_rows]
    if args.max_validation_rows:
        validation = validation[:args.max_validation_rows]
    transformer = args.model not in WORD_MODELS
    settings = {"model": args.model, "seed": args.seed, "max_length": args.max_length,
                "batch_size": args.batch_size or (16 if transformer else 64),
                "learning_rate": args.learning_rate or (2e-5 if transformer else 1e-3),
                "epochs_budget": args.epochs, "patience": args.patience, "weight_decay": 0.01,
                "gradient_clip": 1.0, "amp": args.amp, "selection_metric": "validation_macro_f1",
                "training_rows": len(train_rows), "validation_rows": len(validation)}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{args.model}_validation_{stamp}"
    artifact = ROOT / "artifacts" / run_id
    artifact.mkdir(parents=True, exist_ok=False)
    vocabulary, tokenizer = None, None
    if transformer:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        config = json.loads(REGISTRY.read_text(encoding="utf-8"))[args.model]
        settings.update(config)
        cache = str(ROOT / ".cache/huggingface")
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"],
                                                  cache_dir=cache, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(
            config["model_id"], revision=config["revision"], cache_dir=cache,
            num_labels=len(categories), id2label=dict(enumerate(categories)),
            label2id={name: i for i, name in enumerate(categories)},
            trust_remote_code=False, use_safetensors=True)
        model.config.save_pretrained(artifact)
        tokenizer.save_pretrained(artifact / "tokenizer")
    else:
        vocabulary = build_vocabulary([row["text"] for row in train_rows])
        weights, embedding = embedding_weights(vocabulary, args.embedding_dim, args.embedding_file)
        settings.update(embedding=embedding, vocabulary_size=len(vocabulary),
                        hidden_size=args.hidden_size, dropout=args.dropout,
                        freeze_embeddings=args.freeze_embeddings)
        model = WordClassifier(args.model, weights, len(categories), args.hidden_size,
                               args.dropout, args.freeze_embeddings)
        write_json(artifact / "vocabulary.json", vocabulary)
    start = perf_counter()
    training = encode_rows(train_rows, categories, settings, vocabulary, tokenizer)
    evaluation = encode_rows(validation, categories, settings, vocabulary, tokenizer)
    tokenization_seconds = perf_counter() - start
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(training, batch_size=settings["batch_size"], shuffle=True, generator=generator)
    total_steps = len(loader) * args.epochs
    warmup = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(
        (step + 1) / warmup, max(0, (total_steps - step) / max(1, total_steps - warmup)))) if transformer else None
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    best, best_epoch, stale, history = -1.0, 0, 0, []
    start = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        for features, targets in loader:
            features = {key: value.to(device) for key, value in features.items()}
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=args.amp and device.type == "cuda"):
                logits = get_logits(model, features)
                loss = nn.functional.cross_entropy(logits, targets)
            if not torch.isfinite(loss):
                raise ValueError("Training loss is non-finite")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            loss_sum += loss.item() * len(targets)
        probabilities, _ = predict(model, evaluation, device, settings["batch_size"])
        validation_score = score(evaluation.targets.numpy(), probabilities.argmax(axis=1), list(range(len(categories))))
        entry = {"epoch": epoch, "train_loss": loss_sum / len(training), **validation_score}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if validation_score["macro_f1"] > best:
            best, best_epoch, stale = validation_score["macro_f1"], epoch, 0
            torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, artifact / "weights.pt")
        else:
            stale += 1
        if stale >= args.patience:
            break
    fit_seconds = perf_counter() - start
    model.load_state_dict(torch.load(artifact / "weights.pt", map_location=device, weights_only=True))
    probabilities, predict_seconds = predict(model, evaluation, device, settings["batch_size"])
    metadata = {"run_id": run_id, "evaluation_split": "validation", "settings": settings,
                "categories": categories, "best_epoch": best_epoch, "history": history,
                "development_subset": bool(args.max_train_rows or args.max_validation_rows),
                "artifact_directory": artifact.relative_to(ROOT).as_posix(),
                "dataset_summary_sha256": sha256(ROOT / "data/processed/summary.json"),
                "dataset_files_sha256": data_hashes(), "source_sha256": source_hashes(),
                "training_rows": len(train_rows), "evaluation_rows": len(validation),
                "fit_seconds": fit_seconds, "tokenization_seconds": tokenization_seconds,
                "predict_seconds": predict_seconds,
                "prediction_ms_per_message": 1000 * predict_seconds / len(validation),
                "environment": environment(device), "official_test_evaluated": False}
    metadata = save_outputs(ROOT / "results/runs" / run_id, metadata, validation, categories, probabilities)
    write_json(artifact / "metadata.json", metadata)
    return metadata


def artifact_directory(metadata):
    path = (ROOT / metadata["artifact_directory"]).resolve()
    if not path.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Checkpoint must be under artifacts/")
    return path


def freeze(validation_run, output):
    metadata = json.loads((ROOT / "results/runs" / validation_run / "metrics.json").read_text(encoding="utf-8"))
    if metadata["evaluation_split"] != "validation" or metadata["development_subset"]:
        raise ValueError("Only a full validation run can be frozen")
    if metadata["dataset_files_sha256"] != data_hashes() or metadata["source_sha256"] != source_hashes():
        raise ValueError("Training data or implementation changed since the validation run")
    artifact = artifact_directory(metadata)
    protocol = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "official_test_evaluated_at_freeze": False, "validation_run": metadata,
                "dataset_files_sha256": data_hashes("test"),
                "artifact_sha256": {path.relative_to(artifact).as_posix(): sha256(path)
                                     for path in sorted(artifact.rglob("*")) if path.is_file()},
                "policy": "Commit this protocol before test; no test-based parameter selection or retraining."}
    output = Path(output).resolve()
    if not output.is_relative_to((ROOT / "results").resolve()):
        raise ValueError("Protocol must be saved under results/")
    if output.exists():
        raise ValueError("Protocol already exists; do not overwrite frozen settings")
    write_json(output, protocol)
    return {"protocol": str(output), "next_step": "Commit this protocol, then use the test subcommand."}


def test(protocol_path, device_name="auto", threads=4):
    path = Path(protocol_path).resolve()
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True, check=True).stdout
    if committed != path.read_bytes():
        raise ValueError("Protocol must match its committed version before test")
    protocol = json.loads(committed)
    metadata = protocol["validation_run"]
    if metadata["evaluation_split"] != "validation" or metadata["development_subset"]:
        raise ValueError("Protocol must reference a full validation run")
    if protocol["dataset_files_sha256"] != data_hashes("test") or metadata["source_sha256"] != source_hashes():
        raise ValueError("Frozen dataset or implementation hashes changed")
    artifact = artifact_directory(metadata)
    actual_hashes = {p.relative_to(artifact).as_posix(): sha256(p) for p in sorted(artifact.rglob("*")) if p.is_file()}
    if actual_hashes != protocol["artifact_sha256"]:
        raise ValueError("Frozen checkpoint/tokenizer hashes changed")
    seed_all(metadata["settings"]["seed"], threads)
    device = choose_device(device_name)
    settings, categories = metadata["settings"], metadata["categories"]
    state = torch.load(artifact / "weights.pt", map_location="cpu", weights_only=True)
    vocabulary, tokenizer = None, None
    if settings["model"] in WORD_MODELS:
        vocabulary = json.loads((artifact / "vocabulary.json").read_text(encoding="utf-8"))
        model = WordClassifier(settings["model"], state["embedding.weight"], len(categories),
                               settings["hidden_size"], settings["dropout"], settings["freeze_embeddings"])
    else:
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        config = AutoConfig.from_pretrained(artifact, local_files_only=True, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_config(config, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(artifact / "tokenizer", local_files_only=True, trust_remote_code=False)
    model.load_state_dict(state)
    model.to(device)
    rows = read_records(ROOT / "data/processed/test.csv")
    start = perf_counter()
    evaluation = encode_rows(rows, categories, settings, vocabulary, tokenizer)
    tokenization_seconds = perf_counter() - start
    probabilities, predict_seconds = predict(model, evaluation, device, settings["batch_size"])
    run_id = f"{settings['model']}_test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    result = {**metadata, "run_id": run_id, "evaluation_split": "test", "evaluation_rows": len(rows),
              "official_test_evaluated": True, "protocol": relative, "protocol_sha256": sha256(path),
              "dataset_files_sha256": data_hashes("test"), "environment": environment(device),
              "predict_seconds": predict_seconds, "tokenization_seconds": tokenization_seconds,
              "prediction_ms_per_message": 1000 * predict_seconds / len(rows)}
    return save_outputs(ROOT / "results/runs" / run_id, result, rows, categories, probabilities)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    training.add_argument("--model", choices=(*WORD_MODELS, *registry), required=True)
    training.add_argument("--epochs", type=int, default=15)
    training.add_argument("--patience", type=int, default=3)
    training.add_argument("--batch-size", type=int)
    training.add_argument("--learning-rate", type=float)
    training.add_argument("--max-length", type=int, default=64)
    training.add_argument("--embedding-dim", type=int, default=100)
    training.add_argument("--embedding-file", type=Path)
    training.add_argument("--freeze-embeddings", action="store_true")
    training.add_argument("--hidden-size", type=int, default=128)
    training.add_argument("--dropout", type=float, default=0.3)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--amp", action="store_true")
    training.add_argument("--max-train-rows", type=int)
    training.add_argument("--max-validation-rows", type=int)
    frozen = commands.add_parser("freeze")
    frozen.add_argument("--validation-run", required=True)
    frozen.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("test")
    evaluation.add_argument("--protocol", type=Path, required=True)
    for command in (training, evaluation):
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
        command.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        if args.command == "train":
            for key in ("epochs", "patience", "max_length", "embedding_dim", "hidden_size", "threads",
                        "batch_size", "max_train_rows", "max_validation_rows"):
                value = getattr(args, key)
                if value is not None and value < 1:
                    raise ValueError(f"{key} must be positive")
            if not 0 <= args.dropout < 1 or (args.learning_rate is not None and (not math.isfinite(args.learning_rate) or args.learning_rate <= 0)):
                raise ValueError("Invalid dropout or learning rate")
            result = train(args)
            print(json.dumps({key: result[key] for key in ("run_id", "accuracy", "macro_f1", "best_epoch", "fit_seconds", "development_subset")}, indent=2))
        elif args.command == "freeze":
            print(json.dumps(freeze(args.validation_run, args.output), indent=2))
        else:
            print(json.dumps(test(args.protocol, args.device, args.threads), indent=2))
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
