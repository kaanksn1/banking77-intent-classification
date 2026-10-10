"""Train-only Word2Vec/FastText representations with a fixed linear classifier."""

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from time import perf_counter

import gensim
from gensim.models import FastText, Word2Vec
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from banking77.data import ROOT, read_records
from banking77.neural_models import tokenize
from banking77.train_neural import data_hashes, save_outputs, sha256, write_json

MODELS = ("word2vec_cbow", "word2vec_skipgram", "fasttext")
SOURCES = ("src/banking77/train_embeddings.py", "src/banking77/neural_models.py",
           "src/banking77/train_neural.py")


def stable_hash(word):
    import hashlib
    return int.from_bytes(hashlib.sha256(word.encode("utf-8")).digest()[:8], "little")


def source_hashes():
    return {name: sha256(ROOT / name) for name in SOURCES}


class MeanEmbeddingClassifier:
    def __init__(self, kind, seed=42, epochs=20):
        if kind not in MODELS:
            raise ValueError("Unknown embedding method")
        self.kind, self.seed, self.epochs = kind, seed, epochs

    def fit(self, texts, labels):
        sentences = [tokenize(text) or ["<empty>"] for text in texts]
        options = dict(sentences=sentences, vector_size=100, window=5, min_count=1,
                       workers=1, seed=self.seed, epochs=self.epochs, hashfxn=stable_hash,
                       sg=int(self.kind == "word2vec_skipgram"), negative=5)
        self.embedding = FastText(**options, bucket=20000) if self.kind == "fasttext" else Word2Vec(**options)
        self.head = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=self.seed)
        self.head.fit(self.transform(texts), labels)
        self.classes_ = self.head.classes_
        return self

    def transform(self, texts):
        rows = []
        for text in texts:
            words = tokenize(text)
            vectors = [self.embedding.wv.get_vector(word) for word in words
                       if self.kind == "fasttext" or word in self.embedding.wv.key_to_index]
            rows.append(np.mean(vectors, axis=0) if vectors else np.zeros(100, dtype=np.float32))
        return np.asarray(rows, dtype=np.float32)

    def predict_proba(self, texts):
        return self.head.predict_proba(self.transform(texts))


def train(kind, epochs=20):
    train_rows = read_records(ROOT / "data/processed/train.csv")
    validation = read_records(ROOT / "data/processed/validation.csv")
    categories = json.loads((ROOT / "data/processed/categories.json").read_text(encoding="utf-8"))
    start = perf_counter()
    model = MeanEmbeddingClassifier(kind, epochs=epochs).fit(
        [row["text"] for row in train_rows], [row["category"] for row in train_rows])
    fit_seconds = perf_counter() - start
    start = perf_counter()
    probabilities = model.predict_proba([row["text"] for row in validation])
    probabilities = probabilities[:, [model.classes_.tolist().index(label) for label in categories]]
    predict_seconds = perf_counter() - start
    run_id = f"{kind}_validation_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    artifact = ROOT / "artifacts" / run_id
    artifact.mkdir(parents=True, exist_ok=False)
    joblib.dump(model, artifact / "pipeline.joblib")
    settings = {"model": kind, "seed": 42, "embedding_dimension": 100, "embedding_epochs": epochs,
                "window": 5, "min_count": 1, "workers": 1, "negative_samples": 5,
                "sg": int(kind == "word2vec_skipgram"), "head": "LogisticRegression",
                "C": 1.0, "solver": "lbfgs", "max_iter": 2000, "pooling": "mean",
                "embedding_training_source": "prepared_train_only"}
    if kind == "fasttext":
        settings.update(bucket=20000, min_n=3, max_n=6, out_of_vocabulary="synthesized from character ngrams")
    else:
        settings["out_of_vocabulary"] = "omit unknown words; zero vector for an all-unknown message"
    result = {"run_id": run_id, "evaluation_split": "validation", "settings": settings,
              "categories": categories, "development_subset": False,
              "dataset_summary_sha256": sha256(ROOT / "data/processed/summary.json"),
              "dataset_files_sha256": data_hashes(), "source_sha256": source_hashes(),
              "artifact_directory": artifact.relative_to(ROOT).as_posix(),
              "training_rows": len(train_rows), "evaluation_rows": len(validation),
              "vocabulary_size": len(model.embedding.wv.key_to_index),
              "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
              "prediction_ms_per_message": 1000 * predict_seconds / len(validation),
              "optimizer_iterations": int(model.head.n_iter_.max()),
              "converged": bool(model.head.n_iter_.max() < model.head.max_iter),
              "environment": {"python": platform.python_version(), "gensim": gensim.__version__, "numpy": np.__version__},
              "official_test_evaluated": False}
    result = save_outputs(ROOT / "results/runs" / run_id, result, validation, categories, probabilities)
    write_json(artifact / "metadata.json", result)
    return result


def freeze(run_id, output):
    metadata = json.loads((ROOT / "results/runs" / run_id / "metrics.json").read_text(encoding="utf-8"))
    if metadata["evaluation_split"] != "validation" or metadata["settings"]["model"] not in MODELS:
        raise ValueError("Expected an embedding validation run")
    if metadata["dataset_files_sha256"] != data_hashes() or metadata["source_sha256"] != source_hashes():
        raise ValueError("Data or embedding code changed since validation")
    artifact = (ROOT / metadata["artifact_directory"]).resolve()
    if not artifact.is_relative_to((ROOT / "artifacts").resolve()):
        raise ValueError("Checkpoint must be under artifacts/")
    protocol = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "official_test_evaluated_at_freeze": False, "validation_run": metadata,
                "dataset_files_sha256": data_hashes("test"),
                "checkpoint_sha256": sha256(artifact / "pipeline.joblib")}
    output = output.resolve()
    if not output.is_relative_to((ROOT / "results").resolve()) or output.exists():
        raise ValueError("Use a new protocol path under results/")
    write_json(output, protocol)
    return {"protocol": str(output), "next_step": "Commit before running the test subcommand"}


def test(path):
    path = path.resolve()
    relative = path.relative_to(ROOT).as_posix()
    committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, capture_output=True, check=True).stdout
    if committed != path.read_bytes():
        raise ValueError("Protocol must match committed version")
    protocol = json.loads(committed)
    metadata = protocol["validation_run"]
    if metadata["source_sha256"] != source_hashes() or protocol["dataset_files_sha256"] != data_hashes("test"):
        raise ValueError("Frozen data or implementation changed")
    artifact = (ROOT / metadata["artifact_directory"]).resolve()
    if not artifact.is_relative_to((ROOT / "artifacts").resolve()) or sha256(artifact / "pipeline.joblib") != protocol["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint changed")
    model = joblib.load(artifact / "pipeline.joblib")
    rows = read_records(ROOT / "data/processed/test.csv")
    categories = metadata["categories"]
    start = perf_counter()
    probabilities = model.predict_proba([row["text"] for row in rows])
    probabilities = probabilities[:, [model.classes_.tolist().index(label) for label in categories]]
    seconds = perf_counter() - start
    run_id = f"{metadata['settings']['model']}_test_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"
    result = {**metadata, "run_id": run_id, "evaluation_split": "test", "evaluation_rows": len(rows),
              "official_test_evaluated": True, "dataset_files_sha256": data_hashes("test"),
              "protocol": relative, "protocol_sha256": sha256(path),
              "predict_seconds": seconds, "prediction_ms_per_message": 1000 * seconds / len(rows)}
    return save_outputs(ROOT / "results/runs" / run_id, result, rows, categories, probabilities)


def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--model", choices=MODELS, required=True)
    training.add_argument("--epochs", type=int, default=20)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--validation-run", required=True)
    freezing.add_argument("--output", type=Path, required=True)
    evaluation = commands.add_parser("test")
    evaluation.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "train":
            if args.epochs < 1:
                raise ValueError("epochs must be positive")
            result = train(args.model, args.epochs)
        elif args.command == "freeze":
            result = freeze(args.validation_run, args.output)
        else:
            result = test(args.protocol)
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
