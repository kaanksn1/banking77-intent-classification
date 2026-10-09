"""Download upstream CSVs and create a deterministic, duplicate-aware split."""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.request import Request, urlopen

from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
SOURCE_CONFIG = ROOT / "configs" / "data.json"


def normalize_text(text):
    """Used only to find duplicates; the model still receives the original text."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def read_records(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"text", "category"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing text/category columns: {path}")
        return list(reader)


def write_records(path, records):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        # LF matches .gitattributes, so regenerated files are byte-identical on every OS.
        writer = csv.DictWriter(handle, fieldnames=["id", "text", "category"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def clean_training_records(records, categories):
    """Remove blank texts and exact normalized duplicates; reject conflicting labels."""
    label_set = set(categories)
    unique = {}
    blank_count = 0
    for index, row in enumerate(records):
        text, label = row["text"], row["category"]
        if label not in label_set:
            raise ValueError(f"Unknown category: {label}")
        key = normalize_text(text)
        if not key:
            blank_count += 1
            continue
        if key in unique and unique[key]["category"] != label:
            raise ValueError(f"Conflicting labels for normalized training text at row {index}")
        if key not in unique:
            unique[key] = {"id": f"train-{index:05d}", "text": text, "category": label}
    return list(unique.values()), {
        "blank_training_rows_removed": blank_count,
        "duplicate_training_rows_removed": len(records) - blank_count - len(unique),
    }


def split_training_records(records, validation_fraction=0.15, seed=42):
    """Split by label after deduplication, so identical texts cannot cross splits."""
    train, validation = train_test_split(
        records,
        test_size=validation_fraction,
        random_state=seed,
        stratify=[row["category"] for row in records],
    )
    train_keys = {normalize_text(row["text"]) for row in train}
    validation_keys = {normalize_text(row["text"]) for row in validation}
    if train_keys & validation_keys:
        raise ValueError("Training and validation texts overlap")
    return train, validation


def download(raw_dir):
    config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    revision = config["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("configs/data.json must contain a pinned 40-character commit SHA")
    files = {
        "train.csv": "banking_data/train.csv",
        "test.csv": "banking_data/test.csv",
        "categories.json": "banking_data/categories.json",
        "LICENSE": "LICENSE",
    }
    hashes = {}
    for filename, upstream_path in files.items():
        target = raw_dir / filename
        url = f"https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/{revision}/{upstream_path}"
        request = Request(url, headers={"User-Agent": "banking77-course-project"})
        with urlopen(request, timeout=60) as response:
            content = response.read()
        expected = config.get("sha256", {}).get(filename)
        actual = hashlib.sha256(content).hexdigest()
        if expected and actual != expected:
            raise ValueError(f"Checksum mismatch: {filename}")
        target.write_bytes(content)
        hashes[filename] = actual
    return {"revision": revision, "sha256": hashes}


def prepare(raw_dir, output_dir, validation_fraction=0.15, seed=42):
    config = json.loads(SOURCE_CONFIG.read_text(encoding="utf-8"))
    raw_hashes = {}
    for filename, expected in config["sha256"].items():
        actual = hashlib.sha256((raw_dir / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Source file changed: {filename}; download the pinned source again")
        raw_hashes[filename] = actual
    categories = json.loads((raw_dir / "categories.json").read_text(encoding="utf-8"))
    if len(categories) != 77 or len(set(categories)) != 77:
        raise ValueError("Expected 77 unique BANKING77 categories")
    original_train = read_records(raw_dir / "train.csv")
    original_test = read_records(raw_dir / "test.csv")
    clean_train, cleaning = clean_training_records(original_train, categories)
    train, validation = split_training_records(clean_train, validation_fraction, seed)
    test = []
    for index, row in enumerate(original_test):
        if row["category"] not in categories or not normalize_text(row["text"]):
            raise ValueError(f"Invalid official test row: {index}")
        test.append({"id": f"test-{index:05d}", **row})
    splits = {"train": train, "validation": validation, "test": test}
    for name, rows in splits.items():
        if {row["category"] for row in rows} != set(categories):
            raise ValueError(f"Missing classes in {name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        write_records(output_dir / f"{name}.csv", rows)
    (output_dir / "categories.json").write_text(
        json.dumps(categories, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    # Preserve the official test set; report its pre-existing text overlaps explicitly.
    keys = {name: {normalize_text(row["text"]) for row in rows} for name, rows in splits.items()}
    summary = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "original_rows": {"train": len(original_train), "test": len(original_test)},
        "rows": {name: len(rows) for name, rows in splits.items()},
        "class_counts": {name: dict(sorted(Counter(row["category"] for row in rows).items())) for name, rows in splits.items()},
        "cleaning": cleaning,
        "normalized_text_overlap": {
            "train_validation": len(keys["train"] & keys["validation"]),
            "train_test": len(keys["train"] & keys["test"]),
            "validation_test": len(keys["validation"] & keys["test"]),
        },
        "source_revision": config["revision"],
        "raw_sha256": raw_hashes,
        "protocol": "Deduplicated official train, stratified validation; official test unchanged. Overlaps reported, not silently removed.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use already downloaded source files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    args = parser.parse_args()
    if not 0 < args.validation_fraction < 1:
        parser.error("--validation-fraction must be between 0 and 1")
    raw_dir, output_dir = ROOT / "data/raw", ROOT / "data/processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not args.offline:
        download(raw_dir)
    summary = prepare(raw_dir, output_dir, args.validation_fraction, args.seed)
    print(json.dumps({"rows": summary["rows"], "cleaning": summary["cleaning"], "overlap": summary["normalized_text_overlap"]}, indent=2))


if __name__ == "__main__":
    main()
