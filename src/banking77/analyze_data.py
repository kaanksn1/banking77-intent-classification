"""Report dataset statistics, duplicates, label conflicts and split overlaps (read-only)."""

import json
import re
import statistics
from collections import Counter, defaultdict

from banking77.data import ROOT, normalize_text, read_records

SPLITS = ("train", "validation", "test")


def loose_key(text):
    """Normalized text without punctuation; finds near-duplicates the strict key misses."""
    return " ".join(re.sub(r"[^\w\s]", " ", normalize_text(text)).split())


def describe(rows):
    chars = [len(row["text"]) for row in rows]
    words = [len(row["text"].split()) for row in rows]
    counts = Counter(row["category"] for row in rows)
    keys = [normalize_text(row["text"]) for row in rows]
    labels = defaultdict(set)
    for key, row in zip(keys, rows):
        labels[key].add(row["category"])
    return {
        "rows": len(rows),
        "classes": len(counts),
        "class_count_min": min(counts.values()),
        "class_count_max": max(counts.values()),
        "class_count_mean": round(statistics.mean(counts.values()), 2),
        "imbalance_ratio": round(max(counts.values()) / min(counts.values()), 2),
        "smallest_classes": counts.most_common()[-3:],
        "largest_classes": counts.most_common(3),
        "chars": {"mean": round(statistics.mean(chars), 2), "median": statistics.median(chars), "min": min(chars), "max": max(chars)},
        "words": {"mean": round(statistics.mean(words), 2), "median": statistics.median(words), "min": min(words), "max": max(words)},
        "empty_text": sum(1 for row in rows if not row["text"].strip()),
        "missing_label": sum(1 for row in rows if not row["category"].strip()),
        "exact_duplicates": len(rows) - len({row["text"] for row in rows}),
        "normalized_duplicates": len(rows) - len(set(keys)),
        "conflicting_label_texts": sum(1 for found in labels.values() if len(found) > 1),
        "outer_whitespace": sum(1 for row in rows if row["text"] != row["text"].strip()),
        "repeated_inner_whitespace": sum(1 for row in rows if re.search(r"\s{2,}", row["text"].strip())),
        "non_ascii": sum(1 for row in rows if not row["text"].isascii()),
    }


def overlaps(splits):
    """Count shared texts between splits under the strict and the punctuation-free key."""
    result = {}
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1:]:
            strict = {normalize_text(r["text"]): r["category"] for r in splits[first]}
            other = {normalize_text(r["text"]): r["category"] for r in splits[second]}
            shared = strict.keys() & other.keys()
            loose_a = {loose_key(r["text"]) for r in splits[first]}
            loose_b = {loose_key(r["text"]) for r in splits[second]}
            result[f"{first}/{second}"] = {
                "strict_shared_texts": len(shared),
                "strict_label_conflicts": sum(strict[key] != other[key] for key in shared),
                "punctuation_free_shared_texts": len(loose_a & loose_b),
            }
    return result


def analyze():
    raw = ROOT / "data/raw"
    processed = ROOT / "data/processed"
    categories = json.loads((processed / "categories.json").read_text(encoding="utf-8"))
    splits = {name: read_records(processed / f"{name}.csv") for name in SPLITS}
    raw_train = read_records(raw / "train.csv")
    raw_test = read_records(raw / "test.csv")
    ids = {name: [row["id"] for row in rows] for name, rows in splits.items()}
    all_ids = [i for values in ids.values() for i in values]
    train_keys = {normalize_text(row["text"]) for row in splits["train"]}
    return {
        "raw_train": describe(raw_train),
        "raw_test": describe(raw_test),
        "splits": {name: describe(rows) for name, rows in splits.items()},
        "overlaps": overlaps(splits),
        "ids_unique_across_splits": len(all_ids) == len(set(all_ids)),
        "all_77_classes_in_every_split": all({r["category"] for r in rows} == set(categories) for rows in splits.values()),
        "test_rows_overlapping_train": sum(normalize_text(row["text"]) in train_keys for row in splits["test"]),
    }


if __name__ == "__main__":
    print(json.dumps(analyze(), indent=2, ensure_ascii=False))
