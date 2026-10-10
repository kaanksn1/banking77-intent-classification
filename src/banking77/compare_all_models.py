"""16 yöntemin resmî test tahminlerini tek tabloda karşılaştırır.

Bu modül eğitim yapmaz ve hiçbir ayar seçmez. Sabitlenmiş protokollerle üretilmiş,
repoda kayıtlı `predictions.csv` dosyalarını okur, aynı 3.080 test mesajına hizalar,
skorları yeniden hesaplar ve eşleştirilmiş istatistikleri üretir.
Komut: `python -m banking77.compare_all_models`
"""

import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from banking77.benchmark_models import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    bootstrap_difference,
    macro_f1_codes,
    nearest_train_similarity,
    paired_comparison,
    subset_scores,
)
from banking77.data import ROOT, read_records

RESULTS = ROOT / "results"
NEAR_DUPLICATE_THRESHOLD = 0.95
ALPHA = 0.05
TOLERANCE = 1e-9

FAMILY_CLASSICAL = "Classical TF-IDF baselines"
FAMILY_EMBEDDING = "Embeddings + linear head"
FAMILY_SCRATCH = "Neural networks trained from scratch"
FAMILY_TRANSFORMER = "Pretrained Transformers (fine-tuned)"
FAMILY_TEAM = "Team contribution"


@dataclass(frozen=True)
class Method:
    key: str
    name: str
    family: str
    directory: str  # results/ altındaki standart çıktı klasörü
    source: str  # süre ve çıktı hash'i için kaynak: classical | neural | transformer
    source_key: str  # kaynak dosyadaki anahtar


METHODS = (
    Method("nb", "Naive Bayes", FAMILY_CLASSICAL, "naive_bayes_test", "classical", "nb_selected"),
    Method("lr", "Logistic Regression", FAMILY_CLASSICAL, "logistic_regression_test", "classical", "lr_selected"),
    Method("svm", "Linear SVM", FAMILY_CLASSICAL, "linear_svm_test", "classical", "svm"),
    Method("glove", "GloVe (frozen) + mean + linear", FAMILY_EMBEDDING, "neural_test/glove", "neural", "glove"),
    Method("word2vec_cbow", "Word2Vec CBOW + mean + LR", FAMILY_EMBEDDING,
           "neural_test/word2vec_cbow", "neural", "word2vec_cbow"),
    Method("word2vec_skipgram", "Word2Vec Skip-gram + mean + LR", FAMILY_EMBEDDING,
           "neural_test/word2vec_skipgram", "neural", "word2vec_skipgram"),
    Method("fasttext", "FastText + mean + LR", FAMILY_EMBEDDING, "neural_test/fasttext", "neural", "fasttext"),
    Method("rnn", "RNN", FAMILY_SCRATCH, "neural_test/rnn", "neural", "rnn"),
    Method("cnn", "CNN", FAMILY_SCRATCH, "neural_test/cnn", "neural", "cnn"),
    Method("lstm", "LSTM", FAMILY_SCRATCH, "neural_test/lstm", "neural", "lstm"),
    Method("bilstm", "BiLSTM", FAMILY_SCRATCH, "neural_test/bilstm", "neural", "bilstm"),
    Method("bert", "BERT-base", FAMILY_TRANSFORMER, "transformer_test/bert", "transformer", "bert"),
    Method("distilbert", "DistilBERT-base", FAMILY_TRANSFORMER, "transformer_test/distilbert",
           "transformer", "distilbert"),
    Method("roberta", "RoBERTa-base", FAMILY_TRANSFORMER, "transformer_test/roberta", "transformer", "roberta"),
    Method("albert", "ALBERT-base-v2", FAMILY_TRANSFORMER, "transformer_test/albert", "transformer", "albert"),
    Method("nb_cnn", "NB + CNN soft voting", FAMILY_TEAM, "neural_test/nb_cnn", "neural", "nb_cnn"),
)
BY_KEY = {method.key: method for method in METHODS}

# Önceden belirlenmiş karşılaştırmalar: (referans, aday, neden)
KEY_COMPARISONS = (
    ("svm", "roberta", "best Transformer vs best classical TF-IDF model"),
    ("nb_cnn", "roberta", "best Transformer vs team contribution"),
    ("cnn", "roberta", "best Transformer vs best network trained from scratch"),
    ("lr", "svm", "the two linear classical models"),
    ("svm", "cnn", "best classical model vs CNN"),
    ("cnn", "nb_cnn", "team contribution vs its stronger component"),
    ("nb", "nb_cnn", "team contribution vs its weaker component"),
    ("distilbert", "roberta", "distilled vs robustly pre-trained Transformer"),
    ("word2vec_cbow", "word2vec_skipgram", "CBOW vs Skip-gram on the same data"),
    ("lstm", "bilstm", "one vs two reading directions"),
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(directory):
    path = RESULTS / directory / "predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def holm_adjust(p_values):
    """Holm-Bonferroni düzeltmesi; sözlük {anahtar: p} alır ve düzeltilmiş p döndürür."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted, running = {}, 0.0
    for rank, (key, p) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * p))
        adjusted[key] = running
    return adjusted


def load_sources():
    return {
        "classical": json.loads((RESULTS / "model_comparison_test.json").read_text(encoding="utf-8")),
        "neural": json.loads((RESULTS / "neural_test.json").read_text(encoding="utf-8")),
        "transformer": json.loads((RESULTS / "transformer_test.json").read_text(encoding="utf-8")),
        "transformer_validation": json.loads(
            (RESULTS / "transformer_validation.json").read_text(encoding="utf-8")),
    }


def source_run(sources, method):
    """Yönteme ait kaynak kaydı (metrik, süre, ortam)."""
    if method.source == "classical":
        return sources["classical"]["models"][method.source_key]
    runs = {run["key"]: run for run in sources[method.source]["runs"]}
    return runs[method.source_key]


def timing_record(sources, method):
    """Süre ve donanım; ölçüm kapsamı yönteme göre değişir ve raporda açıklanır."""
    record = source_run(sources, method)
    if method.source == "classical":
        return {
            "fit_seconds": record["fit_seconds"],
            "predict_ms_per_message": record["prediction_ms_per_message"],
            "device": "CPU (benchmark machine, sequential runs)",
            "pretraining": "none (learned from our training split)",
        }
    if method.source == "neural":
        metrics = record["metrics"]
        settings = metrics.get("settings", {})
        embedding = settings.get("embedding", {})
        if method.key == "glove":
            pretraining = "external GloVe 6B 100d vectors, frozen"
        elif method.key in ("word2vec_cbow", "word2vec_skipgram", "fasttext"):
            pretraining = "embeddings learned from our training split only"
        elif method.key == "nb_cnn":
            pretraining = "none (components learned from our training split)"
        elif embedding.get("initialization") == "random":
            pretraining = "none (random embeddings)"
        else:
            pretraining = "see protocol"
        return {
            "fit_seconds": metrics["fit_seconds"],
            "predict_ms_per_message": metrics["prediction_ms_per_message"],
            "device": "CPU, Windows (4 threads)",
            "pretraining": pretraining,
        }
    # Transformer: eğitim süresi validation kaydında, tahmin süresi test kaydındadır
    validation = {run["key"]: run for run in sources["transformer_validation"]["runs"]}[method.key]
    metrics = record["metrics"]
    return {
        "fit_seconds": validation["metrics"]["fit_seconds"],
        "predict_ms_per_message": metrics["prediction_ms_per_message"],
        "device": "AMD Radeon RX 7800 XT GPU (ROCm, WSL2)",
        "pretraining": "large external pre-training (public checkpoint)",
    }


def expected_hash(sources, method):
    """Kaynak JSON'da kayıtlı predictions.csv hash'i; klasik modeller için yok."""
    if method.source == "classical":
        return None
    record = source_run(sources, method)
    return record.get("output_sha256", {}).get("predictions.csv")


def load_predictions(sources):
    """Tüm yöntemlerin tahminlerini yükler, hizalamayı ve skorları doğrular."""
    tables, checks = {}, []
    reference = None
    for method in METHODS:
        rows = read_rows(method.directory)
        identity = [(r["id"], r["text"], r["true_label"]) for r in rows]
        if reference is None:
            reference = identity
        if identity != reference:
            raise ValueError(f"{method.key}: test mesajları veya etiketleri diğer yöntemlerle aynı değil")
        file_hash = sha256(RESULTS / method.directory / "predictions.csv")
        wanted = expected_hash(sources, method)
        if wanted is not None and wanted != file_hash:
            raise ValueError(f"{method.key}: predictions.csv hash'i kaynak kayıtla eşleşmiyor")
        true = [r["true_label"] for r in rows]
        predicted = [r["predicted_label"] for r in rows]
        recorded = json.loads((RESULTS / method.directory / "metrics.json").read_text(encoding="utf-8"))
        accuracy = float(accuracy_score(true, predicted))
        macro = float(f1_score(true, predicted, average="macro", zero_division=0))
        if abs(accuracy - recorded["accuracy"]) > TOLERANCE or abs(macro - recorded["macro_f1"]) > TOLERANCE:
            raise ValueError(f"{method.key}: yeniden hesaplanan skor kayıtlı metrikle eşleşmiyor")
        for row in rows:
            if (row["true_label"] == row["predicted_label"]) != (row["correct"] == "True"):
                raise ValueError(f"{method.key}: correct sütunu etiketlerle tutarsız ({row['id']})")
        tables[method.key] = rows
        checks.append({
            "key": method.key,
            "rows": len(rows),
            "predictions_sha256": file_hash,
            "hash_matches_source_record": wanted is not None,
            "recomputed_accuracy": accuracy,
            "recomputed_macro_f1": macro,
        })
    return tables, checks


def encode(tables):
    """Etiketleri tamsayı kodlarına çevirir; tüm yöntemler için aynı sınıf sırası."""
    first = next(iter(tables.values()))
    classes = sorted({row["true_label"] for row in first})
    index = {label: i for i, label in enumerate(classes)}
    true = np.array([index[r["true_label"]] for r in first])
    predicted = {
        key: np.array([index[r["predicted_label"]] for r in rows]) for key, rows in tables.items()
    }
    overlaps = np.array([r["overlaps_training"] == "True" for r in first])
    return classes, true, predicted, overlaps


def bootstrap_scores(true, predicted, classes, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """Her yöntem için aynı yeniden örneklemelerle macro F1 ve accuracy %95 aralığı."""
    rng = np.random.default_rng(seed)
    size, count = len(true), len(classes)
    macro = {key: np.empty(resamples) for key in predicted}
    for i in range(resamples):
        index = rng.integers(0, size, size)
        for key, codes in predicted.items():
            macro[key][i] = macro_f1_codes(true[index], codes[index], count)
    return {
        key: [float(v) for v in np.percentile(values, [2.5, 97.5])] for key, values in macro.items()
    }


def pairwise_tests(true, predicted):
    """Tüm 120 çift için exact McNemar ve Holm düzeltmesi."""
    correct = {key: codes == true for key, codes in predicted.items()}
    raw = {}
    for a, b in itertools.combinations(predicted, 2):
        raw[f"{a}|{b}"] = paired_comparison(correct[a], correct[b])
    adjusted = holm_adjust({pair: result["mcnemar_exact_p"] for pair, result in raw.items()})
    for pair, result in raw.items():
        result["holm_adjusted_p"] = adjusted[pair]
    return raw


def lookup(pairs, a, b):
    """İki yöntem için (a referans, b aday) sonucu; saklanan sıraya göre yönü çevirir."""
    if f"{a}|{b}" in pairs:
        result = pairs[f"{a}|{b}"]
        return result["reference_only_correct"], result["candidate_only_correct"], result
    result = pairs[f"{b}|{a}"]
    return result["candidate_only_correct"], result["reference_only_correct"], result


def key_comparisons(true, predicted, classes, pairs):
    results = []
    for reference, candidate, reason in KEY_COMPARISONS:
        broken, fixed, result = lookup(pairs, reference, candidate)
        interval = bootstrap_difference(true, predicted[reference], predicted[candidate], len(classes))
        results.append({
            "reference": reference,
            "candidate": candidate,
            "reason": reason,
            "candidate_fixed": fixed,
            "candidate_broken": broken,
            "mcnemar_exact_p": result["mcnemar_exact_p"],
            "holm_adjusted_p": result["holm_adjusted_p"],
            "macro_f1_difference": interval["observed"],
            "ci95_low": interval["ci95_low"],
            "ci95_high": interval["ci95_high"],
        })
    return results


def ablation(sources, true, predicted, classes, pairs):
    """NB + CNN: validation ağırlık taraması, test tek model skorları ve tamamlayıcılık."""
    run = {r["key"]: r for r in sources["neural"]["runs"]}["nb_cnn"]["metrics"]
    grid = run["weight_candidates"]
    count = len(classes)
    nb_ok, cnn_ok, ens_ok = (predicted[k] == true for k in ("nb", "cnn", "nb_cnn"))
    position = {"better": [], "not_separable": [], "worse": []}
    for method in METHODS:
        if method.key == "nb_cnn":
            continue
        broken, fixed, result = lookup(pairs, method.key, "nb_cnn")
        if result["holm_adjusted_p"] >= ALPHA:
            position["not_separable"].append(method.key)
        elif fixed > broken:
            position["better"].append(method.key)
        else:
            position["worse"].append(method.key)
    return {
        "position_vs_other_methods": position,
        "selected_cnn_weight": run["settings"]["cnn_weight"],
        "selected_nb_weight": run["settings"]["nb_weight"],
        "validation_grid": grid,
        "test": {
            key: {
                "accuracy": float((predicted[key] == true).mean()),
                "macro_f1": macro_f1_codes(true, predicted[key], count),
            }
            for key in ("nb", "cnn", "nb_cnn")
        },
        "both_correct": int((nb_ok & cnn_ok).sum()),
        "only_nb_correct": int((nb_ok & ~cnn_ok).sum()),
        "only_cnn_correct": int((~nb_ok & cnn_ok).sum()),
        "both_wrong": int((~nb_ok & ~cnn_ok).sum()),
        "oracle_accuracy": float((nb_ok | cnn_ok).mean()),
        "ensemble_fixed_both_wrong": int((ens_ok & ~nb_ok & ~cnn_ok).sum()),
        "ensemble_broken_when_one_right": int((~ens_ok & (nb_ok | cnn_ok)).sum()),
    }


def difficulty(tables, true, predicted, classes):
    """Hata analizi: tüm yöntemlerin yanlış yaptığı mesajlar ve en zor kategoriler."""
    keys = list(predicted)
    wrong = np.stack([predicted[k] != true for k in keys])  # yöntem x mesaj
    per_class = []
    for code, label in enumerate(classes):
        mask = true == code
        per_class.append({
            "category": label,
            "mean_error_rate": float(wrong[:, mask].mean()),
            "best_error_rate": float(wrong[:, mask].mean(axis=1).min()),
        })
    per_class.sort(key=lambda item: (-item["mean_error_rate"], item["category"]))
    all_wrong = np.where(wrong.all(axis=0))[0]
    first = tables[keys[0]]
    transformer_keys = [k for k in keys if BY_KEY[k].family == FAMILY_TRANSFORMER]
    transformer_wrong = np.where(np.stack([predicted[k] != true for k in transformer_keys]).all(axis=0))[0]

    def view(i):
        return {
            "id": first[i]["id"],
            "text": first[i]["text"],
            "true_label": first[i]["true_label"],
            "predictions": {k: tables[k][i]["predicted_label"] for k in ("svm", "cnn", "nb_cnn", "roberta")},
        }

    pair_counts = Counter(
        (classes[true[i]], classes[predicted["roberta"][i]]) for i in np.where(predicted["roberta"] != true)[0]
    )
    return {
        "methods": len(keys),
        "wrong_for_all_methods": int(len(all_wrong)),
        "wrong_for_all_transformers": int(len(transformer_wrong)),
        "wrong_for_all_examples": [view(i) for i in all_wrong[:6]],
        "hardest_categories": per_class[:10],
        "easiest_categories_count_zero_error": sum(1 for c in per_class if c["mean_error_rate"] == 0.0),
        "roberta_top_confusions": [
            {"true_label": t, "predicted_label": p, "count": n} for (t, p), n in pair_counts.most_common(5)
        ],
    }


def near_duplicate(tables, true_labels):
    """Train'e karakter n-gram benzerliği >= eşik olan test mesajları çıkarılınca skorlar."""
    train = read_records(ROOT / "data/processed/train.csv")
    test = read_records(ROOT / "data/processed/test.csv")
    similarity = nearest_train_similarity([r["text"] for r in train], [r["text"] for r in test])
    flagged = {row["id"] for row, score in zip(test, similarity) if score >= NEAR_DUPLICATE_THRESHOLD}
    per_method = {}
    for key, rows in tables.items():
        kept = [r for r in rows if r["id"] not in flagged]
        per_method[key] = subset_scores([r["true_label"] for r in kept], [r["predicted_label"] for r in kept])
    return {
        "threshold": NEAR_DUPLICATE_THRESHOLD,
        "flagged_messages": len(flagged),
        "flagged_share": len(flagged) / len(test),
        "without_near_duplicates": per_method,
    }


def build_summary():
    sources = load_sources()
    tables, checks = load_predictions(sources)
    classes, true, predicted, overlaps = encode(tables)
    count = len(classes)
    intervals = bootstrap_scores(true, predicted, classes)
    pairs = pairwise_tests(true, predicted)

    methods = []
    for method in METHODS:
        codes = predicted[method.key]
        keep = ~overlaps
        entry = {
            "key": method.key,
            "name": method.name,
            "family": method.family,
            "accuracy": float((codes == true).mean()),
            "macro_f1": macro_f1_codes(true, codes, count),
            "macro_f1_ci95": intervals[method.key],
            "nonoverlapping_accuracy": float((codes[keep] == true[keep]).mean()),
            "nonoverlapping_macro_f1": macro_f1_codes(true[keep], codes[keep], count),
            "timing": timing_record(sources, method),
        }
        methods.append(entry)
    methods.sort(key=lambda m: (-m["macro_f1"], m["name"]))
    best = methods[0]["key"]
    for rank, entry in enumerate(methods, start=1):
        entry["rank"] = rank
        if entry["key"] == best:
            entry["vs_best"] = None
            continue
        broken, fixed, result = lookup(pairs, best, entry["key"])
        entry["vs_best"] = {
            "best_only_correct": broken,
            "method_only_correct": fixed,
            "mcnemar_exact_p": result["mcnemar_exact_p"],
            "holm_adjusted_p": result["holm_adjusted_p"],
            "separable": result["holm_adjusted_p"] < ALPHA,
        }
    first_rows = next(iter(tables.values()))
    return {
        "command": "python -m banking77.compare_all_models",
        "evaluation_split": "test",
        "evaluation_rows": len(first_rows),
        "overlapping_evaluation_rows": int(overlaps.sum()),
        "categories": count,
        "methods_count": len(METHODS),
        "dataset_summary_sha256": sha256(ROOT / "data/processed/summary.json"),
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "alpha": ALPHA,
        "best": best,
        "methods": methods,
        "integrity_checks": checks,
        "key_comparisons": key_comparisons(true, predicted, classes, pairs),
        "pairwise_mcnemar": pairs,
        "ablation": ablation(sources, true, predicted, classes, pairs),
        "difficulty": difficulty(tables, true, predicted, classes),
        "near_duplicate_sensitivity": near_duplicate(tables, true),
    }


def _pct(value):
    return f"{value * 100:.2f}%"


def _p(value):
    return f"{value:.1e}" if value < 1e-3 else f"{value:.3f}"


def _seconds(value):
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def _signed(value):
    return f"{value:+.4f}"


def _names(keys):
    return ", ".join(BY_KEY[k].name for k in keys) if keys else "no method"


def render_markdown(summary):
    methods = summary["methods"]
    by_key = {m["key"]: m for m in methods}
    best = by_key[summary["best"]]
    inseparable = [m for m in methods if m["vs_best"] and not m["vs_best"]["separable"]]
    lines = [
        "# All-method comparison (official test, final)",
        "",
        f"Sixteen methods on the same {summary['evaluation_rows']} official test messages "
        f"({summary['categories']} categories, 40 messages each). Every setting was frozen on validation "
        "before the official test; nothing in this report selects a setting, a checkpoint or a weight.",
        f"Command: `{summary['command']}`. Dataset summary SHA-256: `{summary['dataset_summary_sha256']}`.",
        "The script trains nothing: it reads the committed `predictions.csv` of each method, checks that all "
        "files contain the same message ids, texts and true labels, recomputes accuracy and macro F1, and "
        "compares the file hashes with the records written when the predictions were produced.",
        "",
        "## Results",
        "",
        "Sorted by macro F1. The interval is a paired bootstrap 95% interval "
        f"({summary['bootstrap']['resamples']} resamples of test messages, seed {summary['bootstrap']['seed']}, "
        "the same resamples for every method). The last column is the exact McNemar test against the best "
        "method with Holm correction over all 120 method pairs.",
        "",
        "| # | Method | Family | Accuracy | Macro F1 [95% interval] | Macro F1 without 7 overlapping messages "
        "| only method right / only best right | Holm p |",
        "| ---: | --- | --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for m in methods:
        low, high = m["macro_f1_ci95"]
        if m["vs_best"] is None:
            versus, holm = "best", "-"
        else:
            v = m["vs_best"]
            versus = f"{v['method_only_correct']} / {v['best_only_correct']}"
            holm = _p(v["holm_adjusted_p"])
        lines.append(
            f"| {m['rank']} | {m['name']} | {m['family']} | {_pct(m['accuracy'])} | "
            f"{m['macro_f1']:.4f} [{low:.4f}, {high:.4f}] | {m['nonoverlapping_macro_f1']:.4f} | "
            f"{versus} | {holm} |"
        )
    lines += [
        "",
        f"Reading the table: {best['name']} has the highest macro F1 ({best['macro_f1']:.4f}). "
        f"{len(methods) - 1 - len(inseparable)} of the other {len(methods) - 1} methods are separable from it "
        f"(Holm-adjusted p < {summary['alpha']})"
        + (
            "; not separable from it: " + ", ".join(m["name"] for m in inseparable) + "."
            if inseparable else "."
        ),
        "Methods that are close to each other in macro F1 are compared pair by pair below; a single test split "
        "does not give a firm ranking when the intervals overlap.",
        "",
        "## Pre-specified paired comparisons",
        "",
        "Chosen before looking at the pairwise results; they answer the questions the presentation asks. "
        "\"Fixed\" counts messages the candidate gets right and the reference gets wrong.",
        "",
        "| Reference | Candidate | Why this pair | Fixed / broken | Macro F1 difference [95% interval] | p | Holm p |",
        "| --- | --- | --- | ---: | --- | ---: | ---: |",
    ]
    for c in summary["key_comparisons"]:
        verdict = "excludes 0" if c["ci95_low"] > 0 or c["ci95_high"] < 0 else "includes 0"
        lines.append(
            f"| {BY_KEY[c['reference']].name} | {BY_KEY[c['candidate']].name} | {c['reason']} | "
            f"{c['candidate_fixed']} / {c['candidate_broken']} | {_signed(c['macro_f1_difference'])} "
            f"[{_signed(c['ci95_low'])}, {_signed(c['ci95_high'])}] ({verdict}) | "
            f"{_p(c['mcnemar_exact_p'])} | {_p(c['holm_adjusted_p'])} |"
        )
    a = summary["ablation"]
    test = a["test"]
    lines += [
        "",
        "## Team contribution: NB + CNN soft voting",
        "",
        f"The CNN weight ({a['selected_cnn_weight']:.1f}; Naive Bayes {a['selected_nb_weight']:.1f}) was chosen "
        "on validation macro F1 from a grid of 0.0 to 1.0 in steps of 0.1; the two end points are the single "
        "models.",
        "",
        "| CNN weight | Validation accuracy | Validation macro F1 |",
        "| ---: | ---: | ---: |",
    ]
    for item in a["validation_grid"]:
        marker = " (selected)" if abs(item["cnn_weight"] - a["selected_cnn_weight"]) < 1e-9 else ""
        lines.append(
            f"| {item['cnn_weight']:.1f}{marker} | {_pct(item['accuracy'])} | {item['macro_f1']:.4f} |"
        )
    both, only_nb, only_cnn, wrong = (
        a["both_correct"], a["only_nb_correct"], a["only_cnn_correct"], a["both_wrong"])
    lines += [
        "",
        "| Official test | Accuracy | Macro F1 |",
        "| --- | ---: | ---: |",
        f"| Naive Bayes alone | {_pct(test['nb']['accuracy'])} | {test['nb']['macro_f1']:.4f} |",
        f"| CNN alone | {_pct(test['cnn']['accuracy'])} | {test['cnn']['macro_f1']:.4f} |",
        f"| NB + CNN soft voting | {_pct(test['nb_cnn']['accuracy'])} | {test['nb_cnn']['macro_f1']:.4f} |",
        "",
        f"The two components make different mistakes: both correct {both}, only Naive Bayes correct {only_nb}, "
        f"only CNN correct {only_cnn}, both wrong {wrong}. An oracle that always picks the right component "
        f"would reach {_pct(a['oracle_accuracy'])} accuracy; the ensemble recovers part of that gap "
        f"({a['ensemble_fixed_both_wrong']} messages that both components got wrong are right in the "
        f"ensemble, {a['ensemble_broken_when_one_right']} messages that at least one component had right are "
        "wrong in the ensemble). The ensemble is a team-built combination of two standard baselines; it is "
        "not a new architecture, and it does not reach the best Transformer.",
        "",
        "Against every other method (exact McNemar, Holm-adjusted over all 120 pairs, "
        f"p < {summary['alpha']}): NB + CNN is better than "
        + _names(a["position_vs_other_methods"]["better"]) + "; it is not separable from "
        + _names(a["position_vs_other_methods"]["not_separable"]) + "; it is worse than "
        + _names(a["position_vs_other_methods"]["worse"]) + ".",
        "",
        "## Cost",
        "",
        "Training and prediction time are not comparable across rows without the device and the scope below. "
        "Classical models were timed as medians of three sequential repeats; neural models and embeddings on a "
        "4-thread CPU; Transformers on one AMD GPU. Neural fit time includes validation and checkpoint writing "
        "per epoch and excludes first downloads and tokenisation; Transformer fit time is the recorded "
        "training run whose checkpoint was evaluated here. External pre-training cost is not included.",
        "",
        "| Method | Fit (s) | Predict (ms/message) | Device | Pre-training |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for m in methods:
        t = m["timing"]
        lines.append(
            f"| {m['name']} | {_seconds(t['fit_seconds'])} | {t['predict_ms_per_message']:.3f} | "
            f"{t['device']} | {t['pretraining']} |"
        )
    d = summary["difficulty"]
    lines += [
        "",
        "## Where the methods fail",
        "",
        f"- {d['wrong_for_all_methods']} of {summary['evaluation_rows']} test messages are wrong for all "
        f"{d['methods']} methods; {d['wrong_for_all_transformers']} are wrong for all four Transformers.",
        "- Hardest categories (share of the 40 test messages that a method gets wrong, averaged over the "
        f"{d['methods']} methods; the best method's own error rate in brackets):",
    ]
    for item in d["hardest_categories"]:
        lines.append(
            f"  - `{item['category']}`: {item['mean_error_rate']:.1%} (best method {item['best_error_rate']:.1%})"
        )
    lines += ["", f"Most frequent confusions of {best['name']}:", "",
              "| True category | Predicted category | Messages |", "| --- | --- | ---: |"]
    for item in d["roberta_top_confusions"]:
        lines.append(f"| `{item['true_label']}` | `{item['predicted_label']}` | {item['count']} |")
    lines += [
        "",
        "Real messages that every method gets wrong (first six in file order; examples, not causal explanations):",
        "",
        "| Message | True category | Linear SVM | CNN | NB + CNN | RoBERTa-base |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in d["wrong_for_all_examples"]:
        p = item["predictions"]
        text = item["text"].replace("|", "\\|")
        lines.append(
            f"| {text} | `{item['true_label']}` | `{p['svm']}` | `{p['cnn']}` | `{p['nb_cnn']}` | `{p['roberta']}` |"
        )
    near = summary["near_duplicate_sensitivity"]
    lines += [
        "",
        "## Sensitivity to overlap with the training data",
        "",
        f"{summary['overlapping_evaluation_rows']} test messages are exact normalised copies of training messages; "
        "the macro F1 column in the results table excludes them. A looser check removes the "
        f"{near['flagged_messages']} test messages ({near['flagged_share']:.1%}) whose character 3-5-gram TF-IDF "
        f"cosine similarity to a training message is at least {near['threshold']:.2f}.",
        "",
        "| Method | Macro F1 (all) | Macro F1 (without near-duplicates) | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for m in methods:
        reduced = near["without_near_duplicates"][m["key"]]["macro_f1"]
        lines.append(f"| {m['name']} | {m['macro_f1']:.4f} | {reduced:.4f} | {_signed(reduced - m['macro_f1'])} |")
    lines += [
        "",
        "## Limitations",
        "",
        "- One official test split of 3,080 messages (40 per category) and one training run (one seed) per "
        "method; differences of about one point are not firm rankings.",
        "- Tuning budgets differ by family: validation grids for the classical models, one configuration with the "
        "epoch chosen on validation macro F1 for each network, and a common five-epoch budget for the "
        "Transformers; the comparison is between protocols, not between optimal versions of each method.",
        "- The classical test results were seen before the neural scope was added, so the whole study is not a "
        "fully blind test; the new settings were frozen on validation before their official test.",
        "- Word2Vec and FastText vectors are trained on our 8,499 training messages only, GloVe is frozen and "
        "externally pre-trained, and the Transformers use large external pre-training; the families do not "
        "have the same information.",
        "- Times come from different devices and measurement scopes (see Cost) and are not a speed ranking.",
        "- Hyper-parameter tuning for the classical models and the NB + CNN weight use validation; the "
        "validation set was not added to training for any method.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(summary):
    (RESULTS / "all_models_test.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=int) + "\n", encoding="utf-8")
    (RESULTS / "ALL_MODELS_TEST.md").write_text(render_markdown(summary), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    summary = build_summary()
    write_outputs(summary)
    if not args.no_figures:
        from banking77.plot_all_models import make_figures

        make_figures(summary)
    best = summary["methods"][0]
    print(f"{summary['methods_count']} methods, best: {best['name']} macro F1 {best['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
