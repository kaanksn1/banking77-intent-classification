"""Reproduce the owner's alpha-only Naive Bayes validation benchmark."""

import csv
import json
from collections import Counter

from banking77.data import ROOT
from banking77.train_naive_bayes import run_naive_bayes

ALPHAS = (1.0, 0.5, 0.1, 0.05, 0.01)


def select_best_run(runs):
    """Select by validation macro F1; ties keep the first scheduled alpha."""
    if not runs or any(run["evaluation_split"] != "validation" for run in runs):
        raise ValueError("Model selection requires nonempty validation-only runs")
    return max(runs, key=lambda run: run["macro_f1"])


def read_predictions(run):
    path = ROOT / "results/runs" / run["run_id"] / "predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_summary(runs):
    best = select_best_run(runs)
    baseline = next(run for run in runs if run["settings"]["alpha"] == 1.0)
    data_identity = baseline["dataset_files_sha256"]
    if any(run["dataset_files_sha256"] != data_identity for run in runs):
        raise ValueError("Prepared data changed during the alpha benchmark")
    baseline_rows = {row["id"]: row for row in read_predictions(baseline)}
    best_rows = read_predictions(best)
    if set(baseline_rows) != {row["id"] for row in best_rows}:
        raise ValueError("Baseline and selected predictions must cover the same records")
    errors = [row for row in best_rows if row["correct"] == "False"]
    pairs = Counter((row["true_label"], row["predicted_label"]) for row in errors)
    return {
        "command": "python -m banking77.benchmark_naive_bayes",
        "selection_metric": "validation macro_f1",
        "tie_break": "first alpha in the scheduled order",
        "feature_settings_unchanged": True,
        "official_test_evaluated": False,
        "runs": runs,
        "best_tested_alpha": best["settings"]["alpha"],
        "selected_run_id": best["run_id"],
        "baseline_errors": sum(row["correct"] == "False" for row in baseline_rows.values()),
        "best_errors": len(errors),
        "corrected_messages": sum(
            baseline_rows[row["id"]]["correct"] == "False" and row["correct"] == "True"
            for row in best_rows
        ),
        "regressed_messages": sum(
            baseline_rows[row["id"]]["correct"] == "True" and row["correct"] == "False"
            for row in best_rows
        ),
        "most_frequent_best_errors": [
            {"true_label": actual, "predicted_label": predicted, "count": count}
            for (actual, predicted), count in pairs.most_common(5)
        ],
        "best_error_examples": errors[:3],
    }


def write_summary(summary):
    target = ROOT / "results"
    target.mkdir(exist_ok=True)
    (target / "naive_bayes_alpha_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    baseline = summary["runs"][0]
    lines = [
        "# Naive Bayes alpha deneyi", "",
        f"Aynı {baseline['training_rows']:,} eğitim ve {baseline['evaluation_rows']:,} validation kaydı kullanıldı.",
        "TF-IDF unigram + bigram ve sublinear TF sabit tutuldu. Yalnızca alpha değiştirildi.",
        "Resmî testte değerlendirme yapılmadı. Bu rapor script tarafından üretilir.", "",
        "Komut: `python -m banking77.benchmark_naive_bayes`.", "",
        f"Veri özeti SHA-256: `{baseline['dataset_summary_sha256']}`.", "",
        "| Alpha | Validation accuracy | Validation macro F1 | Eğitim (s) | Tahmin (s) |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['settings']['alpha']:g} | {run['accuracy']:.2%} | {run['macro_f1']:.4f} | "
            f"{run['fit_seconds']:.4f} | {run['predict_seconds']:.4f} |"
        )
    lines += [
        "", f"Denenen değerlerde en yüksek validation macro F1: alpha={summary['best_tested_alpha']:g}.",
        "Eşit macro F1 durumunda listedeki ilk alpha seçilir. Varsayılan alpha=1.0 korunur.",
        f"Yanlış sayısı {summary['baseline_errors']} → {summary['best_errors']}; "
        f"{summary['corrected_messages']} hata düzeldi, {summary['regressed_messages']} doğru tahmin bozuldu.",
        "Süreler bu bilgisayardaki tek ölçümdür; genel bir hız üstünlüğü göstermez.", "",
        "## Sık karışan kategoriler", "",
        "| Gerçek kategori | Tahmin | Sayı |", "| --- | --- | ---: |",
    ]
    for pair in summary["most_frequent_best_errors"]:
        lines.append(f"| {pair['true_label']} | {pair['predicted_label']} | {pair['count']} |")
    lines += ["", "## Yanlış tahmin örnekleri", ""]
    for row in summary["best_error_examples"]:
        lines += [f"- `{row['id']}`: {row['text']}",
                  f"  Gerçek: `{row['true_label']}`; tahmin: `{row['predicted_label']}`."]
    lines += [
        "", "Ayarlar, ortam ve veri dosyası hash'leri: [JSON raporu](naive_bayes_alpha_validation.json).",
        "Seçilen çalıştırmanın tam tahminleri ve confusion matrix'i yerel `results/runs/` altındadır.",
        "Bunlar aynı komutla yeniden üretilir; eğitilmiş modeller Git'e eklenmez.",
        "Teknik açıklama ve hata yorumları: [Naive Bayes notu](../docs/NAIVE_BAYES.md).", "",
    ]
    (target / "NAIVE_BAYES_ALPHA.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    # No test option: model selection always uses validation only.
    runs = [run_naive_bayes(split="validation", ngram_max=2, alpha=alpha) for alpha in ALPHAS]
    summary = build_summary(runs)
    write_summary(summary)
    print(json.dumps({
        "evaluation_split": "validation",
        "best_tested_alpha": summary["best_tested_alpha"],
        "selected_run_id": summary["selected_run_id"],
        "report": "results/NAIVE_BAYES_ALPHA.md",
        "official_test_evaluated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
