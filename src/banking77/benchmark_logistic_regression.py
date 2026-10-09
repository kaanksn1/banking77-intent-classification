"""Reproduce teammate 3's Logistic Regression C/solver validation benchmark."""

import csv
import json
from collections import Counter

from scipy.stats import binomtest

from banking77.data import ROOT
from banking77.logistic_regression import SOLVERS
from banking77.train_logistic_regression import run_logistic_regression

C_VALUES = (0.1, 1.0, 10.0, 100.0, 1000.0)
BASELINE = {"solver": "lbfgs", "C": 1.0}
SOLVER_NOTES = {
    "lbfgs": "multinomial, scikit-learn varsayılanı",
    "saga": "multinomial, stokastik ortalama gradyan",
    "liblinear-ovr": "one-vs-rest, her sınıf için ikili model",
}


def schedule():
    """Solver-major order: lbfgs (default) first, so ties keep the default."""
    return [{"solver": solver, "C": C} for solver in SOLVERS for C in C_VALUES]


def select_best_run(runs):
    """Select by validation macro F1; ties keep the first scheduled setting."""
    if not runs or any(run["evaluation_split"] != "validation" for run in runs):
        raise ValueError("Model selection requires nonempty validation-only runs")
    return max(runs, key=lambda run: run["macro_f1"])


def find_run(runs, solver, C):
    return next(run for run in runs if run["settings"]["solver"] == solver and run["settings"]["C"] == C)


def read_predictions(run):
    path = ROOT / "results/runs" / run["run_id"] / "predictions.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def paired_comparison(reference, candidate):
    """Count messages that flip between two runs and test them with exact McNemar."""
    reference_rows = {row["id"]: row for row in read_predictions(reference)}
    candidate_rows = read_predictions(candidate)
    if set(reference_rows) != {row["id"] for row in candidate_rows}:
        raise ValueError("Compared runs must cover the same records")
    corrected = sum(reference_rows[r["id"]]["correct"] == "False" and r["correct"] == "True" for r in candidate_rows)
    regressed = sum(reference_rows[r["id"]]["correct"] == "True" and r["correct"] == "False" for r in candidate_rows)
    discordant = corrected + regressed
    p_value = binomtest(corrected, discordant, 0.5).pvalue if discordant else 1.0
    return {
        "reference_run_id": reference["run_id"],
        "candidate_run_id": candidate["run_id"],
        "reference_errors": sum(row["correct"] == "False" for row in reference_rows.values()),
        "candidate_errors": sum(row["correct"] == "False" for row in candidate_rows),
        "corrected_messages": corrected,
        "regressed_messages": regressed,
        "mcnemar_exact_p": p_value,
    }


def build_summary(runs):
    best = select_best_run(runs)
    baseline = find_run(runs, **BASELINE)
    data_identity = baseline["dataset_files_sha256"]
    if any(run["dataset_files_sha256"] != data_identity for run in runs):
        raise ValueError("Prepared data changed during the benchmark")
    best_lbfgs = select_best_run([run for run in runs if run["settings"]["solver"] == "lbfgs"])
    best_rows = read_predictions(best)
    errors = [row for row in best_rows if row["correct"] == "False"]
    pairs = Counter((row["true_label"], row["predicted_label"]) for row in errors)
    top_pairs = pairs.most_common(5)
    pair_examples = []
    for (actual, predicted), _ in top_pairs[:3]:
        example = next(r for r in errors if r["true_label"] == actual and r["predicted_label"] == predicted)
        pair_examples.append(example)
    report_path = ROOT / "results/runs" / best["run_id"] / "classification_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_class = [
        {"label": label, **{key: values[key] for key in ("precision", "recall", "f1-score", "support")}}
        for label, values in report.items()
        if isinstance(values, dict) and label not in ("macro avg", "weighted avg", "micro avg")
    ]
    weakest = sorted(per_class, key=lambda item: (item["f1-score"], item["label"]))[:5]
    return {
        "command": "python -m banking77.benchmark_logistic_regression",
        "selection_metric": "validation macro_f1",
        "tie_break": "first setting in the scheduled order (solver: lbfgs, saga, liblinear-ovr; C ascending)",
        "c_values": list(C_VALUES),
        "solvers": list(SOLVERS),
        "feature_settings_unchanged": True,
        "official_test_evaluated": False,
        "runs": runs,
        "baseline": BASELINE,
        "selected": {"solver": best["settings"]["solver"], "C": best["settings"]["C"]},
        "selected_run_id": best["run_id"],
        "best_lbfgs": {"C": best_lbfgs["settings"]["C"], "run_id": best_lbfgs["run_id"]},
        "baseline_vs_selected": paired_comparison(baseline, best),
        "best_lbfgs_vs_selected": paired_comparison(best_lbfgs, best),
        "most_frequent_selected_errors": [
            {"true_label": actual, "predicted_label": predicted, "count": count}
            for (actual, predicted), count in top_pairs
        ],
        "confused_pair_examples": pair_examples,
        "weakest_classes": weakest,
    }


def _setting(run):
    return f"{run['settings']['solver']}, C={run['settings']['C']:g}"


def write_summary(summary):
    target = ROOT / "results"
    target.mkdir(exist_ok=True)
    (target / "logistic_regression_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    runs = summary["runs"]
    first = runs[0]
    best = find_run(runs, **summary["selected"])
    lines = [
        "# Logistic Regression C ve solver deneyi", "",
        f"Aynı {first['training_rows']:,} eğitim ve {first['evaluation_rows']:,} validation kaydı kullanıldı.",
        "TF-IDF unigram + bigram ve sublinear TF sabit tutuldu (Naive Bayes ile aynı özellikler).",
        "Yalnızca C ve solver değiştirildi. L2 düzenlileştirme, max_iter=2000, random_state=42.",
        "Resmî testte değerlendirme yapılmadı. Bu rapor script tarafından üretilir.", "",
        "Komut: `python -m banking77.benchmark_logistic_regression`.", "",
        f"Veri özeti SHA-256: `{first['dataset_summary_sha256']}`.", "",
        "C, düzenlileştirmenin tersidir: küçük C daha güçlü düzenlileştirme (daha basit model),",
        "büyük C eğitim verisine daha sıkı uyum demektir.", "",
        "## C karşılaştırması (lbfgs, multinomial)", "",
        "| C | Validation accuracy | Validation macro F1 | Eğitim (s) | Tahmin (s) | İterasyon |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in runs:
        if run["settings"]["solver"] == "lbfgs":
            lines.append(
                f"| {run['settings']['C']:g} | {run['accuracy']:.2%} | {run['macro_f1']:.4f} | "
                f"{run['fit_seconds']:.2f} | {run['predict_seconds']:.4f} | {run['optimizer_iterations']} |"
            )
    lines += [
        "", "## Solver karşılaştırması (tüm C değerleri)", "",
        "| Solver | Çok sınıf yöntemi | C | Accuracy | Macro F1 | Eğitim (s) | Yakınsadı |",
        "| --- | --- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for run in runs:
        s = run["settings"]
        lines.append(
            f"| {s['solver']} | {s['multiclass']} | {s['C']:g} | {run['accuracy']:.2%} | "
            f"{run['macro_f1']:.4f} | {run['fit_seconds']:.2f} | {'evet' if run['converged'] else 'hayır'} |"
        )
    base = summary["baseline_vs_selected"]
    solver_cmp = summary["best_lbfgs_vs_selected"]
    best_lbfgs = find_run(runs, "lbfgs", summary["best_lbfgs"]["C"])
    lines += [
        "", "## Seçim", "",
        f"En yüksek validation macro F1: **{_setting(best)}** "
        f"(accuracy {best['accuracy']:.2%}, macro F1 {best['macro_f1']:.4f}).",
        "Eşit macro F1 durumunda çizelgedeki ilk ayar seçilir.", "",
        f"- Başlangıç ayarına göre (lbfgs, C=1): yanlış sayısı {base['reference_errors']} → {base['candidate_errors']}; "
        f"{base['corrected_messages']} hata düzeldi, {base['regressed_messages']} doğru tahmin bozuldu "
        f"(McNemar exact p = {base['mcnemar_exact_p']:.2g}).",
        f"- En iyi lbfgs ayarına göre ({_setting(best_lbfgs)}): yanlış sayısı "
        f"{solver_cmp['reference_errors']} → {solver_cmp['candidate_errors']}; "
        f"{solver_cmp['corrected_messages']} düzeldi, {solver_cmp['regressed_messages']} bozuldu "
        f"(McNemar exact p = {solver_cmp['mcnemar_exact_p']:.2g}).",
        "Süreler bu bilgisayardaki tek ölçümdür; genel bir hız üstünlüğü göstermez.", "",
        "## Sık karışan kategoriler (seçilen ayar)", "",
        "| Gerçek kategori | Tahmin | Sayı |", "| --- | --- | ---: |",
    ]
    for pair in summary["most_frequent_selected_errors"]:
        lines.append(f"| {pair['true_label']} | {pair['predicted_label']} | {pair['count']} |")
    lines += ["", "## En sık üç karışmadan birer örnek", ""]
    for row in summary["confused_pair_examples"]:
        lines += [f"- `{row['id']}`: {row['text']}",
                  f"  Gerçek: `{row['true_label']}`; tahmin: `{row['predicted_label']}`."]
    lines += [
        "", "## En düşük F1'li beş kategori (seçilen ayar)", "",
        "| Kategori | Precision | Recall | F1 | Support |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in summary["weakest_classes"]:
        lines.append(
            f"| {item['label']} | {item['precision']:.3f} | {item['recall']:.3f} | "
            f"{item['f1-score']:.3f} | {item['support']:g} |"
        )
    lines += [
        "", "Ayarlar, ortam ve veri dosyası hash'leri: [JSON raporu](logistic_regression_validation.json).",
        "Seçilen çalıştırmanın tam tahminleri ve confusion matrix'i yerel `results/runs/` altındadır.",
        "Bunlar aynı komutla yeniden üretilir; eğitilmiş modeller Git'e eklenmez.",
        "Teknik açıklama ve hata yorumları: [Logistic Regression notu](../docs/LOGISTIC_REGRESSION.md).", "",
    ]
    (target / "LOGISTIC_REGRESSION_C.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    # No test option: model selection always uses validation only.
    runs = [run_logistic_regression(split="validation", ngram_max=2, **setting) for setting in schedule()]
    summary = build_summary(runs)
    write_summary(summary)
    print(json.dumps({
        "evaluation_split": "validation",
        "selected": summary["selected"],
        "selected_run_id": summary["selected_run_id"],
        "report": "results/LOGISTIC_REGRESSION_C.md",
        "official_test_evaluated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
