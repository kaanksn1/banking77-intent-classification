"""Reproduce teammate 4's Linear SVM C/loss validation benchmark."""

import csv
import json
from collections import Counter

import joblib
from scipy.stats import binomtest

from banking77.data import ROOT
from banking77.linear_svm import LOSSES
from banking77.train_linear_svm import run_linear_svm

C_VALUES = (0.01, 0.1, 1.0, 10.0, 100.0)
BASELINE = {"loss": "squared_hinge", "C": 1.0}
LOSS_NOTES = {
    "squared_hinge": "max(0, 1 − y·f(x))², scikit-learn varsayılanı",
    "hinge": "max(0, 1 − y·f(x)), klasik soft-margin SVM",
}


def schedule():
    """Loss-major order: squared_hinge (default) first, so ties keep the default."""
    return [{"loss": loss, "C": C} for loss in LOSSES for C in C_VALUES]


def select_best_run(runs):
    """Select by validation macro F1; ties keep the first scheduled setting."""
    if not runs or any(run["evaluation_split"] != "validation" for run in runs):
        raise ValueError("Model selection requires nonempty validation-only runs")
    return max(runs, key=lambda run: run["macro_f1"])


def find_run(runs, loss, C):
    return next(run for run in runs if run["settings"]["loss"] == loss and run["settings"]["C"] == C)


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


def feature_contributions(pipeline, text, true_label, predicted_label, top=3):
    """Per-feature push toward the wrong class: (w_predicted − w_true) × tfidf.

    The sum over all features plus the intercept difference is exactly the
    decision-score gap between the predicted and the true class.
    """
    vectorizer = pipeline.named_steps["tfidf"]
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    row = vectorizer.transform([text])
    weights = classifier.coef_[classes.index(predicted_label)] - classifier.coef_[classes.index(true_label)]
    names = vectorizer.get_feature_names_out()
    pushes = sorted(
        ((names[j], float(weights[j] * value)) for j, value in zip(row.indices, row.data)),
        key=lambda item: item[1],
    )
    return {
        "toward_predicted": [{"feature": f, "value": v} for f, v in reversed(pushes[-top:]) if v > 0],
        "toward_true": [{"feature": f, "value": v} for f, v in pushes[:top] if v < 0],
    }


def build_summary(runs):
    best = select_best_run(runs)
    baseline = find_run(runs, **BASELINE)
    data_identity = baseline["dataset_files_sha256"]
    if any(run["dataset_files_sha256"] != data_identity for run in runs):
        raise ValueError("Prepared data changed during the benchmark")
    other_loss = next(loss for loss in LOSSES if loss != best["settings"]["loss"])
    best_other_loss = select_best_run([run for run in runs if run["settings"]["loss"] == other_loss])
    best_rows = read_predictions(best)
    errors = [row for row in best_rows if row["correct"] == "False"]
    pairs = Counter((row["true_label"], row["predicted_label"]) for row in errors)
    top_pairs = pairs.most_common(5)
    pipeline = joblib.load(ROOT / "artifacts" / f"{best['run_id']}.joblib")
    pair_examples = []
    for (actual, predicted), _ in top_pairs[:3]:
        example = next(r for r in errors if r["true_label"] == actual and r["predicted_label"] == predicted)
        pair_examples.append({**example, **feature_contributions(pipeline, example["text"], actual, predicted)})
    report_path = ROOT / "results/runs" / best["run_id"] / "classification_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    per_class = [
        {"label": label, **{key: values[key] for key in ("precision", "recall", "f1-score", "support")}}
        for label, values in report.items()
        if isinstance(values, dict) and label not in ("macro avg", "weighted avg", "micro avg")
    ]
    weakest = sorted(per_class, key=lambda item: (item["f1-score"], item["label"]))[:5]
    return {
        "command": "python -m banking77.benchmark_linear_svm",
        "selection_metric": "validation macro_f1",
        "tie_break": "first setting in the scheduled order (loss: squared_hinge, hinge; C ascending)",
        "c_values": list(C_VALUES),
        "losses": list(LOSSES),
        "feature_settings_unchanged": True,
        "official_test_evaluated": False,
        "runs": runs,
        "baseline": BASELINE,
        "selected": {"loss": best["settings"]["loss"], "C": best["settings"]["C"]},
        "selected_run_id": best["run_id"],
        "best_other_loss": {"loss": other_loss, "C": best_other_loss["settings"]["C"], "run_id": best_other_loss["run_id"]},
        "baseline_vs_selected": paired_comparison(baseline, best),
        "best_other_loss_vs_selected": paired_comparison(best_other_loss, best),
        "selected_errors": len(errors),
        "most_frequent_selected_errors": [
            {"true_label": actual, "predicted_label": predicted, "count": count}
            for (actual, predicted), count in top_pairs
        ],
        "confused_pair_examples": pair_examples,
        "weakest_classes": weakest,
    }


def _setting(run):
    return f"{run['settings']['loss']}, C={run['settings']['C']:g}"


def _comparison_line(label, comparison):
    if comparison["reference_run_id"] == comparison["candidate_run_id"]:
        return f"- {label}: seçilen ayarla aynı çalıştırma."
    return (
        f"- {label}: yanlış sayısı {comparison['reference_errors']} → {comparison['candidate_errors']}; "
        f"{comparison['corrected_messages']} hata düzeldi, {comparison['regressed_messages']} doğru tahmin bozuldu "
        f"(McNemar exact p = {comparison['mcnemar_exact_p']:.2g})."
    )


def _features(items):
    return ", ".join(f"`{item['feature']}` ({item['value']:+.2f})" for item in items) or "—"


def write_summary(summary):
    target = ROOT / "results"
    target.mkdir(exist_ok=True)
    (target / "linear_svm_validation.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )
    runs = summary["runs"]
    first = runs[0]
    best = find_run(runs, **summary["selected"])
    other = summary["best_other_loss"]
    lines = [
        "# Linear SVM C ve loss deneyi", "",
        f"Aynı {first['training_rows']:,} eğitim ve {first['evaluation_rows']:,} validation kaydı kullanıldı.",
        "TF-IDF unigram + bigram ve sublinear TF sabit tutuldu (Naive Bayes ve Logistic Regression ile aynı özellikler).",
        "Yalnızca C ve loss değiştirildi. L2 düzenlileştirme, one-vs-rest, dual=True, max_iter=10000, random_state=42.",
        "Resmî testte değerlendirme yapılmadı. Bu rapor script tarafından üretilir.", "",
        "Komut: `python -m banking77.benchmark_linear_svm`.", "",
        f"Veri özeti SHA-256: `{first['dataset_summary_sha256']}`.", "",
        "C, hata cezasının ağırlığıdır: küçük C geniş margin ve güçlü düzenlileştirme (daha basit model),",
        "büyük C eğitim hatalarına daha ağır ceza ve eğitim verisine daha sıkı uyum demektir.", "",
    ]
    lines += [f"- `{loss}`: {note}" for loss, note in LOSS_NOTES.items()]
    lines += [
        "",
        "| Loss | C | Validation accuracy | Validation macro F1 | Eğitim (s) | Tahmin (s) | İterasyon | Yakınsadı |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for run in runs:
        s = run["settings"]
        lines.append(
            f"| {s['loss']} | {s['C']:g} | {run['accuracy']:.2%} | {run['macro_f1']:.4f} | "
            f"{run['fit_seconds']:.2f} | {run['predict_seconds']:.4f} | {run['optimizer_iterations']} | "
            f"{'evet' if run['converged'] else 'hayır'} |"
        )
    lines += [
        "", "## Seçim", "",
        f"En yüksek validation macro F1: **{_setting(best)}** "
        f"(accuracy {best['accuracy']:.2%}, macro F1 {best['macro_f1']:.4f}).",
        "Eşit macro F1 durumunda çizelgedeki ilk ayar seçilir.", "",
        _comparison_line("Başlangıç ayarına göre (squared_hinge, C=1)", summary["baseline_vs_selected"]),
        _comparison_line(f"Diğer loss'un en iyi ayarına göre ({other['loss']}, C={other['C']:g})",
                         summary["best_other_loss_vs_selected"]),
        "Süreler bu bilgisayardaki tek ölçümdür; genel bir hız üstünlüğü göstermez.", "",
        "## Sık karışan kategoriler (seçilen ayar)", "",
        f"Seçilen ayarda yanlış sınıflandırılan validation mesajı: {summary['selected_errors']} / {first['evaluation_rows']:,}.", "",
        "| Gerçek kategori | Tahmin | Sayı |", "| --- | --- | ---: |",
    ]
    for pair in summary["most_frequent_selected_errors"]:
        lines.append(f"| {pair['true_label']} | {pair['predicted_label']} | {pair['count']} |")
    lines += [
        "", "## En sık üç karışmadan birer örnek", "",
        "Katkı = `(w_tahmin − w_gerçek) × tfidf`; pozitif değer yanlış kategoriye, negatif değer doğru kategoriye iter.", "",
    ]
    for row in summary["confused_pair_examples"]:
        lines += [f"- `{row['id']}`: {row['text']}",
                  f"  Gerçek: `{row['true_label']}`; tahmin: `{row['predicted_label']}`.",
                  f"  Yanlışa iten: {_features(row['toward_predicted'])}. Doğruya iten: {_features(row['toward_true'])}."]
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
        "", "Ayarlar, ortam ve veri dosyası hash'leri: [JSON raporu](linear_svm_validation.json).",
        "Seçilen çalıştırmanın tam tahminleri ve confusion matrix'i yerel `results/runs/` altındadır.",
        "Bunlar aynı komutla yeniden üretilir; eğitilmiş modeller Git'e eklenmez.",
        "Teknik açıklama ve hata yorumları: [Linear SVM notu](../docs/LINEAR_SVM.md).", "",
    ]
    (target / "LINEAR_SVM_C.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")


def main():
    # No test option: model selection always uses validation only.
    runs = [run_linear_svm(split="validation", ngram_max=2, **setting) for setting in schedule()]
    summary = build_summary(runs)
    write_summary(summary)
    print(json.dumps({
        "evaluation_split": "validation",
        "selected": summary["selected"],
        "selected_run_id": summary["selected_run_id"],
        "report": "results/LINEAR_SVM_C.md",
        "official_test_evaluated": False,
    }, indent=2))


if __name__ == "__main__":
    main()
