"""Figures for the shared model comparison (reads the benchmark summary)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter  # noqa: E402

# Sabit kategori renkleri: ilk üç slot, üç modelin tüm ikilileri için doğrulanmış palet
COLORS = {"Naive Bayes": "#2a78d6", "Logistic Regression": "#eb6834", "Linear SVM": "#1baf7a"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
MODEL_ORDER = ("Naive Bayes", "Logistic Regression", "Linear SVM")


def _rows(summary):
    """Her model için (başlangıç, seçilen) girdileri; SVM'de ikisi aynıdır."""
    by_model = {}
    for entry in summary["models"].values():
        by_model.setdefault(entry["model"], {})[entry["stage"]] = entry
    rows = []
    for model in MODEL_ORDER:
        stages = by_model[model]
        if "initial = selected" in stages:
            rows.append((model, stages["initial = selected"], None))
        else:
            rows.append((model, stages["initial"], stages["selected"]))
    return rows


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _dot_panel(ax, rows, getter, fmt, title, log=False, pad=0.12):
    """Satır başına bir model: boş daire = başlangıç, dolu daire = seçilen."""
    _style(ax)
    ys = list(range(len(rows)))[::-1]
    values = []
    for y, (model, initial, selected) in zip(ys, rows):
        color = COLORS[model]
        start = getter(initial)
        values.append(start)
        if selected is None:
            ax.plot(start, y, "o", color=color, markersize=10, markeredgecolor=SURFACE, markeredgewidth=1.5)
            ax.annotate(fmt(start) + " (initial = selected)", (start, y), textcoords="offset points",
                        xytext=(0, 12), ha="center", fontsize=9, color=INK)
            continue
        end = getter(selected)
        values.append(end)
        ax.plot([start, end], [y, y], "-", color=color, linewidth=2, alpha=0.5, zorder=1)
        ax.plot(start, y, "o", markerfacecolor=SURFACE, markeredgecolor=color, markeredgewidth=2,
                markersize=10, zorder=3)
        ax.plot(end, y, "o", color=color, markersize=10, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        ax.annotate(fmt(start), (start, y), textcoords="offset points", xytext=(0, -18), ha="center",
                    fontsize=9, color=INK)
        ax.annotate(fmt(end), (end, y), textcoords="offset points", xytext=(0, 12), ha="center",
                    fontsize=9, color=INK)
    ax.set_yticks(ys)
    ax.set_yticklabels([row[0] for row in rows], color=INK, fontsize=10)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    if log:
        ax.set_xscale("log")
        low, high = min(values), max(values)
        ax.set_xlim(low / 3, high * 3)
        # Log ekseninde yalnızca düz sayılarla etiketlenen sabit işaretler kullanılır
        ticks = [t for t in (0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10) if low / 3 <= t <= high * 3]
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        low, high = min(values), max(values)
        span = max(high - low, 0.02)
        ax.set_xlim(low - span * pad * 6, high + span * pad * 6)
    ax.set_title(title, loc="left", color=INK, fontsize=11)


def _legend(fig):
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK_SECONDARY,
                   markeredgewidth=2, markersize=9, label="initial setting"),
        plt.Line2D([], [], marker="o", linestyle="", color=INK_SECONDARY, markersize=9, label="selected setting"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, labelcolor=INK_SECONDARY, fontsize=9)


def scores_figure(summary, path):
    rows = _rows(summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    _dot_panel(axes[0], rows, lambda e: e["macro_f1"], lambda v: f"{v:.4f}", "Macro F1 (main metric)")
    _dot_panel(axes[1], rows, lambda e: e["accuracy"], lambda v: f"{100 * v:.2f}%", "Accuracy (supporting)")
    axes[1].set_yticklabels([])
    fig.suptitle("Validation scores, 1,500 messages, 77 categories", x=0.02, ha="left", color=INK, fontsize=12)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def timing_figure(summary, path):
    rows = _rows(summary)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE)
    _dot_panel(axes[0], rows, lambda e: e["fit_seconds"], lambda v: f"{v:.2f} s", "Training time (log scale)",
               log=True)
    _dot_panel(axes[1], rows, lambda e: 1000 * e["predict_seconds"], lambda v: f"{v:.1f} ms",
               "Prediction time for all 1,500 messages (ms)")
    axes[1].set_yticklabels([])
    fig.suptitle("Median of repeated runs on one machine, models run sequentially", x=0.02, ha="left",
                 color=INK, fontsize=12)
    _legend(fig)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def confusions_figure(summary, path):
    table = summary["confusion_pair_table"][:8]
    keys = ("nb_selected", "lr_selected", "svm")
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.xaxis.grid(True, color=GRID)
    height = 0.24
    for j, (key, model) in enumerate(zip(keys, MODEL_ORDER)):
        positions = [i + (j - 1) * (height + 0.03) for i in range(len(table))]
        counts = [item["counts"][key] for item in table]
        ax.barh(positions, counts, height=height, color=COLORS[model], label=model, edgecolor=SURFACE,
                linewidth=1)
        for position, count in zip(positions, counts):
            ax.annotate(str(count), (count, position), textcoords="offset points", xytext=(4, 0),
                        va="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(table)))
    ax.set_yticklabels([f"{t['true_label']} → {t['predicted_label']}" for t in table], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Misclassified validation messages", color=INK_SECONDARY)
    ax.legend(frameon=False, labelcolor=INK_SECONDARY, loc="lower right")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.text(0.01, 0.97, "Most confused category pairs (true → predicted), selected settings",
             color=INK, fontsize=12, va="center")
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def make_figures(summary, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    scores_figure(summary, directory / "scores.png")
    timing_figure(summary, directory / "timing.png")
    confusions_figure(summary, directory / "confusions.png")
