"""16 yöntemlik karşılaştırma için şekiller (compare_all_models özetini okur)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from banking77.data import ROOT  # noqa: E402

FIGURES = ROOT / "results/figures"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Üst üste binen etiketler için elle verilen kaydırmalar (nokta cinsinden)
LABEL_OFFSETS = {"roberta": (-8, 2), "nb_cnn": (-8, 4), "albert": (-4, 9), "bert": (8, -3),
                 "distilbert": (-8, -11), "fasttext": (-8, 4), "glove": (-8, -4)}
FAMILY_COLORS = {
    "Classical TF-IDF baselines": "#2a78d6",
    "Embeddings + linear head": "#8a63d2",
    "Neural networks trained from scratch": "#eb6834",
    "Pretrained Transformers (fine-tuned)": "#1baf7a",
    "Team contribution": "#c2185b",
}


def _style(ax, grid_axis="x"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    ax.grid(True, axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _legend(ax, location):
    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=color, label=family)
        for family, color in FAMILY_COLORS.items()
    ]
    ax.legend(handles=handles, loc=location, frameon=False, fontsize=8, labelcolor=INK_SECONDARY)


def scores_figure(summary):
    methods = summary["methods"]
    fig, ax = plt.subplots(figsize=(8, 6), facecolor=SURFACE)
    _style(ax)
    ys = list(range(len(methods)))[::-1]
    for y, m in zip(ys, methods):
        color = FAMILY_COLORS[m["family"]]
        low, high = m["macro_f1_ci95"]
        ax.plot([low, high], [y, y], color=color, linewidth=2, alpha=0.5)
        ax.plot(m["macro_f1"], y, "o", color=color, markersize=7)
        ax.text(high + 0.004, y, f"{m['macro_f1']:.3f}", va="center", fontsize=8, color=INK_SECONDARY)
    ax.set_yticks(ys)
    ax.set_yticklabels([m["name"] for m in methods], fontsize=9, color=INK)
    ax.set_xlabel("Macro F1 on the official test (dot) with paired bootstrap 95% interval", color=INK_SECONDARY)
    ax.set_title("Sixteen methods on the same 3,080 test messages", loc="left", color=INK, fontsize=12)
    _legend(ax, "lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "all_scores.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def cost_figure(summary):
    methods = summary["methods"]
    fig, ax = plt.subplots(figsize=(7.5, 5), facecolor=SURFACE)
    _style(ax, "both")
    for m in methods:
        color = FAMILY_COLORS[m["family"]]
        ax.plot(m["timing"]["fit_seconds"], m["macro_f1"], "o", color=color, markersize=8)
        offset = LABEL_OFFSETS.get(m["key"], (5, 3))
        ax.annotate(m["name"], (m["timing"]["fit_seconds"], m["macro_f1"]), xytext=offset,
                    textcoords="offset points", fontsize=7, color=INK_SECONDARY,
                    ha="right" if offset[0] < 0 else "left")
    ax.set_xscale("log")
    ax.set_xlim(0.06, 1800)
    ax.set_xlabel("Fit time in seconds (log scale; device and scope differ, see report)", color=INK_SECONDARY)
    ax.set_ylabel("Macro F1 on the official test", color=INK_SECONDARY)
    ax.set_title("Accuracy against training cost", loc="left", color=INK, fontsize=12)
    _legend(ax, "lower right")
    fig.tight_layout()
    fig.savefig(FIGURES / "all_cost.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def ablation_figure(summary):
    a = summary["ablation"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4), facecolor=SURFACE, gridspec_kw={"width_ratios": [3, 2]})
    _style(left, "y")
    weights = [g["cnn_weight"] for g in a["validation_grid"]]
    values = [g["macro_f1"] for g in a["validation_grid"]]
    left.plot(weights, values, "-o", color=FAMILY_COLORS["Team contribution"])
    left.axvline(a["selected_cnn_weight"], color=INK_SECONDARY, linestyle="--", linewidth=1)
    left.set_xlabel("CNN weight (0 = Naive Bayes only, 1 = CNN only)", color=INK_SECONDARY)
    left.set_ylabel("Validation macro F1", color=INK_SECONDARY)
    left.set_title("Weight chosen on validation", loc="left", color=INK, fontsize=11)
    _style(right, "x")
    names = ["Naive Bayes", "CNN", "NB + CNN"]
    keys = ["nb", "cnn", "nb_cnn"]
    colors = [FAMILY_COLORS["Classical TF-IDF baselines"], FAMILY_COLORS["Neural networks trained from scratch"],
              FAMILY_COLORS["Team contribution"]]
    scores = [a["test"][k]["macro_f1"] for k in keys]
    ys = [2, 1, 0]
    for y, value, color in zip(ys, scores, colors):
        right.plot([0.8, value], [y, y], color=color, linewidth=1.5, alpha=0.5)
        right.plot(value, y, "o", color=color, markersize=9)
        right.text(value, y + 0.22, f"{value:.3f}", ha="center", fontsize=9, color=INK)
    right.set_yticks(ys)
    right.set_yticklabels(names, color=INK)
    right.set_xlim(0.8, 0.93)
    right.set_ylim(-0.6, 2.7)
    right.set_xlabel("Macro F1 (axis starts at 0.80)", color=INK_SECONDARY)
    right.set_title("Official test macro F1", loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "all_ablation.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def make_figures(summary):
    FIGURES.mkdir(parents=True, exist_ok=True)
    scores_figure(summary)
    cost_figure(summary)
    ablation_figure(summary)
