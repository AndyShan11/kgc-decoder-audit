from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 7.5,
        "axes.titlesize": 8,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.55,
        "grid.linewidth": 0.35,
    }
)


BLUE = "#2f80b9"
RED = "#d64b3c"
PURPLE = "#7a5ab5"
GREY = "#9a9a9a"
DARK = "#222222"


def save_edges_per_relation() -> None:
    data = [
        ("UMLS", 113, 0.022),
        ("Kinship", 342, 0.143),
        ("FB15k-237", 1148, 0.005),
        ("CoDEx-M", 3627, 0.010),
        ("WN18RR", 7894, 0.012),
        ("YAGO3-10", 29163, 0.006),
    ]

    fig, ax = plt.subplots(figsize=(3.35, 2.15))
    xs = np.array([row[1] for row in data], dtype=float)
    ys = np.array([row[2] for row in data], dtype=float)

    ax.scatter(xs, ys, s=20, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Training edges per relation (log scale)", labelpad=2)
    ax.set_ylabel("Decoder delta (CX-DM MRR)", labelpad=2)
    ax.set_ylim(-0.012, 0.158)
    ax.set_xlim(70, 50000)
    ax.grid(True, axis="both", color="#d8d8d8", alpha=0.85)
    ax.axhline(0, color="#555555", linewidth=0.5)

    offsets = {
        "UMLS": (8, 8, "left"),
        "Kinship": (8, 4, "left"),
        "FB15k-237": (-14, 14, "right"),
        "CoDEx-M": (-10, 24, "right"),
        "WN18RR": (-2, 20, "center"),
        "YAGO3-10": (10, 14, "left"),
    }
    for name, x, y in data:
        dx, dy, ha = offsets[name]
        ax.annotate(
            name,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=6.8,
            color=DARK,
            arrowprops=dict(
                arrowstyle="-",
                color="#666666",
                lw=0.35,
                shrinkA=1.5,
                shrinkB=2.5,
            ),
        )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(ROOT / "fig2_edges_per_rel.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def save_wn18rr_ablation() -> None:
    labels = [
        "LS=0",
        "LS=0.1",
        "L=1",
        "d=128",
        "Baseline\nDM",
        "L=3",
        "ComplEx",
    ]
    mrr = np.array([0.4266, 0.4573, 0.4414, 0.4655, 0.4673, 0.4743, 0.4780])
    colors = [RED, RED, "#7fa6c9", "#7fa6c9", GREY, BLUE, PURPLE]

    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    x = np.arange(len(labels))
    bars = ax.bar(x, mrr, color=colors, width=0.68, edgecolor="white", linewidth=0.4)

    ax.set_ylabel("Test MRR", labelpad=2)
    ax.set_ylim(0.39, 0.493)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.grid(True, axis="y", color="#d8d8d8", alpha=0.8)
    ax.set_axisbelow(True)
    ax.axhline(mrr[4], color="#666666", linewidth=0.65, linestyle="--", zorder=0)

    for bar, value in zip(bars, mrr):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.0021,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=6.4,
            color=DARK,
        )

    ax.annotate(
        "No LS: -0.041",
        xy=(x[0], mrr[0]),
        xytext=(x[0] - 0.15, 0.486),
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=RED,
        arrowprops=dict(arrowstyle="-", color=RED, lw=0.55, shrinkA=0, shrinkB=3),
    )
    ax.annotate(
        "ComplEx: +0.011",
        xy=(x[-1], mrr[-1]),
        xytext=(x[-1] - 0.15, 0.488),
        ha="center",
        va="bottom",
        fontsize=6.8,
        color=PURPLE,
        arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.55, shrinkA=0, shrinkB=3),
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.25)
    fig.savefig(ROOT / "fig3_ablation_wn18rr.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    save_edges_per_relation()
    save_wn18rr_ablation()
