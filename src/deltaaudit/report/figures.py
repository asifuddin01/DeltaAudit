"""All figures, and only here (CLAUDE.md style rule). Static PNGs for the write-up.

Colors: reference data-viz palette, light mode (categorical slots 1-2 are a validated pair).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834"]
DPI = 200


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TEXT_2)
    ax.yaxis.label.set_color(TEXT_2)


def _ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(x[~np.isnan(x)])
    return x, np.arange(1, len(x) + 1) / len(x)


def fig_sys_variation(desc: pd.DataFrame, title: str, out: Path) -> Path:
    """Fig 1: (A) sys_cos histogram with median; (B) split-half ceiling ECDFs, raw vs pspec."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.4), facecolor=SURFACE)
    sys_cos = desc["sys_cos"].dropna()
    a.hist(sys_cos, bins=40, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
    med = float(sys_cos.median())
    a.axvline(med, color=TEXT_2, linewidth=1, linestyle="--")
    a.annotate(f"median {med:.2f}", (med, 0.97), xycoords=("data", "axes fraction"),
               xytext=(4, 0), textcoords="offset points", ha="left", va="top",
               fontsize=8, color=TEXT)  # fmt: skip
    a.set_xlabel("cos(Δ_p, mean Δ)  — systematic variation")
    a.set_ylabel("perturbations")
    a.set_title(
        "A  Similarity to the average effect", loc="left", fontsize=9, color=TEXT
    )

    for col, color, label in (
        ("ceil_raw", SERIES[0], "raw Pearson Δ"),
        ("ceil_pspec", SERIES[1], "perturbation-specific"),
    ):
        x, y = _ecdf(desc[col].to_numpy())
        m = float(np.median(x))
        b.plot(x, y, color=color, linewidth=2, label=f"{label} (median {m:.2f})")
        b.plot(
            [m], [0.5], marker="o", markersize=5, color=color, markeredgecolor=SURFACE
        )
    b.set_xlabel("split-half Pearson r  (noise ceiling)")
    b.set_ylabel("fraction of perturbations ≤ r")
    b.set_ylim(0, 1.02)
    b.set_title("B  Noise ceiling (dots: medians)", loc="left", fontsize=9, color=TEXT)
    leg = b.legend(frameon=False, fontsize=8, loc="upper left")
    for t in leg.get_texts():
        t.set_color(TEXT)

    for ax in (a, b):
        _style(ax)
    fig.suptitle(f"{title}  (n = {len(desc):,} perturbations; target gene excluded)",
                 x=0.01, ha="left", fontsize=10, color=TEXT)  # fmt: skip
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    plt.close(fig)
    return out
