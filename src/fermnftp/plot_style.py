"""Shared plotting style for paper assets.

The visual target is a compact AI-conference paper style: white background,
minimal ink, colorblind-friendly colors, subtle grids, and no significance
markers in figures unless explicitly requested later.
"""

from __future__ import annotations

from typing import Any

from cycler import cycler


AI_CONFERENCE_COLORS = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#56B4E9",  # sky blue
    "#E69F00",  # orange
    "#332288",  # indigo
    "#88CCEE",  # light blue
]

MODEL_COLORS = {
    "Persistence": "#8C8C8C",
    "Blind Ridge": "#0072B2",
    "Controlled Ridge": "#D55E00",
    "Ridge": "#D55E00",
    "GRU": "#0072B2",
    "TCN": "#009E73",
    "Transformer": "#CC79A7",
    "Process": "#8C8C8C",
    "Naive": "#0072B2",
    "Gated": "#009E73",
    "Attention": "#CC79A7",
    "Residual Process": "#8C8C8C",
    "Residual Naive Raman": "#0072B2",
    "TargetGate": "#009E73",
}

SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "coolwarm"

SIGNIFICANCE_ANNOTATIONS_ALLOWED = False
SIGNIFICANCE_POLICY = (
    "No p-values, stars, ns labels, significance brackets, error bars, or confidence bands are drawn "
    "inside paper figures by default. Variability fields may remain in CSV files for future redrawing."
)


def apply_ai_conference_style(plt: Any) -> None:
    """Apply a shared Matplotlib style for all generated paper figures."""

    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "figure.titlesize": 11.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#303030",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.45,
            "grid.alpha": 0.75,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.0,
            "patch.linewidth": 0.5,
            "legend.frameon": False,
            "legend.handlelength": 1.6,
            "legend.borderaxespad": 0.4,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "axes.prop_cycle": cycler(color=AI_CONFERENCE_COLORS),
        }
    )


def polish_axis(ax: Any, *, grid_axis: str = "y") -> None:
    """Apply small finishing touches to an axis."""

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis == "none":
        ax.grid(False)
    else:
        ax.grid(True, axis=grid_axis, color="#D9D9D9", linewidth=0.45, alpha=0.75)


def no_significance_note() -> str:
    """Return a standard note for figure explanation files."""

    return "，；CSV ，。"
