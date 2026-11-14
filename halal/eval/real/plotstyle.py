"""Shared matplotlib style for the real-data plots: one palette in a fixed order, recessive axes, no dual axes."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from halal.eval.real import dataset  # noqa: E402

PLOTS = dataset.REPO / "runs/real_frames/plots"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def apply() -> None:
    """Set the rcParams used by every plot."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": "#52514e", "ytick.labelcolor": "#52514e",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "axes.prop_cycle": plt.cycler(color=SERIES), "lines.linewidth": 1.6,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
        "figure.dpi": 110,
    })


def save(fig: plt.Figure, name: str) -> Path:
    """Write a figure under runs/real_frames/plots and close it."""
    PLOTS.mkdir(parents=True, exist_ok=True)
    path = PLOTS / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
