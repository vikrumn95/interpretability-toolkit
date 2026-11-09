"""Matplotlib plots over structured results (rows from ``metrics.score`` / ``to_grid``).

Functions take data, not experiments, and return the ``Figure`` so callers can
adjust or save it. Color choices: recovery is signed with a meaningful zero, so
heatmaps use a diverging blue/red map centred on a neutral gray at 0; series use
a fixed categorical order; NaN cells are hatched (not colored) so "undefined"
never reads as "zero".
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

from .experiment import ExperimentResult, Readout
from .metrics import target_prob, to_grid

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list(
    "recovery", ["#9e2a2a", "#e34948", "#f0efec", "#2a78d6", "#184f95"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)


def plot_heatmap(grid: np.ndarray, row_labels: Sequence, col_labels: Sequence, *,
                 title: str = "", xlabel: str = "position", ylabel: str = "layer",
                 colorbar_label: str = "recovery", vlim: float | None = None, ax=None):
    """Diverging heatmap centred at 0. ``vlim`` sets symmetric limits (default: max(|grid|, 1))."""
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(
        figsize=(max(4, 0.55 * len(col_labels) + 2), max(3, 0.32 * len(row_labels) + 1.5)))
    finite = grid[np.isfinite(grid)]
    lim = vlim or max(1.0, float(np.abs(finite).max()) if finite.size else 1.0)
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=DIVERGING, aspect="auto",
                   norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim))
    for i, j in zip(*np.where(~np.isfinite(grid))):
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///",
                               edgecolor=GRID, linewidth=0))
    ax.set_xticks(range(len(col_labels)), [str(c) for c in col_labels], rotation=45, ha="right")
    ax.set_yticks(range(len(row_labels)), [str(r) for r in row_labels])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10, loc="left")
    _style(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label(colorbar_label, color=INK_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=INK_2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def plot_recovery_heatmap(rows: Iterable[dict], component: str, *, value: str = "recovery",
                          str_tokens: Sequence[str] | None = None, head=None, title: str | None = None,
                          ax=None):
    """Layer x position heatmap of ``value`` for one component (mean over pairs).

    ``str_tokens`` (e.g. ``result.baselines[0].clean_str_tokens``) labels positions.
    """
    rows = list(rows)
    grid, layers, positions = to_grid(rows, value, "layer", "position", component=component, head=head)
    if str_tokens is not None:
        cols = [f"{p}: {str_tokens[p]!r}" if p is not None else "all" for p in positions]
    else:
        cols = ["all" if p is None else p for p in positions]
    return plot_heatmap(grid, layers, cols, title=title or f"{component}: {value}",
                        colorbar_label=value, ax=ax)


def plot_prediction_comparison(readouts: dict[str, Readout],
                               target_labels: Sequence[str] = ("clean target", "corrupted target"),
                               *, title: str = "Target probabilities", ax=None):
    """Grouped bars: p(clean_target) and p(corrupted_target) for each named run.

    ``readouts`` maps a run label (e.g. "clean", "corrupted", "patched L6 pos 4")
    to its ``Readout``. The top-1 token of each run is shown under its label.
    """
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(1.6 * len(readouts) + 2, 3.2))
    labels = list(readouts)
    x = np.arange(len(labels))
    width = 0.38
    for k, name in enumerate(target_labels):
        vals = [target_prob(readouts[l], k) for l in labels]
        bars = ax.bar(x + (k - 0.5) * (width + 0.02), vals, width, color=SERIES[k], label=name)
        ax.bar_label(bars, fmt="%.3g", fontsize=7, color=INK_2, padding=2)
    ax.set_xticks(x, [f"{l}\ntop: {readouts[l].top_token!r}" for l in labels], fontsize=8)
    ax.set_ylabel("probability")
    ax.set_ylim(0, max(0.05, ax.get_ylim()[1] * 1.1))
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    ax.set_title(title, fontsize=10, loc="left")
    _style(ax)
    fig.tight_layout()
    return fig


def baseline_readouts(result: ExperimentResult, pair: int = 0) -> dict[str, Readout]:
    """Convenience: {'clean': ..., 'corrupted': ...} for one pair, for plot_prediction_comparison."""
    b = result.baselines[pair]
    return {"clean": b.clean, "corrupted": b.corrupted}


def plot_metric_by_config(rows: Iterable[dict], *, x: str = "layer", value: str = "recovery",
                          series: str = "component", title: str = "", ax=None, **where):
    """Line plot of mean ``value`` vs ``x`` with one line per ``series`` (± std band across rows).

    ``where`` filters rows by equality, e.g. ``position=8``.
    """
    rows = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    if not rows:
        raise ValueError(f"No rows match {where}")
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6, 3.2))
    keys = list(dict.fromkeys(r[series] for r in rows))
    for i, key in enumerate(keys):
        sub = [r for r in rows if r[series] == key]
        xs = sorted({r[x] for r in sub})
        vals = [np.array([r[value] for r in sub if r[x] == xv], dtype=float) for xv in xs]
        mean = np.array([np.nanmean(v) if np.isfinite(v).any() else np.nan for v in vals])
        std = np.array([np.nanstd(v) if np.isfinite(v).any() else np.nan for v in vals])
        color = SERIES[i % len(SERIES)]
        ax.plot(xs, mean, color=color, linewidth=2, marker="o", markersize=6, label=str(key))
        if any(len(v) > 1 for v in vals):
            ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel(x)
    ax.set_ylabel(value)
    if len(keys) > 1:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_2)
    ax.set_title(title or f"{value} by {x}", fontsize=10, loc="left")
    _style(ax)
    fig.tight_layout()
    return fig


def save_figure(fig, path: str | Path, dpi: int = 150, close: bool = True) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor="white", bbox_inches="tight")
    if close:
        plt.close(fig)
    return path
