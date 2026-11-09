import math

import numpy as np

from interp_toolkit import plotting
from interp_toolkit.experiment import Readout


def synthetic_rows():
    return [{"component": c, "layer": l, "position": p, "head": None, "pair": k,
             "recovery": (l / 3) * (p / 4) if (l, p) != (1, 2) else math.nan, "score": l - p}
            for c in ("resid_pre", "mlp_out") for l in range(4) for p in range(5) for k in range(2)]


def test_heatmap_renders_and_saves(tmp_path):
    fig = plotting.plot_recovery_heatmap(synthetic_rows(), "resid_pre",
                                         str_tokens=["<bos>", "a", "b", "c", "d"])
    path = plotting.save_figure(fig, tmp_path / "figs" / "heat.png")
    assert path.exists() and path.stat().st_size > 1000


def test_raw_heatmap_all_nan():
    fig = plotting.plot_heatmap(np.full((2, 3), np.nan), [0, 1], [0, 1, 2])
    assert fig.axes


def test_prediction_comparison(tmp_path):
    readouts = {"clean": Readout([2.0, 0.0], 3.0, 1, " a"), "corrupted": Readout([0.0, 2.0], 3.0, 2, " b"),
                "patched": Readout([1.0, 1.0], 3.0, 1, " a")}
    fig = plotting.plot_prediction_comparison(readouts)
    assert plotting.save_figure(fig, tmp_path / "pred.png").exists()


def test_metric_by_config(tmp_path):
    fig = plotting.plot_metric_by_config(synthetic_rows(), x="layer", value="score", position=4)
    assert len(fig.axes[0].lines) >= 2
    assert plotting.save_figure(fig, tmp_path / "line.svg").exists()
