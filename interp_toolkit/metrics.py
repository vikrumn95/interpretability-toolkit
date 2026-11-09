"""Metrics over recorded readouts. Pure functions: no model, no forward passes.

Assumptions and limitations
---------------------------
logit_diff = logit(clean_target) - logit(corrupted_target)
    Invariant to adding a constant to all logits, so it ignores the softmax
    normaliser. Only measures the competition between the two chosen tokens;
    a patch can move probability to a third token without changing it.

target probability p(t) = softmax(logits)[t]
    Sensitive to every other token. Saturates near 0 and 1, so equal
    probability changes can correspond to very different logit changes.

recovery = (patched - corrupted) / (clean - corrupted)
    0 means "no different from the corrupted run" and 1 means "matches the clean
    run". Values outside [0, 1] are possible and meaningful (overshoot, or a
    move in the opposite direction). It is undefined when clean ≈ corrupted;
    such cases return NaN (see ``min_denominator``) and aggregation skips them,
    while reporting how many were skipped. Recovery is only as meaningful as the
    gap it normalises by: a tiny but nonzero clean-corrupted gap produces large,
    noisy values, so choose ``min_denominator`` on the scale of the metric.

Aggregation across prompt pairs averages per-pair values. Grouping by
``position`` assumes positions mean the same thing across pairs, which only
holds when pairs share a token template.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict
from typing import Callable, Iterable, Sequence

import numpy as np
import torch

from .experiment import ExperimentResult, Readout

# ---------- readout-level metrics ----------


def logit_diff(r: Readout) -> float:
    return r.target_logits[0] - r.target_logits[1]


def target_prob(r: Readout, which: int = 0) -> float:
    """Probability of target ``which`` (0 = clean_target, 1 = corrupted_target)."""
    return math.exp(r.target_logits[which] - r.logsumexp)


METRICS: dict[str, Callable[[Readout], float]] = {
    "logit_diff": logit_diff,
    "clean_target_prob": lambda r: target_prob(r, 0),
    "corrupted_target_prob": lambda r: target_prob(r, 1),
}


def recovery(patched, clean, corrupted, min_denominator: float = 1e-6):
    """Normalized patching recovery; NaN where ``|clean - corrupted| < min_denominator``.

    Works on Python floats or NumPy arrays (broadcasting).
    """
    patched, clean, corrupted = (np.asarray(x, dtype=float) for x in (patched, clean, corrupted))
    denom = clean - corrupted
    ok = np.abs(denom) >= min_denominator
    out = np.where(ok, (patched - corrupted) / np.where(ok, denom, 1.0), np.nan)
    return float(out) if out.ndim == 0 else out


# ---------- tensor-level helpers for ad-hoc use with raw logits ----------


def logit_diff_from_logits(logits: torch.Tensor, a: int, b: int, pos: int = -1) -> torch.Tensor:
    """logit[a] - logit[b] at ``pos`` for each batch element. logits: [batch, pos, vocab]."""
    return logits[:, pos, a] - logits[:, pos, b]


def prob_from_logits(logits: torch.Tensor, token: int, pos: int = -1) -> torch.Tensor:
    return logits[:, pos].float().softmax(-1)[:, token]


# ---------- experiment-level scoring and aggregation ----------


def score(result: ExperimentResult, metric: str | Callable[[Readout], float] = "logit_diff",
          min_denominator: float = 1e-6) -> list[dict]:
    """One flat row per patched run, with the metric, baselines, recovery, and probability change.

    ``prob_change`` is p(clean_target | patched) - p(clean_target | corrupted).
    """
    fn = METRICS[metric] if isinstance(metric, str) else metric
    base = {b.pair: (fn(b.clean), fn(b.corrupted), target_prob(b.corrupted, 0)) for b in result.baselines}
    rows = []
    for rec in result.records:
        clean_s, corr_s, corr_p = base[rec.pair]
        s = fn(rec.readout)
        p = target_prob(rec.readout, 0)
        rows.append({
            "pair": rec.pair, "component": rec.component, "layer": rec.layer,
            "position": rec.position, "head": rec.head,
            "score": s, "clean_score": clean_s, "corrupted_score": corr_s,
            "recovery": recovery(s, clean_s, corr_s, min_denominator),
            "clean_target_prob": p, "prob_change": p - corr_p,
            "top_token": rec.readout.top_token,
        })
    return rows


def aggregate(rows: Iterable[dict], value: str = "recovery",
              by: Sequence[str] = ("component", "layer", "position", "head")) -> list[dict]:
    """Mean/std of ``value`` over rows sharing the ``by`` keys (typically: across pairs).

    NaNs are excluded; ``n`` counts valid values and ``n_nan`` the excluded ones.
    """
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        groups[tuple(r[k] for k in by)].append(r[value])
    out = []
    for key, vals in groups.items():
        v = np.asarray(vals, dtype=float)
        valid = v[~np.isnan(v)]
        out.append({**dict(zip(by, key)),
                    "mean": float(valid.mean()) if valid.size else float("nan"),
                    "std": float(valid.std()) if valid.size else float("nan"),
                    "n": int(valid.size), "n_nan": int(v.size - valid.size)})
    return out


def to_grid(rows: Iterable[dict], value: str = "recovery", row_key: str = "layer",
            col_key: str = "position", **where) -> tuple[np.ndarray, list, list]:
    """Pivot rows into a 2D array (NaN-mean over duplicates, e.g. across pairs).

    ``where`` filters rows by equality first, e.g. ``to_grid(rows, component="resid_pre")``.
    Returns ``(grid, row_labels, col_labels)``; missing cells are NaN.
    """
    sel = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
    if not sel:
        raise ValueError(f"No rows match {where}")
    row_labels = sorted({r[row_key] for r in sel}, key=_sort_key)
    col_labels = sorted({r[col_key] for r in sel}, key=_sort_key)
    cells: dict[tuple, list[float]] = defaultdict(list)
    for r in sel:
        cells[(r[row_key], r[col_key])].append(r[value])
    grid = np.full((len(row_labels), len(col_labels)), np.nan)
    ri = {k: i for i, k in enumerate(row_labels)}
    ci = {k: i for i, k in enumerate(col_labels)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells stay NaN
        for (rk, ck), vals in cells.items():
            grid[ri[rk], ci[ck]] = np.nanmean(np.asarray(vals, dtype=float))
    return grid, row_labels, col_labels


def _sort_key(x):
    return (x is None, x if x is not None else 0)
