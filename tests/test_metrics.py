import math

import numpy as np
import pytest
import torch

from interp_toolkit.experiment import (ExperimentResult, PairBaseline, PatchRecord, PromptPair, Readout,
                                       SweepConfig)
from interp_toolkit.metrics import (aggregate, logit_diff, logit_diff_from_logits, prob_from_logits,
                                    recovery, score, target_prob, to_grid)


def R(a, b, lse=0.0):
    return Readout([a, b], lse, 0, "x")


def test_logit_diff_and_prob():
    r = R(2.0, -1.0, lse=math.log(math.exp(2) + math.exp(-1) + 1))  # 3-token vocab: logits 2, -1, 0
    assert logit_diff(r) == 3.0
    assert target_prob(r, 0) == pytest.approx(math.exp(2) / (math.exp(2) + math.exp(-1) + 1))
    assert target_prob(r, 0) + target_prob(r, 1) < 1


def test_recovery_values():
    assert recovery(5.0, 5.0, 1.0) == 1.0
    assert recovery(1.0, 5.0, 1.0) == 0.0
    assert recovery(3.0, 5.0, 1.0) == 0.5
    assert recovery(7.0, 5.0, 1.0) == 1.5  # overshoot
    assert recovery(-1.0, 5.0, 1.0) == -0.5  # moves away from clean
    assert recovery(3.0, -2.0, 2.0) == pytest.approx(-0.25)  # works for negative gaps


def test_recovery_zero_denominator():
    assert math.isnan(recovery(1.0, 2.0, 2.0))
    assert math.isnan(recovery(1.0, 2.0, 2.0 + 1e-9))
    assert math.isnan(recovery(1.0, 2.0, 1.95, min_denominator=0.1))
    out = recovery(np.array([1.0, 2.0]), np.array([2.0, 3.0]), np.array([0.0, 3.0]))
    assert out[0] == 0.5 and np.isnan(out[1])


def test_tensor_helpers():
    logits = torch.tensor([[[0.0, 1.0, 3.0], [2.0, 0.0, 1.0]]])
    assert logit_diff_from_logits(logits, 0, 1).tolist() == [2.0]
    assert prob_from_logits(logits, 0).item() == pytest.approx(torch.tensor([2.0, 0.0, 1.0]).softmax(0)[0].item())


@pytest.fixture
def synthetic_result():
    pairs = [PromptPair("a", "b", " x", " y")] * 2
    baselines = [PairBaseline(0, R(4.0, 0.0), R(0.0, 0.0), [1, 2], [], []),
                 PairBaseline(1, R(2.0, 0.0), R(2.0, 0.0), [1, 2], [], [])]  # zero gap
    records = [PatchRecord(p, "resid_pre", layer, pos, None, R(float(layer + pos), 0.0))
               for p in (0, 1) for layer in (0, 1) for pos in (0, 1)]
    return ExperimentResult(SweepConfig(), pairs, baselines, records)


def test_score_rows(synthetic_result):
    rows = score(synthetic_result)
    assert len(rows) == 8
    r = next(r for r in rows if r["pair"] == 0 and r["layer"] == 1 and r["position"] == 1)
    assert (r["score"], r["clean_score"], r["corrupted_score"], r["recovery"]) == (2.0, 4.0, 0.0, 0.5)
    assert all(math.isnan(r["recovery"]) for r in rows if r["pair"] == 1)
    assert "prob_change" in r and "clean_target_prob" in r


def test_score_custom_metric(synthetic_result):
    rows = score(synthetic_result, metric=lambda r: r.target_logits[0] * 2)
    assert rows[0]["clean_score"] == 8.0


def test_aggregate_skips_nan(synthetic_result):
    agg = aggregate(score(synthetic_result))
    assert len(agg) == 4
    a = next(a for a in agg if a["layer"] == 1 and a["position"] == 1)
    assert (a["mean"], a["n"], a["n_nan"]) == (0.5, 1, 1)
    by_layer = aggregate(score(synthetic_result), value="score", by=("layer",))
    assert {a["layer"]: a["mean"] for a in by_layer} == {0: 0.5, 1: 1.5}


def test_to_grid(synthetic_result):
    grid, rows, cols = to_grid(score(synthetic_result), component="resid_pre")
    assert rows == [0, 1] and cols == [0, 1]
    np.testing.assert_allclose(grid, [[0.0, 0.25], [0.25, 0.5]])
    with pytest.raises(ValueError):
        to_grid(score(synthetic_result), component="z")
