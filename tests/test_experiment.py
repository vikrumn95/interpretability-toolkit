import pytest
import torch
from transformer_lens import patching

from interp_toolkit import (ExperimentResult, PromptPair, SweepConfig, active_hook_count,
                            enumerate_interventions, run_patching_experiment)


def test_grid_size():
    cfg = SweepConfig(components=["resid_pre", "z"], layers=[0, 1, 2], positions=[0, -1], per_head=True)
    grid = enumerate_interventions(cfg, n_layers=12, n_heads=12, seq_len=6)
    assert len(grid) == 3 * 2 + 3 * 2 * 12
    assert len(enumerate_interventions(SweepConfig(), 12, 12, 6)) == 12 * 6
    one = enumerate_interventions(SweepConfig(per_position=False), 12, 12, 6)
    assert len(one) == 12 and all(iv.position is None for iv in one)
    with pytest.raises(ValueError, match="Position 9"):
        enumerate_interventions(SweepConfig(positions=[9]), 12, 12, 6)


def test_sweep_produces_expected_records(model, pair):
    cfg = SweepConfig(components=["resid_pre", "attn_out", "z"], layers=[0, 11], positions=[1, -1],
                      per_head=True)
    res = run_patching_experiment(model, [pair, pair], cfg)
    per_pair = 2 * 2 + 2 * 2 + 2 * 2 * 12
    assert len(res.records) == 2 * per_pair
    assert len(res.baselines) == 2
    assert res.metadata["n_forward_passes"] == 2 * per_pair + 4
    assert active_hook_count(model) == 0
    assert {r.position for r in res.records} == {1, 5}  # -1 normalised to absolute index


def test_matches_transformerlens_patching(model, pair, tokens):
    """Independent oracle: TL's own resid_pre patching sweep gives identical logit diffs."""
    res = run_patching_experiment(model, [pair], SweepConfig(components=["resid_pre"]))
    clean, corrupted = tokens
    a, b = res.baselines[0].target_ids
    _, cache = model.run_with_cache(clean)
    expected = patching.get_act_patch_resid_pre(model, corrupted, cache,
                                                 lambda lg: lg[0, -1, a] - lg[0, -1, b])
    ours = torch.zeros_like(expected)
    for r in res.records:
        ours[r.layer, r.position] = r.readout.target_logits[0] - r.readout.target_logits[1]
    assert torch.allclose(ours, expected, atol=1e-5)


def test_baselines_and_full_patch_endpoints(model, pair):
    """Patching all positions of resid_pre gives exactly the clean readout."""
    res = run_patching_experiment(model, [pair], SweepConfig(layers=[4], per_position=False))
    assert res.records[0].readout == res.baselines[0].clean
    assert res.baselines[0].clean != res.baselines[0].corrupted


def test_deterministic(model, pair):
    cfg = SweepConfig(components=["mlp_out"], layers=[3, 8], positions=[-1])
    assert run_patching_experiment(model, [pair], cfg).records == \
        run_patching_experiment(model, [pair], cfg).records


def test_save_load_roundtrip(model, pair, tmp_path):
    res = run_patching_experiment(model, [pair], SweepConfig(components=["z"], layers=[2], positions=[-1],
                                                             per_head=True))
    loaded = ExperimentResult.load(res.save(tmp_path / "exp" / "result.json"))
    assert loaded.config == res.config
    assert loaded.pairs == res.pairs
    assert loaded.baselines == res.baselines
    assert loaded.records == res.records
    assert loaded.metadata["versions"]["transformer-lens"]


def test_input_validation(model):
    with pytest.raises(ValueError, match="different lengths"):
        run_patching_experiment(model, [PromptPair("The cat sat", "The big dog sat", " on", " in")])
    with pytest.raises(ValueError, match="single tokens"):
        run_patching_experiment(model, [PromptPair("A b", "A c", " zqxjvwk", " x")])
    with pytest.raises(ValueError, match="Unknown component"):
        SweepConfig(components=["foo"])
