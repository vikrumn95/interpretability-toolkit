import pytest
import torch

from interp_toolkit import (CaptureSpec, Patch, active_hook_count, capture, hook_name, run,
                            run_with_patches)


@pytest.fixture(scope="module")
def clean_acts(model, tokens):
    _, acts = capture(model, tokens[0], CaptureSpec(
        components=["resid_pre", "resid_post", "z", "pattern", "attn_out"]))
    return acts


@pytest.fixture(scope="module")
def baselines(model, tokens):
    clean, corrupted = tokens
    return run(model, clean).logits, run(model, corrupted).logits


def patched_activation(model, corrupted, patches, name):
    """Run with patches and capture the (post-patch) activation at ``name``."""
    out = run(model, corrupted, patches=patches, capture_spec=CaptureSpec(hook_names=[name]))
    return out.activations[name], out.logits


@pytest.mark.parametrize("layer", [0, 6, 11])
def test_full_residual_patch_reproduces_clean_run(model, tokens, clean_acts, baselines, layer):
    """Replacing the entire residual stream at a layer boundary makes everything downstream
    a deterministic function of clean values, so logits must equal the clean run exactly."""
    _, corrupted = tokens
    clean_logits, corrupted_logits = baselines
    assert not torch.equal(clean_logits, corrupted_logits)
    out = run_with_patches(model, corrupted, [Patch.at("resid_pre", layer, clean_acts.get("resid_pre", layer))])
    assert torch.equal(out, clean_logits)


def test_final_resid_post_patch_reproduces_clean_run(model, tokens, clean_acts, baselines):
    out = run_with_patches(model, tokens[1], [Patch.at("resid_post", 11, clean_acts.get("resid_post", 11))])
    assert torch.equal(out, baselines[0])


def test_position_patch_modifies_only_selected_positions(model, tokens, clean_acts):
    clean, corrupted = tokens
    name = hook_name("attn_out", 3)
    src = clean_acts[name]
    _, corr_acts = capture(model, corrupted, CaptureSpec(hook_names=[name]))
    act, _ = patched_activation(model, corrupted, [Patch(name, src, positions=[2])], name)
    assert torch.equal(act[:, 2], src[:, 2])
    others = [p for p in range(clean.shape[1]) if p != 2]
    assert torch.equal(act[:, others], corr_acts[name][:, others])


def test_head_and_position_patch(model, tokens, clean_acts):
    _, corrupted = tokens
    name = hook_name("z", 4)
    src = clean_acts[name]
    _, corr = capture(model, corrupted, CaptureSpec(hook_names=[name]))
    act, _ = patched_activation(model, corrupted, [Patch(name, src, positions=[3], heads=[5, 7])], name)
    expected = corr[name].clone()
    expected[:, 3, [5, 7]] = src[:, 3, [5, 7]]
    assert torch.equal(act, expected)


def test_pattern_patch_uses_head_and_query_dims(model, tokens, clean_acts):
    _, corrupted = tokens
    name = hook_name("pattern", 2)
    src = clean_acts[name]
    _, corr = capture(model, corrupted, CaptureSpec(hook_names=[name]))
    act, _ = patched_activation(model, corrupted, [Patch(name, src, positions=[-1], heads=[0])], name)
    expected = corr[name].clone()
    expected[:, 0, -1, :] = src[:, 0, -1, :]
    assert torch.equal(act, expected)


def test_patch_changes_output_and_does_not_mutate_source(model, tokens, clean_acts, baselines):
    src = clean_acts.get("z", 9)
    before = src.clone()
    out = run_with_patches(model, tokens[1], [Patch.at("z", 9, src, heads=[0, 1, 2])])
    assert not torch.equal(out, baselines[1])
    assert torch.equal(src, before)


def test_canonical_hook_name_resolves_layout(model, tokens, clean_acts):
    # blocks.4.attn.o.hook_in is the same HookPoint as blocks.4.attn.hook_z
    src = clean_acts.get("z", 4)
    a = run_with_patches(model, tokens[1], [Patch("blocks.4.attn.o.hook_in", src, heads=[3])])
    b = run_with_patches(model, tokens[1], [Patch.at("z", 4, src, heads=[3])])
    assert torch.equal(a, b)


def test_multiple_compatible_patches(model, tokens, clean_acts, baselines):
    """Non-overlapping patches compose; patching every position piecewise == patching all."""
    _, corrupted = tokens
    T = corrupted.shape[1]
    src = clean_acts.get("resid_pre", 5)
    pieces = [Patch.at("resid_pre", 5, src, positions=[p]) for p in range(T)]
    assert torch.equal(run_with_patches(model, corrupted, pieces), baselines[0])
    mixed = [Patch.at("z", 2, clean_acts.get("z", 2), heads=[1]),
             Patch.at("z", 2, clean_acts.get("z", 2), heads=[2]),
             Patch.at("attn_out", 7, clean_acts.get("attn_out", 7), positions=[1])]
    assert torch.isfinite(run_with_patches(model, corrupted, mixed)).all()


def test_overlapping_patches_rejected(model, tokens, clean_acts):
    src = clean_acts.get("z", 1)
    with pytest.raises(ValueError, match="overlapping"):
        run_with_patches(model, tokens[1], [Patch.at("z", 1, src, heads=[0]),
                                            Patch.at("z", 1, src, positions=[2])])
    # Overlap is detected through aliases too (same HookPoint, different name).
    with pytest.raises(ValueError, match="overlapping"):
        run_with_patches(model, tokens[1], [Patch.at("z", 1, src),
                                            Patch("blocks.1.attn.o.hook_in", src)])


@pytest.mark.parametrize("make, match", [
    (lambda a: Patch("blocks.0.nope", a.get("resid_pre", 0)), "Unknown hook"),
    (lambda a: Patch.at("resid_pre", 0, a.get("resid_pre", 0), heads=[0]), "no head dimension"),
    (lambda a: Patch.at("z", 0, a.get("z", 0), heads=[12]), "head 12 out of range"),
    (lambda a: Patch.at("z", 0, a.get("z", 0), positions=[50]), "position 50 out of range"),
    (lambda a: Patch("blocks.0.ln1.hook_scale", torch.zeros(1), positions=[0]), "no known layout"),
])
def test_config_validation(model, tokens, clean_acts, make, match):
    with pytest.raises(ValueError, match=match):
        run_with_patches(model, tokens[1], [make(clean_acts)])
    assert active_hook_count(model) == 0


@pytest.mark.parametrize("src_fn, match", [
    (lambda s: s[:, :2], "shape"),
    (lambda s: s.to(torch.float64), "dtype"),
    (lambda s: torch.empty(s.shape, device="meta"), "call patch.to"),
])
def test_runtime_validation_and_cleanup(model, tokens, clean_acts, baselines, src_fn, match):
    """Errors raised inside a hook propagate, hooks are still removed, later runs are clean."""
    bad = Patch.at("resid_pre", 3, src_fn(clean_acts.get("resid_pre", 3)))
    with pytest.raises(ValueError, match=match):
        run_with_patches(model, tokens[1], [bad])
    assert active_hook_count(model) == 0
    assert torch.equal(run(model, tokens[1]).logits, baselines[1])


def test_hooks_removed_and_later_runs_unaffected(model, tokens, clean_acts, baselines):
    _, corrupted = tokens
    for layer in range(0, 12, 3):
        run_with_patches(model, corrupted, [Patch.at("resid_pre", layer, clean_acts.get("resid_pre", layer))])
        run(model, corrupted, patches=[Patch.at("z", layer, clean_acts.get("z", layer), heads=[0])],
            capture_spec=CaptureSpec(components=["resid_post"]))
    assert active_hook_count(model) == 0
    assert torch.equal(run(model, corrupted).logits, baselines[1])
