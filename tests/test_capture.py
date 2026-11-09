import pytest
import torch

from interp_toolkit import CaptureSpec, active_hook_count, capture, hook_name, run


def test_component_shapes(model, tokens):
    clean, _ = tokens
    T = clean.shape[1]
    _, acts = capture(model, clean, CaptureSpec(
        components=["resid_pre", "attn_out", "mlp_out", "mlp_post", "z", "pattern"], layers=[0, 5]))
    assert acts.get("resid_pre", 5).shape == (1, T, 768)
    assert acts.get("attn_out", 0).shape == (1, T, 768)
    assert acts.get("mlp_out", 0).shape == (1, T, 768)
    assert acts.get("mlp_post", 0).shape == (1, T, 3072)
    assert acts.get("z", 5).shape == (1, T, 12, 64)
    assert acts.get("pattern", 5).shape == (1, 12, T, T)


def test_selective_capture_retains_only_requested(model, tokens):
    spec = CaptureSpec(components=["resid_pre"], layers=[2, 7], hook_names=["blocks.3.attn.hook_q"])
    _, acts = capture(model, tokens[0], spec)
    assert acts.names == [hook_name("resid_pre", 2), hook_name("resid_pre", 7), "blocks.3.attn.hook_q"]


def test_position_selection(model, tokens):
    clean, _ = tokens
    T = clean.shape[1]
    _, full = capture(model, clean, CaptureSpec(components=["resid_pre", "pattern"], layers=[4]))
    _, sel = capture(model, clean, CaptureSpec(components=["resid_pre", "pattern"], layers=[4],
                                               positions=[1, -1]))
    assert sel.positions == (1, T - 1)
    assert torch.equal(sel.get("resid_pre", 4), full.get("resid_pre", 4)[:, [1, T - 1]])
    # Patterns are sliced on the query dimension; all key positions remain.
    assert torch.equal(sel.get("pattern", 4), full.get("pattern", 4)[:, :, [1, T - 1]])


def test_capture_matches_transformerlens_cache(model, tokens):
    clean, _ = tokens
    _, cache = model.run_with_cache(clean)
    _, acts = capture(model, clean, CaptureSpec(components=["z"], layers=[3]))
    assert torch.equal(acts.get("z", 3), cache["blocks.3.attn.hook_z"])


def test_full_capture_and_alias_dedup(model, tokens):
    _, acts = capture(model, tokens[0], CaptureSpec.all())
    # Every standard component at every layer fires and is captured. (Some hook points,
    # e.g. attn.hook_result, only fire when enabled via model.set_use_attn_result.)
    from interp_toolkit import COMPONENTS
    for c in COMPONENTS.values():
        for layer in range(model.cfg.n_layers):
            assert c.hook_name(layer) in acts
    naive = sum(t.numel() * t.element_size() for t in acts.tensors.values())
    assert 0 < acts.nbytes() < naive  # aliases share storage and are counted once


def test_captured_tensors_detached_and_offloaded(model, tokens):
    _, acts = capture(model, tokens[0], CaptureSpec(components=["resid_pre"], layers=[0], device="cpu"))
    t = acts.get("resid_pre", 0)
    assert not t.requires_grad and t.device.type == "cpu"


def test_hooks_removed_after_capture(model, tokens):
    capture(model, tokens[0], CaptureSpec.all())
    assert active_hook_count(model) == 0


def test_capture_does_not_change_logits(model, tokens):
    clean, _ = tokens
    logits, _ = capture(model, clean, CaptureSpec.all())
    assert torch.equal(logits, run(model, clean).logits)


@pytest.mark.parametrize("kwargs, match", [
    ({}, "Empty CaptureSpec"),
    ({"components": ["nope"]}, "Unknown component"),
])
def test_spec_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        CaptureSpec(**kwargs)


def test_resolve_validation(model, tokens):
    with pytest.raises(ValueError, match="Layer 12 out of range"):
        capture(model, tokens[0], CaptureSpec(components=["resid_pre"], layers=[12]))
    with pytest.raises(ValueError, match="Unknown hook names"):
        capture(model, tokens[0], CaptureSpec(hook_names=["blocks.0.not_a_hook"]))
    with pytest.raises(ValueError, match="Position 99"):
        capture(model, tokens[0], CaptureSpec(components=["resid_pre"], positions=[99]))
