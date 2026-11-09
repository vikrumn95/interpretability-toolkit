import torch

from interp_toolkit import (activation_shapes, list_hook_points, model_info, run,
                            top_predictions)


def test_forward_pass_valid(model, tokens):
    clean, _ = tokens
    out = run(model, clean)
    assert out.logits.shape == (1, clean.shape[1], model.cfg.d_vocab)
    assert torch.isfinite(out.logits).all()
    assert out.activations is None


def test_inference_reproducible(model, tokens):
    clean, _ = tokens
    assert torch.equal(run(model, clean).logits, run(model, clean).logits)


def test_string_and_tensor_inputs_agree(model, tokens):
    clean, _ = tokens
    assert torch.equal(run(model, "The capital of France is").logits, run(model, clean).logits)


def test_top_predictions(model, tokens):
    preds = top_predictions(model, run(model, tokens[0]).logits, k=5)
    probs = [p for _, _, p in preds]
    assert len(preds) == 5 and probs == sorted(probs, reverse=True) and sum(probs) <= 1.0


def test_model_info(model):
    info = model_info(model)
    assert (info.n_layers, info.n_heads, info.d_model, info.d_head) == (12, 12, 768, 64)
    assert info.n_params == 124_439_808
    assert info.device == "cpu"


def test_list_hook_points(model):
    all_names = list_hook_points(model)
    canonical = list_hook_points(model, canonical_only=True)
    assert len(canonical) < len(all_names)  # aliases collapse
    assert "blocks.0.hook_resid_pre" in all_names
    z = list_hook_points(model, pattern=r"attn\.hook_z$")
    assert len(z) == model.cfg.n_layers
    assert all(n.startswith("blocks.3.") for n in list_hook_points(model, layer=3))


def test_activation_shapes(model):
    shapes = activation_shapes(model, "Hi there", names=[
        "blocks.0.hook_resid_pre", "blocks.0.attn.hook_z", "blocks.0.attn.hook_pattern"])
    assert shapes == {"blocks.0.hook_resid_pre": (1, 3, 768),
                      "blocks.0.attn.hook_z": (1, 3, 12, 64),
                      "blocks.0.attn.hook_pattern": (1, 12, 3, 3)}
