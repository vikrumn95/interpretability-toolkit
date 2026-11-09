"""Model loading, deterministic setup, and a single entry point for inference.

The toolkit passes TransformerLens' ``TransformerBridge`` around directly rather
than wrapping it: every TL utility keeps working on the same object, and there
is no parallel API to keep in sync.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from transformer_lens.model_bridge import TransformerBridge

from .capture import Activations, CaptureSpec, capture
from .interventions import Patch, patch_hooks

DEFAULT_MODEL = "openai-community/gpt2"


def resolve_device(device: str | None = None) -> str:
    """``None`` -> "cpu" (the default); "auto" -> cuda, then mps, then cpu."""
    if device is None:
        return "cpu"
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def set_determinism(seed: int = 0) -> None:
    """Seed all RNGs and request deterministic kernels.

    GPT-2 inference in eval mode uses no randomness, so seeding mostly matters for
    any sampling a caller does. Deterministic kernels matter on GPU, where some
    reductions are otherwise nondeterministic. ``warn_only=True`` keeps ops that
    lack a deterministic implementation running instead of raising.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Required by cuBLAS for deterministic matmuls; must be set before CUDA init.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def load_model(
    name: str = DEFAULT_MODEL,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
    deterministic: bool = True,
) -> TransformerBridge:
    """Load a Hugging Face model through TransformerBridge, in eval mode.

    Weights are downloaded and cached by Hugging Face on first use (~/.cache/huggingface).
    The bridge keeps the *unprocessed* HF weights (no LayerNorm folding), so logits
    match the HF model. Call ``model.enable_compatibility_mode()`` if you need legacy
    HookedTransformer numerics; it is not needed for activation patching.
    """
    if deterministic:
        set_determinism(seed)
    model = TransformerBridge.boot_transformers(name, device=resolve_device(device), dtype=dtype)
    model.eval()
    return model


def model_device(model: TransformerBridge) -> torch.device:
    return next(model.parameters()).device


def to_tokens(model: TransformerBridge, prompt: str | Sequence[str] | torch.Tensor,
              prepend_bos: bool = True) -> torch.Tensor:
    """Tokenize to a [batch, pos] tensor on the model's device (tensors pass through)."""
    if isinstance(prompt, torch.Tensor):
        tokens = prompt if prompt.dim() == 2 else prompt.unsqueeze(0)
    else:
        tokens = model.to_tokens(prompt, prepend_bos=prepend_bos)
    return tokens.to(model_device(model))


@dataclass
class RunResult:
    tokens: torch.Tensor  # [batch, pos]
    logits: torch.Tensor  # [batch, pos, d_vocab]
    activations: Activations | None = None


@torch.no_grad()
def run(
    model: TransformerBridge,
    prompt: str | Sequence[str] | torch.Tensor,
    *,
    capture_spec: CaptureSpec | None = None,
    patches: Sequence[Patch] = (),
    prepend_bos: bool = True,
) -> RunResult:
    """Run inference, optionally capturing activations and/or applying patches.

    With neither option this is a plain forward pass with no hooks attached.
    When both are given, patch hooks are registered first, so captured values at
    a patched site reflect the patched tensor.
    """
    tokens = to_tokens(model, prompt, prepend_bos)
    with model.hooks(fwd_hooks=patch_hooks(model, patches, tokens.shape[1])):
        if capture_spec is None:
            return RunResult(tokens, model(tokens))
        logits, acts = capture(model, tokens, capture_spec)
        return RunResult(tokens, logits, acts)


def top_predictions(model: TransformerBridge, logits: torch.Tensor, k: int = 5,
                    pos: int = -1, batch: int = 0) -> list[tuple[str, int, float]]:
    """Top-k next-token predictions at one position as (token, id, probability)."""
    probs = logits[batch, pos].float().softmax(-1)
    values, ids = probs.topk(k)
    return [(model.to_string(i), int(i), float(p)) for p, i in zip(values, ids)]
