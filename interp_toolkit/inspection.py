"""Architecture metadata, hook-point listing, and activation shapes."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import torch

from .components import COMPONENTS


@dataclass(frozen=True)
class ModelInfo:
    name: str
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_mlp: int
    d_vocab: int
    n_ctx: int
    n_params: int  # as reported by TransformerLens (tied weights counted once)
    device: str
    dtype: str
    n_hook_points: int  # distinct HookPoint objects (aliases not double-counted)

    def as_dict(self) -> dict:
        return asdict(self)


def model_info(model) -> ModelInfo:
    cfg = model.cfg
    p = next(model.parameters())
    return ModelInfo(
        name=getattr(cfg, "model_name", "?"), n_layers=cfg.n_layers, n_heads=cfg.n_heads,
        d_model=cfg.d_model, d_head=cfg.d_head, d_mlp=cfg.d_mlp, d_vocab=cfg.d_vocab,
        n_ctx=cfg.n_ctx, n_params=int(model.n_params_total),
        device=str(p.device), dtype=str(p.dtype),
        n_hook_points=len({id(h) for h in model.hook_dict.values()}),
    )


def list_hook_points(model, pattern: str | None = None, layer: int | None = None,
                     canonical_only: bool = False) -> list[str]:
    """Hook names exposed by the model, optionally filtered by regex and/or layer.

    The bridge exposes two names for many sites: a canonical module-path name
    (``blocks.0.hook_in``) and a legacy TransformerLens alias
    (``blocks.0.hook_resid_pre``) for the same HookPoint. ``canonical_only``
    keeps one name per HookPoint (the HookPoint's own ``.name``).
    """
    hook_dict = model.hook_dict
    names = list(hook_dict)
    if canonical_only:
        names = [n for n in names if hook_dict[n].name == n]
    if layer is not None:
        names = [n for n in names if n.startswith(f"blocks.{layer}.")]
    if pattern is not None:
        rx = re.compile(pattern)
        names = [n for n in names if rx.search(n)]
    return names


def list_components() -> dict[str, str]:
    """Named components usable in CaptureSpec / Patch.at / SweepConfig."""
    return {name: c.description for name, c in COMPONENTS.items()}


@torch.no_grad()
def activation_shapes(model, prompt: str | torch.Tensor = "Hello world",
                      names: list[str] | None = None) -> dict[str, tuple[int, ...]]:
    """Shapes of activations for one forward pass, recorded without retaining tensors.

    Uses hooks that store only ``tensor.shape``, so this is cheap even for all hooks.
    """
    tokens = model.to_tokens(prompt) if isinstance(prompt, str) else prompt
    shapes: dict[str, tuple[int, ...]] = {}

    def record(t, hook):
        if isinstance(t, torch.Tensor):
            shapes[hook.name] = tuple(t.shape)

    targets = names if names is not None else list_hook_points(model, canonical_only=True)
    model.run_with_hooks(tokens, fwd_hooks=[(n, record) for n in targets])
    # Report under the names the caller used (hook.name is the canonical name).
    hook_dict = model.hook_dict
    return {n: shapes[hook_dict[n].name] for n in targets if hook_dict[n].name in shapes}
