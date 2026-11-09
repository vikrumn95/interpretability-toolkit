"""Activation patching: replace selected activations with values from another run.

How it hooks into the forward pass
----------------------------------
Every HookPoint is an identity ``nn.Module`` sitting on the data path. When the
model reaches it, PyTorch calls its forward hooks in registration order; if a
hook *returns* a tensor, that tensor replaces the activation and is what the
rest of the network sees. A patch hook therefore receives the corrupted run's
activation and returns a new tensor equal to it everywhere except the selected
(positions x heads) region, which is copied from ``source``.

Design choices
--------------
* ``source`` must have the *full* shape of the target activation (as captured
  without position slicing). The mask then selects the same region in both, so
  there is no ambiguity about how a sliced source maps onto positions.
* The hook builds a new tensor (``torch.where``) instead of writing in place.
  The incoming tensor may be shared with other consumers (e.g. the residual
  stream feeds both the next sublayer and the skip connection), and the source
  must never be mutated, since it is reused across many patched runs.
* Hooks are registered through ``model.hooks(...)``, TransformerLens' context
  manager, which removes them in a ``finally`` block even if the run raises.
  Nothing persists on the model after the call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch

from .components import Component, get_component, resolve_component


@dataclass(frozen=True, eq=False)
class Patch:
    """Replace ``hook_name``'s activation with ``source`` at the selected region.

    ``positions=None`` / ``heads=None`` mean "all". Head selection requires a
    per-head component (q, k, v, z, pattern).
    """

    hook_name: str
    source: torch.Tensor
    positions: tuple[int, ...] | None = None
    heads: tuple[int, ...] | None = None

    def __post_init__(self):
        for f in ("positions", "heads"):
            v = getattr(self, f)
            if v is not None:
                object.__setattr__(self, f, tuple(int(i) for i in v))

    @classmethod
    def at(cls, component: str, layer: int, source: torch.Tensor,
           positions: Sequence[int] | None = None, heads: Sequence[int] | None = None) -> "Patch":
        return cls(get_component(component).hook_name(layer), source, positions, heads)

    def to(self, device) -> "Patch":
        return Patch(self.hook_name, self.source.to(device), self.positions, self.heads)


@dataclass(frozen=True)
class _ResolvedPatch:
    patch: Patch
    pos_dim: int | None
    head_dim: int | None
    positions: tuple[int, ...] | None  # absolute
    heads: tuple[int, ...] | None


def validate_patches(model, patches: Sequence[Patch], seq_len: int) -> list[_ResolvedPatch]:
    """Check patches against the model and input length before any forward pass.

    Checks: hook exists; position/head targeting is supported for the hook's
    layout; indices are in range; no two patches write overlapping regions of
    the same HookPoint (the result would depend on registration order).
    Shape, dtype, and device of ``source`` are checked against the live activation
    inside the hook, which is the only place the true shape is known.
    """
    resolved: list[_ResolvedPatch] = []
    hook_dict = model.hook_dict  # TL 4.x rebuilds this dict on every access; read it once
    for p in patches:
        hp = hook_dict.get(p.hook_name)
        if hp is None:
            raise ValueError(f"Unknown hook {p.hook_name!r}")
        comp: Component | None = None
        if p.positions is not None or p.heads is not None:
            found = resolve_component(p.hook_name, model)
            if found is None:
                raise ValueError(f"{p.hook_name!r} has no known layout; only whole-tensor patches "
                                 "(positions=None, heads=None) are supported for it")
            comp = found[0]
        positions = None
        if p.positions is not None:
            positions = tuple(_check_index(i, seq_len, "position") for i in p.positions)
        heads = None
        if p.heads is not None:
            if comp.head_dim is None:
                raise ValueError(f"{p.hook_name!r} ({comp.name}) has no head dimension")
            heads = tuple(_check_index(h, model.cfg.n_heads, "head") for h in p.heads)
        r = _ResolvedPatch(p, comp.pos_dim if comp else None, comp.head_dim if comp else None,
                           positions, heads)
        for other in resolved:
            if hook_dict[other.patch.hook_name] is hp and _overlaps(r, other):
                raise ValueError(f"Patches on {p.hook_name!r} and {other.patch.hook_name!r} "
                                 "write overlapping regions of the same activation")
        resolved.append(r)
    return resolved


def _check_index(i: int, size: int, what: str) -> int:
    if not -size <= i < size:
        raise ValueError(f"{what} {i} out of range (size {size})")
    return i % size


def _overlaps(a: _ResolvedPatch, b: _ResolvedPatch) -> bool:
    def inter(x, y):
        return x is None or y is None or bool(set(x) & set(y))
    return inter(a.positions, b.positions) and inter(a.heads, b.heads)


def _mask(r: _ResolvedPatch, act: torch.Tensor) -> torch.Tensor | None:
    """Boolean mask broadcastable to ``act`` selecting the patched region (None = all)."""
    if r.positions is None and r.heads is None:
        return None
    mask = torch.ones([1] * act.dim(), dtype=torch.bool, device=act.device)
    for dim, idx in ((r.pos_dim, r.positions), (r.head_dim, r.heads)):
        if idx is None:
            continue
        sel = torch.zeros(act.shape[dim], dtype=torch.bool, device=act.device)
        sel[list(idx)] = True
        shape = [1] * act.dim()
        shape[dim] = act.shape[dim]
        mask = mask & sel.view(shape)
    return mask


def _make_hook(r: _ResolvedPatch) -> Callable:
    src = r.patch.source

    def hook(act: torch.Tensor, hook) -> torch.Tensor:
        if act.shape != src.shape:
            raise ValueError(f"Patch source shape {tuple(src.shape)} != activation shape "
                             f"{tuple(act.shape)} at {hook.name!r}")
        if act.device != src.device:
            raise ValueError(f"Patch source on {src.device}, activation on {act.device} at "
                             f"{hook.name!r}; call patch.to(device)")
        if act.dtype != src.dtype:
            raise ValueError(f"Patch source dtype {src.dtype} != activation dtype {act.dtype}")
        mask = _mask(r, act)
        return src.clone() if mask is None else torch.where(mask, src, act)

    return hook


def patch_hooks(model, patches: Sequence[Patch], seq_len: int) -> list[tuple[str, Callable]]:
    """Validate patches and return ``(hook_name, fn)`` pairs for ``model.hooks()`` / ``run_with_hooks``."""
    return [(r.patch.hook_name, _make_hook(r)) for r in validate_patches(model, patches, seq_len)]


@torch.no_grad()
def run_with_patches(model, tokens: torch.Tensor, patches: Sequence[Patch]) -> torch.Tensor:
    """Forward pass on ``tokens`` with ``patches`` applied; returns logits."""
    return model.run_with_hooks(tokens, fwd_hooks=patch_hooks(model, patches, tokens.shape[1]))


def active_hook_count(model) -> int:
    """Hook functions currently attached to HookPoints (should be 0 between runs).

    Counts PyTorch's own hook registries, i.e. what would actually execute,
    rather than TransformerLens' bookkeeping lists.
    """
    seen = {id(hp): hp for hp in model.hook_dict.values()}
    return sum(len(hp._forward_hooks) + len(hp._forward_pre_hooks) + len(hp._backward_hooks)
               for hp in seen.values())
