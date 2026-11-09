"""Configurable activation capture on top of ``TransformerBridge.run_with_cache``.

TransformerLens already does the mechanics: it registers a caching hook on each
selected HookPoint, stores ``tensor.detach()`` (optionally moved to another
device and sliced along positions), and removes the hooks when the call returns.
This module adds:

* a declarative ``CaptureSpec`` (components x layers x positions, or raw names),
* validation with clear errors before any forward pass runs,
* an ``Activations`` container keyed by exactly the names you asked for, with
  metadata and a storage-deduplicated size measurement.

Memory notes: capture runs under ``torch.no_grad()`` so no autograd graph is
retained; tensors are detached; ``positions`` slicing happens inside the hook,
so only the selected positions are retained, not the full tensor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from .components import COMPONENTS, get_component


@dataclass(frozen=True)
class CaptureSpec:
    """What to capture.

    Selective capture: ``components`` x ``layers`` (``layers=None`` means all
    layers), plus any raw TransformerLens ``hook_names``.
    Full capture: ``CaptureSpec.all()``, i.e. every hook point the model exposes.
    ``positions`` (``None`` = all) slices the position dimension of every
    captured tensor (query position for attention patterns).
    ``device`` offloads captured tensors, e.g. to "cpu" when running on GPU.
    """

    components: tuple[str, ...] = ()
    layers: tuple[int, ...] | None = None
    positions: tuple[int, ...] | None = None
    hook_names: tuple[str, ...] = ()
    capture_all: bool = False
    device: str | None = None

    @classmethod
    def all(cls, positions: Sequence[int] | None = None, device: str | None = None) -> "CaptureSpec":
        return cls(capture_all=True, positions=_tuple(positions), device=device)

    def __post_init__(self):
        # Accept lists for convenience; store tuples so specs are hashable/immutable.
        for f in ("components", "hook_names"):
            object.__setattr__(self, f, tuple(getattr(self, f)))
        object.__setattr__(self, "layers", _tuple(self.layers))
        object.__setattr__(self, "positions", _tuple(self.positions))
        if not self.capture_all and not self.components and not self.hook_names:
            raise ValueError("Empty CaptureSpec: give components/hook_names, or use CaptureSpec.all()")
        for c in self.components:
            get_component(c)

    def resolve_names(self, model) -> list[str] | None:
        """Concrete hook names to capture (``None`` means every hook point)."""
        if self.capture_all:
            return None
        n_layers = model.cfg.n_layers
        layers = range(n_layers) if self.layers is None else self.layers
        for layer in layers:
            if not -n_layers <= layer < n_layers:
                raise ValueError(f"Layer {layer} out of range for {n_layers}-layer model")
        names = [COMPONENTS[c].hook_name(layer % n_layers) for c in self.components for layer in layers]
        names += list(self.hook_names)
        hook_dict = model.hook_dict  # rebuilt on every access in TL 4.x; read once
        missing = [n for n in names if n not in hook_dict]
        if missing:
            raise ValueError(f"Unknown hook names: {missing}. See inspection.list_hook_points().")
        return list(dict.fromkeys(names))  # dedupe, keep order


@dataclass
class Activations:
    """Captured tensors plus the metadata needed to interpret them."""

    tensors: dict[str, torch.Tensor]
    positions: tuple[int, ...] | None  # absolute positions retained; None = all
    seq_len: int
    spec: CaptureSpec = field(repr=False)

    def __getitem__(self, name: str) -> torch.Tensor:
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"{name!r} was not captured; captured: {len(self.tensors)} names") from None

    def __contains__(self, name: str) -> bool:
        return name in self.tensors

    def __len__(self) -> int:
        return len(self.tensors)

    @property
    def names(self) -> list[str]:
        return list(self.tensors)

    def get(self, component: str, layer: int) -> torch.Tensor:
        return self[get_component(component).hook_name(layer)]

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {k: tuple(v.shape) for k, v in self.tensors.items()}

    def nbytes(self) -> int:
        """Bytes actually retained. Alias names share storage, so count each storage once."""
        seen: dict[tuple[str, int], int] = {}
        for t in self.tensors.values():
            s = t.untyped_storage()
            seen[(str(t.device), s.data_ptr())] = s.nbytes()
        return sum(seen.values())

    def to(self, device: str | torch.device) -> "Activations":
        return Activations({k: v.to(device) for k, v in self.tensors.items()},
                           self.positions, self.seq_len, self.spec)


def normalize_positions(positions: Sequence[int] | None, seq_len: int) -> tuple[int, ...] | None:
    """Validate positions against ``seq_len`` and convert negatives to absolute indices."""
    if positions is None:
        return None
    out = []
    for p in positions:
        if not -seq_len <= p < seq_len:
            raise ValueError(f"Position {p} out of range for sequence length {seq_len}")
        out.append(p % seq_len)
    return tuple(out)


@torch.no_grad()
def capture(model, tokens: torch.Tensor, spec: CaptureSpec) -> tuple[torch.Tensor, Activations]:
    """Run one forward pass on ``tokens`` and return ``(logits, Activations)``."""
    names = spec.resolve_names(model)
    seq_len = tokens.shape[1]
    positions = normalize_positions(spec.positions, seq_len)
    logits, cache = model.run_with_cache(
        tokens,
        names_filter=names,
        pos_slice=list(positions) if positions is not None else None,
        device=spec.device,
        return_cache_object=False,
    )
    # run_with_cache may also store the canonical name next to an alias you asked for
    # (same tensor). Keep exactly the requested names for a predictable API.
    tensors = cache if names is None else {n: cache[n] for n in names}
    return logits, Activations(dict(tensors), positions, seq_len, spec)


def _tuple(x):
    return None if x is None else tuple(x)
