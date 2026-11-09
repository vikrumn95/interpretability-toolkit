"""Named activation sites and their tensor layouts.

TransformerLens exposes hundreds of hook points, but patching "position 4" or
"head 7" requires knowing *which tensor dimension* holds positions or heads for
a given hook. That differs across hooks:

    resid_pre / attn_out / mlp_out   [batch, pos, d_model]
    q / k / v / z                    [batch, pos, head, d_head]
    pattern                          [batch, head, query_pos, key_pos]

This module is the single place that layout knowledge lives. Raw TransformerLens
hook names are accepted everywhere; for hooks not listed here, only whole-tensor
operations are supported (no position/head targeting).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Component:
    name: str
    template: str  # hook name with a {layer} placeholder
    pos_dim: int  # dimension indexing token positions (query position for patterns)
    head_dim: int | None  # dimension indexing attention heads, if any
    description: str

    def hook_name(self, layer: int) -> str:
        return self.template.format(layer=layer)


COMPONENTS: dict[str, Component] = {
    c.name: c
    for c in [
        Component("resid_pre", "blocks.{layer}.hook_resid_pre", 1, None,
                  "Residual stream entering the block [batch, pos, d_model]"),
        Component("resid_mid", "blocks.{layer}.hook_resid_mid", 1, None,
                  "Residual stream after attention, before the MLP [batch, pos, d_model]"),
        Component("resid_post", "blocks.{layer}.hook_resid_post", 1, None,
                  "Residual stream leaving the block [batch, pos, d_model]"),
        Component("attn_out", "blocks.{layer}.hook_attn_out", 1, None,
                  "Attention sublayer output, summed over heads [batch, pos, d_model]"),
        Component("mlp_out", "blocks.{layer}.hook_mlp_out", 1, None,
                  "MLP sublayer output [batch, pos, d_model]"),
        Component("mlp_post", "blocks.{layer}.mlp.hook_post", 1, None,
                  "MLP hidden activations after the nonlinearity [batch, pos, d_mlp]"),
        Component("q", "blocks.{layer}.attn.hook_q", 1, 2,
                  "Per-head queries [batch, pos, head, d_head]"),
        Component("k", "blocks.{layer}.attn.hook_k", 1, 2,
                  "Per-head keys [batch, pos, head, d_head]"),
        Component("v", "blocks.{layer}.attn.hook_v", 1, 2,
                  "Per-head values [batch, pos, head, d_head]"),
        Component("z", "blocks.{layer}.attn.hook_z", 1, 2,
                  "Per-head attention output before W_O [batch, pos, head, d_head]"),
        Component("pattern", "blocks.{layer}.attn.hook_pattern", 2, 1,
                  "Attention probabilities [batch, head, query_pos, key_pos]"),
    ]
}

_TEMPLATE_RE = {
    name: re.compile("^" + re.escape(c.template).replace(r"\{layer\}", r"(\d+)") + "$")
    for name, c in COMPONENTS.items()
}


def get_component(name: str) -> Component:
    try:
        return COMPONENTS[name]
    except KeyError:
        raise ValueError(f"Unknown component {name!r}; known: {sorted(COMPONENTS)}") from None


def hook_name(component: str, layer: int) -> str:
    """Hook name for a named component at a layer, e.g. ('z', 3) -> 'blocks.3.attn.hook_z'."""
    return get_component(component).hook_name(layer)


def resolve_component(name: str, model: Any = None) -> tuple[Component, int] | None:
    """Map a hook name back to (Component, layer), or None if its layout is unknown.

    Matches the component templates (TransformerLens' legacy alias names) directly.
    If ``model`` is given, a canonical bridge name such as ``blocks.3.hook_in`` is
    also recognised by checking whether it is the *same HookPoint object* as a
    known component's hook at that layer.
    """
    for cname, rx in _TEMPLATE_RE.items():
        m = rx.match(name)
        if m:
            return COMPONENTS[cname], int(m.group(1))
    if model is not None:
        m = re.match(r"^blocks\.(\d+)\.", name)
        hook_dict = model.hook_dict  # a fresh dict per access in TL 4.x; read it once
        hp = hook_dict.get(name)
        if m and hp is not None:
            layer = int(m.group(1))
            for c in COMPONENTS.values():
                if hook_dict.get(c.hook_name(layer)) is hp:
                    return c, layer
    return None
