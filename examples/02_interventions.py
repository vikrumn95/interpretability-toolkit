"""Patch activations from one run into another, at the level of layers, positions, and heads."""

import torch

from interp_toolkit import CaptureSpec, Patch, active_hook_count, capture, load_model, run, to_tokens
from interp_toolkit.metrics import logit_diff_from_logits

model = load_model()
source = to_tokens(model, "The capital of France is")
target = to_tokens(model, "The capital of Italy is")
paris, rome = model.to_single_token(" Paris"), model.to_single_token(" Rome")

_, src_acts = capture(model, source, CaptureSpec(components=["resid_pre", "z"]))
base = run(model, target).logits


def ld(logits):
    return logit_diff_from_logits(logits, paris, rome).item()


print(f"source run logit diff (Paris - Rome): {ld(run(model, source).logits):+.3f}")
print(f"target run logit diff:                {ld(base):+.3f}")

# 1. Whole residual stream at layer 6: everything downstream now sees source values,
#    so the output equals the source run exactly.
p = Patch.at("resid_pre", 6, src_acts.get("resid_pre", 6))
print("full resid_pre[6] patch == source run:", torch.equal(run(model, target, patches=[p]).logits,
                                                           run(model, source).logits))

# 2. One position of the residual stream (position 4 = the differing token).
p = Patch.at("resid_pre", 6, src_acts.get("resid_pre", 6), positions=[4])
print(f"resid_pre[6] @ pos 4:                 {ld(run(model, target, patches=[p]).logits):+.3f}")

# 3. Two heads at the last position, in two layers, applied together.
patches = [Patch.at("z", 9, src_acts.get("z", 9), positions=[-1], heads=[0, 1]),
           Patch.at("z", 10, src_acts.get("z", 10), positions=[-1], heads=[0])]
print(f"z heads (9.0, 9.1, 10.0) @ last pos:  {ld(run(model, target, patches=patches).logits):+.3f}")

# Hooks are gone and ordinary inference is unchanged afterwards.
print("hooks attached:", active_hook_count(model), "| unpatched output unchanged:",
      torch.equal(run(model, target).logits, base))
