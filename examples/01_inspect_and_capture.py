"""Load GPT-2, inspect its structure, and capture activations (full vs selective)."""

from interp_toolkit import (CaptureSpec, activation_shapes, list_components, list_hook_points,
                            load_model, model_info, run, top_predictions)

model = load_model()  # CPU by default; load_model(device="auto") for GPU/MPS
print(model_info(model))

print("\nNamed components:")
for name, desc in list_components().items():
    print(f"  {name:<11} {desc}")

print("\nHook points in block 0 (canonical names):", len(list_hook_points(model, layer=0, canonical_only=True)))
print(activation_shapes(model, "Hello world", names=["blocks.0.hook_resid_pre", "blocks.0.attn.hook_z",
                                                     "blocks.0.attn.hook_pattern"]))

# Plain inference: no hooks attached.
out = run(model, "The quick brown fox jumps over the")
print("\nTop predictions:", top_predictions(model, out.logits, k=3))

# Selective capture: two components, three layers, last position only.
spec = CaptureSpec(components=["resid_pre", "z"], layers=[0, 6, 11], positions=[-1])
acts = run(model, "The quick brown fox jumps over the", capture_spec=spec).activations
print("\nSelective:", acts.shapes())
print(f"retained {acts.nbytes() / 1e3:.1f} kB")

# Full capture: every hook point.
full = run(model, "The quick brown fox jumps over the", capture_spec=CaptureSpec.all()).activations
print(f"Full: {len(full)} names, retained {full.nbytes() / 1e6:.2f} MB")
