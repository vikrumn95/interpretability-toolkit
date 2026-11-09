"""Measure the overhead of full vs selective activation capture."""

from interp_toolkit import CaptureSpec, benchmark_capture, format_results, load_model, to_tokens

model = load_model()
tokens = to_tokens(model, "The quick brown fox jumps over the lazy dog. " * 4)
results = benchmark_capture(model, tokens, {
    "no capture": None,
    "resid_pre, last pos": CaptureSpec(components=["resid_pre"], positions=[-1]),
    "resid_pre": CaptureSpec(components=["resid_pre"]),
    "all components": CaptureSpec(components=["resid_pre", "resid_mid", "resid_post", "attn_out",
                                              "mlp_out", "mlp_post", "q", "k", "v", "z", "pattern"]),
    "full (every hook)": CaptureSpec.all(),
}, repeats=10)
print(f"sequence length {tokens.shape[1]}\n")
print(format_results(results))
