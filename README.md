# interp-toolkit

A small toolkit for **activation-patching experiments** on transformer language
models, built on [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
4.x (`TransformerBridge`). It provides activation capture, validated
interventions, a hypothesis-agnostic sweep runner, metrics, plots, and overhead
measurement. It is infrastructure for experiments; it contains no research
findings.

Default model: `openai-community/gpt2` (124M parameters). It runs on CPU, and a
full sweep of one prompt pair over the residual stream (12 layers × 6 positions)
takes about 3 s.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"            # or: pip install -r requirements-lock.txt, then -e .
.venv/bin/python -m pytest                    # 58 tests, ~15 s on CPU (first run downloads GPT-2, ~550 MB)
.venv/bin/python examples/03_patching_sweep.py
```

Weights are downloaded from Hugging Face on first use and cached in
`~/.cache/huggingface`. No API keys are needed. `requirements-lock.txt` records the
exact versions this was verified with (torch 2.15, transformer-lens 4.3.0,
transformers 5.22).

## Concepts

**Residual stream.** GPT-2 is a stack of blocks, and each block *adds* its output to
a running vector per token: `resid_mid = resid_pre + attn_out`,
`resid_post = resid_mid + mlp_out`. That running sum, `[batch, pos, d_model]`, is
the residual stream. Everything a later layer knows about earlier computation
must pass through it. That is why replacing the *whole* residual stream at one
layer replaces everything downstream.

**Hooks.** TransformerLens wraps the model so that every interesting intermediate
value passes through a `HookPoint`, an identity module. A forward hook registered
on a HookPoint is called with the activation; **if it returns a tensor, that
tensor replaces the activation** for the rest of the forward pass. Capture is a
hook that stores `tensor.detach()` and returns nothing. Patching is a hook that
returns a modified tensor. Hooks must be removed afterwards or they silently
affect every later run. This toolkit only registers hooks through TL's
`model.hooks()` context manager, which removes them in a `finally` block.

```
tokens ─▶ embed ─▶ [hook_resid_pre] ─▶ attn ─▶ [hook_attn_out] ─(+)─▶ [hook_resid_mid] ─▶ mlp ─▶ ...
                          ▲                          ▲
                   patch hook returns          capture hook stores
                   clean value here            detached copy here
```

**Activation patching.** Run a *clean* prompt and a *corrupted* prompt that differ
minimally (e.g. one word). Then re-run the corrupted prompt with one activation
replaced by its value from the clean run. If that one replacement moves the
output back toward the clean output, the activation carries information the
difference depends on. Sweeping over layers, positions, and components localizes
where that happens. Positions only line up when both prompts tokenize to the
same length, so the runner requires this.

**Selective capture.** Full capture stores every hook point (about 500 tensors
for GPT-2). A patching sweep only needs the hooks it will patch. Capturing just
those, and optionally just some positions, cuts retained memory by orders of
magnitude (see `examples/04_benchmark.py`).

**Patching recovery.** `recovery = (patched − corrupted) / (clean − corrupted)` for a
scalar score such as a logit difference. 0 means "patch did nothing", 1 means
"fully restores clean behavior", and values outside [0, 1] mean overshoot or a
move in the opposite direction.

## Architecture

```
interp_toolkit/
  model.py          load_model() · run() · top_predictions()         ── entry points
  inspection.py     model_info() · list_hook_points() · activation_shapes()
  components.py     name → hook template + tensor layout (pos / head dims)
  capture.py        CaptureSpec → capture() → Activations             ── wraps run_with_cache
  interventions.py  Patch → validate_patches() → patch_hooks()        ── hook fns for model.hooks()
  experiment.py     PromptPair + SweepConfig → run_patching_experiment() → ExperimentResult (JSON)
  metrics.py        Readout → logit_diff / target_prob / recovery → score() → aggregate() / to_grid()
  plotting.py       rows / grids → matplotlib Figures
  benchmark.py      CaptureSpec configs → timing + memory table
```

Data flows one way: **execution** (model, capture, interventions, experiment)
produces records, **metrics** turn records into rows, and **plotting** draws rows.
Metrics and plotting never touch the model, so saved results can be re-scored
and re-plotted without re-running anything.

**What comes from TransformerLens, and what is custom.**

| Reused from TransformerLens | Written here (and why) |
|---|---|
| Model loading (`boot_transformers`) | `components.py`: which dim is position/head per hook. TL doesn't expose this as data, and per-position/per-head targeting needs it |
| `HookPoint`s, `model.hooks()` (registration and cleanup) | `Patch` and validation: TL's patch setters are internal to its sweep functions, can't be combined, and don't validate |
| `run_with_cache` (`names_filter`, `pos_slice`, device offload) | `CaptureSpec` / `Activations`: declarative selection, early validation, alias-aware memory size |
| `patching.get_act_patch_resid_pre` (used as a **test oracle**) | Sweep runner: structured per-intervention records, several pairs, several components, JSON persistence |

## Public API

```python
from interp_toolkit import *          # plotting: from interp_toolkit import plotting

model = load_model()                  # GPT-2, CPU, eval mode, seeded; device="auto" for GPU/MPS
run(model, "text")                    # RunResult(tokens, logits, activations=None); no hooks attached
run(model, "text", capture_spec=CaptureSpec(...), patches=[Patch(...)])   # both, composable
top_predictions(model, logits, k=5)   # [(token, id, prob), ...]

model_info(model); list_components(); list_hook_points(model, pattern=r"hook_z$", layer=3)
activation_shapes(model, "text", names=[...])        # shapes only, nothing retained

CaptureSpec(components=["resid_pre", "z"], layers=[0, 6], positions=[-1], device="cpu")
CaptureSpec(hook_names=["blocks.3.ln1.hook_scale"])  # any raw TL hook name
CaptureSpec.all()                                    # every hook point
logits, acts = capture(model, tokens, spec)          # acts["blocks.0.hook_resid_pre"], acts.get("z", 6),
                                                     # acts.shapes(), acts.nbytes(), acts.positions

Patch.at("z", 9, source, positions=[-1], heads=[0, 1])   # source = full-shape clean activation
Patch("blocks.6.hook_resid_pre", source)                 # whole tensor
run_with_patches(model, tokens, [p1, p2])                # logits; hooks removed afterwards
active_hook_count(model)                                 # 0 between runs

pairs = [PromptPair(clean, corrupted, clean_target=" Paris", corrupted_target=" Rome")]
cfg = SweepConfig(components=["resid_pre", "z"], layers=None, positions=None,
                  per_position=True, per_head=True, readout_position=-1)
result = run_patching_experiment(model, pairs, cfg)       # .baselines, .records, .metadata
result.save("out.json"); ExperimentResult.load("out.json")

rows = score(result, metric="logit_diff")    # or "clean_target_prob", or any Readout -> float
aggregate(rows, value="recovery", by=("component", "layer", "position"))
grid, row_labels, col_labels = to_grid(rows, "recovery", "layer", "position", component="resid_pre")

plotting.plot_recovery_heatmap(rows, "resid_pre", str_tokens=result.baselines[0].clean_str_tokens)
plotting.plot_prediction_comparison({"clean": r1, "corrupted": r2, "patched": r3})
plotting.plot_metric_by_config(rows, x="layer", value="recovery", position=5)
plotting.save_figure(fig, "outputs/fig.png")

print(format_results(benchmark_capture(model, tokens, {"none": None, "sel": spec, "full": CaptureSpec.all()})))
```

Named components: `resid_pre`, `resid_mid`, `resid_post`, `attn_out`, `mlp_out`,
`mlp_post`, `q`, `k`, `v`, `z` (per-head output before `W_O`), and `pattern`
(position = query position).

## Examples

| Script | Shows |
|---|---|
| `examples/01_inspect_and_capture.py` | model metadata, hook listing, shapes, selective vs full capture |
| `examples/02_interventions.py` | whole-tensor, single-position, and multi-head patches; hook cleanup |
| `examples/03_patching_sweep.py` | sweep → JSON → reload → score → aggregate → figures in `outputs/` |
| `examples/04_benchmark.py` | capture overhead table |

The prompt pairs in the examples exercise the pipeline. They are not a study,
and their outputs are not findings.

## What the tests verify

- Inference is valid, bitwise reproducible, and identical for string and token inputs.
- Captured shapes are correct per component; selective capture retains exactly the requested names; position slicing follows the right dimension (query dimension for patterns); captured values equal TL's own cache.
- Patches modify **only** the intended (position × head) region, checked by capturing the post-patch activation; the source tensor is never mutated.
- **Replacing the full residual stream** (`resid_pre` at layers 0/6/11, or the final `resid_post`) **reproduces the clean logits exactly** (`torch.equal`, not `allclose`).
- Patching every position separately, as N non-overlapping patches, equals patching the whole tensor.
- Overlapping patches, including through alias names, and bad heads, positions, shapes, dtypes, or devices are rejected; hooks are removed even when a hook raises, and later unpatched runs are unchanged.
- The sweep runner matches TransformerLens' independent `get_act_patch_resid_pre` on every layer × position cell, produces the expected record count, is deterministic, and round-trips through JSON.
- Metrics on synthetic inputs, including NaN handling; all plot types render and save; benchmark sizes are exact.

## Limitations and tradeoffs

- **One forward pass per intervention, batch size 1.** Simple and easy to verify,
  but slow for large sweeps: about 27 ms per run on CPU for short prompts. Batching
  interventions (stacking N copies of the corrupted prompt with a different patch
  per batch row) would be the first optimization; it is deliberately not done.
- **Two target tokens per pair, single-token only.** Readouts store two target
  logits plus the logsumexp, which is enough for logit difference, probabilities,
  and recovery, but not for arbitrary full-distribution metrics such as KL to
  the clean run. Recording full logits would make that possible at about 200 KB per run.
- **Patch sources must be full-shape.** A position-sliced capture cannot be used
  directly as a patch source. This avoids ambiguity about how sliced indices map
  to positions.
- **Positions are absolute.** Aggregating by position across pairs is only
  meaningful when pairs share a token template. Pairs must have equal clean and
  corrupted lengths.
- **Recovery is unstable when clean ≈ corrupted.** Below `min_denominator` it is
  NaN and is excluded (and counted) by `aggregate`; just above it, values can be
  large and noisy. Set `min_denominator` on the scale of your metric.
- **Raw HF weights.** `TransformerBridge` defaults to unprocessed weights (no
  LayerNorm folding). This doesn't matter for patching, but activations and
  weights differ from legacy `HookedTransformer` unless you call
  `model.enable_compatibility_mode()`.
- **Hooks that need enabling.** `attn.hook_result` (per-head output after `W_O`)
  only fires after `model.set_use_attn_result(True)`. Use `z` for per-head patching.
- **Memory measurement.** `retained_bytes` is exact. CPU peak memory is not
  measurable per run (`ru_maxrss` is a lifetime high-water mark, and RSS deltas are
  hidden by allocator reuse). CUDA reports a true per-run peak; MPS reports the
  current allocation.
- **TL 4.x detail:** `model.hook_dict` builds a new dict on every access
  (about 0.15 ms). Code here reads it once per call, and the benchmark caught a
  20 ms overhead from doing otherwise.
- **Determinism.** CPU inference is bitwise deterministic here. On GPU,
  `load_model` enables deterministic algorithms (`warn_only=True`), but bitwise
  equality across hardware or library versions is not guaranteed.
