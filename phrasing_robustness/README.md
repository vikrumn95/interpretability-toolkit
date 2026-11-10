# GPT-2 capital-city phrasing robustness study

## Question

For a pretrained GPT-2 Small model, are the **internal activation locations that influence country-to-capital predictions** similar across different ways of phrasing the same query?

This is an exploratory, educational mechanistic-interpretability experiment, **not a claim of discovering a novel circuit**.

### Predeclared hypothesis

Some residual-stream layers and attention heads will show similar causal influence across wording changes, even though the country token moves to a different absolute position.

### Experimental conditions

We use three completion templates:

1. `The capital of {country} is`
2. `The capital city of {country} is`
3. `{country}'s capital is`

For each candidate country pair, the first country's prompt is the *clean* input and the second country's prompt is the *corrupted* input. Neither input is objectively corrupted; these terms describe a controlled difference.

**Screening takes place before the patching sweeps.** We require:

- Both prompts tokenize to equal length and differ at exactly one token position (the country), so patching at that position is interpretable.
- Both capital targets are individual GPT-2 next-tokens; the target strings include leading spaces.
- The clean prompt favors the clean capital over the corrupted capital, and vice versa for the corrupted prompt, with a clean-to-corrupted logit-difference gap of at least `1.0` by default.
- The pair passes these checks under **all three** phrasings. This keeps the country-pair set identical across template comparisons.

**Important selection caveat:** Conditioning on working prompts in all three templates can inflate apparent phrasing robustness. The `screening.csv` file preserves every rejected candidate and its reason. The baseline preference is a *pairwise logit comparison*, not a guarantee that the correct capital is GPT-2's top-1 prediction; see recorded target ranks and top-1 predictions.

### Experiments

**Phase A — Residual-stream sweep:** For every valid pair and phrasing, patch one `resid_pre` activation vector at a time across all 12 GPT-2 layers and all prompt positions. Compare normalized recovery for two semantic positions across wording changes: the *country token* and the *final token*. A layer-0 country-token patch is a control: because the country token is the only input-token difference, this should reproduce the clean logit difference.

**Phase B — Attention-head sweep (optional):** Patch each head's `z` output at the final token, across all layers and heads. Compare the 144 head-specific effects across phrasings with Spearman rank correlation and top-10 overlap. These measures describe similarity, not proof that an attention head implements a capital-city circuit.

**Score:** `logit(clean_capital) - logit(corrupted_capital)`. Normalized recovery is `(patch_score - corrupted_score) / (clean_score - corrupted_score)`. Scores outside `[0, 1]` are possible; they are **not clipped** in saved data.

### Run it

From the root of your existing `interpretability-toolkit` checkout:

```bash
# Install the toolkit/environment if necessary:
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'

# Copy this entire directory to experiments/phrasing_robustness/ in the checkout.

# Small pilot, first 3 candidate pairs:
.venv/bin/python experiments/phrasing_robustness/run_experiment.py --pilot \
  --out experiments/phrasing_robustness/pilot_results
.venv/bin/python experiments/phrasing_robustness/analyze_results.py \
  --out experiments/phrasing_robustness/pilot_results

# Full residual-stream study:
.venv/bin/python experiments/phrasing_robustness/run_experiment.py \
  --out experiments/phrasing_robustness/results
.venv/bin/python experiments/phrasing_robustness/analyze_results.py \
  --out experiments/phrasing_robustness/results

# Optional: rerun with attention-head sweeps as well (overwrites the same result set):
.venv/bin/python experiments/phrasing_robustness/run_experiment.py --heads \
  --out experiments/phrasing_robustness/results
.venv/bin/python experiments/phrasing_robustness/analyze_results.py \
  --out experiments/phrasing_robustness/results
```

Or use `--device auto` to select an available accelerator. The scripts use the current repository's `PromptPair`, `SweepConfig`, `run_patching_experiment`, `score`, `ExperimentResult`, and plotting API; they do not change the toolkit itself.

**The scripts have been syntax-checked but were not executed against the actual GPT-2 model in this environment.** The first pilot is intended to validate that the live dependencies and prompts behave as expected.

### Files recorded

| File | Description |
|---|---|
| `manifest.json` | Model, templates, predeclared hypothesis, candidate pool, exact inclusion rules, selected pairs, repository commit |
| `screening.csv` | Every attempted prompt pair and template, baseline logit differences, target ranks, top-1 outputs, exclusion reasons |
| `residual_<template>.json` | Original model readouts and intervention configurations for the complete residual sweep |
| `residual_<template>.csv` | Each residual patch's scores and recovery |
| `residual_<template>.png` | Layer-by-position recovery heatmap for one wording |
| `heads_<template>.json/.csv/.png` | Same for the optional individual-head sweeps |
| `all_patch_scores.csv` | Flat combined results across wordings and interventions |
| `baselines.csv` | Original clean/corrupted readouts for every included prompt pair |
| `sanity_checks.csv` | Whether layer-0 country-token patches reproduce clean scores |
| `semantic_layer_summary.csv` | Mean, median, and standard deviation by template, layer, and semantic token role |
| `compare_country_token.png` / `compare_final_token.png` | Cross-template mean recovery versus layer; error bars are SE, **not confidence intervals** |
| `attention_head_summary.csv` | Aggregate recovery for each of GPT-2's 144 heads, if enabled |
| `head_comparisons.csv` | Head rank correlation and top-10 overlap between templates, if enabled |
| `report.md` | Automatically generated initial summary with placeholders for your interpretation |

Do not alter the input candidate set or thresholds after examining patch results without clearly labeling a new exploratory analysis. Keep `screening.csv` and all raw JSON outputs. Do not claim that high recovery identifies a factual knowledge representation; the experiment measures **causal effect on a chosen output score**.

### How to interpret the results

Look at: (1) whether the layer-0 country-token control passes; (2) whether effects move from earlier country-token representations to later final-token representations; (3) which head rankings are consistent across the three phrasings; (4) how variable the effects are across country pairs. Compare template differences only at matched **semantic roles**, not necessarily at the same absolute token index. If most pairs fail screening, say so; don't quietly replace them with easier examples.

### Limitations / next checks

- Sample is small, hand-picked, and conditionally filtered; results do not establish population-level generality.
- Only one direction (country A to country B) is patched per pair; reversing direction is a useful follow-up.
- Head scans use only the final token's `z` value, not every possible head-position combination.
- The selected targets are single-token capitals. This excludes many real-world capitals.
- A high normalized recovery can reflect broad information transplantation rather than an interpretable specialized mechanism.
- Layer/head sweep scores are observational comparisons among interventions; further ablations or counterfactual tests would be needed for stronger circuit claims.
