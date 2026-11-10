# GPT-2 capital-city phrasing robustness study

**Question.** Does GPT-2 small (`openai-community/gpt2`) use similar internal components to retrieve capitals across different phrasings of the same query?

This is an exploratory, educational activation-patching experiment built on the `interp_toolkit` package in this repository. It is **not a claim of a novel circuit**.

**Status.** The pilot and the full run are complete, including the attention-head sweeps. The written analysis is in [`results/findings.md`](results/findings.md) (full run) and [`pilot_results/findings.md`](pilot_results/findings.md) (pilot).

## Results at a glance

- **Screening:** 5 of 14 candidate pairs passed under all three phrasings (France→Italy, Germany→Spain, Japan→China, Russia→Canada, Iran→Iraq).
- **Residual stream:** the country signal stays at the country token through layer 8. It moves to the final position in **block 9** in all 15 pair × phrasing combinations.
- **Attention heads:** the same heads dominate in every phrasing. Recovery is shown as `capital_of` / `capital_city` / `possessive`:
  - **L9H8:** 0.50 / 0.45 / 0.35
  - **L11H2:** 0.14 / 0.27 / 0.26
  - **L10H0:** 0.17 / 0.10 / 0.10
- **Negative heads:** L10H7 and L11H10 push against the clean capital.
- **Effect of phrasing:** it changes how much each head contributes, not which heads are involved.
- **What this supports:** shared routing of country-dependent information to the prediction position. It does not support a capital-specific retrieval circuit.
- **Main caveat:** the capital is never GPT-2's top-1 prediction for these prompts (p ≈ 0.0002–0.006). The metric measures a relative preference between two low-probability tokens.
- **Recommended next experiment:** a relation-swap control, such as official language instead of capital, to test whether these heads are specific to capitals. See `results/findings.md` §4.

The pilot's 3 pairs are the first 3 pairs of the full run, and its records are identical to theirs. It is a subset of the full run, not a replication.

## Design

### Templates

| Name | Template |
|---|---|
| `capital_of` | `The capital of {country} is` |
| `capital_city` | `The capital city of {country} is` |
| `possessive` | `{country}'s capital is` |

In each pair, the first country's prompt is the *clean* input and the second country's prompt is the *corrupted* input. "Corrupted" only names one side of a controlled one-token difference.

### Screening (before any patching)

A candidate pair is kept only if, under **all three** templates:

- **Prompt shape:** the two prompts have equal token length and differ at exactly one position, the country.
- **Targets:** both capitals are single GPT-2 tokens, with a leading space.
- **Preference:**
  - The clean prompt prefers the clean capital, and the corrupted prompt prefers the corrupted capital.
  - The gap between them is at least `--min-gap`, 1.0 logits by default.

Why the 9 rejected pairs failed:
- **Multi-token capital** (Hungary, Vietnam).
- **Wrong baseline preference** (Norway, Netherlands, and Portugal under `capital_city`).
- **Possessive tokenization:** `{country}'s` splits into a different number of tokens than for the other country. Seven pairs failed this check, and for four of them (Austria, Finland, Cuba, Turkey) it was the only failure.

`screening.csv` records every candidate and why it was rejected.

**Selection caveat:** keeping only pairs that work in every template can make phrasing robustness look better than it is.

### Sweeps

- **Residual sweep:** patches one `resid_pre` vector at a time, over 12 layers × all positions. Results are compared across templates at matched *semantic* positions, the country token and the final token, because absolute indices differ.
  - **Control:** the layer-0 country-token patch must reproduce the clean score. This is guaranteed by construction, so it checks the pipeline, not the model.
- **Head sweep (`--heads`):** patches each head's `z` output at the final token, 144 heads in total.

**Score:** `logit(clean capital) − logit(corrupted capital)` at the final token.

**Recovery:** `(patched − corrupted) / (clean − corrupted)`. It is not clipped.

## Running it

From the repository root, with the toolkit installed (`python3 -m venv .venv && .venv/bin/pip install -e '.[dev]'`):

```bash
# Pilot: first 3 candidate pairs
.venv/bin/python phrasing_robustness/run_experiment.py --pilot --heads --out phrasing_robustness/pilot_results
.venv/bin/python phrasing_robustness/analyze_results.py --out phrasing_robustness/pilot_results

# Full run (writes to phrasing_robustness/results/ by default)
.venv/bin/python phrasing_robustness/run_experiment.py --heads
.venv/bin/python phrasing_robustness/analyze_results.py
```

**Options:**
- `--device auto` uses an available accelerator.
- `--max-pairs N` caps the number of passing pairs.

**Run time:** the full run with heads takes a few minutes on CPU.

**Analysis:** `analyze_results.py` reads only the saved JSON, so it doesn't load the model and can be rerun freely.

**Determinism:** inference is deterministic on CPU, so rerunning reproduces the committed results.

Re-running `run_experiment.py` overwrites the output folder. Don't change the candidate pool or thresholds after looking at patch results unless you label the new run as a separate exploratory analysis.

## Output files

Each of `results/` and `pilot_results/` contains:

| File | Contents |
|---|---|
| `findings.md` | Written analysis: residual and head observations, interpretation, limitations, next steps. Written by hand, not generated |
| `report.md` | Auto-generated summary from `analyze_results.py` |
| `manifest.json` | Model, templates, hypothesis, candidate pool, screening rule, selected pairs, toolkit commit |
| `screening.csv` | Every candidate × template: baseline scores, target ranks, top-1 tokens, accept/reject reason |
| `residual_<template>.json` / `.csv` / `.png` | Raw residual-sweep records, scored rows, layer × position heatmap |
| `heads_<template>.json` / `.csv` / `.png` | Same for the head sweep (layer × head heatmap) |
| `all_patch_scores.csv` | All scored patches across templates |
| `baselines.csv` | Clean/corrupted readouts and top-1 tokens per pair and template |
| `sanity_checks.csv` | Layer-0 country-token control |
| `semantic_layer_summary.csv` | Recovery by template × layer × role (country / final token) |
| `compare_country_token.png` / `compare_final_token.png` | Mean recovery vs layer per template (error bars are SE, not CIs) |
| `attention_head_summary.csv` | Per-head mean / median / SD recovery per template |
| `head_comparisons.csv` | Spearman correlation and top-10 overlap between templates |

**Two caveats on `report.md`:**
- **Excluded heads:** its top-head lists and top-10 overlaps rank signed means, so negative heads (L10H7, L11H10) don't appear.
- **Inflated Spearman:** its Spearman values are computed over all 144 heads, many of them near zero.

`findings.md` gives matched comparisons that avoid both issues.

## Limitations

- **Sample:**
  - **Small and hand-picked:** 5 pairs, filtered by baseline behavior.
  - **One patch direction:** each pair is patched only from clean to corrupted.
  - **Single-token capitals only.**
- **Weak target behavior:** the capitals are low-probability tokens, never the model's top prediction.
- **Sufficiency only:** clean-into-corrupted patching shows a component is enough to restore the preference, not that the model needs it.
- **Partial coverage:** heads were tested at the final token only, and MLPs were not decomposed.
- **Descriptive only:** there are no significance tests, and high recovery measures causal effect on a chosen score, not the location of stored factual knowledge.
