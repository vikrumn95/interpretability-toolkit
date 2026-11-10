# Capital-city phrasing robustness — results

## Design
- Model: openai-community/gpt2
- Prompt templates: capital_of, capital_city, possessive
- Passing pairs (same across all templates): 5 / 14
- Baseline threshold: opposite preferences, gap >= 1.0 logits
- Repository commit: a470cd8b41d024ee1cd21e77a3725f8a9019d4b7
- Hypothesis (recorded before patches): Some influential layer/head locations for capital-city recall will be shared across phrasings when the same country pairs are used; similarity metrics are descriptive and not statistical significance tests.

## Data quality / controls
- Layer-0 country-token restoration: 15 / 15 passed (abs logit-difference error < 0.001).
- See screening.csv for **all** candidates and exclusion reasons.
- See baselines.csv for the actual top predicted tokens (pairwise preference is not top-1 correctness).

## Residual-stream results (descriptive)
Each mean is across the same passing country pairs. Recovery is not a fraction of a circuit explained.

### capital_of
- country_token: highest mean recoveries: L0 (1.00), L1 (1.00), L2 (0.99)
- final_token: highest mean recoveries: L11 (0.84), L10 (0.67), L9 (0.15)

### capital_city
- country_token: highest mean recoveries: L0 (1.00), L1 (1.00), L2 (0.99)
- final_token: highest mean recoveries: L11 (0.73), L10 (0.61), L9 (0.16)

### possessive
- country_token: highest mean recoveries: L0 (1.00), L1 (0.99), L2 (0.98)
- final_token: highest mean recoveries: L11 (0.64), L10 (0.51), L9 (0.14)

## Attention-head results

- capital_of: L9H8 (0.50), L10H0 (0.17), L11H2 (0.14), L8H11 (0.07), L10H10 (0.03)
- capital_city: L9H8 (0.45), L11H2 (0.27), L10H0 (0.10), L6H4 (0.05), L8H11 (0.04)
- possessive: L9H8 (0.35), L11H2 (0.26), L10H0 (0.10), L8H11 (0.08), L11H3 (0.06)

### Similarity across templates

- capital_of vs capital_city: Spearman 0.92, top-10 overlap 9/10
- capital_of vs possessive: Spearman 0.77, top-10 overlap 9/10
- capital_city vs possessive: Spearman 0.78, top-10 overlap 8/10

## Limitations

- Results are conditional on pre-screened prompts that work in all three templates; this can exaggerate apparent robustness.
- Input token positions differ across templates; cross-template plots compare semantic locations (country/final).
- Attention-head scores are for z at the final token only; they do not isolate a full circuit.
- Patching can change predictions without directly identifying human-readable stored knowledge.
- Country pairs are not a representative random sample; no generalization or novelty claim is justified.

## Reproducibility

- Original forward-pass readouts and configs: residual_*.json and heads_*.json (when enabled)
- Flat patch-level outcomes: all_patch_scores.csv
- Screening decision record: screening.csv
- Controls: sanity_checks.csv
- Metadata and predeclared hypothesis: manifest.json
