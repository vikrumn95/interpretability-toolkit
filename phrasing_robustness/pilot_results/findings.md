# Findings (pilot): does GPT-2 use similar internal components to retrieve capitals across phrasings?

**Run:** pilot (`pilot_results/`), GPT-2 small (`openai-community/gpt2`), toolkit commit `a470cd8`, CPU, deterministic.
**Pairs:** the first 3 candidates (France→Italy, Germany→Spain, Japan→China). All 3 passed screening under all three phrasings.
**Phrasings:** `capital_of` = "The capital of {country} is" · `capital_city` = "The capital city of {country} is" · `possessive` = "{country}'s capital is".
**Score:** logit(clean capital) − logit(corrupted capital) at the final token. **Recovery** = (patched − corrupted) / (clean − corrupted).

> **Relationship to the full run.** The pilot's 3 pairs are the first 3 pairs of the full run in `results/`. Inference is deterministic, and every one of the pilot's 1,944 patch records is **identical** to the matching record in the full run (max difference 0.0). The pilot is therefore a **subset** of the full study, not an independent replication. It validated the pipeline; conclusions should rest on `results/findings.md`.

### How to read the numbers

- **Layer numbering.** "Layer L" in the residual sweep is `resid_pre` of block L, the stream *entering* block L, so a change between L and L+1 is caused by block L.
- **What is patched.** The head sweep patches one head's output (`z`) at the final token.
- **Statistics.** Values are means over 3 pairs, with the SD across pairs in parentheses.

### Data quality

- **Control.** The layer-0 country-token control passed 9/9. This is guaranteed by construction, since the country is the only differing token, so it confirms the pipeline, not the model.
- **The capitals are not GPT-2's answers.** GPT-2's top-1 prediction is never the capital (" now", " home", " the"). The clean capital ranks **26th–93rd** under `capital_of`, **165th–537th** under `capital_city` and **144th–497th** under `possessive`, with p ≈ 0.002–0.006 under `capital_of`. The metric is a relative preference between two low-probability capitals.

---

## 1. Residual-stream observations

| Layer (`resid_pre`) | capital_of country | capital_of final | capital_city country | capital_city final | possessive country | possessive final |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| 3 | 0.98 | 0.01 | 0.98 | 0.01 | 0.95 | 0.01 |
| 6 | 0.98 | 0.04 | 0.98 | 0.03 | 0.97 | 0.02 |
| 7 | 0.96 | 0.07 | 0.94 | 0.08 | 0.95 | 0.05 |
| 8 | 0.96 | 0.07 | 0.94 | 0.09 | 0.95 | 0.06 |
| 9 | 0.87 | 0.13 | 0.89 | 0.13 | 0.83 | 0.12 |
| 10 | 0.29 | 0.70 | 0.41 | 0.64 | 0.41 | 0.51 |
| 11 | 0.16 | 0.84 | 0.31 | 0.74 | 0.28 | 0.61 |

**Shared pattern.**
- **Before the handoff.** The country token alone carries almost all of the patchable effect through layer 8 (≥ 0.94). Final-token recovery stays ≤ 0.13 through layer 9.
- **The handoff.** It happens in **block 9** for every phrasing and pair (9/9). Between layers 9 and 10, final-token recovery jumps to 0.51–0.70, and country-token recovery falls to 0.29–0.41.

**Differences.**
- **Final-token recovery at layer 11** is ordered `capital_of` (0.84) > `capital_city` (0.74) > `possessive` (0.61).
- **Intermediate tokens (possessive only).** `possessive` routes a small share through its intermediate tokens `'s` and ` capital` (up to about 0.07). Country + final recovery sums to about 1.0 in the two "capital" phrasings but drops to about 0.89 in `possessive` by layer 11.
- **Structural zeros.** The zeros before the country token in the "capital" phrasings follow from causal attention, not from the model.

## 2. Attention-head observations

| Head | capital_of | capital_city | possessive |
|---|---:|---:|---:|
| L9H8 | +0.53 (0.01) | +0.48 (0.04) | +0.37 (0.09) |
| L11H2 | +0.14 (0.03) | +0.27 (0.04) | +0.28 (0.05) |
| L10H0 | +0.14 (0.02) | +0.09 (0.02) | +0.08 (0.01) |
| L8H11 | +0.05 (0.01) | +0.03 (0.01) | +0.07 (0.00) |
| L11H3 | +0.02 (0.01) | +0.03 (0.01) | +0.06 (0.01) |
| L6H4 | +0.03 (0.01) | +0.04 (0.01) | +0.01 (0.00) |
| L10H10 | +0.03 (0.02) | +0.02 (0.01) | +0.02 (0.01) |
| L10H7 | −0.07 (0.04) | −0.04 (0.02) | −0.02 (0.02) |
| L11H10 | −0.06 (0.01) | −0.06 (0.02) | −0.02 (0.01) |

**The same heads lead in every phrasing.**
- **L9H8** is the strongest head in all three phrasings, and the top head in 8 of 9 pair × phrasing combinations. In Japan→China under `possessive`, L11H2 is on top.
- **{L9H8, L10H0, L11H2}** is the top-3 set in 8 of 9 combinations. In Germany→Spain under `possessive`, L8H11 replaces L10H0.

**The weighting shifts.** From `capital_of` to `possessive`, L9H8 falls (0.53 → 0.37) and L11H2 rises (0.14 → 0.28). Summed per layer, layer 9 contributes 0.53 / 0.49 / 0.38 and layer 11 contributes 0.11 / 0.24 / 0.36.

**The two sweeps agree.** Summed layer-9 head effects (0.53 / 0.49 / 0.38) closely match the final-token residual jump across block 9 (0.57 / 0.51 / 0.38).

**Negative heads.** L10H7 and L11H10 work against the clean capital, and most strongly in `capital_of` / `capital_city`.

**Similarity metrics.** `report.md` lists Spearman 0.87 / 0.62 / 0.65 and top-10 overlaps of 8 / 8 / 7. These are computed on 3-pair means over all 144 heads, where near-zero heads dominate the ranking.
- **Matched comparisons:**
  - Same pair, different phrasing: Spearman 0.70.
  - Same phrasing, different pair: Spearman 0.40.
  - Pearson, dominated by the large heads: 0.92–0.98.
- **Restricted to the ~10 heads with |recovery| ≥ 0.02,** Spearman across phrasings is 0.85–0.90. The lower all-head values for `possessive` mostly reflect noise among inactive heads, not disagreement among the heads that matter.

## 3. Interpretation and limitations

**What the observations suggest.** The same route appears in every phrasing:
- **Before block 9,** the country signal stays at the country token.
- **In blocks 9–11,** it moves to the final position through the same few heads (L9H8, then L11H2 and L10H0).
- **Phrasing changes the weights** (the possessive form leans more on layer 11), not which components are involved.

**What the pilot does not establish, beyond the full run's limitations.**
- **Independence.** Its 3 pairs are reused in the full run, so the full run "agreeing" with the pilot is partly guaranteed by the overlap.
- **Realistic variability.** SDs across 3 pairs are smaller than across 5. For example, L9H8 under `capital_of` has SD 0.01 here vs 0.09 in the full run, where Iran→Iraq is an outlier (L9H8 0.35, L10H0 0.34).
- **How selective screening is.** All 3 pilot candidates passed, but 9 of 14 candidates failed in the full run. Seven of the 14 failed the possessive phrasing's tokenization checks. The pilot gave no warning of that selection effect.
- **The shared limitations of the full run** also apply:
  - **Sufficiency only.** Only clean→corrupted patches were run, so there is no evidence about necessity.
  - **Partial coverage.** Heads were tested at the final token only, and there was no MLP decomposition.
  - **Weak target behavior.** The capitals are low-probability tokens.
  - **Descriptive only.** There is no inferential statistics.

**Alternative explanations,** also as in the full run:
- **General subject movers.** The heads could be general mechanisms that move the subject to the final position, not capital-specific ones.
- **Copy-suppression heads.** L10H7 and L11H10 match heads described in prior GPT-2 small work as negative name-mover / copy-suppression heads (Wang et al., 2022; McDougall et al., 2023).
- **Surface similarity.** All three phrasings share "capital … is".
- **Distance confound.** In `possessive` the country is 3 tokens from the end rather than 1.

## 4. Conclusions and next steps

**Answer (pilot only).** On 3 country pairs, the evidence points to **shared components across phrasings**:
- **The same transfer point:** block 9.
- **The same dominant heads:** L9H8, L10H0, L11H2.

The relative weighting depends on phrasing. Taken alone, this is an early signal, not a result. Like the full run, it supports shared *routing of country-dependent information*, not a shared capital-retrieval circuit.

**Follow-up.** The pilot's natural follow-up, a larger run over the full candidate pool, has been done (`results/`). It reproduced the same pattern on 5 pairs and showed more variation between pairs. The next informative experiment is the **relation-swap control** described in `results/findings.md`: keep the countries and sentence frames, change the relation (for example, official language), and check whether L9H8 / L10H0 / L11H2 still dominate. That tells apart capital-specific routing from a general mechanism that moves the subject to the prediction position.
