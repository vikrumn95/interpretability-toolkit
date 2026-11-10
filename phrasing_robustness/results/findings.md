# Findings: does GPT-2 use similar internal components to retrieve capitals across phrasings?

**Run:** full study (`results/`), GPT-2 small (`openai-community/gpt2`), toolkit commit `a470cd8`, CPU, deterministic.
**Pairs:** 5 of 14 candidates passed screening under all three phrasings: France→Italy, Germany→Spain, Japan→China, Russia→Canada, Iran→Iraq.
**Phrasings:** `capital_of` = "The capital of {country} is" · `capital_city` = "The capital city of {country} is" · `possessive` = "{country}'s capital is".
**Score:** logit(clean capital) − logit(corrupted capital) at the final token. **Recovery** = (patched − corrupted) / (clean − corrupted): 0 = no effect, 1 = fully clean.

### How to read the numbers

- **Layer numbering.** "Layer L" in the residual sweep means `resid_pre` of block L, the residual stream *entering* block L. A change between layer L and L+1 is therefore caused by block L. Head `LxHy` is head y of block x.
- **What is patched.** The residual sweep patches one token position at one layer. The head sweep patches one head's output (`z`) **at the final token only**.
- **Statistics.** All values are means over the 5 pairs, with the SD across pairs in parentheses. Nothing here is a significance test.

### Data quality

- **Control.** The layer-0 country-token control passed 15/15 (patched score = clean score exactly). This confirms the pipeline is correct. It is not evidence about the model: the country is the only differing input token, so the control must pass by construction.
- **The capitals are not GPT-2's answers.** The model's top-1 prediction is never a capital. It is " now", " home", " the" or " a". The clean capital ranks **22nd–93rd** under `capital_of`, **165th–537th** under `capital_city` and **41st–497th** under `possessive`, with p ≈ 0.002–0.006 under `capital_of`. Everything below measures how the model ranks *two unlikely candidates against each other*, not whether it produces the capital.

---

## 1. Residual-stream observations

Mean recovery at the country token and at the final token (" is"):

| Layer (`resid_pre`) | capital_of country | capital_of final | capital_city country | capital_city final | possessive country | possessive final |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 1.00 | 0.00 | 1.00 | 0.00 | 1.00 | 0.00 |
| 3 | 0.99 | 0.02 | 0.98 | 0.03 | 0.94 | 0.02 |
| 6 | 0.97 | 0.04 | 0.97 | 0.05 | 0.95 | 0.03 |
| 7 | 0.93 | 0.07 | 0.91 | 0.10 | 0.92 | 0.06 |
| 8 | 0.93 | 0.08 | 0.91 | 0.11 | 0.92 | 0.07 |
| 9 | 0.86 | 0.15 | 0.86 | 0.16 | 0.80 | 0.14 |
| 10 | 0.32 | 0.67 | 0.41 | 0.61 | 0.40 | 0.51 |
| 11 | 0.19 | 0.84 | 0.31 | 0.73 | 0.27 | 0.64 |

(Figures: `compare_country_token.png`, `compare_final_token.png`, `residual_*.png`.)

**Shared pattern: one handoff, in the same block, for every phrasing.**
- **Country token.** Through layer 8, patching the country token's residual alone restores almost all of the clean preference (≥ 0.91). Through layer 9, the country-specific information that affects the score sits mainly at the country position.
- **Final token.** Recovery stays small through layer 9 (≤ 0.16) and then jumps between layers 9 and 10. Across that same step, country-token recovery falls by roughly half.
- **Consistency.** The handoff happens in **block 9** for every phrasing and every pair: 15/15 pair × phrasing combinations have country recovery > 0.7 at layer 9, < 0.55 at layer 10, and final recovery > 0.4 at layer 10.
- **Gradual leakage before it.** A small, steady rise in final-token recovery over layers 3–9 (to about 0.15) suggests some earlier, weaker transfer.

**Differences between phrasings.**
- **Final-token recovery at layer 11** is ordered `capital_of` 0.84 > `capital_city` 0.73 > `possessive` 0.64. Correspondingly, more of the effect remains at the country token at layer 11 in `capital_city` (0.31) and `possessive` (0.27) than in `capital_of` (0.19).
- **Intermediate tokens (possessive only).** `possessive` is the only phrasing with tokens between the country and the final position (`'s`, ` capital`). Those carry small effects (up to 0.07 at ` capital`, layer 11). Country + final recovery sums to about 1.00 at every layer in the two "capital" phrasings but falls to about 0.91 in `possessive`. The remaining ~9% flows through the intermediate tokens.
- **Structural zeros.** In `capital_of` and `capital_city`, the tokens *before* the country show exactly 0. This follows from causal attention, which stops earlier tokens from seeing the country. It is not a finding about the model.

**Variation across pairs** is modest. Japan→China and Iran→Iraq keep more of the effect at the country token in late layers (0.40–0.50 at layer 10 under `capital_city`, vs 0.32–0.37 for the others).

## 2. Attention-head observations

Mean recovery when one head's output at the final token is patched. SD across the 5 pairs is in parentheses.

| Head | capital_of | capital_city | possessive |
|---|---:|---:|---:|
| L9H8 | +0.50 (0.09) | +0.45 (0.10) | +0.35 (0.07) |
| L11H2 | +0.14 (0.03) | +0.27 (0.07) | +0.26 (0.06) |
| L10H0 | +0.17 (0.09) | +0.10 (0.05) | +0.10 (0.05) |
| L8H11 | +0.07 (0.02) | +0.04 (0.01) | +0.08 (0.01) |
| L11H3 | +0.02 (0.01) | +0.03 (0.01) | +0.06 (0.01) |
| L6H4 | +0.03 (0.01) | +0.05 (0.01) | +0.02 (0.01) |
| L10H10 | +0.03 (0.01) | +0.02 (0.01) | +0.02 (0.01) |
| L10H7 | −0.10 (0.07) | −0.05 (0.03) | −0.03 (0.02) |
| L11H10 | −0.06 (0.01) | −0.06 (0.02) | −0.02 (0.01) |

(Figures: `heads_*.png`.)

**The strongest heads are the same in every phrasing.**
- **L9H8** is the largest single effect in all three phrasings, and the top head in 13 of 15 pair × phrasing combinations. The other two have L11H2 on top.
- **{L9H8, L10H0, L11H2}** is exactly the top-3 set in 12 of 15 combinations. In the remaining three, L8H11 or L6H4 takes the third slot.
- **Only about 9 of 144 heads** have |mean recovery| ≥ 0.02 in any phrasing. The rest are essentially zero.

**What changes is the weighting, not the identity of the heads.**
- From `capital_of` to `possessive`, L9H8's share falls (0.50 → 0.35) and layer 11's rises: L11H2 goes 0.14 → 0.26 and L11H3 goes 0.02 → 0.06. Summed over heads, layer 9 contributes 0.51 / 0.46 / 0.37 and layer 11 contributes 0.11 / 0.24 / 0.33 (`capital_of` / `capital_city` / `possessive`).
- `possessive` therefore relies less on the block-9 handoff and more on layer 11. This matches its lower final-token recovery at `resid_pre` 11 in Section 1, since layer 11 acts *after* that point in the residual stream.

**Two negative heads, L10H7 and L11H10,** push the score toward the *corrupted* capital when given clean inputs. Both are strongest in `capital_of`.

**The head and residual sweeps agree.** They are separate experiments, but the summed head effects for each layer match the size of the residual-stream jump at the final token closely:

| Template | Residual jump across block 9 | Sum of layer-9 heads | Residual jump across block 10 | Sum of layer-10 heads |
|---|---:|---:|---:|---:|
| capital_of | +0.52 | +0.51 | +0.17 | +0.14 |
| capital_city | +0.46 | +0.46 | +0.12 | +0.11 |
| possessive | +0.37 | +0.37 | +0.13 | +0.13 |

Patching is not additive in general, so this agreement is descriptive. But it indicates that the block-9 jump at the final token comes almost entirely from attention, with L9H8 the main contributor.

**Similarity metrics, and a caveat about them.** `report.md` lists Spearman correlations of 0.92 / 0.77 / 0.78 and top-10 overlaps of 9 / 9 / 8 across phrasing pairs. Two caveats apply:

- **These Spearman values are inflated relative to anything they could be compared with.** They are computed on 5-pair *means*, and over all 144 heads, most of which are near zero. A fair comparison uses single pairs:
  - **Same pair, different phrasing:** Spearman 0.70.
  - **Same phrasing, different pair:** Spearman 0.40.
  - **Pearson,** which is dominated by the few large heads, is 0.89–0.94 in every comparison.

  So the large heads are shared everywhere. The ranking of the many small heads depends more on *which countries* are involved than on *how the question is phrased*.
- **The top-10 lists rank signed values,** so the negative heads L10H7 and L11H10 never appear in them.

## 3. Interpretation and limitations

**What the observations suggest.** For these pairs and phrasings, the country-dependent signal follows the same route each time:
- **Before block 9,** it stays at the country token.
- **In blocks 9–11,** a small, consistent set of attention heads moves it to the final position, chiefly L9H8, then L10H0 and L11H2, with L8H11 and L11H3 smaller.
- **Phrasing matters, but modestly.** It shifts how much each head contributes, mainly from L9H8 toward layer 11 in the possessive form, but not which heads are involved. This supports the predeclared hypothesis that some influential locations are shared across phrasings.

**What this does not establish.**
- **That these heads *retrieve capitals*.** Patching a head's output at the final token shows that the head *carries* information that changes the capital preference. It cannot tell whether that information is the country's identity or an already-computed capital. Likewise, patching the country token's residual transplants everything computed there, so this experiment cannot say where the country→capital association is computed (for example, in MLPs at the country token in earlier layers, or late at the final token).
- **Necessity.** Every patch puts *clean* activations into the *corrupted* run, which tests whether a component is *sufficient* to move the score. No component was removed (ablated), so nothing here shows any head is *necessary*. Redundant "backup" heads could hide behind the ones that appear.
- **Complete coverage.** Heads were tested only at the final token, and MLPs were not tested on their own. Effects routed through other positions (such as the possessive's intermediate tokens) or through MLP layers are not attributed to any component.
- **Generality.**
  - n = 5 hand-picked pairs, patched in one direction only.
  - Screening kept only pairs that worked under all three phrasings, which biases the set toward robust cases and can inflate apparent similarity.
  - 7 of the 14 candidates failed the possessive phrasing's tokenization checks, and 4 of those failed for no other reason, so the surviving set is partly determined by tokenization.
  - The capitals are low-probability tokens for GPT-2 small (see Data quality), so these are effects on a weak, relative preference.
- **Statistical support.** All comparisons are descriptive. The SE bars in the comparison plots are not confidence intervals, and no significance testing was done.

**Alternative explanations.**
- **A general "move the subject" mechanism.** The same heads might move the subject entity's identity to the final position for *any* question about the subject (language, currency, continent). If so, the shared components are subject movers, not capital retrievers. This experiment cannot distinguish the two.
- **Known general-purpose heads.** L10H7 and L11H10 have the same indices as heads previously described in GPT-2 small as "negative name mover" / copy-suppression heads, in work on indirect-object identification (Wang et al., 2022) and copy suppression (McDougall et al., 2023). Their appearance here fits a general mechanism that dampens whatever the model is about to copy, rather than anything capital-specific.
- **Shallow similarity between the phrasings.** All three phrasings contain the word "capital" and end in " is". The similarity may reflect near-identical surface forms rather than robustness to phrasing more broadly.
- **Position and distance confounds.** In `possessive` the country is 3 tokens from the final position, versus 1 in the other two phrasings. The shift from L9H8 toward layer 11 could reflect that distance, or the extra intermediate tokens, rather than the change of wording.

## 4. Conclusions and next steps

**Answer to the research question (within this evidence).** Yes, in a limited sense. For these 5 country pairs, GPT-2 small routes the country-dependent signal through largely the **same internal components** in all three phrasings:
- **The same transfer point:** country position → final position, in block 9, in all 15 pair × phrasing combinations.
- **The same dominant heads:** L9H8, L10H0, L11H2.

Phrasing changes the **relative weighting** but not the set of components; the possessive form leans less on L9H8 and more on layer 11.

The evidence supports "shared components that move country-dependent information to the prediction position". It does **not** support a shared *capital-retrieval circuit*, because these experiments cannot show where the capital itself is computed, or whether these heads are specific to capitals at all.

**Recommended follow-up: a relation-swap control.** Keep the country pairs and sentence frames but change the relation. For example, use "The official language of {country} is" and "{country}'s official language is", with single-token targets such as " French" vs " Italian" and " German" vs " Spanish", screened the same way. Then run the same residual and final-token head sweeps.
- **If** L9H8 / L10H0 / L11H2 again dominate, with a similar block-9 handoff, they are general subject movers, and anything capital-specific must happen elsewhere (most likely in MLPs, which would be the next thing to test).
- **If** the dominant heads change, the routing found here is at least partly specific to the capital relation.

Write down the comparison criteria before running. For example: top-3 head overlap, and the ratio of L9H8's recovery under the language relation to its recovery under the capital relation.
