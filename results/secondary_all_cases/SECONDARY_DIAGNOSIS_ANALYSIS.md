# Exploratory Secondary-Diagnosis Analysis

## Scope and status

This is a **post hoc exploratory analysis** of the already-evaluated locked 2025 test set. It must not replace or be described as part of the prespecified primary outcome. The main paper result remains 30-class `Diagnosis_1_Code` performance.

The locked test contains 1,500 cases. A gold `Diagnosis_2_Code` is available for 332 cases (22.1%). Of these, 78 repeat the primary diagnosis code and 59 use code 71. The remaining 254 cases contain a diagnosis code distinct from the primary code.

## Definitions

- **Primary—all:** primary-diagnosis accuracy on all 1,500 cases.
- **Primary—secondary subset:** primary accuracy on the same 332 cases used for secondary evaluation.
- **Secondary exact:** positional accuracy of the predicted secondary code among the 332 cases with a gold secondary code; omission is incorrect.
- **Secondary macro F1:** unweighted F1 across the 27 gold secondary-code categories; supports range from 1 to 59, so this estimate is unstable for rare codes.
- **Both ordered:** both primary and secondary codes must be correct in their assigned slots.
- **Presence recall:** proportion of gold-secondary cases for which a valid two-digit secondary code was emitted.
- **False-positive rate:** proportion of the 1,168 cases without a gold secondary code for which a valid secondary code was emitted.

Predicted strings such as `59 (Laceration)` are normalized to code 59. Missing values, ICD codes, and malformed numeric outputs are treated as invalid/missing predictions.

## Model comparison

| Model | Primary—all | Primary—secondary subset (95% CI) | Secondary exact (95% CI) | Secondary macro F1 | Both ordered | Presence recall | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mistral zero-shot | 0.254 | 0.178 [0.148, 0.211] | 0.093 [0.066, 0.120] | 0.061 | 0.027 | 0.852 | 0.732 |
| Mistral 10-shot | 0.430 | 0.380 [0.334, 0.422] | 0.160 [0.127, 0.196] | 0.120 | 0.087 | 0.509 | 0.076 |
| Qwen zero-shot | 0.309 | 0.235 [0.196, 0.274] | 0.117 [0.087, 0.145] | 0.086 | 0.060 | 0.774 | 0.874 |
| Qwen 10-shot | 0.540 | 0.449 [0.401, 0.497] | 0.211 [0.172, 0.250] | 0.195 | 0.145 | 0.795 | 0.216 |
| GPT-5.4 zero-shot | 0.592 | 0.497 [0.452, 0.548] | 0.358 [0.316, 0.404] | 0.303 | 0.244 | 0.801 | 0.104 |
| GPT-5.4 10-shot | 0.700 | 0.590 [0.542, 0.639] | 0.455 [0.410, 0.500] | 0.416 | 0.352 | 0.753 | 0.056 |
| Hosted Llama zero-shot | 0.423 | 0.352 [0.307, 0.395] | 0.130 [0.102, 0.157] | 0.123 | 0.081 | 0.750 | 0.193 |
| Hosted Llama 10-shot | 0.594 | 0.533 [0.482, 0.584] | 0.232 [0.196, 0.271] | 0.245 | 0.169 | 0.590 | 0.051 |
| Mistral SFT 1K | 0.533 | 0.467 [0.422, 0.515] | 0.352 [0.310, 0.395] | 0.237 | 0.217 | 0.627 | 0.031 |
| Mistral SFT 3K | 0.885 | 0.768 [0.723, 0.810] | 0.651 [0.602, 0.696] | 0.588 | 0.578 | 0.789 | 0.026 |
| Mistral SFT 6K | 0.846 | 0.714 [0.669, 0.759] | 0.533 [0.488, 0.581] | 0.484 | 0.464 | 0.696 | 0.019 |
| Qwen SFT 1K | 0.813 | 0.651 [0.602, 0.696] | 0.358 [0.316, 0.398] | 0.260 | 0.268 | 0.578 | 0.021 |
| Qwen SFT 3K | 0.886 | 0.780 [0.735, 0.822] | 0.563 [0.515, 0.605] | 0.527 | 0.512 | 0.720 | 0.015 |
| Qwen SFT 6K | 0.907 | 0.804 [0.768, 0.843] | 0.617 [0.569, 0.663] | 0.584 | 0.563 | 0.759 | 0.020 |

## Main observations

- **Qwen SFT 6K:** primary accuracy on the 332-case subset was 0.804, versus positional secondary accuracy 0.617; the paired difference was -0.187 [-0.241, -0.142].
- **Mistral SFT 3K:** primary accuracy on the 332-case subset was 0.768, versus positional secondary accuracy 0.651; the paired difference was -0.117 [-0.166, -0.063].
- **GPT-5.4 10-shot:** primary accuracy on the 332-case subset was 0.590, versus positional secondary accuracy 0.455; the paired difference was -0.136 [-0.190, -0.084].
- Primary performance is lower on multi-diagnosis cases than on the complete test set for the strongest SFT models, indicating that competing injuries form a harder subset.
- Secondary exact accuracy is lower than primary accuracy even when both are compared on the same 332 narratives. Common failure modes are omission of Diagnosis 2 and promotion of the secondary diagnosis into the primary slot.
- Same-code secondary records often describe the same injury type at a second body part. A model that emits only one diagnosis code is counted wrong positionally even when its primary diagnosis category is correct.
- Code 71 accounts for a substantial share of secondary labels and often accompanies symptoms, illnesses, or incompletely specified additional conditions. These cases require careful review against the NEISS manual before making a strong secondary-coding claim.

### Secondary-label and narrative subgroups

| Model | Repeated code (n=78) | Distinct code (n=254) | Code 71 (n=59) | Other codes (n=273) | Semicolon present (n=83) | No semicolon (n=249) |
|---|---:|---:|---:|---:|---:|---:|
| Qwen SFT 6K | 0.615 | 0.618 | 0.475 | 0.648 | 0.675 | 0.598 |
| Mistral SFT 3K | 0.769 | 0.614 | 0.525 | 0.678 | 0.699 | 0.635 |
| GPT-5.4 10-shot | 0.474 | 0.449 | 0.136 | 0.524 | 0.446 | 0.458 |

## Narrative audit

A targeted read of failures from the strongest SFT outputs identified the following recurring patterns:

1. **Secondary omission:** the model returns the salient primary injury but leaves the optional secondary field blank, including narratives that list two diagnoses after the `DX` marker.
2. **Priority reversal:** the model identifies both diagnoses but swaps their primary/secondary order, especially when ordinary clinical severity appears to conflict with the recorded NEISS ordering.
3. **Duplicate diagnosis categories:** two injuries at different body parts share one diagnosis code; models frequently emit the code once rather than duplicate it in Diagnosis 2.
4. **Code 71 ambiguity:** additional symptoms or non-injury conditions are sometimes represented by code 71 in the source record, while the prompt instructs models not to code unrelated illnesses. This creates a potential task-definition/label-alignment issue.
5. **Mechanism/injury combinations:** electrical burn plus shock, poisoning plus internal injury, and head injury plus fracture require retaining two related but distinct codes rather than collapsing to one.

These qualitative categories are a targeted exploratory audit, not a completed blinded human error annotation. Counts suitable for publication require a prespecified review form and independent adjudication.

## Publication guidance

- Keep primary diagnosis as the only confirmatory headline outcome.
- Present this secondary analysis in the analysis section or appendix and label it post hoc/exploratory.
- Report the 332-case denominator and distinguish all secondary cases from the 254 distinct-secondary cases.
- Before claiming complete NEISS coding, independently verify secondary labels—especially duplicate codes and code 71—and evaluate secondary body-part pairing.
- Do not use these locked-test findings to redesign the prompt and then re-report performance on the same test set.

## Reproducibility

All 14 saved prediction files were aligned by `case_id` to `data\neiss_temporal_2021_2025\locked_test.csv`. Confidence intervals use 1,000 paired resamples stratified by gold secondary diagnosis code with seed `20260721`. Numeric results are available in `secondary_diagnosis_summary.csv`; error-type and narrative-marker summaries are in the accompanying CSV files.
