# Diagnosis-term-only masking results

Exploratory frozen-model evaluation; no retraining or tuning.

The lexicon masked 1,565 spans in 1,091/1,500 reports. All other characters were preserved. The remaining 409 reports were unchanged. Coverage is not exhaustive or clinician-validated.

## Primary diagnosis, all 1,500 cases

| Model | Original accuracy | Masked accuracy | Original macro F1 | Masked macro F1 | F1 drop (95% CI) |
|---|---:|---:|---:|---:|---:|
| TF-IDF 6K | 0.872 | 0.661 | 0.875 | 0.651 | 0.223 [0.207, 0.244] |
| Qwen SFT 6K | 0.907 | 0.645 | 0.905 | 0.631 | 0.274 [0.258, 0.294] |
| Mistral SFT 3K | 0.885 | 0.645 | 0.884 | 0.634 | 0.251 [0.232, 0.271] |

Intervals use 1,000 paired, diagnosis-stratified bootstrap samples (seed 20260718), conditional on fixed models; they do not reflect training variability.

## Changed reports only

| Model | N | Original accuracy | Masked accuracy | Original macro F1 | Masked macro F1 |
|---|---:|---:|---:|---:|---:|
| TF-IDF 6K | 1091 | 0.885 | 0.596 | 0.853 | 0.595 |
| Qwen SFT 6K | 1091 | 0.909 | 0.550 | 0.884 | 0.561 |
| Mistral SFT 3K | 1091 | 0.890 | 0.561 | 0.851 | 0.567 |

## Distinct primary/secondary diagnoses (254 cases)

| Model | Primary original / masked | Secondary original / masked | Both ordered original / masked | Exact swaps original / masked |
|---|---:|---:|---:|---:|
| TF-IDF 6K | 0.728 / 0.508 | Not applicable | Not applicable | Not applicable |
| Qwen SFT 6K | 0.776 / 0.520 | 0.618 / 0.181 | 0.551 / 0.114 | 22 / 3 |
| Mistral SFT 3K | 0.728 / 0.504 | 0.614 / 0.201 | 0.535 / 0.118 | 20 / 5 |

## Interpretation and verification

This intervention isolates listed diagnosis spans more precisely than tail removal, but it does not remove all diagnostic information. Residual synonyms, symptoms, and injury mechanisms remain. Matching does not distinguish diagnostic nouns from incidental or verb uses of the same words, and negated mentions are also masked. Mask tokens can shift the input distribution, and diagnostic terms are themselves useful evidence. Neither a drop nor retained accuracy proves contextual reasoning. Class coverage and per-class recall are provided separately.

- TF-IDF 6K: 0 primary-prediction disagreements versus archived results among 409 unchanged reports.
- Qwen SFT 6K: 1 primary-prediction disagreements versus archived results among 409 unchanged reports.
- Mistral SFT 3K: 1 primary-prediction disagreements versus archived results among 409 unchanged reports.
