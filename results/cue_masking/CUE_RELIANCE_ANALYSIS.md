# Cue-reliance analysis

## Scope

This is a **post hoc exploratory analysis** of the locked 2025 test set. It was not used for model selection or tuning. Marker-only masking removes the tokens introducing an explicit diagnosis statement; diagnosis-tail masking removes the first marker and everything after it.

An explicit marker occurred in **1495 of 1,500 narratives (99.7%)**.

## Results

| Model | Mask | Macro F1 original | Macro F1 masked | F1 drop (95% CI) |
|---|---|---:|---:|---:|
| Qwen SFT 6K | marker only | 0.905 | 0.897 | 0.009 [0.000, 0.017] |
| Qwen SFT 6K | diagnosis tail | 0.905 | 0.550 | 0.355 [0.335, 0.380] |
| Mistral SFT 3K | marker only | 0.884 | 0.879 | 0.005 [-0.003, 0.014] |
| Mistral SFT 3K | diagnosis tail | 0.884 | 0.546 | 0.338 [0.318, 0.361] |
| TF-IDF + logistic regression | marker only | 0.875 | 0.867 | 0.007 [0.000, 0.015] |
| TF-IDF + logistic regression | diagnosis tail | 0.875 | 0.588 | 0.287 [0.265, 0.309] |

## Most diagnosis-tail-sensitive categories

- **Qwen SFT 6K:** 52 Concussion: 0.98 [0.94, 1.00]; 54 Crushing injury: 0.86 [0.76, 0.96]; 72 Avulsion: 0.84 [0.74, 0.94]; 53 Contusion/abrasion: 0.80 [0.68, 0.90]; 64 Strain/sprain: 0.76 [0.64, 0.86].
- **Mistral SFT 3K:** 52 Concussion: 0.98 [0.94, 1.00]; 57 Fracture: 0.90 [0.80, 0.98]; 72 Avulsion: 0.84 [0.72, 0.94]; 64 Strain/sprain: 0.80 [0.68, 0.90]; 54 Crushing injury: 0.78 [0.66, 0.90].
- **TF-IDF + logistic regression:** 52 Concussion: 0.82 [0.72, 0.92]; 57 Fracture: 0.72 [0.60, 0.84]; 72 Avulsion: 0.68 [0.54, 0.80]; 54 Crushing injury: 0.66 [0.54, 0.78]; 55 Dislocation: 0.64 [0.50, 0.78].

## Interpretation boundary

- Marker-only masking is a formatting control; it leaves the diagnosis words intact.
- Diagnosis-tail masking is deliberately severe and can remove both explicit labels and legitimate clinical details. Its performance drop is therefore an upper-bound estimate of reliance on the tail, not a pure causal estimate of label-word reliance.
- The locked test set had already been examined; all findings are exploratory.
- No API narratives were uploaded and no paid model was rerun for this analysis.

## Reproducibility

- Seed: `20260718`
- Paired diagnosis-stratified bootstrap resamples: `1000`
- Locked-test SHA-256: `19797d36307469b7522fc21947e72c74fa76dae8264b499ef74ca91ab5044d15`
