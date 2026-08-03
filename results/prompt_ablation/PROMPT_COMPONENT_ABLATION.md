# Prompt-component ablation

## Scope

This nested ablation uses only the 300-case **2024 development set** and the unfine-tuned base models. It does not consume the locked 2025 test set and is not part of the main test benchmark.

## Development results

| Model | Prompt condition | Accuracy | Macro F1 (95% CI) | Parse errors |
|---|---|---:|---:|---:|
| Mistral-7B | minimal | 0.000 | 0.000 [0.000, 0.000] | 0.017 |
| Mistral-7B | codebook | 0.480 | 0.422 [0.381, 0.457] | 0.000 |
| Mistral-7B | operational rules | 0.453 | 0.400 [0.355, 0.442] | 0.000 |
| Mistral-7B | demonstrations | 0.473 | 0.408 [0.363, 0.444] | 0.000 |
| Qwen2.5-7B | minimal | 0.000 | 0.000 [0.000, 0.000] | 0.000 |
| Qwen2.5-7B | codebook | 0.547 | 0.494 [0.452, 0.527] | 0.000 |
| Qwen2.5-7B | operational rules | 0.523 | 0.473 [0.429, 0.506] | 0.000 |
| Qwen2.5-7B | demonstrations | 0.650 | 0.621 [0.572, 0.658] | 0.000 |

## Adjacent-condition paired tests

| Model | Comparison | Earlier-only correct | Later-only correct | Exact McNemar p |
|---|---|---:|---:|---:|
| Mistral-7B | minimal -> codebook | 0 | 144 | 8.968e-44 |
| Mistral-7B | codebook -> operational_rules | 33 | 25 | 0.3581 |
| Mistral-7B | operational_rules -> demonstrations | 32 | 38 | 0.5504 |
| Qwen2.5-7B | minimal -> codebook | 0 | 164 | 8.553e-50 |
| Qwen2.5-7B | codebook -> operational_rules | 26 | 19 | 0.3713 |
| Qwen2.5-7B | operational_rules -> demonstrations | 9 | 47 | 2.572e-07 |

## Interpretation boundary

- Results are development-set diagnostics, not additional held-out test results.
- Components are nested, so each row estimates the cumulative prompt rather than an isolated factorial effect.
- The rules are called **operational rules**, not verified manual rules. Independent line-by-line provenance verification remains pending.
- The ten demonstrations are fixed; alternative example selections were not tested.
