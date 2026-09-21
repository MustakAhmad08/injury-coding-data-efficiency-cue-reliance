# Recorded experimental settings

This document describes the evaluated runs, not a newly optimized recipe. The machine-readable [SFT settings](../configs/recorded_sft_settings.json) are extracted from the original six training manifests. Changing these settings creates a replication variant.

## Data and seed policy

Training: 2021-2023, 6,000 records (200 per category); development: 2024, 300 (10 per category); balanced test: 2025, 1,500 (50 per category). Training subsets of 1K/3K/6K are nested. The 1K subset has 33 or 34 cases per category; the 3K subset has 100. The split and primary model seed is 20260718.

The proportional test is a separate, disjoint 1,500-case sample using sampling seed 20260918 and largest-remainder class quotas. It approximates unweighted 2025 source-record frequencies, not national prevalence. See [its protocol](../analysis/robustness/proportional_test/README.md). Do not substitute it for the original locked test.

## Fixed generative SFT recipe

| Setting | Recorded value |
|---|---|
| Base models | Qwen/Qwen2.5-7B-Instruct; mistralai/Mistral-7B-Instruct-v0.3 |
| Qwen revision | a09a35458c702b33eeacc393d103063234e8bc28 |
| Mistral revision | c170c708c41dac9275d15a8fff4eca08d52bab71 |
| Quantization | 4-bit NF4, double quantization, bfloat16 computation |
| LoRA | rank 16, alpha 32, dropout 0.05, no bias |
| Target modules | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Training budgets | 1K, 3K, 6K per family; one seed per condition |
| Maximum sequence length | 1,024 tokens |
| Batch / accumulation | 8 per device / 2 steps; effective batch 16 on one GPU |
| Optimizer | paged_adamw_8bit |
| Learning rate / epochs | 2e-4 / up to 3 epochs |
| Scheduler / warm-up | cosine / 0.05 |
| Weight decay / gradient clipping | 0.01 / 1.0 |
| Evaluation / checkpoint interval | every 50 steps |
| Checkpoint selection | lowest development loss; early stopping patience 3, threshold 0 |
| Local generation | greedy; up to 256 output tokens; inference batch 8 |

The SFT recipe was held fixed; it was not selected through the encoder learning-rate sweep. Training uses formatted narrative-to-JSON examples, including structured auxiliary fields and templated rationales. The principal endpoint is primary diagnosis. Consult `Injury_LLM_SFT.py` for exact formatting and loss construction; do not infer response-only loss from the output schema.

## Development-selected baselines

TF-IDF logistic regression uses lowercase Unicode word features, sublinear term frequency, minimum document frequency 2, at most 100,000 features, and L2 regularization. The development grid covers n-grams (1,1)/(1,2) and C=0.25/1/4; selected settings are (1,2), C=4. See `scripts/run_tfidf_logreg_baseline.py`.

RoBERTa-base has 124,668,702 parameters and a 30-class head; all parameters were fine-tuned on 6K. The model revision is `e2da8e2f811d1448a5b465c236feacd80ffbac7b`. Each of three learning rates (1e-5, 2e-5, 5e-5) was trained for five epochs with seeds 20260718/19/20. Each checkpoint was selected by development macro-F1; the learning rate was selected by mean development macro-F1 across seeds (5e-5). Seed 20260718 uses epoch 5. Batch size is 32, evaluation batch 64, maximum length 512, weight decay 0.01, warm-up ratio 0.06, and gradient clipping 1.0. Exact settings and selection records are in `methods/roberta/config.json` and `methods/roberta/results/validation_selection.json`.

Primary reporting uses RoBERTa seed 20260718 to match the other models. That reporting decision was made after evaluation for alignment, not by maximizing test performance. Other seeds remain supplementary. Matching seed numbers does not equalize tuning budgets or random trajectories across architectures.

## Environment and limits

Original SFT manifests record torch 2.11.0+cu128, transformers 4.52.4, peft 0.13.2, trl 0.11.4, bitsandbytes 0.48.2, and datasets 2.21.0; hardware was an RTX 5090 with CUDA 12.8. These are recorded versions, not a complete environment lock. `requirements.txt` specifies installation ranges; a compatible CUDA-specific PyTorch installation is still required. Analysis-specific environment records are retained with the analyses. Hardware and library changes can affect results even with fixed seeds.

Hosted GPT used gpt-5.4-2026-03-05; the hosted Llama condition used a mutable institutional alias. Provider access and availability constrain reruns. A mutable alias cannot guarantee identical future weights. Static release checks validate packaged artifacts, not independent end-to-end replication.
