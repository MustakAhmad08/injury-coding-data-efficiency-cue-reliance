"""Evaluate prespecified difficult diagnosis pairs on aligned locked-test cases.

This is a natural-case contrast audit, not a synthetic minimal-pair experiment.
It uses the existing predictions and creates a reviewer worksheet so a qualified
co-author can verify a deterministic sample before publication.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "methods_runs"
TEST_FILE = ROOT / "data" / "neiss_temporal_2021_2025" / "locked_test.csv"
OUTPUT_DIR = RUNS / "difficult_contrasts"
SEED = 20260718
RESAMPLES = 1_000

CONTRASTS = [
    ("scald_vs_thermal_burn", 48, "Scald burn", 51, "Thermal burn"),
    ("electrical_burn_vs_shock", 46, "Electrical burn", 67, "Electric shock"),
    ("concussion_vs_internal_injury", 52, "Concussion", 62, "Internal injury"),
    ("laceration_vs_avulsion", 59, "Laceration", 72, "Avulsion"),
    ("ingestion_vs_aspiration", 41, "Ingestion", 42, "Aspiration"),
]

MODEL_ORDER = [
    "tfidf_logreg_6000",
    "mistral_0shot",
    "mistral_10shot",
    "qwen_0shot",
    "qwen_10shot",
    "gpt54_0shot",
    "gpt54_10shot",
    "hosted_llama_0shot",
    "hosted_llama_10shot",
    "mistral_sft_1000",
    "mistral_sft_3000",
    "mistral_sft_6000",
    "qwen_sft_1000",
    "qwen_sft_3000",
    "qwen_sft_6000",
]

DISPLAY = {
    "tfidf_logreg_6000": "TF-IDF + logistic regression",
    "mistral_0shot": "Mistral zero-shot",
    "mistral_10shot": "Mistral 10-shot",
    "qwen_0shot": "Qwen zero-shot",
    "qwen_10shot": "Qwen 10-shot",
    "gpt54_0shot": "GPT-5.4 zero-shot",
    "gpt54_10shot": "GPT-5.4 10-shot",
    "hosted_llama_0shot": "Hosted Llama zero-shot",
    "hosted_llama_10shot": "Hosted Llama 10-shot",
    "mistral_sft_1000": "Mistral SFT 1K",
    "mistral_sft_3000": "Mistral SFT 3K",
    "mistral_sft_6000": "Mistral SFT 6K",
    "qwen_sft_1000": "Qwen SFT 1K",
    "qwen_sft_3000": "Qwen SFT 3K",
    "qwen_sft_6000": "Qwen SFT 6K",
}


def percentile_interval(values: list[float]) -> tuple[float, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return float(low), float(high)


def main() -> None:
    test = pd.read_csv(TEST_FILE)
    if len(test) != 1_500 or test["case_id"].nunique() != 1_500:
        raise ValueError("Locked-test integrity failure")
    test = test.set_index("case_id", drop=False)

    predictions: dict[str, pd.DataFrame] = {}
    for model in MODEL_ORDER:
        path = RUNS / model / "test_results.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        required = {"case_id", "true_diag1", "pred_diag1"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{path} lacks {sorted(required - set(frame.columns))}")
        frame["pred_diag1"] = pd.to_numeric(frame["pred_diag1"], errors="coerce").fillna(-1).astype(int)
        frame["true_diag1"] = frame["true_diag1"].astype(int)
        frame = frame.set_index("case_id").sort_index()
        if not frame.index.equals(test.index.sort_values()):
            raise ValueError(f"Case IDs do not align for {model}")
        predictions[model] = frame

    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    aggregate_values: dict[str, list[float]] = defaultdict(list)
    for contrast_id, code_a, label_a, code_b, label_b in CONTRASTS:
        selected = test[test["diag1"].isin([code_a, code_b])].sort_index()
        if len(selected) != 100 or selected["diag1"].value_counts().to_dict() != {code_a: 50, code_b: 50}:
            raise ValueError(f"Unexpected support for {contrast_id}")
        y_true = selected["diag1"].astype(int).to_numpy()
        indices_a = np.flatnonzero(y_true == code_a)
        indices_b = np.flatnonzero(y_true == code_b)
        for model, frame in predictions.items():
            y_pred = frame.loc[selected.index, "pred_diag1"].to_numpy(dtype=int)
            correct = y_pred == y_true
            recall_a = float(np.mean(y_pred[indices_a] == code_a))
            recall_b = float(np.mean(y_pred[indices_b] == code_b))
            cross = int(
                np.sum((y_true == code_a) & (y_pred == code_b))
                + np.sum((y_true == code_b) & (y_pred == code_a))
            )
            accuracy = float(np.mean(correct))
            macro_f1 = float(
                f1_score(y_true, y_pred, labels=[code_a, code_b], average="macro", zero_division=0)
            )
            boot_acc: list[float] = []
            boot_f1: list[float] = []
            for _ in range(RESAMPLES):
                sampled = np.concatenate(
                    [
                        rng.choice(indices_a, size=len(indices_a), replace=True),
                        rng.choice(indices_b, size=len(indices_b), replace=True),
                    ]
                )
                boot_acc.append(float(np.mean(y_pred[sampled] == y_true[sampled])))
                boot_f1.append(
                    float(
                        f1_score(
                            y_true[sampled],
                            y_pred[sampled],
                            labels=[code_a, code_b],
                            average="macro",
                            zero_division=0,
                        )
                    )
                )
            acc_low, acc_high = percentile_interval(boot_acc)
            f1_low, f1_high = percentile_interval(boot_f1)
            rows.append(
                {
                    "contrast_id": contrast_id,
                    "code_a": code_a,
                    "label_a": label_a,
                    "code_b": code_b,
                    "label_b": label_b,
                    "model_key": model,
                    "model": DISPLAY[model],
                    "n": len(selected),
                    "recall_a": recall_a,
                    "recall_b": recall_b,
                    "accuracy": accuracy,
                    "accuracy_ci_low": acc_low,
                    "accuracy_ci_high": acc_high,
                    "macro_f1": macro_f1,
                    "macro_f1_ci_low": f1_low,
                    "macro_f1_ci_high": f1_high,
                    "cross_pair_errors": cross,
                    "cross_pair_error_rate": cross / len(selected),
                    "other_errors": int(np.sum(~correct)) - cross,
                }
            )
            aggregate_values[model].append(accuracy)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(OUTPUT_DIR / "contrast_metrics.csv", index=False)

    review_parts = []
    for contrast_id, code_a, label_a, code_b, label_b in CONTRASTS:
        for code, label in ((code_a, label_a), (code_b, label_b)):
            sample = (
                test[test["diag1"].eq(code)]
                .sample(n=5, random_state=SEED + code)
                [["case_id", "year", "narrative", "diag1", "diagnosis_category"]]
                .copy()
            )
            sample["contrast_id"] = contrast_id
            sample["expected_label"] = label
            review_parts.append(sample)
    review = pd.concat(review_parts, ignore_index=True)
    review["review_label_supported"] = ""
    review["review_contrast_is_rule_relevant"] = ""
    review["reviewer_suggested_code"] = ""
    review["reviewer_notes"] = ""
    review.to_csv(OUTPUT_DIR / "contrast_case_review.csv", index=False)

    focus = ["qwen_sft_6000", "mistral_sft_3000", "tfidf_logreg_6000", "gpt54_10shot"]
    report = [
        "# Difficult Diagnosis Contrast Analysis",
        "",
        "## Scope",
        "",
        "This analysis evaluates five prespecified diagnosis pairs using the naturally",
        "occurring locked-test cases (50 cases per code). It is a **post hoc natural-case",
        "contrast audit**, not a controlled synthetic minimal-pair experiment. The",
        "50-case reviewer worksheet remains unvalidated until a qualified co-author",
        "completes its review fields.",
        "",
        "## Focus-model results",
        "",
        "| Contrast | Qwen SFT 6K | Mistral SFT 3K | TF-IDF logistic | GPT-5.4 10-shot |",
        "|---|---:|---:|---:|---:|",
    ]
    for contrast_id, _, label_a, _, label_b in CONTRASTS:
        values = []
        for model in focus:
            row = metrics[
                metrics["contrast_id"].eq(contrast_id) & metrics["model_key"].eq(model)
            ].iloc[0]
            values.append(
                f"{row['accuracy']:.3f} [{row['accuracy_ci_low']:.3f}, "
                f"{row['accuracy_ci_high']:.3f}]"
            )
        report.append(f"| {label_a} vs {label_b} | " + " | ".join(values) + " |")
    report.extend(
        [
            "",
            "Accuracy treats predictions outside the two target codes as errors. The full",
            "CSV also reports directional recall, macro F1, direct cross-pair errors, and",
            "other errors for all 15 conditions.",
            "",
            "## Mean accuracy across the five contrast subsets",
            "",
            "| Model | Mean contrast accuracy |",
            "|---|---:|",
        ]
    )
    for model in focus:
        report.append(f"| {DISPLAY[model]} | {np.mean(aggregate_values[model]):.3f} |")
    report.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- These subsets identify performance on difficult category pairs but do not",
            "  prove sensitivity to a single changed linguistic feature.",
            "- Do not call this a validated minimal-pair experiment until the review",
            "  worksheet is completed and a genuinely controlled contrast set is created.",
            "- Because the locked-test results were already examined, report this as post",
            "  hoc exploratory analysis.",
            "",
            "## Reproducibility",
            "",
            f"- Seed: `{SEED}`",
            f"- Bootstrap: `{RESAMPLES}` diagnosis-stratified resamples per pair",
            "- Source predictions: the frozen `test_results.csv` files for all 15",
            "  primary-diagnosis conditions",
            "",
        ]
    )
    (OUTPUT_DIR / "DIFFICULT_CONTRASTS.md").write_text("\n".join(report), encoding="utf-8")
    metadata = {
        "seed": SEED,
        "bootstrap_resamples": RESAMPLES,
        "status": "post hoc natural-case contrast audit; human validation pending",
        "contrasts": [
            {
                "contrast_id": item[0],
                "code_a": item[1],
                "label_a": item[2],
                "code_b": item[3],
                "label_b": item[4],
            }
            for item in CONTRASTS
        ],
        "models": MODEL_ORDER,
    }
    (OUTPUT_DIR / "contrast_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(metrics)} contrast-model rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
