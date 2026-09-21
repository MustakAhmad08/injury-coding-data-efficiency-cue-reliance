"""Compute ACL-style metrics, bootstrap CIs, and paired McNemar tests."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction",
        action="append",
        required=True,
        metavar="NAME=CSV",
        help="Repeat once per model condition; CSV must contain case_id, true_diag1, pred_diag1.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def load_predictions(specifications: list[str]) -> dict[str, pd.DataFrame]:
    loaded = {}
    for specification in specifications:
        if "=" not in specification:
            raise ValueError(f"Prediction must use NAME=CSV syntax: {specification}")
        name, raw_path = specification.split("=", 1)
        frame = pd.read_csv(raw_path)
        required = {"case_id", "true_diag1", "pred_diag1"}
        if not required.issubset(frame.columns):
            raise ValueError(f"{raw_path} lacks {sorted(required - set(frame.columns))}")
        if frame["case_id"].duplicated().any():
            raise ValueError(f"Duplicate case IDs in {raw_path}")
        frame = frame[["case_id", "true_diag1", "pred_diag1"]].copy()
        frame["true_diag1"] = frame["true_diag1"].astype(int)
        frame["pred_diag1"] = pd.to_numeric(frame["pred_diag1"], errors="coerce").fillna(-1).astype(int)
        loaded[name] = frame.set_index("case_id").sort_index()
    if len(loaded) < 2:
        raise ValueError("At least two prediction files are required for paired comparisons")
    reference_name = next(iter(loaded))
    reference = loaded[reference_name]
    for name, frame in loaded.items():
        if not reference.index.equals(frame.index):
            raise ValueError(f"Case IDs differ between {reference_name} and {name}")
        if not np.array_equal(reference["true_diag1"], frame["true_diag1"]):
            raise ValueError(f"Gold labels differ between {reference_name} and {name}")
    return loaded


def aggregate_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.prediction)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = list(predictions)
    reference = predictions[names[0]]
    y_true = reference["true_diag1"].to_numpy()
    labels = np.array(sorted(np.unique(y_true)))

    summary_rows = []
    per_category_rows = []
    for name, frame in predictions.items():
        y_pred = frame["pred_diag1"].to_numpy()
        summary_rows.append({"condition": name, **aggregate_metrics(y_true, y_pred, labels)})
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=None, zero_division=0,
        )
        for code, p, r, score, count in zip(labels, precision, recall, f1, support):
            per_category_rows.append({
                "condition": name, "diagnosis_code": int(code),
                "precision": float(p), "recall": float(r), "f1": float(score),
                "support": int(count),
            })

    rng = np.random.default_rng(args.seed)
    bootstrap = {name: {metric: [] for metric in ("accuracy", "macro_precision", "macro_recall", "macro_f1")} for name in names}
    category_indices = {label: np.flatnonzero(y_true == label) for label in labels}
    for _ in range(args.bootstrap_resamples):
        sampled = np.concatenate([
            rng.choice(indices, size=len(indices), replace=True)
            for indices in category_indices.values()
        ])
        sampled_true = y_true[sampled]
        for name, frame in predictions.items():
            sampled_pred = frame["pred_diag1"].to_numpy()[sampled]
            values = aggregate_metrics(sampled_true, sampled_pred, labels)
            for metric, value in values.items():
                bootstrap[name][metric].append(value)

    ci_rows = []
    for name in names:
        for metric, values in bootstrap[name].items():
            low, high = np.percentile(values, [2.5, 97.5])
            ci_rows.append({
                "condition": name, "metric": metric,
                "estimate": next(row[metric] for row in summary_rows if row["condition"] == name),
                "ci_low": float(low), "ci_high": float(high),
                "resamples": args.bootstrap_resamples,
            })

    mcnemar_rows = []
    for first, second in itertools.combinations(names, 2):
        first_correct = predictions[first]["pred_diag1"].to_numpy() == y_true
        second_correct = predictions[second]["pred_diag1"].to_numpy() == y_true
        first_only = int(np.sum(first_correct & ~second_correct))
        second_only = int(np.sum(~first_correct & second_correct))
        discordant = first_only + second_only
        p_value = 1.0 if discordant == 0 else float(
            binomtest(first_only, discordant, p=0.5, alternative="two-sided").pvalue
        )
        mcnemar_rows.append({
            "condition_a": first, "condition_b": second,
            "a_correct_b_wrong": first_only,
            "a_wrong_b_correct": second_only,
            "discordant_pairs": discordant,
            "exact_p_value": p_value,
        })
    adjusted = holm_adjust([row["exact_p_value"] for row in mcnemar_rows])
    for row, value in zip(mcnemar_rows, adjusted):
        row["holm_adjusted_p_value"] = value

    pd.DataFrame(summary_rows).to_csv(args.output_dir / "model_summary.csv", index=False)
    pd.DataFrame(per_category_rows).to_csv(args.output_dir / "per_category_metrics.csv", index=False)
    pd.DataFrame(ci_rows).to_csv(args.output_dir / "bootstrap_95ci.csv", index=False)
    pd.DataFrame(mcnemar_rows).to_csv(args.output_dir / "mcnemar_all_pairs.csv", index=False)
    metadata = {
        "conditions": names,
        "n_cases": int(len(y_true)),
        "diagnosis_codes": labels.tolist(),
        "bootstrap_resamples": args.bootstrap_resamples,
        "bootstrap_method": "paired stratified case resampling with replacement; percentile interval",
        "mcnemar_method": "exact two-sided binomial test with Holm family-wise correction",
        "seed": args.seed,
    }
    (args.output_dir / "evaluation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8",
    )
    print(f"Wrote evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
