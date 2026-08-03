"""Create a reproducible manual-review sample from SFT diagnosis failures."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260718)
    return parser.parse_args()


def linguistic_flags(narrative: str, has_second_diagnosis: bool) -> str:
    text = str(narrative).upper()
    flags = []
    if any(token in text for token in (" NO ", "DENIES", "NEGATIVE", "WITHOUT")):
        flags.append("negation")
    if any(token in text for token in ("POSSIBLE", "PROBABLE", "RULE OUT", "SUSPECT")):
        flags.append("uncertainty")
    if has_second_diagnosis or any(token in text for token in (" ALSO ", " AND ", "MULTIPLE")):
        flags.append("multiple_findings")
    if not any(token in text for token in (" DX", "DIAGNOS", "ASSESS", "IMPRESSION")):
        flags.append("implicit_diagnosis")
    if len(text.split()) > 45:
        flags.append("long_narrative")
    return ";".join(flags)


def main() -> None:
    args = parse_args()
    test = pd.read_csv(args.test_file)
    required_test = {"case_id", "narrative", "diag1", "diagnosis_category", "diag2"}
    if not required_test.issubset(test.columns):
        raise ValueError(f"Test file lacks {sorted(required_test - set(test.columns))}")
    test = test[["case_id", "year", "narrative", "diag1", "diagnosis_category", "body1", "diag2"]]

    failures = []
    for specification in args.prediction:
        name, raw_path = specification.split("=", 1)
        predictions = pd.read_csv(raw_path)
        required = {"case_id", "true_diag1", "pred_diag1"}
        if not required.issubset(predictions.columns):
            raise ValueError(f"{raw_path} lacks {sorted(required - set(predictions.columns))}")
        merged = test.merge(predictions[list(required)], on="case_id", validate="one_to_one")
        merged = merged[merged["true_diag1"] != merged["pred_diag1"]].copy()
        merged["condition"] = name
        failures.append(merged)
    all_failures = pd.concat(failures, ignore_index=True)
    if all_failures.empty:
        raise ValueError("No SFT failure cases were found")

    all_failures["automatic_linguistic_flags"] = all_failures.apply(
        lambda row: linguistic_flags(row["narrative"], pd.notna(row["diag2"])), axis=1,
    )
    all_failures["error_pair"] = (
        all_failures["true_diag1"].astype("Int64").astype(str)
        + "→"
        + pd.to_numeric(all_failures["pred_diag1"], errors="coerce").astype("Int64").astype(str)
    )

    rng = np.random.default_rng(args.seed)
    all_failures["_priority"] = rng.random(len(all_failures))
    all_failures = all_failures.sort_values(["condition", "true_diag1", "_priority"])
    groups = [list(group.index) for _, group in all_failures.groupby(["condition", "true_diag1"], sort=True)]
    selected_indices = []
    while len(selected_indices) < min(args.sample_size, len(all_failures)):
        added = False
        for queue in groups:
            if queue and len(selected_indices) < args.sample_size:
                selected_indices.append(queue.pop(0))
                added = True
        if not added:
            break
    review = all_failures.loc[selected_indices].drop(columns=["_priority"]).copy()
    review["review_primary_pattern"] = ""
    review["review_secondary_pattern"] = ""
    review["review_error_cause"] = ""
    review["reviewer_notes"] = ""

    args.output_dir.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.output_dir / "sft_failure_review.csv", index=False)
    pair_summary = (
        all_failures.groupby(["condition", "true_diag1", "pred_diag1", "error_pair"], dropna=False)
        .size().rename("count").reset_index().sort_values(["condition", "count"], ascending=[True, False])
    )
    pair_summary.to_csv(args.output_dir / "error_pair_counts.csv", index=False)
    (args.output_dir / "ERROR_ANALYSIS_PROTOCOL.md").write_text(
        "# SFT failure-analysis protocol\n\n"
        "Review each sampled failure without changing the gold label. Assign one primary "
        "linguistic pattern from: `negation`, `uncertainty`, `multiple injuries`, "
        "`implicit diagnosis`, `abbreviation`, `temporal sequence`, `rare terminology`, "
        "or `other`. Optionally assign a secondary pattern. In `review_error_cause`, record "
        "whether the error appears due to narrative ambiguity, competing injuries, coding-rule "
        "confusion, label noise, or model failure. Automated flags are review aids, not findings. "
        "Report pattern counts and representative de-identified examples after adjudication.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(review)} manual-review rows to {args.output_dir}")


if __name__ == "__main__":
    main()
