"""Exploratory secondary-diagnosis analysis on the locked 2025 test outputs.

This analysis is intentionally separate from the confirmatory primary-diagnosis
results. It aligns every model's saved case-level output to the same test rows,
evaluates Diagnosis 2 only where a gold secondary diagnosis is available, and
writes aggregate artifacts without copying narratives into the report.
"""

from __future__ import annotations

import csv
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
TEST_PATH = ROOT / "data" / "neiss_temporal_2021_2025" / "locked_test.csv"
RUNS_DIR = ROOT / "methods_runs"
OUTPUT_DIR = RUNS_DIR / "secondary_diagnosis_analysis"
BOOTSTRAP_RESAMPLES = 1_000
BOOTSTRAP_SEED = 20260721

MODEL_ORDER = [
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

DISPLAY_NAMES = {
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

MISSING_STRINGS = {"", "none", "null", "nan", "na", "n/a"}
LEADING_CODE = re.compile(r"^\s*(\d{2})(?:\.0+)?(?:\s*(?:\(|-|$))", re.IGNORECASE)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def normalize_code(value: object) -> int | None:
    """Return a two-digit diagnosis code or None for missing/invalid output."""
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in MISSING_STRINGS:
        return None
    match = LEADING_CODE.match(text)
    return int(match.group(1)) if match else None


def raw_present(value: object) -> bool:
    return value is not None and str(value).strip().lower() not in MISSING_STRINGS


def safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def macro_scores(gold: list[int], predicted: list[int | None]) -> tuple[float, float]:
    labels = sorted(set(gold))
    recalls: list[float] = []
    f1_scores: list[float] = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        recalls.append(recall)
        f1_scores.append(f1)
    return mean(recalls), mean(f1_scores)


def bootstrap_primary_secondary(
    rows: list[dict[str, object]], seed: int
) -> dict[str, float]:
    """Paired bootstrap, stratified by the gold secondary diagnosis code."""
    strata: dict[int, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        strata[int(row["true_diag2"])].append(index)
    rng = random.Random(seed)
    primary_values: list[float] = []
    secondary_values: list[float] = []
    differences: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled: list[int] = []
        for indices in strata.values():
            sampled.extend(rng.choice(indices) for _ in indices)
        primary = mean(
            [float(rows[i]["pred_diag1"] == rows[i]["true_diag1"]) for i in sampled]
        )
        secondary = mean(
            [float(rows[i]["pred_diag2"] == rows[i]["true_diag2"]) for i in sampled]
        )
        primary_values.append(primary)
        secondary_values.append(secondary)
        differences.append(secondary - primary)
    return {
        "primary_subset_ci_low": percentile(primary_values, 0.025),
        "primary_subset_ci_high": percentile(primary_values, 0.975),
        "secondary_exact_ci_low": percentile(secondary_values, 0.025),
        "secondary_exact_ci_high": percentile(secondary_values, 0.975),
        "secondary_minus_primary_ci_low": percentile(differences, 0.025),
        "secondary_minus_primary_ci_high": percentile(differences, 0.975),
    }


def summarize_model(
    model_key: str,
    source_rows: list[dict[str, str]],
    test_by_id: dict[str, dict[str, str]],
    seed: int,
) -> tuple[dict[str, object], Counter[str], dict[str, float]]:
    if len(source_rows) != len(test_by_id):
        raise ValueError(f"{model_key}: expected {len(test_by_id)} rows, found {len(source_rows)}")
    if len({row["case_id"] for row in source_rows}) != len(source_rows):
        raise ValueError(f"{model_key}: duplicate case IDs")
    if {row["case_id"] for row in source_rows} != set(test_by_id):
        raise ValueError(f"{model_key}: case IDs do not match the locked test")

    rows: list[dict[str, object]] = []
    for source in source_rows:
        case_id = source["case_id"]
        test = test_by_id[case_id]
        true_diag1 = normalize_code(test["diag1"])
        true_diag2 = normalize_code(test["diag2"])
        if normalize_code(source.get("true_diag1")) != true_diag1:
            raise ValueError(f"{model_key}: primary gold mismatch for case {case_id}")
        if normalize_code(source.get("true_diag2")) != true_diag2:
            raise ValueError(f"{model_key}: secondary gold mismatch for case {case_id}")
        rows.append(
            {
                "case_id": case_id,
                "narrative": test["narrative"],
                "true_diag1": true_diag1,
                "true_diag2": true_diag2,
                "pred_diag1": normalize_code(source.get("pred_diag1")),
                "pred_diag2": normalize_code(source.get("pred_diag2")),
                "pred_diag2_raw_present": raw_present(source.get("pred_diag2")),
            }
        )

    eligible = [row for row in rows if row["true_diag2"] is not None]
    distinct = [row for row in eligible if row["true_diag2"] != row["true_diag1"]]
    no_secondary = [row for row in rows if row["true_diag2"] is None]
    predicted_secondary = [row for row in rows if row["pred_diag2"] is not None]
    true_positive_presence = sum(
        row["true_diag2"] is not None and row["pred_diag2"] is not None for row in rows
    )
    presence_precision = safe_div(true_positive_presence, len(predicted_secondary))
    presence_recall = safe_div(true_positive_presence, len(eligible))
    presence_f1 = safe_div(2 * presence_precision * presence_recall, presence_precision + presence_recall)

    primary_all = mean(
        [float(row["pred_diag1"] == row["true_diag1"]) for row in rows]
    )
    primary_subset = mean(
        [float(row["pred_diag1"] == row["true_diag1"]) for row in eligible]
    )
    secondary_exact = mean(
        [float(row["pred_diag2"] == row["true_diag2"]) for row in eligible]
    )
    secondary_exact_distinct = mean(
        [float(row["pred_diag2"] == row["true_diag2"]) for row in distinct]
    )
    secondary_any_position_distinct = mean(
        [
            float(row["true_diag2"] in (row["pred_diag1"], row["pred_diag2"]))
            for row in distinct
        ]
    )
    ordered_pair = mean(
        [
            float(
                row["pred_diag1"] == row["true_diag1"]
                and row["pred_diag2"] == row["true_diag2"]
            )
            for row in eligible
        ]
    )
    unordered_pair = mean(
        [
            float(
                row["pred_diag2"] is not None
                and Counter((row["pred_diag1"], row["pred_diag2"]))
                == Counter((row["true_diag1"], row["true_diag2"]))
            )
            for row in eligible
        ]
    )
    gold_secondary = [int(row["true_diag2"]) for row in eligible]
    pred_secondary = [row["pred_diag2"] for row in eligible]
    secondary_macro_recall, secondary_macro_f1 = macro_scores(gold_secondary, pred_secondary)

    error_types: Counter[str] = Counter()
    for row in eligible:
        p1, p2 = row["pred_diag1"], row["pred_diag2"]
        g1, g2 = row["true_diag1"], row["true_diag2"]
        if p1 == g1 and p2 == g2:
            error_types["ordered_pair_correct"] += 1
        elif g1 != g2 and p1 == g2 and p2 == g1:
            error_types["swapped_pair"] += 1
        elif g1 != g2 and p1 == g2:
            error_types["secondary_promoted_to_primary"] += 1
        elif p2 is None:
            error_types["secondary_omitted"] += 1
        elif p2 == g2:
            error_types["secondary_correct_primary_wrong"] += 1
        else:
            error_types["wrong_secondary_code"] += 1

    invalid_nonempty = sum(
        row["pred_diag2_raw_present"] and row["pred_diag2"] is None for row in rows
    )
    false_positive_rate = mean(
        [float(row["pred_diag2"] is not None) for row in no_secondary]
    )
    summary: dict[str, object] = {
        "model_key": model_key,
        "model": DISPLAY_NAMES.get(model_key, model_key),
        "n_all": len(rows),
        "n_gold_secondary": len(eligible),
        "n_gold_distinct_secondary": len(distinct),
        "primary_accuracy_all": primary_all,
        "primary_accuracy_secondary_subset": primary_subset,
        "secondary_exact_accuracy": secondary_exact,
        "secondary_exact_accuracy_distinct": secondary_exact_distinct,
        "secondary_any_position_accuracy_distinct": secondary_any_position_distinct,
        "secondary_macro_recall": secondary_macro_recall,
        "secondary_macro_f1": secondary_macro_f1,
        "ordered_pair_accuracy": ordered_pair,
        "unordered_pair_accuracy": unordered_pair,
        "secondary_presence_precision": presence_precision,
        "secondary_presence_recall": presence_recall,
        "secondary_presence_f1": presence_f1,
        "secondary_false_positive_rate_when_absent": false_positive_rate,
        "invalid_nonempty_secondary_outputs": invalid_nonempty,
        "secondary_minus_primary_same_subset": secondary_exact - primary_subset,
    }
    summary.update(bootstrap_primary_secondary(eligible, seed))

    same_code_rows = [row for row in eligible if row["true_diag2"] == row["true_diag1"]]
    code71_rows = [row for row in eligible if row["true_diag2"] == 71]
    non_code71_rows = [row for row in eligible if row["true_diag2"] != 71]
    semicolon_rows = [row for row in eligible if ";" in str(row["narrative"])]
    no_semicolon_rows = [row for row in eligible if row not in semicolon_rows]

    def subgroup_exact(subgroup: list[dict[str, object]]) -> float:
        return mean(
            [float(row["pred_diag2"] == row["true_diag2"]) for row in subgroup]
        )

    narrative_metrics = {
        "same_code_n": len(same_code_rows),
        "same_code_secondary_exact": subgroup_exact(same_code_rows),
        "distinct_code_n": len(distinct),
        "distinct_code_secondary_exact": subgroup_exact(distinct),
        "code71_n": len(code71_rows),
        "code71_secondary_exact": subgroup_exact(code71_rows),
        "non_code71_n": len(non_code71_rows),
        "non_code71_secondary_exact": subgroup_exact(non_code71_rows),
        "semicolon_n": len(semicolon_rows),
        "semicolon_secondary_exact": subgroup_exact(semicolon_rows),
        "no_semicolon_n": len(no_semicolon_rows),
        "no_semicolon_secondary_exact": subgroup_exact(no_semicolon_rows),
    }
    return summary, error_types, narrative_metrics


def format_ci(point: float, low: float, high: float) -> str:
    return f"{point:.3f} [{low:.3f}, {high:.3f}]"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    test_rows = read_csv(TEST_PATH)
    test_by_id = {row["case_id"]: row for row in test_rows}
    if len(test_by_id) != len(test_rows):
        raise ValueError("Locked test contains duplicate case IDs")

    available = [
        key for key in MODEL_ORDER if (RUNS_DIR / key / "test_results.csv").exists()
    ]
    if available != MODEL_ORDER:
        missing = sorted(set(MODEL_ORDER) - set(available))
        raise FileNotFoundError(f"Missing expected model outputs: {missing}")

    summaries: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    narrative_rows: list[dict[str, object]] = []
    for model_index, model_key in enumerate(MODEL_ORDER):
        source_rows = read_csv(RUNS_DIR / model_key / "test_results.csv")
        summary, error_types, narrative_metrics = summarize_model(
            model_key,
            source_rows,
            test_by_id,
            BOOTSTRAP_SEED + model_index,
        )
        summaries.append(summary)
        for error_type, count in sorted(error_types.items()):
            error_rows.append(
                {
                    "model_key": model_key,
                    "model": DISPLAY_NAMES[model_key],
                    "error_type": error_type,
                    "count": count,
                    "proportion_of_gold_secondary": count / int(summary["n_gold_secondary"]),
                }
            )
        narrative_rows.append(
            {
                "model_key": model_key,
                "model": DISPLAY_NAMES[model_key],
                **narrative_metrics,
            }
        )

    write_csv(OUTPUT_DIR / "secondary_diagnosis_summary.csv", summaries)
    write_csv(OUTPUT_DIR / "secondary_error_types.csv", error_rows)
    write_csv(OUTPUT_DIR / "secondary_narrative_marker_results.csv", narrative_rows)

    eligible_test = [row for row in test_rows if normalize_code(row["diag2"]) is not None]
    secondary_distribution = Counter(normalize_code(row["diag2"]) for row in eligible_test)
    same_code_n = sum(
        normalize_code(row["diag1"]) == normalize_code(row["diag2"])
        for row in eligible_test
    )
    code71_n = secondary_distribution[71]

    report: list[str] = [
        "# Exploratory Secondary-Diagnosis Analysis",
        "",
        "## Scope and status",
        "",
        "This is a **post hoc exploratory analysis** of the already-evaluated locked 2025 test set. "
        "It must not replace or be described as part of the prespecified primary outcome. The main "
        "paper result remains 30-class `Diagnosis_1_Code` performance.",
        "",
        f"The locked test contains {len(test_rows):,} cases. A gold `Diagnosis_2_Code` is available "
        f"for {len(eligible_test):,} cases ({len(eligible_test) / len(test_rows):.1%}). Of these, "
        f"{same_code_n} repeat the primary diagnosis code and {code71_n} use code 71. The remaining "
        f"{len(eligible_test) - same_code_n} cases contain a diagnosis code distinct from the primary code.",
        "",
        "## Definitions",
        "",
        "- **Primary—all:** primary-diagnosis accuracy on all 1,500 cases.",
        "- **Primary—secondary subset:** primary accuracy on the same 332 cases used for secondary evaluation.",
        "- **Secondary exact:** positional accuracy of the predicted secondary code among the 332 cases with a gold secondary code; omission is incorrect.",
        "- **Secondary macro F1:** unweighted F1 across the 27 gold secondary-code categories; supports range from 1 to 59, so this estimate is unstable for rare codes.",
        "- **Both ordered:** both primary and secondary codes must be correct in their assigned slots.",
        "- **Presence recall:** proportion of gold-secondary cases for which a valid two-digit secondary code was emitted.",
        "- **False-positive rate:** proportion of the 1,168 cases without a gold secondary code for which a valid secondary code was emitted.",
        "",
        "Predicted strings such as `59 (Laceration)` are normalized to code 59. Missing values, "
        "ICD codes, and malformed numeric outputs are treated as invalid/missing predictions.",
        "",
        "## Model comparison",
        "",
        "| Model | Primary—all | Primary—secondary subset (95% CI) | Secondary exact (95% CI) | Secondary macro F1 | Both ordered | Presence recall | False-positive rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        report.append(
            "| {model} | {pall:.3f} | {psub} | {sexact} | {mf1:.3f} | {pair:.3f} | {recall:.3f} | {fpr:.3f} |".format(
                model=row["model"],
                pall=row["primary_accuracy_all"],
                psub=format_ci(
                    row["primary_accuracy_secondary_subset"],
                    row["primary_subset_ci_low"],
                    row["primary_subset_ci_high"],
                ),
                sexact=format_ci(
                    row["secondary_exact_accuracy"],
                    row["secondary_exact_ci_low"],
                    row["secondary_exact_ci_high"],
                ),
                mf1=row["secondary_macro_f1"],
                pair=row["ordered_pair_accuracy"],
                recall=row["secondary_presence_recall"],
                fpr=row["secondary_false_positive_rate_when_absent"],
            )
        )

    by_key = {str(row["model_key"]): row for row in summaries}
    report.extend(
        [
            "",
            "## Main observations",
            "",
        ]
    )
    for key in ("qwen_sft_6000", "mistral_sft_3000", "gpt54_10shot"):
        row = by_key[key]
        report.append(
            f"- **{row['model']}:** primary accuracy on the 332-case subset was "
            f"{row['primary_accuracy_secondary_subset']:.3f}, versus positional secondary accuracy "
            f"{row['secondary_exact_accuracy']:.3f}; the paired difference was "
            f"{row['secondary_minus_primary_same_subset']:.3f} "
            f"[{row['secondary_minus_primary_ci_low']:.3f}, "
            f"{row['secondary_minus_primary_ci_high']:.3f}]."
        )
    report.extend(
        [
            "- Primary performance is lower on multi-diagnosis cases than on the complete test set for the strongest SFT models, indicating that competing injuries form a harder subset.",
            "- Secondary exact accuracy is lower than primary accuracy even when both are compared on the same 332 narratives. Common failure modes are omission of Diagnosis 2 and promotion of the secondary diagnosis into the primary slot.",
            "- Same-code secondary records often describe the same injury type at a second body part. A model that emits only one diagnosis code is counted wrong positionally even when its primary diagnosis category is correct.",
            "- Code 71 accounts for a substantial share of secondary labels and often accompanies symptoms, illnesses, or incompletely specified additional conditions. These cases require careful review against the NEISS manual before making a strong secondary-coding claim.",
            "",
            "### Secondary-label and narrative subgroups",
            "",
            "| Model | Repeated code (n=78) | Distinct code (n=254) | Code 71 (n=59) | Other codes (n=273) | Semicolon present (n=83) | No semicolon (n=249) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    narrative_by_key = {str(row["model_key"]): row for row in narrative_rows}
    for key in ("qwen_sft_6000", "mistral_sft_3000", "gpt54_10shot"):
        row = narrative_by_key[key]
        report.append(
            f"| {row['model']} | {row['same_code_secondary_exact']:.3f} | "
            f"{row['distinct_code_secondary_exact']:.3f} | "
            f"{row['code71_secondary_exact']:.3f} | "
            f"{row['non_code71_secondary_exact']:.3f} | "
            f"{row['semicolon_secondary_exact']:.3f} | "
            f"{row['no_semicolon_secondary_exact']:.3f} |"
        )
    report.extend(
        [
            "",
            "## Narrative audit",
            "",
            "A targeted read of failures from the strongest SFT outputs identified the following recurring patterns:",
            "",
            "1. **Secondary omission:** the model returns the salient primary injury but leaves the optional secondary field blank, including narratives that list two diagnoses after the `DX` marker.",
            "2. **Priority reversal:** the model identifies both diagnoses but swaps their primary/secondary order, especially when ordinary clinical severity appears to conflict with the recorded NEISS ordering.",
            "3. **Duplicate diagnosis categories:** two injuries at different body parts share one diagnosis code; models frequently emit the code once rather than duplicate it in Diagnosis 2.",
            "4. **Code 71 ambiguity:** additional symptoms or non-injury conditions are sometimes represented by code 71 in the source record, while the prompt instructs models not to code unrelated illnesses. This creates a potential task-definition/label-alignment issue.",
            "5. **Mechanism/injury combinations:** electrical burn plus shock, poisoning plus internal injury, and head injury plus fracture require retaining two related but distinct codes rather than collapsing to one.",
            "",
            "These qualitative categories are a targeted exploratory audit, not a completed blinded human error annotation. Counts suitable for publication require a prespecified review form and independent adjudication.",
            "",
            "## Publication guidance",
            "",
            "- Keep primary diagnosis as the only confirmatory headline outcome.",
            "- Present this secondary analysis in the analysis section or appendix and label it post hoc/exploratory.",
            "- Report the 332-case denominator and distinguish all secondary cases from the 254 distinct-secondary cases.",
            "- Before claiming complete NEISS coding, independently verify secondary labels—especially duplicate codes and code 71—and evaluate secondary body-part pairing.",
            "- Do not use these locked-test findings to redesign the prompt and then re-report performance on the same test set.",
            "",
            "## Reproducibility",
            "",
            f"All 14 saved prediction files were aligned by `case_id` to `{TEST_PATH.relative_to(ROOT)}`. "
            f"Confidence intervals use {BOOTSTRAP_RESAMPLES:,} paired resamples stratified by gold "
            f"secondary diagnosis code with seed `{BOOTSTRAP_SEED}`. Numeric results are available "
            "in `secondary_diagnosis_summary.csv`; error-type and narrative-marker summaries are in "
            "the accompanying CSV files.",
            "",
        ]
    )
    (OUTPUT_DIR / "SECONDARY_DIAGNOSIS_ANALYSIS.md").write_text(
        "\n".join(report), encoding="utf-8"
    )

    print(f"Validated {len(test_rows)} aligned cases across {len(summaries)} models.")
    print(f"Gold secondary diagnoses: {len(eligible_test)}; distinct: {len(eligible_test) - same_code_n}.")
    print(f"Wrote analysis to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
