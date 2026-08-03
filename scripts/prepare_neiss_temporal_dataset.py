"""Build reproducible, category-balanced temporal splits from annual NEISS files.

Design (defaults):
  * train:      2021-2023, 200 cases x 30 diagnosis categories = 6,000
  * validation: 2024,       10 cases x 30 diagnosis categories =   300
  * test:       2025,       50 cases x 30 diagnosis categories = 1,500

The test files are accompanied by SHA-256 checksums. Once an output directory has
been created, this script will verify it rather than silently resample it. Use a
new output directory for a new experimental split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DIAGNOSIS_LABELS = {
    41: "Ingestion", 42: "Aspiration", 46: "Electrical burn",
    47: "Burn, not specified", 48: "Scald burn", 49: "Chemical burn",
    50: "Amputation", 51: "Thermal burn", 52: "Concussion",
    53: "Contusion/abrasion", 54: "Crushing injury", 55: "Dislocation",
    56: "Foreign body", 57: "Fracture", 58: "Hematoma",
    59: "Laceration", 60: "Dental injury", 61: "Nerve damage",
    62: "Internal injury", 63: "Puncture", 64: "Strain/sprain",
    65: "Anoxia", 66: "Hemorrhage", 67: "Electric shock",
    68: "Poisoning", 69: "Submersion", 70: "Other",
    71: "Unknown/not stated", 72: "Avulsion", 73: "Radiation injury",
    74: "Dermatitis/conjunctivitis",
}

REQUIRED_COLUMNS = [
    "CPSC_Case_Number", "Treatment_Date", "Body_Part", "Diagnosis",
    "Other_Diagnosis", "Body_Part_2", "Diagnosis_2",
    "Other_Diagnosis_2", "Narrative_1",
]

INSTRUCTION = (
    "Read the emergency-department injury narrative. Assign the primary NEISS "
    "diagnosis code and body-part code, include the secondary codes when present, "
    "and give a one-sentence rationale for the primary diagnosis."
)

ABLATION_SIZES = (1000, 3000, 6000)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path,
        default=Path(os.environ.get("NEISS_SOURCE_DIR", "NEISS_10353")),
        help="Directory containing annual NEISS source files (or set NEISS_SOURCE_DIR).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "neiss_temporal_2021_2025",
    )
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--categories", type=int, default=30, choices=range(20, 31))
    parser.add_argument("--train-per-category", type=int, default=200)
    parser.add_argument("--validation-per-category", type=int, default=10)
    parser.add_argument("--test-per-category", type=int, default=50)
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Verify an existing dataset and its locked test checksums.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_narrative(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip()).upper()


def read_year(source_dir: Path, year: int) -> pd.DataFrame:
    path = source_dir / f"NEISS_{year}.XLSX"
    if not path.is_file():
        raise FileNotFoundError(f"Missing annual source file: {path}")
    frame = pd.read_excel(path, usecols=REQUIRED_COLUMNS)
    frame = frame.rename(columns={
        "CPSC_Case_Number": "case_id", "Treatment_Date": "treatment_date",
        "Body_Part": "body1", "Diagnosis": "diag1",
        "Other_Diagnosis": "other_diagnosis", "Body_Part_2": "body2",
        "Diagnosis_2": "diag2", "Other_Diagnosis_2": "other_diagnosis_2",
        "Narrative_1": "narrative",
    })
    frame = frame.dropna(subset=["case_id", "diag1", "body1", "narrative"]).copy()
    frame["year"] = year
    frame["case_id"] = frame["case_id"].astype("int64")
    frame["diag1"] = frame["diag1"].astype("int64")
    frame["body1"] = frame["body1"].astype("int64")
    frame["narrative"] = frame["narrative"].map(normalized_narrative)
    frame = frame[frame["narrative"].str.len().gt(0)]
    frame = frame.drop_duplicates(subset=["case_id"], keep="first")
    return frame


def balanced_sample(
    frame: pd.DataFrame,
    codes: list[int],
    per_category: int,
    seed: int,
    balance_years: bool = False,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for code in codes:
        category = frame[frame["diag1"].eq(code)]
        if balance_years:
            years = sorted(category["year"].unique())
            available = category["year"].value_counts().reindex(years).astype(int)
            exact = available / available.sum() * per_category
            allocated = np.floor(exact).astype(int)
            remainder = per_category - int(allocated.sum())
            for year in (exact - allocated).sort_values(ascending=False).index[:remainder]:
                allocated.loc[year] += 1
            allocations = allocated.to_dict()
            for year, count in allocations.items():
                stratum = category[category["year"].eq(year)]
                if len(stratum) < count:
                    raise ValueError(f"Code {code}, year {year}: need {count}, found {len(stratum)}")
                pieces.append(stratum.sample(n=count, random_state=seed + code * 101 + year))
        else:
            if len(category) < per_category:
                raise ValueError(f"Code {code}: need {per_category}, found {len(category)}")
            pieces.append(category.sample(n=per_category, random_state=seed + code * 101))
    result = pd.concat(pieces, ignore_index=True)
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


def capped_balanced_allocations(
    frame: pd.DataFrame, codes: list[int], total: int,
) -> dict[int, int]:
    """Allocate a total nearly equally, capping categories at available rows."""
    available = frame["diag1"].value_counts().reindex(codes, fill_value=0).astype(int).to_dict()
    allocations = {code: 0 for code in codes}
    for _ in range(total):
        eligible = [code for code in codes if allocations[code] < available[code]]
        if not eligible:
            raise ValueError(f"Only {sum(available.values())} rows are available for a {total}-row ablation")
        code = min(eligible, key=lambda item: (allocations[item], item))
        allocations[code] += 1
    return allocations


def build_nested_ablations(
    train_pool: pd.DataFrame, codes: list[int], seed: int,
) -> dict[int, pd.DataFrame]:
    """Create deterministic nested category-stratified learning-curve subsets."""
    ordered = {}
    for code in codes:
        category = train_pool[train_pool["diag1"].eq(code)]
        ordered[code] = category.sample(frac=1, random_state=seed + code * 1009).reset_index(drop=True)

    subsets = {}
    previous_ids: set[int] = set()
    for size in ABLATION_SIZES:
        allocations = capped_balanced_allocations(train_pool, codes, size)
        pieces = [ordered[code].iloc[:count] for code, count in allocations.items()]
        subset = pd.concat(pieces, ignore_index=True).sample(frac=1, random_state=seed + size)
        subset = subset.reset_index(drop=True)
        current_ids = set(subset["case_id"])
        if not previous_ids.issubset(current_ids):
            raise AssertionError(f"The {size}-row ablation is not nested")
        if set(subset["year"].unique()) != {2021, 2022, 2023}:
            raise AssertionError(f"The {size}-row ablation does not cover every training year")
        previous_ids = current_ids
        subsets[size] = subset
    return subsets


def nullable_int(value: object) -> int | None:
    return None if pd.isna(value) else int(value)


def rationale_for(row: pd.Series) -> str:
    code = int(row["diag1"])
    label = DIAGNOSIS_LABELS.get(code, f"diagnosis {code}")
    detail = row.get("other_diagnosis")
    if pd.notna(detail) and str(detail).strip():
        detail_text = re.sub(r"[.!?]+", ",", normalized_narrative(detail)).strip(" ,")
        return f'The documented diagnosis "{detail_text}" supports NEISS code {code} ({label}).'
    return f"The documented injury is consistent with {label}, supporting NEISS diagnosis code {code}."


def add_instruction_fields(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["diagnosis_category"] = result["diag1"].map(DIAGNOSIS_LABELS)
    result["rationale"] = result.apply(rationale_for, axis=1)
    result["instruction"] = INSTRUCTION

    def response(row: pd.Series) -> str:
        payload = {
            "Diagnosis_1_Code": int(row["diag1"]),
            "Diagnosis_1_Label": row["diagnosis_category"],
            "Body_Part_1_Code": int(row["body1"]),
            "Diagnosis_2_Code": nullable_int(row["diag2"]),
            "Body_Part_2_Code": nullable_int(row["body2"]),
            "Rationale": row["rationale"],
        }
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    result["response"] = result.apply(response, axis=1)
    result["text"] = result.apply(
        lambda row: (
            f"<s>[INST] {row['instruction']}\n\n"
            f"Injury Narrative: {row['narrative']} [/INST] {row['response']} </s>"
        ),
        axis=1,
    )
    return result


def write_jsonl(frame: pd.DataFrame, path: Path, columns: Iterable[str]) -> None:
    def json_value(value: object) -> object:
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if isinstance(value, np.generic):
            return value.item()
        return value

    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in frame[list(columns)].to_dict(orient="records"):
            clean = {key: json_value(value) for key, value in record.items()}
            stream.write(json.dumps(clean, ensure_ascii=False) + "\n")


def verify_existing(output_dir: Path) -> dict:
    manifest_path = output_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No dataset manifest found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for filename, expected in manifest["output_sha256"].items():
        path = output_dir / filename
        actual = sha256(path) if path.is_file() else "MISSING"
        if actual != expected:
            failures.append(f"{filename}: expected {expected}, got {actual}")
    if failures:
        raise RuntimeError("Dataset verification failed:\n" + "\n".join(failures))
    return manifest


def assert_split(split: pd.DataFrame, year_set: set[int], codes: list[int], count: int) -> None:
    if set(split["year"].unique()) != year_set:
        raise AssertionError(f"Unexpected years: {sorted(split['year'].unique())}")
    observed = split["diag1"].value_counts().to_dict()
    expected = {code: count for code in codes}
    if observed != expected:
        raise AssertionError(f"Category imbalance: {observed}")
    if split["case_id"].duplicated().any():
        raise AssertionError("Duplicate case IDs found within a split")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.verify_only or (output_dir / "dataset_manifest.json").exists():
        manifest = verify_existing(output_dir)
        print(f"Verified existing dataset at {output_dir}")
        print(json.dumps(manifest["split_counts"], indent=2))
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    annual = {year: read_year(args.source_dir, year) for year in range(2021, 2026)}
    train_pool = pd.concat([annual[year] for year in (2021, 2022, 2023)], ignore_index=True)
    validation_pool = annual[2024]
    test_pool = annual[2025]
    heldout_narratives = set(validation_pool["narrative"]) | set(test_pool["narrative"])
    train_pool = train_pool[~train_pool["narrative"].isin(heldout_narratives)].copy()

    eligible = []
    for code in DIAGNOSIS_LABELS:
        if (
            (train_pool["diag1"].eq(code).sum() >= args.train_per_category)
            and (validation_pool["diag1"].eq(code).sum() >= args.validation_per_category)
            and (test_pool["diag1"].eq(code).sum() >= args.test_per_category)
        ):
            eligible.append(code)
    prevalence = test_pool[test_pool["diag1"].isin(eligible)]["diag1"].value_counts()
    codes = [int(code) for code in prevalence.index[: args.categories]]
    if len(codes) != args.categories:
        raise ValueError(f"Only {len(codes)} categories meet the requested minimum counts")
    codes.sort()

    train = balanced_sample(
        train_pool, codes, args.train_per_category, args.seed, balance_years=True,
    )
    validation = balanced_sample(
        validation_pool, codes, args.validation_per_category, args.seed + 1,
    )
    test = balanced_sample(test_pool, codes, args.test_per_category, args.seed + 2)
    # Build learning-curve subsets from the exact 6K main training set so the
    # 1K and 3K conditions are nested subsets and 6K is the full training set.
    ablations = build_nested_ablations(train, codes, args.seed + 10)

    assert_split(train, {2021, 2022, 2023}, codes, args.train_per_category)
    assert_split(validation, {2024}, codes, args.validation_per_category)
    assert_split(test, {2025}, codes, args.test_per_category)
    case_sets = [set(frame["case_id"]) for frame in (train, validation, test)]
    if any(case_sets[i] & case_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise AssertionError("Case leakage detected across temporal splits")

    train = add_instruction_fields(train)
    validation = add_instruction_fields(validation)
    test = add_instruction_fields(test)
    ablations = {size: add_instruction_fields(frame) for size, frame in ablations.items()}
    export_columns = [
        "case_id", "treatment_date", "year", "narrative", "diag1",
        "diagnosis_category", "body1", "diag2", "body2", "other_diagnosis",
        "other_diagnosis_2", "rationale", "instruction", "response", "text",
    ]
    csv_columns = [column for column in export_columns if column != "text"]
    for name, frame in (("train", train), ("validation", validation), ("locked_test", test)):
        frame[csv_columns].to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")
        write_jsonl(frame, output_dir / f"{name}.jsonl", export_columns)

    ablation_dir = output_dir / "ablations"
    ablation_dir.mkdir(exist_ok=True)
    for size, frame in ablations.items():
        frame[csv_columns].to_csv(ablation_dir / f"train_{size}.csv", index=False, encoding="utf-8")
        write_jsonl(frame, ablation_dir / f"train_{size}.jsonl", export_columns)

    review = ablations[6000].sample(n=100, random_state=args.seed + 100).copy()
    review = review[[
        "case_id", "year", "narrative", "diag1", "diagnosis_category", "body1", "rationale",
    ]]
    review["review_code_matches_narrative"] = ""
    review["reviewer_suggested_code"] = ""
    review["reviewer_notes"] = ""
    review.to_csv(output_dir / "human_review_100.csv", index=False, encoding="utf-8")
    (output_dir / "HUMAN_REVIEW_README.md").write_text(
        "# Independent 100-example quality review\n\n"
        "A co-author who did not construct the split should review each row without consulting "
        "model predictions. Enter `yes` or `no` in `review_code_matches_narrative`, provide a "
        "replacement in `reviewer_suggested_code` when needed, and document ambiguity in "
        "`reviewer_notes`. Report the agreement rate as yes decisions divided by 100.\n",
        encoding="utf-8",
    )

    source_hashes = {
        f"NEISS_{year}.XLSX": sha256(args.source_dir / f"NEISS_{year}.XLSX")
        for year in range(2021, 2026)
    }
    output_files = [
        "train.csv", "train.jsonl", "validation.csv", "validation.jsonl",
        "locked_test.csv", "locked_test.jsonl", "human_review_100.csv",
        "HUMAN_REVIEW_README.md",
    ]
    output_files.extend(
        f"ablations/train_{size}.{suffix}"
        for size in ABLATION_SIZES for suffix in ("csv", "jsonl")
    )
    manifest = {
        "design": "category-balanced temporal holdout",
        "seed": args.seed,
        "years": {"train": [2021, 2022, 2023], "validation": [2024], "test": [2025]},
        "selected_diagnosis_codes": codes,
        "selected_diagnosis_labels": {str(code): DIAGNOSIS_LABELS[code] for code in codes},
        "excluded_diagnosis_codes": sorted(set(DIAGNOSIS_LABELS) - set(codes)),
        "per_category": {
            "train": args.train_per_category,
            "validation": args.validation_per_category,
            "test": args.test_per_category,
        },
        "split_counts": {"train": len(train), "validation": len(validation), "test": len(test)},
        "ablation_counts": {str(size): len(frame) for size, frame in ablations.items()},
        "ablation_category_counts": {
            str(size): {str(code): int(count) for code, count in frame["diag1"].value_counts().sort_index().items()}
            for size, frame in ablations.items()
        },
        "human_review_count": len(review),
        "source_sha256": source_hashes,
        "output_sha256": {filename: sha256(output_dir / filename) for filename in output_files},
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )
    (output_dir / "LOCKED_TEST_SET.sha256").write_text(
        "".join(
            f"{manifest['output_sha256'][name]}  {name}\n"
            for name in ("locked_test.csv", "locked_test.jsonl")
        ),
        encoding="ascii",
    )
    verified = verify_existing(output_dir)
    print(f"Created and verified dataset at {output_dir}")
    print(json.dumps(verified["split_counts"], indent=2))
    print(f"Selected diagnosis codes: {codes}")


if __name__ == "__main__":
    main()
