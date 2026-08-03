"""Train and evaluate a validation-selected TF-IDF logistic-regression baseline.

The tune stage reads only the 2021-2023 training set and 2024 validation set.
The final-eval stage requires explicit confirmation before reading the locked
2025 test set. The modeled outcome is primary diagnosis code only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import version as package_version
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "neiss_temporal_2021_2025"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "methods_runs" / "tfidf_logreg_6000"
SEED = 20260718

# Prespecified before the baseline test evaluation. The grid is intentionally
# small: word unigram versus unigram+bigram features, crossed with three values
# of inverse L2 regularization strength. Macro F1 on 2024 is the selection metric.
NGRAM_RANGES = ((1, 1), (1, 2))
C_VALUES = (0.25, 1.0, 4.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("tune", "final-eval"))
    parser.add_argument(
        "--train-file",
        type=Path,
        default=DATA_DIR / "ablations" / "train_6000.csv",
    )
    parser.add_argument(
        "--validation-file", type=Path, default=DATA_DIR / "validation.csv"
    )
    parser.add_argument(
        "--test-file", type=Path, default=DATA_DIR / "locked_test.csv"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--confirm-locked-test",
        action="store_true",
        help="Required for final-eval; confirms that model selection is frozen.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing outputs for the selected stage.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split(path: Path, expected_years: set[int]) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"case_id", "year", "narrative", "diag1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} lacks required columns: {sorted(missing)}")
    if frame["case_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate case IDs")
    if frame["narrative"].isna().any() or frame["narrative"].astype(str).str.strip().eq("").any():
        raise ValueError(f"{path} contains missing narratives")
    observed_years = set(frame["year"].astype(int).unique())
    if observed_years != expected_years:
        raise ValueError(
            f"{path} has years {sorted(observed_years)}; expected {sorted(expected_years)}"
        )
    frame = frame.copy()
    frame["diag1"] = frame["diag1"].astype(int)
    frame["narrative"] = frame["narrative"].astype(str)
    return frame


def build_pipeline(ngram_range: tuple[int, int], c_value: float, seed: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    analyzer="word",
                    token_pattern=r"(?u)\b\w+\b",
                    ngram_range=ngram_range,
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                    norm="l2",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    solver="lbfgs",
                    max_iter=2_000,
                    random_state=seed,
                ),
            ),
        ]
    )


def validate_design(train: pd.DataFrame, validation: pd.DataFrame) -> list[int]:
    if set(train["case_id"]) & set(validation["case_id"]):
        raise ValueError("Case leakage detected between training and validation")
    train_labels = sorted(train["diag1"].unique().tolist())
    validation_labels = sorted(validation["diag1"].unique().tolist())
    if train_labels != validation_labels:
        raise ValueError("Training and validation diagnosis categories differ")
    if len(train) != 6_000 or len(validation) != 300 or len(train_labels) != 30:
        raise ValueError(
            "Expected the frozen 6,000/300-row, 30-category train/validation design"
        )
    return [int(label) for label in train_labels]


def tune(args: argparse.Namespace) -> None:
    pipeline_path = args.output_dir / "tfidf_logreg_pipeline.joblib"
    selection_path = args.output_dir / "validation_selection.json"
    if not args.overwrite and (pipeline_path.exists() or selection_path.exists()):
        raise FileExistsError("Tune outputs already exist; add --overwrite to replace them")

    train = load_split(args.train_file, {2021, 2022, 2023})
    validation = load_split(args.validation_file, {2024})
    labels = validate_design(train, validation)
    train_text = train["narrative"].tolist()
    train_gold = train["diag1"].tolist()
    validation_text = validation["narrative"].tolist()
    validation_gold = validation["diag1"].tolist()

    trials: list[dict[str, object]] = []
    best_pipeline: Pipeline | None = None
    best_key: tuple[float, float] | None = None
    best_trial: dict[str, object] | None = None
    for ngram_range in NGRAM_RANGES:
        for c_value in C_VALUES:
            started = time.perf_counter()
            pipeline = build_pipeline(ngram_range, c_value, args.seed)
            pipeline.fit(train_text, train_gold)
            predicted = pipeline.predict(validation_text)
            macro_f1 = float(
                f1_score(validation_gold, predicted, labels=labels, average="macro", zero_division=0)
            )
            accuracy = float(accuracy_score(validation_gold, predicted))
            classifier = pipeline.named_steps["classifier"]
            trial = {
                "ngram_range": list(ngram_range),
                "C": c_value,
                "min_df": 2,
                "max_features": 100_000,
                "sublinear_tf": True,
                "validation_macro_f1": macro_f1,
                "validation_accuracy": accuracy,
                "vocabulary_size": len(pipeline.named_steps["tfidf"].vocabulary_),
                "maximum_solver_iterations": int(classifier.n_iter_.max()),
                "fit_seconds": time.perf_counter() - started,
            }
            trials.append(trial)
            # Iteration order supplies the prespecified tie-break: prefer the
            # simpler n-gram range and stronger regularization on exact ties.
            key = (macro_f1, accuracy)
            if best_key is None or key > best_key:
                best_key = key
                best_trial = trial
                best_pipeline = pipeline
            print(
                f"ngram={ngram_range}, C={c_value:g}: "
                f"validation macro F1={macro_f1:.4f}, accuracy={accuracy:.4f}",
                flush=True,
            )

    if best_pipeline is None or best_trial is None:
        raise RuntimeError("No validation trial completed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, pipeline_path, compress=3)
    selection = {
        "condition": "tfidf_logreg_6000",
        "stage": "validation_selection",
        "selection_metric": "macro_f1",
        "tie_break": "validation accuracy, then first prespecified grid entry",
        "selected": best_trial,
        "grid_results": trials,
        "training_examples": len(train),
        "validation_examples": len(validation),
        "diagnosis_codes": labels,
        "seed": args.seed,
        "train_file": str(args.train_file.resolve()),
        "validation_file": str(args.validation_file.resolve()),
        "train_sha256": sha256(args.train_file),
        "validation_sha256": sha256(args.validation_file),
        "pipeline_file": pipeline_path.name,
        "pipeline_sha256": sha256(pipeline_path),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "package_versions": {
            name: package_version(name)
            for name in ("numpy", "pandas", "scikit-learn", "joblib")
        },
    }
    selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"Selected: ngram={tuple(best_trial['ngram_range'])}, C={best_trial['C']}")
    print(f"Saved frozen pipeline to: {pipeline_path}")


def final_eval(args: argparse.Namespace) -> None:
    if not args.confirm_locked_test:
        raise PermissionError(
            "Add --confirm-locked-test only after validation selection is frozen"
        )
    pipeline_path = args.output_dir / "tfidf_logreg_pipeline.joblib"
    selection_path = args.output_dir / "validation_selection.json"
    results_path = args.output_dir / "test_results.csv"
    metrics_path = args.output_dir / "test_metrics.json"
    manifest_path = args.output_dir / "run_manifest_tfidf_logreg.json"
    if not pipeline_path.exists() or not selection_path.exists():
        raise FileNotFoundError("Run --stage tune before final evaluation")
    if not args.overwrite and any(path.exists() for path in (results_path, metrics_path, manifest_path)):
        raise FileExistsError("Final-evaluation outputs already exist; add --overwrite to replace them")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if sha256(pipeline_path) != selection["pipeline_sha256"]:
        raise ValueError("Frozen pipeline checksum does not match validation_selection.json")
    test = load_split(args.test_file, {2025})
    if len(test) != 1_500 or test["case_id"].nunique() != 1_500:
        raise ValueError("Expected 1,500 unique locked-test case IDs")
    if sorted(test["diag1"].unique().tolist()) != selection["diagnosis_codes"]:
        raise ValueError("Locked-test diagnosis categories differ from training")

    pipeline: Pipeline = joblib.load(pipeline_path)
    predicted = pipeline.predict(test["narrative"].tolist()).astype(int)
    probabilities = pipeline.predict_proba(test["narrative"].tolist())
    confidence = probabilities.max(axis=1)
    results = pd.DataFrame(
        {
            "case_id": test["case_id"].astype(int),
            "year": test["year"].astype(int),
            "narrative": test["narrative"],
            "true_diag1": test["diag1"].astype(int),
            "true_diagnosis_category": test.get("diagnosis_category", ""),
            "pred_diag1": predicted,
            "diag1_correct": test["diag1"].to_numpy(dtype=int) == predicted,
            "pred_diag1_probability": confidence,
            "parse_error": False,
        }
    )
    results.to_csv(results_path, index=False, encoding="utf-8")

    labels = selection["diagnosis_codes"]
    gold = results["true_diag1"].to_numpy(dtype=int)
    macro_f1 = float(f1_score(gold, predicted, labels=labels, average="macro", zero_division=0))
    metrics = {
        "condition": "tfidf_logreg_6000",
        "method": "word TF-IDF plus multinomial logistic regression",
        "outcome": "Diagnosis_1_Code",
        "n_examples": len(results),
        "diagnosis_accuracy": float(accuracy_score(gold, predicted)),
        "macro_f1": macro_f1,
        "selected_validation_configuration": selection["selected"],
        "classification_report": classification_report(
            gold,
            predicted,
            labels=labels,
            zero_division=0,
            output_dict=True,
        ),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "condition": "tfidf_logreg_6000",
        "stage": "final_evaluation",
        "command": sys.argv,
        "outcome": "Diagnosis_1_Code only",
        "test_file": str(args.test_file.resolve()),
        "test_sha256": sha256(args.test_file),
        "pipeline_file": str(pipeline_path.resolve()),
        "pipeline_sha256": sha256(pipeline_path),
        "selection_file": str(selection_path.resolve()),
        "seed": args.seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Test accuracy: {metrics['diagnosis_accuracy']:.4f}")
    print(f"Test macro F1: {macro_f1:.4f}")
    print(f"Wrote final outputs to: {args.output_dir}")


def main() -> None:
    args = parse_args()
    {"tune": tune, "final-eval": final_eval}[args.stage](args)


if __name__ == "__main__":
    main()
