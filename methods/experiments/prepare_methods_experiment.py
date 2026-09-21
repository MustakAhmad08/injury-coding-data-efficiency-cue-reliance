"""Create fixed few-shot examples and the Mistral/Qwen experiment matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 20260718
MODELS = {
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]

    def repository_path(path: Path) -> str:
        """Store portable paths relative to the repository root."""
        return path.resolve().relative_to(project_root).as_posix()

    data_dir = project_root / "data" / "neiss_temporal_2021_2025"
    ablation_dir = data_dir / "ablations"
    run_root = project_root / "methods_runs"
    run_root.mkdir(exist_ok=True)

    train = pd.read_csv(data_dir / "train.csv")
    codes = sorted(train["diag1"].unique())
    selected_indices = np.linspace(0, len(codes) - 1, 10, dtype=int)
    selected_codes = [int(codes[index]) for index in selected_indices]
    demos = []
    for code in selected_codes:
        row = train[train["diag1"].eq(code)].sample(n=1, random_state=SEED + code).iloc[0]
        demos.append({
            "case_id": int(row["case_id"]),
            "diagnosis_code": code,
            "diagnosis_category": row["diagnosis_category"],
            "instruction": row["instruction"],
            "narrative": row["narrative"],
            "response": row["response"],
        })
    fewshot_path = data_dir / "fewshot_10.jsonl"
    with fewshot_path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in demos:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    test_file = repository_path(data_dir / "locked_test.csv")
    validation_file = repository_path(data_dir / "validation.jsonl")
    conditions = []
    for family, model_id in MODELS.items():
        for shots in (0, 10):
            condition_id = f"{family}_{shots}shot"
            conditions.append({
                "condition_id": condition_id,
                "family": family,
                "method": f"{shots}-shot",
                "model_id": model_id,
                "shots": shots,
                "test_file": test_file,
                "fewshot_file": repository_path(fewshot_path),
                "output_dir": repository_path(run_root / condition_id),
            })
        for size in (1000, 3000, 6000):
            condition_id = f"{family}_sft_{size}"
            conditions.append({
                "condition_id": condition_id,
                "family": family,
                "method": "sft",
                "training_size": size,
                "model_id": model_id,
                "train_file": repository_path(ablation_dir / f"train_{size}.jsonl"),
                "validation_file": validation_file,
                "test_file": test_file,
                "output_dir": repository_path(run_root / condition_id),
            })

    matrix = {
        "seed": SEED,
        "models": MODELS,
        "shared_test_file": test_file,
        "shared_validation_file": validation_file,
        "fewshot_file": repository_path(fewshot_path),
        "conditions": conditions,
        "status": "configured_not_run",
    }
    matrix_path = project_root / "configs" / "experiment_matrix.json"
    matrix_path.parent.mkdir(exist_ok=True)
    matrix_path.write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {fewshot_path} with {len(demos)} demonstrations")
    print(f"Wrote {matrix_path} with {len(conditions)} conditions")


if __name__ == "__main__":
    main()
