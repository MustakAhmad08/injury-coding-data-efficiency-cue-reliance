"""Run one configured baseline or SFT condition without silently opening the test set."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("condition", help="Condition ID from experiment_matrix.json")
    parser.add_argument("--stage", choices=("train", "final-eval"), required=True)
    parser.add_argument(
        "--confirm-locked-test",
        action="store_true",
        help="Required for final-eval because it reads the locked 2025 test set.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the command without executing it.")
    parser.add_argument(
        "--seed",
        type=int,
        help=(
            "Override the matrix seed for a replication run. A seed-specific output "
            "directory is used automatically so existing checkpoints are not overwritten."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    matrix = json.loads((project_root / "configs" / "experiment_matrix.json").read_text(encoding="utf-8"))
    matches = [item for item in matrix["conditions"] if item["condition_id"] == args.condition]
    if len(matches) != 1:
        raise ValueError(f"Unknown or duplicate condition ID: {args.condition}")
    condition = matches[0]
    entrypoint = project_root / "Injury_LLM_SFT.py"
    selected_seed = matrix["seed"] if args.seed is None else args.seed
    output_dir = Path(condition["output_dir"])
    if selected_seed != matrix["seed"]:
        output_dir = output_dir.with_name(f"{output_dir.name}_seed{selected_seed}")
    command = [
        sys.executable, str(entrypoint),
        "--model-name", condition["model_id"],
        "--output-dir", str(output_dir),
        "--seed", str(selected_seed),
    ]

    if condition["method"] in {"0-shot", "10-shot"}:
        if args.stage != "final-eval":
            raise ValueError("Prompting baselines have no training stage; use --stage final-eval")
        command.extend([
            "--baseline-eval",
            "--shots", str(condition["shots"]),
            "--fewshot-file", condition["fewshot_file"],
            "--test-file", condition["test_file"],
        ])
    elif args.stage == "train":
        command.extend([
            "--train-file", condition["train_file"],
            "--validation-file", condition["validation_file"],
        ])
    else:
        command.extend([
            "--eval-only",
            "--test-file", condition["test_file"],
        ])

    if args.stage == "final-eval" and not args.confirm_locked_test:
        raise PermissionError("Add --confirm-locked-test only after all model choices are frozen")
    if args.stage == "final-eval":
        command.extend(["--inference-batch-size", "8"])
    print("Command:")
    print(subprocess.list2cmdline(command))
    if not args.dry_run:
        environment = os.environ.copy()
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        subprocess.run(command, check=True, cwd=project_root, env=environment)


if __name__ == "__main__":
    main()
