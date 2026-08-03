"""Static integrity checks for the anonymous GitHub release."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".py", ".md", ".json", ".jsonl", ".csv", ".txt", ".yml", ".yaml", ".ipynb"
}
FORBIDDEN = (
    re.compile(r"C:\\Users\\", re.I),
    re.compile("One" + "Drive", re.I),
    re.compile("pur" + "due", re.I),
    re.compile("ahmad" + "100", re.I),
    re.compile(r"sk-[A-Za-z0-9_-]{12,}"),
)


def main() -> None:
    required = [
        "Injury_LLM_SFT.py",
        "configs/experiment_matrix.json",
        "scripts/prepare_neiss_temporal_dataset.py",
        "scripts/evaluate_methods.py",
        "scripts/generate_publication_figures.py",
        "results/primary/per_category_metrics.csv",
        "docs/DIAGNOSIS_CODES.md",
    ]
    missing = [path for path in required if not (ROOT / path).is_file()]
    if missing:
        raise AssertionError(f"Missing required files: {missing}")

    matrix = json.loads((ROOT / "configs" / "experiment_matrix.json").read_text(encoding="utf-8"))
    if matrix.get("seed") != 20260718 or len(matrix.get("conditions", [])) != 10:
        raise AssertionError("Frozen local experiment matrix is inconsistent")
    if "replication_seeds" in matrix:
        raise AssertionError("Unexecuted replication plan must not appear in the release matrix")

    forbidden_names = {"api_responses.jsonl", "test_results.csv", "human_review_100.csv"}
    present_forbidden = [
        str(path.relative_to(ROOT))
        for path in ROOT.rglob("*")
        if path.is_file() and path.name in forbidden_names
    ]
    if present_forbidden:
        raise AssertionError(f"Restricted row-level artifacts present: {present_forbidden}")

    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.suffix.lower() not in TEXT_SUFFIXES
            or path.resolve() == Path(__file__).resolve()
        ):
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN:
            if pattern.search(text):
                raise AssertionError(f"Private marker {pattern.pattern!r} in {path.relative_to(ROOT)}")
        if path.suffix.lower() == ".py":
            ast.parse(text, filename=str(path))

    for model in ("mistral", "qwen"):
        prompt_dir = ROOT / "artifacts" / "prompts" / model
        manifest = json.loads((prompt_dir / "prompt_artifact_manifest.json").read_text(encoding="utf-8"))
        entries = list(manifest["system_prompt_files"].values()) + [manifest["demonstration_messages"]]
        for entry in entries:
            artifact = prompt_dir / entry["path"]
            digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
            if digest != entry["sha256"]:
                raise AssertionError(f"Prompt hash mismatch: {artifact.relative_to(ROOT)}")
        demo_lines = (prompt_dir / manifest["demonstration_messages"]["path"]).read_text(
            encoding="utf-8"
        ).splitlines()
        if (
            len(demo_lines) != manifest["demonstration_messages"]["count"]
            or manifest["demonstration_messages"]["count"] != 10
        ):
            raise AssertionError(f"Unexpected demonstration count for {model}")

    with (ROOT / "results" / "primary" / "model_summary.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        summary = list(csv.DictReader(stream))
    if len(summary) != 15:
        raise AssertionError(f"Expected 15 primary conditions, found {len(summary)}")
    best = max(summary, key=lambda row: float(row["macro_f1"]))
    if best["condition"] != "qwen_sft_6000" or abs(float(best["macro_f1"]) - 0.9051688102) > 1e-10:
        raise AssertionError("Headline result does not match the frozen aggregate table")

    with (ROOT / "results" / "primary" / "per_category_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        per_category = list(csv.DictReader(stream))
    conditions = {row["condition"] for row in per_category}
    codes = {row["diagnosis_code"] for row in per_category}
    if len(per_category) != 450 or len(conditions) != 15 or len(codes) != 30:
        raise AssertionError("Per-category results must contain 15 conditions x 30 codes")

    print(
        "Release check passed: structure, anonymity, Python syntax, prompt hashes, "
        "headline metrics, and 15 x 30 per-code results."
    )


if __name__ == "__main__":
    main()
