"""Post hoc diagnosis-cue masking analysis on the frozen 2025 test set.

Stages:
  prepare  - create marker-only and diagnosis-tail masked copies
  tfidf    - evaluate the frozen TF-IDF logistic-regression pipeline
  local    - evaluate one archived SFT adapter
  report   - aggregate original/masked results with paired bootstrap CIs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.sft.train import PeftModel, evaluate_test_set, load_model_and_tokenizer

DATA = ROOT / "data" / "neiss_temporal_2021_2025" / "locked_test.csv"
OUT = ROOT / "methods_runs" / "cue_masking"
SEED = 20260718
N_BOOT = 1000
MARKER = re.compile(
    r"\b(?:DX|DIAGNOSIS|DIAGNOSES|ASSESSMENT|IMPRESSION)\b\s*[:;=\-]*\s*",
    flags=re.IGNORECASE,
)
MODELS = {
    "qwen_sft_6000": {
        "base": "Qwen/Qwen2.5-7B-Instruct",
        "adapter": ROOT.parent / "injury_llm_sft_archive" / "model_artifacts" / "qwen_sft_6000",
        "label": "Qwen SFT 6K",
    },
    "mistral_sft_3000": {
        "base": "mistralai/Mistral-7B-Instruct-v0.3",
        "adapter": ROOT.parent / "injury_llm_sft_archive" / "model_artifacts" / "mistral_sft_3000",
        "label": "Mistral SFT 3K",
    },
    "tfidf_logreg_6000": {
        "label": "TF-IDF + logistic regression",
    },
}
VARIANTS = ("marker_only", "diagnosis_tail")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("prepare", "tfidf", "local", "report"))
    parser.add_argument("--condition", choices=tuple(MODELS))
    parser.add_argument("--confirm-locked-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_confirmation(args: argparse.Namespace) -> None:
    if not args.confirm_locked_test:
        raise PermissionError("This post hoc analysis requires --confirm-locked-test")


def masked_text(text: str, variant: str) -> tuple[str, bool]:
    match = MARKER.search(text)
    if not match:
        return text, False
    if variant == "marker_only":
        return text[: match.start()] + "[MARKER REMOVED] " + text[match.end() :], True
    return text[: match.start()].rstrip() + " [DIAGNOSIS TAIL REMOVED]", True


def prepare(args: argparse.Namespace) -> None:
    require_confirmation(args)
    frame = pd.read_csv(DATA)
    if len(frame) != 1500 or frame["case_id"].nunique() != 1500:
        raise ValueError("Expected the frozen 1,500-case test set")
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for variant in VARIANTS:
        target = OUT / f"{variant}.csv"
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"{target} exists; add --overwrite")
        changed = []
        narratives = []
        for value in frame["narrative"].astype(str):
            new_value, was_changed = masked_text(value, variant)
            narratives.append(new_value)
            changed.append(was_changed)
        masked = frame.copy()
        masked["original_narrative"] = masked["narrative"]
        masked["narrative"] = narratives
        masked["cue_mask_applied"] = changed
        masked.to_csv(target, index=False)
        counts[variant] = int(sum(changed))
    metadata = {
        "analysis": "post_hoc_cue_masking",
        "test_file": str(DATA.resolve()),
        "test_sha256": file_hash(DATA),
        "n_cases": len(frame),
        "seed": SEED,
        "mask_regex": MARKER.pattern,
        "changed_cases": counts,
        "interpretation": (
            "Marker-only is a formatting control. Diagnosis-tail removes all text from "
            "the first explicit marker onward and may remove clinically relevant details."
        ),
    }
    (OUT / "masking_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared masked sets; changed cases: {counts}")


def save_results(results: pd.DataFrame, condition: str, variant: str) -> None:
    directory = OUT / condition / variant
    directory.mkdir(parents=True, exist_ok=True)
    results.to_csv(directory / "test_results.csv", index=False)


def run_tfidf(args: argparse.Namespace) -> None:
    require_confirmation(args)
    pipeline_path = ROOT / "methods_runs" / "tfidf_logreg_6000" / "tfidf_logreg_pipeline.joblib"
    pipeline = joblib.load(pipeline_path)
    for variant in VARIANTS:
        target = OUT / "tfidf_logreg_6000" / variant / "test_results.csv"
        if target.exists() and not args.overwrite:
            raise FileExistsError(f"{target} exists; add --overwrite")
        frame = pd.read_csv(OUT / f"{variant}.csv")
        pred = pipeline.predict(frame["narrative"].astype(str)).astype(int)
        results = pd.DataFrame(
            {
                "case_id": frame["case_id"],
                "true_diag1": frame["diag1"].astype(int),
                "pred_diag1": pred,
                "diag1_correct": pred == frame["diag1"].astype(int).to_numpy(),
                "cue_mask_applied": frame["cue_mask_applied"],
            }
        )
        save_results(results, "tfidf_logreg_6000", variant)
        print(f"TF-IDF {variant}: accuracy={results['diag1_correct'].mean():.4f}")


def run_local(args: argparse.Namespace) -> None:
    require_confirmation(args)
    if args.condition not in {"qwen_sft_6000", "mistral_sft_3000"}:
        raise ValueError("--condition must select a local SFT model")
    spec = MODELS[args.condition]
    adapter = Path(spec["adapter"])
    if not (adapter / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"Archived adapter not found: {adapter}")
    pending = []
    for variant in VARIANTS:
        target = OUT / args.condition / variant / "test_results.csv"
        if target.exists() and not args.overwrite:
            print(f"Skipping existing {target}")
        else:
            pending.append(variant)
    if not pending:
        return
    model, tokenizer = load_model_and_tokenizer(str(spec["base"]))
    model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    for variant in pending:
        results, metrics = evaluate_test_set(
            str(OUT / f"{variant}.csv"),
            model,
            tokenizer,
            data_mode="neiss",
            max_new_tokens=256,
            inference_batch_size=8,
        )
        results["cue_mask_applied"] = pd.read_csv(OUT / f"{variant}.csv")[
            "cue_mask_applied"
        ].astype(bool)
        save_results(results, args.condition, variant)
        metrics_path = OUT / args.condition / variant / "test_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        print(f"{args.condition} {variant}: accuracy={metrics['diagnosis_accuracy']:.4f}")


def metric(gold: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    labels = sorted(np.unique(gold).tolist())
    return (
        float(accuracy_score(gold, pred)),
        float(f1_score(gold, pred, labels=labels, average="macro", zero_division=0)),
    )


def paired_drop_ci(
    gold: np.ndarray, original: np.ndarray, masked: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(SEED)
    by_label = [np.flatnonzero(gold == label) for label in labels]
    acc_drops, f1_drops = [], []
    for _ in range(N_BOOT):
        idx = np.concatenate([rng.choice(group, len(group), replace=True) for group in by_label])
        orig_acc, orig_f1 = metric(gold[idx], original[idx])
        mask_acc, mask_f1 = metric(gold[idx], masked[idx])
        acc_drops.append(orig_acc - mask_acc)
        f1_drops.append(orig_f1 - mask_f1)
    return (
        float(np.quantile(acc_drops, 0.025)),
        float(np.quantile(acc_drops, 0.975)),
        float(np.quantile(f1_drops, 0.025)),
        float(np.quantile(f1_drops, 0.975)),
    )


def category_recall_drop_ci(
    original_correct: np.ndarray, masked_correct: np.ndarray, seed_offset: int
) -> tuple[float, float]:
    rng = np.random.default_rng(SEED + seed_offset)
    drops = []
    for _ in range(N_BOOT):
        idx = rng.choice(len(original_correct), len(original_correct), replace=True)
        drops.append(float(original_correct[idx].mean() - masked_correct[idx].mean()))
    return float(np.quantile(drops, 0.025)), float(np.quantile(drops, 0.975))


def report(args: argparse.Namespace) -> None:
    require_confirmation(args)
    test = pd.read_csv(DATA).set_index("case_id")
    labels = np.array(sorted(test["diag1"].astype(int).unique()))
    rows = []
    category_rows = []
    for condition, spec in MODELS.items():
        original_path = ROOT / "methods_runs" / condition / "test_results.csv"
        original = pd.read_csv(original_path).set_index("case_id").loc[test.index]
        gold = original["true_diag1"].astype(int).to_numpy()
        orig_pred = original["pred_diag1"].fillna(-1).astype(int).to_numpy()
        orig_acc, orig_f1 = metric(gold, orig_pred)
        for variant in VARIANTS:
            masked = pd.read_csv(
                OUT / condition / variant / "test_results.csv"
            ).set_index("case_id").loc[test.index]
            mask_pred = masked["pred_diag1"].fillna(-1).astype(int).to_numpy()
            mask_acc, mask_f1 = metric(gold, mask_pred)
            lo_a, hi_a, lo_f, hi_f = paired_drop_ci(
                gold, orig_pred, mask_pred, labels
            )
            rows.append(
                {
                    "condition": condition,
                    "model": spec["label"],
                    "variant": variant,
                    "original_accuracy": orig_acc,
                    "masked_accuracy": mask_acc,
                    "accuracy_drop": orig_acc - mask_acc,
                    "accuracy_drop_ci_low": lo_a,
                    "accuracy_drop_ci_high": hi_a,
                    "original_macro_f1": orig_f1,
                    "masked_macro_f1": mask_f1,
                    "macro_f1_drop": orig_f1 - mask_f1,
                    "macro_f1_drop_ci_low": lo_f,
                    "macro_f1_drop_ci_high": hi_f,
                }
            )
            for code in labels:
                idx = gold == code
                orig_ok = orig_pred[idx] == gold[idx]
                mask_ok = mask_pred[idx] == gold[idx]
                lo_c, hi_c = category_recall_drop_ci(
                    orig_ok, mask_ok, int(code) * 10 + VARIANTS.index(variant)
                )
                category_rows.append(
                    {
                        "condition": condition,
                        "model": spec["label"],
                        "variant": variant,
                        "diagnosis_code": int(code),
                        "diagnosis_category": str(
                            test.loc[test["diag1"].astype(int) == code, "diagnosis_category"].iloc[0]
                        ),
                        "n": int(idx.sum()),
                        "original_recall": float(orig_ok.mean()),
                        "masked_recall": float(mask_ok.mean()),
                        "recall_drop": float(orig_ok.mean() - mask_ok.mean()),
                        "recall_drop_ci_low": lo_c,
                        "recall_drop_ci_high": hi_c,
                    }
                )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "cue_reliance_summary.csv", index=False)
    category_summary = pd.DataFrame(category_rows)
    category_summary.to_csv(OUT / "cue_reliance_per_category.csv", index=False)
    manifest = json.loads((OUT / "masking_manifest.json").read_text(encoding="utf-8"))
    lines = [
        "# Cue-reliance analysis",
        "",
        "## Scope",
        "",
        "This is a **post hoc exploratory analysis** of the locked 2025 test set. "
        "It was not used for model selection or tuning. Marker-only masking removes "
        "the tokens introducing an explicit diagnosis statement; diagnosis-tail "
        "masking removes the first marker and everything after it.",
        "",
        f"An explicit marker occurred in **{manifest['changed_cases']['marker_only']} "
        f"of 1,500 narratives ({manifest['changed_cases']['marker_only']/1500:.1%})**.",
        "",
        "## Results",
        "",
        "| Model | Mask | Macro F1 original | Macro F1 masked | F1 drop (95% CI) |",
        "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['variant'].replace('_', ' ')} | "
            f"{row['original_macro_f1']:.3f} | {row['masked_macro_f1']:.3f} | "
            f"{row['macro_f1_drop']:.3f} "
            f"[{row['macro_f1_drop_ci_low']:.3f}, {row['macro_f1_drop_ci_high']:.3f}] |"
        )
    lines += ["", "## Most diagnosis-tail-sensitive categories", ""]
    for condition, spec in MODELS.items():
        subset = category_summary[
            (category_summary["condition"] == condition)
            & (category_summary["variant"] == "diagnosis_tail")
        ].sort_values(["recall_drop", "diagnosis_code"], ascending=[False, True]).head(5)
        rendered = "; ".join(
            f"{int(row.diagnosis_code)} {row.diagnosis_category}: "
            f"{row.recall_drop:.2f} [{row.recall_drop_ci_low:.2f}, "
            f"{row.recall_drop_ci_high:.2f}]"
            for row in subset.itertuples()
        )
        lines.append(f"- **{spec['label']}:** {rendered}.")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "- Marker-only masking is a formatting control; it leaves the diagnosis words intact.",
        "- Diagnosis-tail masking is deliberately severe and can remove both explicit labels "
        "and legitimate clinical details. Its performance drop is therefore an upper-bound "
        "estimate of reliance on the tail, not a pure causal estimate of label-word reliance.",
        "- The locked test set had already been examined; all findings are exploratory.",
        "- No API narratives were uploaded and no paid model was rerun for this analysis.",
        "",
        "## Reproducibility",
        "",
        f"- Seed: `{SEED}`",
        f"- Paired diagnosis-stratified bootstrap resamples: `{N_BOOT}`",
        f"- Locked-test SHA-256: `{manifest['test_sha256']}`",
    ]
    (OUT / "CUE_RELIANCE_ANALYSIS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(summary)} cue-masking summary rows")


def main() -> None:
    args = arguments()
    {"prepare": prepare, "tfidf": run_tfidf, "local": run_local, "report": report}[args.stage](args)


if __name__ == "__main__":
    main()
