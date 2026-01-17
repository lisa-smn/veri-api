"""
Evaluiert FactualityAgent auf FRANK-Manifest (für fairen Vergleich mit Baselines).

Input:
- data/frank/frank_subset_manifest.jsonl (gemeinsames Subset)

Output:
- results/evaluation/factuality/<run_id>/
  - predictions.jsonl
  - summary.json (binäre Metriken + dataset_signature)
  - summary.md
  - run_metadata.json

Metriken:
- Binär: TP/FP/TN/FN, Precision, Recall, F1, Balanced Accuracy, AUROC
- Dataset-Signature: SHA256-Hash des Manifests (für Vergleich mit Baselines)
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

from dotenv import load_dotenv

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.openai_client import OpenAIClient
from app.services.agents.factuality.factuality_agent import FactualityAgent

load_dotenv()


# ---------------------------
# Data loading
# ---------------------------


@dataclass
class ManifestExample:
    id: str
    hash: str
    model_name: str
    article_text: str
    summary_text: str
    reference_text: str
    gold_has_error: bool
    gold_score: float
    meta: dict[str, Any]


def load_manifest(manifest_path: Path) -> tuple[list[ManifestExample], str]:
    """Lädt Manifest-JSONL und gibt Dataset-Signature zurück."""
    examples = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                examples.append(
                    ManifestExample(
                        id=data.get("id", ""),
                        hash=data.get("hash", ""),
                        model_name=data.get("model_name", ""),
                        article_text=data.get("article_text", "").strip(),
                        summary_text=data.get("summary_text", "").strip(),
                        reference_text=data.get("reference_text", "").strip(),
                        gold_has_error=bool(data.get("gold_has_error", False)),
                        gold_score=float(data.get("gold_score", 0.0)),
                        meta=data.get("meta", {}),
                    )
                )
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Ungültiges JSONL in {manifest_path} @ Zeile {line_no}: {e}"
                ) from e

    # Load dataset signature from meta.json
    meta_path = manifest_path.with_suffix(".meta.json")
    dataset_signature = None
    if meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
            dataset_signature = meta.get("dataset_signature")

    # Fallback: compute from examples
    if dataset_signature is None:
        ids = sorted([ex.id for ex in examples])
        content = "\n".join(ids)
        dataset_signature = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return examples, dataset_signature


# compute_manifest_hash entfernt - wird jetzt in load_manifest gemacht


# ---------------------------
# Metrics
# ---------------------------


@dataclass
class BinaryMetrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0
    n_failed: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def specificity(self) -> float:
        denom = self.tn + self.fp
        return self.tn / denom if denom > 0 else 0.0

    @property
    def balanced_accuracy(self) -> float:
        return (self.recall + self.specificity) / 2.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0


def compute_mcc(tp: int, fp: int, tn: int, fn: int) -> float:
    """Berechnet Matthews Correlation Coefficient."""
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator > 0 else 0.0


def bootstrap_ci_binary(
    metric_func,
    predictions: list[bool],
    ground_truths: list[bool],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """Bootstrap-CI für binäre Metriken."""
    if not predictions or not ground_truths or len(predictions) != len(ground_truths):
        return {"median": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    rng = random.Random(seed)
    n = len(predictions)
    resamples: list[float] = []

    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        preds_resampled = [predictions[i] for i in indices]
        gts_resampled = [ground_truths[i] for i in indices]

        # Compute confusion matrix
        tp = sum(1 for p, g in zip(preds_resampled, gts_resampled) if p and g)
        fp = sum(1 for p, g in zip(preds_resampled, gts_resampled) if p and not g)
        tn = sum(1 for p, g in zip(preds_resampled, gts_resampled) if not p and not g)
        fn = sum(1 for p, g in zip(preds_resampled, gts_resampled) if not p and g)

        metric_value = metric_func(tp, fp, tn, fn)
        resamples.append(metric_value)

    resamples.sort()
    alpha = 1.0 - confidence
    lower_idx = int(n_resamples * (alpha / 2))
    upper_idx = int(n_resamples * (1 - alpha / 2))

    return {
        "median": resamples[len(resamples) // 2],
        "ci_lower": resamples[lower_idx],
        "ci_upper": resamples[upper_idx],
    }


def compute_auroc(scores: list[float], labels: list[bool]) -> float:
    """
    Berechnet AUROC (Area Under ROC Curve).

    Args:
        scores: Agent-Scores (höher = besser, daher invertieren wir: 1.0 - score)
        labels: Ground-Truth (True = has_error, False = no_error)
    """
    if not scores or not labels or len(scores) != len(labels):
        return 0.0

    # Invertiere Scores: lower score = error (für AUROC)
    inverted_scores = [1.0 - s for s in scores]

    # Sortiere nach Score (absteigend)
    pairs = sorted(zip(inverted_scores, labels), reverse=True)

    # Berechne AUROC (vereinfachte Trapez-Regel)
    tp = sum(labels)
    fp = len(labels) - tp
    if tp == 0 or fp == 0:
        return 0.0

    auc = 0.0
    tp_prev = 0
    fp_prev = 0

    for score, label in pairs:
        if label:
            tp_prev += 1
        else:
            fp_prev += 1
            auc += tp_prev

    return auc / (tp * fp)


# ---------------------------
# Evaluation
# ---------------------------


def run_eval(
    examples: list[ManifestExample],
    llm_model: str,
    max_examples: int | None,
    preds_path: Path,
    issue_threshold: int = 1,
    retries: int = 1,
    sleep_s: float = 1.0,
) -> tuple[BinaryMetrics, list[dict[str, Any]], str]:
    """
    Führt Agent-Evaluation durch.

    Returns:
        (metrics, predictions_list, dataset_signature)
    """
    llm = OpenAIClient(model_name=llm_model)
    agent = FactualityAgent(llm)

    metrics = BinaryMetrics()
    predictions_list: list[dict[str, Any]] = []
    pred_scores: list[float] = []

    if preds_path.exists():
        preds_path.unlink()

    # Dataset signature wird von load_manifest zurückgegeben, nicht hier berechnet
    dataset_signature = ""  # Wird von load_manifest gesetzt

    for idx, ex in enumerate(examples):
        if max_examples is not None and idx >= max_examples:
            break

        if not ex.article_text or not ex.summary_text:
            continue

        try:
            agent_result = agent.run(
                article_text=ex.article_text,
                summary_text=ex.summary_text,
                meta={"source": "eval_manifest", "example_id": ex.id, **ex.meta},
            )

            # Binary prediction: has_error if num_issues >= threshold
            num_issues = len(agent_result.issue_spans)
            pred_has_error = num_issues >= issue_threshold
            gt_has_error = ex.gold_has_error

            # Update metrics
            if pred_has_error and gt_has_error:
                metrics.tp += 1
            elif pred_has_error and not gt_has_error:
                metrics.fp += 1
            elif not pred_has_error and not gt_has_error:
                metrics.tn += 1
            else:
                metrics.fn += 1

            pred_scores.append(agent_result.score)

            rec = {
                "example_id": ex.id,
                "gt_has_error": gt_has_error,
                "gt_score": ex.gold_score,
                "pred_has_error": pred_has_error,
                "agent_score": agent_result.score,
                "num_issues": num_issues,
                "max_severity": max([s.severity for s in agent_result.issue_spans], default="none")
                if agent_result.issue_spans
                else "none",
                "meta": ex.meta,
            }
            predictions_list.append(rec)

            with preds_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (idx + 1) % 25 == 0:
                print(
                    f"[{idx + 1}] gt={gt_has_error} pred={pred_has_error} score={agent_result.score:.3f} issues={num_issues}"
                )

        except Exception as e:
            metrics.n_failed += 1
            print(f"Fehler bei {ex.id}: {e}")
            if retries > 0:
                time.sleep(sleep_s)
            continue

    # Add AUROC (wird später berechnet)
    # dataset_signature wird von load_manifest zurückgegeben

    return metrics, predictions_list, dataset_signature


# ---------------------------
# Output generation
# ---------------------------


def write_summary_json(
    metrics: BinaryMetrics,
    dataset_signature: str,
    n_total: int,
    n_used: int,
    predictions: list[dict[str, Any]],
    bootstrap_n: int,
    seed: int | None,
    out_path: Path,
) -> None:
    # Extract predictions and ground truths for bootstrap
    preds = [p["pred_has_error"] for p in predictions]
    gts = [p["gt_has_error"] for p in predictions]

    # Compute MCC
    mcc = compute_mcc(metrics.tp, metrics.fp, metrics.tn, metrics.fn)

    # Bootstrap CIs
    print(f"Berechne Bootstrap-CIs für binäre Metriken (n={bootstrap_n})...")
    accuracy_ci = bootstrap_ci_binary(
        lambda tp, fp, tn, fn: (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0,
        preds,
        gts,
        n_resamples=bootstrap_n,
        seed=seed,
    )
    precision_ci = bootstrap_ci_binary(
        lambda tp, fp, tn, fn: tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        preds,
        gts,
        n_resamples=bootstrap_n,
        seed=seed,
    )
    recall_ci = bootstrap_ci_binary(
        lambda tp, fp, tn, fn: tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        preds,
        gts,
        n_resamples=bootstrap_n,
        seed=seed,
    )
    f1_ci = bootstrap_ci_binary(
        lambda tp, fp, tn, fn: 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0,
        preds,
        gts,
        n_resamples=bootstrap_n,
        seed=seed,
    )
    balanced_acc_ci = bootstrap_ci_binary(
        lambda tp, fp, tn, fn: (tp / (tp + fn) + tn / (tn + fp)) / 2.0
        if (tp + fn) > 0 and (tn + fp) > 0
        else 0.0,
        preds,
        gts,
        n_resamples=bootstrap_n,
        seed=seed,
    )
    mcc_ci = bootstrap_ci_binary(
        compute_mcc,
        preds,
        gts,
        n_resamples=bootstrap_n,
        seed=seed,
    )

    summary = {
        "n_total": n_total,
        "n_used": n_used,
        "n_failed": metrics.n_failed,
        "dataset_signature": dataset_signature,
        "counts": {
            "tp": metrics.tp,
            "fp": metrics.fp,
            "tn": metrics.tn,
            "fn": metrics.fn,
        },
        "metrics": {
            "accuracy": {
                "value": metrics.accuracy,
                "ci_lower": accuracy_ci["ci_lower"],
                "ci_upper": accuracy_ci["ci_upper"],
            },
            "balanced_accuracy": {
                "value": metrics.balanced_accuracy,
                "ci_lower": balanced_acc_ci["ci_lower"],
                "ci_upper": balanced_acc_ci["ci_upper"],
            },
            "precision": {
                "value": metrics.precision,
                "ci_lower": precision_ci["ci_lower"],
                "ci_upper": precision_ci["ci_upper"],
            },
            "recall": {
                "value": metrics.recall,
                "ci_lower": recall_ci["ci_lower"],
                "ci_upper": recall_ci["ci_upper"],
            },
            "f1": {
                "value": metrics.f1,
                "ci_lower": f1_ci["ci_lower"],
                "ci_upper": f1_ci["ci_upper"],
            },
            "specificity": metrics.specificity,
            "mcc": {
                "value": mcc,
                "ci_lower": mcc_ci["ci_lower"],
                "ci_upper": mcc_ci["ci_upper"],
            },
            "auroc": 0.0,  # Wird später gesetzt
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_summary_md(
    metrics: BinaryMetrics,
    dataset_signature: str,
    n_total: int,
    n_used: int,
    summary_json: dict[str, Any],
    out_path: Path,
) -> None:
    lines = [
        "# Factuality Agent Evaluation Summary (Manifest)",
        "",
        "**Dataset:** FRANK (Manifest)",
        f"**Examples total:** {n_total}",
        f"**Examples used:** {n_used}",
        f"**Examples failed:** {metrics.n_failed}",
        f"**Dataset Signature:** {dataset_signature}",
        "",
        "## Binary Metrics",
        "",
        "| Metric | Value | 95% CI |",
        "|--------|-------|--------|",
    ]

    # Extract CIs from summary_json
    metrics_dict = summary_json.get("metrics", {})

    def format_metric(name: str, value: float) -> str:
        if isinstance(metrics_dict.get(name), dict):
            ci = metrics_dict[name]
            ci_str = f"[{ci.get('ci_lower', 0.0):.4f}, {ci.get('ci_upper', 0.0):.4f}]"
            return f"| {name} | {value:.4f} | {ci_str} |"
        return f"| {name} | {value:.4f} | - |"

    lines.append(f"| TP | {metrics.tp} | - |")
    lines.append(f"| FP | {metrics.fp} | - |")
    lines.append(f"| TN | {metrics.tn} | - |")
    lines.append(f"| FN | {metrics.fn} | - |")
    lines.append(format_metric("Precision", metrics.precision))
    lines.append(format_metric("Recall", metrics.recall))
    lines.append(format_metric("F1", metrics.f1))
    lines.append(format_metric("Specificity", metrics.specificity))
    lines.append(format_metric("Balanced Accuracy", metrics.balanced_accuracy))
    lines.append(format_metric("Accuracy", metrics.accuracy))
    if "mcc" in metrics_dict:
        mcc_val = (
            metrics_dict["mcc"].get("value", 0.0)
            if isinstance(metrics_dict["mcc"], dict)
            else metrics_dict["mcc"]
        )
        lines.append(format_metric("MCC", mcc_val))
    if "auroc" in metrics_dict:
        auroc_val = metrics_dict["auroc"]
        lines.append(f"| AUROC | {auroc_val:.4f} | - |")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_run_metadata(
    run_id: str,
    timestamp: str,
    manifest_path: Path,
    llm_model: str,
    prompt_version: str,
    dataset_signature: str,
    n_total: int,
    n_used: int,
    n_failed: int,
    config_params: dict[str, Any],
    out_path: Path,
) -> None:
    def get_git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent.parent,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "manifest_path": str(manifest_path),
        "dataset_signature": dataset_signature,
        "llm_model": llm_model,
        "prompt_version": prompt_version,
        "n_total": n_total,
        "n_used": n_used,
        "n_failed": n_failed,
        "config": config_params,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluiert FactualityAgent auf FRANK-Manifest")
    ap.add_argument("--manifest", type=str, required=True, help="Pfad zum Manifest-JSONL")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM-Modell")
    ap.add_argument("--max_examples", type=int, help="Maximale Anzahl Beispiele")
    ap.add_argument(
        "--issue_threshold",
        type=int,
        default=1,
        help="Threshold für has_error (num_issues >= threshold)",
    )
    ap.add_argument("--retries", type=int, default=1, help="Anzahl Retries bei Fehlern")
    ap.add_argument("--sleep-s", type=float, default=1.0, help="Sleep zwischen Retries (Sekunden)")
    ap.add_argument(
        "--bootstrap_n", type=int, default=2000, help="Anzahl Bootstrap-Resamples für CIs"
    )
    ap.add_argument("--seed", type=int, help="Random seed für Reproduzierbarkeit")
    ap.add_argument(
        "--out_dir", type=str, help="Output-Verzeichnis (default: results/evaluation/factuality)"
    )

    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        ap.error(f"Manifest nicht gefunden: {manifest_path}")

    print(f"Lade Manifest: {manifest_path}")
    examples, dataset_signature = load_manifest(manifest_path)
    print(f"Geladen: {len(examples)} Beispiele")
    print(f"Dataset-Signature: {dataset_signature}")

    # Run ID
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"factuality_agent_manifest_{ts}_{args.model}"

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir) / run_id
    else:
        out_dir = Path("results") / "evaluation" / "factuality" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / "predictions.jsonl"
    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    metadata_path = out_dir / "run_metadata.json"

    prompt_version = os.getenv("FACTUALITY_PROMPT_VERSION", "v1")

    print(f"Model: {args.model} | Prompt: {prompt_version}")
    print(f"Issue threshold: {args.issue_threshold}")
    print(f"Output: {out_dir}")

    if args.seed is not None:
        random.seed(args.seed)

    metrics, predictions_list, _ = run_eval(
        examples=examples,
        llm_model=args.model,
        max_examples=args.max_examples,
        preds_path=preds_path,
        issue_threshold=args.issue_threshold,
        retries=args.retries,
        sleep_s=args.sleep_s,
    )

    # Use dataset_signature from load_manifest (bereits oben gesetzt)

    # Add AUROC to metrics
    pred_scores = [p["agent_score"] for p in predictions_list]
    gt_labels = [p["gt_has_error"] for p in predictions_list]
    auroc = compute_auroc(pred_scores, gt_labels)

    # Write outputs
    write_summary_json(
        metrics,
        dataset_signature,
        len(examples),
        len(predictions_list),
        predictions_list,
        args.bootstrap_n,
        args.seed,
        summary_json_path,
    )
    # Update AUROC in summary
    summary = json.loads(summary_json_path.read_text())
    summary["metrics"]["auroc"] = auroc
    summary_json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    write_summary_md(
        metrics, dataset_signature, len(examples), len(predictions_list), summary, summary_md_path
    )
    write_run_metadata(
        run_id=run_id,
        timestamp=ts,
        manifest_path=manifest_path,
        llm_model=args.model,
        prompt_version=prompt_version,
        dataset_signature=dataset_signature,
        n_total=len(examples),
        n_used=len(predictions_list),
        n_failed=metrics.n_failed,
        config_params={
            "max_examples": args.max_examples,
            "issue_threshold": args.issue_threshold,
            "retries": args.retries,
            "sleep_s": args.sleep_s,
            "bootstrap_n": args.bootstrap_n,
            "seed": args.seed,
        },
        out_path=metadata_path,
    )

    print("\n" + "=" * 60)
    print("Agent-Evaluation abgeschlossen!")
    print("=" * 60)
    print("\nMetriken:")
    print(f"  TP: {metrics.tp}  FP: {metrics.fp}  TN: {metrics.tn}  FN: {metrics.fn}")
    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1:        {metrics.f1:.4f}")
    print(f"  Balanced Acc: {metrics.balanced_accuracy:.4f}")
    print(f"  AUROC:     {auroc:.4f}")
    print(
        f"  MCC:      {summary.get('metrics', {}).get('mcc', {}).get('value', 0.0) if isinstance(summary.get('metrics', {}).get('mcc'), dict) else 0.0:.4f}"
    )
    print(f"  Dataset Signature: {dataset_signature}")
    print(f"\nArtefakte gespeichert in: {out_dir}")


if __name__ == "__main__":
    main()
