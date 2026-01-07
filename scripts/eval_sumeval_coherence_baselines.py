"""
Berechnet Baselines (ROUGE-L, BERTScore) für Coherence-Evaluation auf SummEval.

Input: Gleiche JSONL wie eval_sumeval_coherence.py
- Falls "ref", "reference" oder "references" vorhanden: Berechnet ROUGE-L und BERTScore
- Falls nicht: Warnung, aber Script läuft weiter (für Vergleichbarkeit)

Output:
- results/evaluation/coherence_baselines/<run_id>/
  - predictions.jsonl
  - summary.json
  - summary.md
  - run_metadata.json

Metriken: Gleiche wie beim Agent (Pearson, Spearman, MAE, RMSE, Bootstrap-CIs)
"""

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

# Optional imports für Baselines
try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

try:
    from bert_score import score as bert_score
    HAS_BERTSCORE = True
except ImportError:
    HAS_BERTSCORE = False


# ---------------------------
# IO helpers
# ---------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Ungültiges JSONL in {path} @ Zeile {line_no}: {e}") from e
    return rows


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_git_commit() -> Optional[str]:
    """Versucht Git-Commit-Hash zu ermitteln."""
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


def find_reference(row: Dict[str, Any]) -> Optional[str]:
    """Sucht nach Referenz-Summary in verschiedenen Feldern."""
    # Direkte Felder
    for key in ["ref", "reference", "references"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val[0].strip()  # Nimm erste Referenz

    # In meta
    meta = row.get("meta", {})
    for key in ["ref", "reference", "references"]:
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, list) and val and isinstance(val[0], str):
            return val[0].strip()

    return None


# ---------------------------
# Metrics (gleiche wie im Agent-Script)
# ---------------------------

def pearson(xs: List[float], ys: List[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx > 0 and deny > 0 else 0.0


def _rank(values: List[float]) -> List[float]:
    idx = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[idx[j + 1]] == values[idx[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[idx[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: List[float], ys: List[float]) -> float:
    if not xs:
        return 0.0
    return pearson(_rank(xs), _rank(ys))


def mae(xs: List[float], ys: List[float]) -> float:
    if not xs:
        return 0.0
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def rmse(xs: List[float], ys: List[float]) -> float:
    if not xs:
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs))


def r_squared(xs: List[float], ys: List[float]) -> float:
    if not xs:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - x) ** 2 for x, y in zip(xs, ys))
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


# ---------------------------
# Bootstrap CIs
# ---------------------------

def bootstrap_ci(
    metric_func,
    xs: List[float],
    ys: List[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: Optional[int] = None,
) -> Dict[str, float]:
    if not xs or not ys or len(xs) != len(ys):
        return {"median": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    rng = random.Random(seed)
    n = len(xs)
    resamples: List[float] = []

    for _ in range(n_resamples):
        indices = [rng.randint(0, n - 1) for _ in range(n)]
        xs_resampled = [xs[i] for i in indices]
        ys_resampled = [ys[i] for i in indices]
        metric_value = metric_func(xs_resampled, ys_resampled)
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


# ---------------------------
# Normalization
# ---------------------------

def normalize_to_0_1(x: float, min_v: float, max_v: float) -> float:
    if max_v <= min_v:
        raise ValueError("gt_max muss > gt_min sein")
    if x < min_v:
        x = min_v
    if x > max_v:
        x = max_v
    return (x - min_v) / (max_v - min_v)


# ---------------------------
# Baseline scores
# ---------------------------

def compute_rouge_l(summary: str, reference: str) -> float:
    """Berechnet ROUGE-L F1-Score."""
    if not HAS_ROUGE:
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, summary)
    return scores["rougeL"].fmeasure  # F1-Score in [0,1]


def compute_bertscore(summary: str, reference: str) -> float:
    """Berechnet BERTScore F1-Score."""
    if not HAS_BERTSCORE:
        return 0.0
    try:
        # bert_score gibt (P, R, F1) zurück, wir nehmen F1
        P, R, F1 = bert_score([summary], [reference], lang="en", verbose=False)
        return float(F1[0].item())  # F1 in [0,1]
    except Exception as e:
        print(f"Warnung: BERTScore-Fehler: {e}")
        return 0.0


# ---------------------------
# Core eval
# ---------------------------

def run_eval(
    rows: List[Dict[str, Any]],
    baseline_type: str,  # "rouge_l" oder "bertscore"
    max_examples: Optional[int],
    preds_path: Path,
    gt_min: float,
    gt_max: float,
    seed: Optional[int],
    bootstrap_n: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Führt die Baseline-Evaluation durch.

    Returns:
        (metrics_dict, predictions_list)
    """
    if seed is not None:
        random.seed(seed)

    gt_norms: List[float] = []
    preds: List[float] = []
    predictions_list: List[Dict[str, Any]] = []
    n_seen = 0
    n_used = 0
    n_skipped = 0
    n_no_ref = 0

    if preds_path.exists():
        preds_path.unlink()

    # Warnung bei fehlenden Bibliotheken
    if baseline_type == "rouge_l" and not HAS_ROUGE:
        print("WARNUNG: rouge-score nicht installiert. Installiere mit: pip install rouge-score")
        print("ROUGE-L wird auf 0.0 gesetzt.")
    if baseline_type == "bertscore" and not HAS_BERTSCORE:
        print("WARNUNG: bert-score nicht installiert. Installiere mit: pip install bert-score")
        print("BERTScore wird auf 0.0 gesetzt.")

    for row in rows:
        if max_examples is not None and n_used >= max_examples:
            break

        n_seen += 1

        gt_raw = row.get("gt", {}).get("coherence")
        summary = row.get("summary")
        reference = find_reference(row)
        meta = row.get("meta", {})
        example_id = meta.get("doc_id") or meta.get("id") or f"example_{n_seen}"

        if gt_raw is None or not isinstance(summary, str) or not summary.strip():
            n_skipped += 1
            continue

        if reference is None:
            n_no_ref += 1
            if n_no_ref == 1:
                print(f"WARNUNG: Keine Referenz gefunden für Beispiel {example_id}. Überspringe...")
            continue

        try:
            gt_raw_f = float(gt_raw)
        except Exception:
            n_skipped += 1
            continue

        gt_norm = normalize_to_0_1(gt_raw_f, min_v=gt_min, max_v=gt_max)

        # Berechne Baseline-Score
        if baseline_type == "rouge_l":
            pred_score = compute_rouge_l(summary, reference)
        elif baseline_type == "bertscore":
            pred_score = compute_bertscore(summary, reference)
        else:
            raise ValueError(f"Unbekannter Baseline-Typ: {baseline_type}")

        # Clamp
        if pred_score < 0.0:
            pred_score = 0.0
        if pred_score > 1.0:
            pred_score = 1.0

        pred_1_5 = 1.0 + 4.0 * pred_score  # Map [0,1] -> [1,5]

        gt_norms.append(gt_norm)
        preds.append(pred_score)
        n_used += 1

        rec = {
            "example_id": example_id,
            "baseline": baseline_type,
            "pred": pred_score,
            "pred_1_5": pred_1_5,
            "gt_raw": gt_raw_f,
            "gt_norm": gt_norm,
        }
        predictions_list.append(rec)

        with preds_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if n_used % 25 == 0:
            print(f"[{n_used}] gt_norm={gt_norm:.3f} pred={pred_score:.3f} (seen={n_seen}, skipped={n_skipped}, no_ref={n_no_ref})")

    if n_no_ref > 0:
        print(f"\nWARNUNG: {n_no_ref} Beispiele ohne Referenz übersprungen.")

    # Calculate metrics
    pearson_r = pearson(preds, gt_norms)
    spearman_rho = spearman(preds, gt_norms)
    mae_val = mae(preds, gt_norms)
    rmse_val = rmse(preds, gt_norms)
    r2_val = r_squared(preds, gt_norms)

    # Bootstrap CIs
    print(f"Berechne Bootstrap-CIs (n={bootstrap_n})...")
    pearson_ci = bootstrap_ci(pearson, preds, gt_norms, n_resamples=bootstrap_n, seed=seed)
    spearman_ci = bootstrap_ci(spearman, preds, gt_norms, n_resamples=bootstrap_n, seed=seed)
    mae_ci = bootstrap_ci(mae, preds, gt_norms, n_resamples=bootstrap_n, seed=seed)
    rmse_ci = bootstrap_ci(rmse, preds, gt_norms, n_resamples=bootstrap_n, seed=seed)

    metrics = {
        "n_seen": n_seen,
        "n_used": len(gt_norms),
        "n_skipped": n_skipped,
        "n_no_ref": n_no_ref,
        "baseline_type": baseline_type,
        "pearson": {
            "value": pearson_r,
            "ci_lower": pearson_ci["ci_lower"],
            "ci_upper": pearson_ci["ci_upper"],
        },
        "spearman": {
            "value": spearman_rho,
            "ci_lower": spearman_ci["ci_lower"],
            "ci_upper": spearman_ci["ci_upper"],
        },
        "mae": {
            "value": mae_val,
            "ci_lower": mae_ci["ci_lower"],
            "ci_upper": mae_ci["ci_upper"],
        },
        "rmse": {
            "value": rmse_val,
            "ci_lower": rmse_ci["ci_lower"],
            "ci_upper": rmse_ci["ci_upper"],
        },
        "r_squared": r2_val,
        "gt_normalization": {
            "raw_min": gt_min,
            "raw_max": gt_max,
            "normalized_to": "0..1",
        },
    }
    return metrics, predictions_list


# ---------------------------
# Output generation
# ---------------------------

def write_summary_json(metrics: Dict[str, Any], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def write_summary_md(metrics: Dict[str, Any], out_path: Path) -> None:
    baseline_name = metrics["baseline_type"].upper().replace("_", "-")
    lines = [
        f"# Coherence Baseline Evaluation Summary ({baseline_name})",
        "",
        f"**Dataset:** SummEval",
        f"**Baseline:** {baseline_name}",
        f"**Examples used:** {metrics['n_used']}",
        f"**Examples skipped:** {metrics['n_skipped']}",
        f"**Examples without reference:** {metrics.get('n_no_ref', 0)}",
        "",
        "## Metrics",
        "",
        "### Correlation",
        "",
        f"- **Pearson r:** {metrics['pearson']['value']:.4f} (95% CI: [{metrics['pearson']['ci_lower']:.4f}, {metrics['pearson']['ci_upper']:.4f}])",
        f"- **Spearman ρ:** {metrics['spearman']['value']:.4f} (95% CI: [{metrics['spearman']['ci_lower']:.4f}, {metrics['spearman']['ci_upper']:.4f}])",
        "",
        "### Error Metrics",
        "",
        f"- **MAE:** {metrics['mae']['value']:.4f} (95% CI: [{metrics['mae']['ci_lower']:.4f}, {metrics['mae']['ci_upper']:.4f}])",
        f"- **RMSE:** {metrics['rmse']['value']:.4f} (95% CI: [{metrics['rmse']['ci_lower']:.4f}, {metrics['rmse']['ci_upper']:.4f}])",
        f"- **R²:** {metrics['r_squared']:.4f}",
        "",
        "### Ground Truth Normalization",
        "",
        f"- Raw scale: [{metrics['gt_normalization']['raw_min']}, {metrics['gt_normalization']['raw_max']}]",
        f"- Normalized to: {metrics['gt_normalization']['normalized_to']}",
    ]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_run_metadata(
    run_id: str,
    timestamp: str,
    data_path: Path,
    baseline_type: str,
    seed: Optional[int],
    bootstrap_n: int,
    n_total: int,
    n_used: int,
    n_no_ref: int,
    config_params: Dict[str, Any],
    out_path: Path,
) -> None:
    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "seed": seed,
        "dataset_path": str(data_path),
        "baseline_type": baseline_type,
        "n_total": n_total,
        "n_used": n_used,
        "n_no_ref": n_no_ref,
        "config": {
            "bootstrap_n": bootstrap_n,
            **config_params,
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Berechnet Baselines (ROUGE-L, BERTScore) für Coherence-Evaluation")
    ap.add_argument("--data", type=str, required=True, help="Pfad zur JSONL-Datei")
    ap.add_argument("--baseline", type=str, choices=["rouge_l", "bertscore"], required=True, help="Baseline-Typ")
    ap.add_argument("--max_examples", type=int, help="Maximale Anzahl Beispiele")
    ap.add_argument("--max", type=int, help="Alias für --max_examples")
    ap.add_argument("--seed", type=int, help="Random seed für Reproduzierbarkeit")
    ap.add_argument("--bootstrap_n", type=int, default=2000, help="Anzahl Bootstrap-Resamples")
    ap.add_argument("--out_dir", type=str, help="Output-Verzeichnis (default: results/evaluation/coherence_baselines)")
    ap.add_argument("--gt-min", type=float, default=1.0, help="GT-Minimum (default: 1.0)")
    ap.add_argument("--gt-max", type=float, default=5.0, help="GT-Maximum (default: 5.0)")

    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        ap.error(f"Datei nicht gefunden: {data_path}")

    max_examples = args.max_examples or args.max
    seed = args.seed
    bootstrap_n = args.bootstrap_n

    rows = load_jsonl(data_path)

    # Run ID
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"coherence_{args.baseline}_{ts}"
    if seed is not None:
        run_id += f"_seed{seed}"

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir) / run_id
    else:
        out_dir = Path("results") / "evaluation" / "coherence_baselines" / run_id
    ensure_dir(out_dir)

    preds_path = out_dir / "predictions.jsonl"
    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    metadata_path = out_dir / "run_metadata.json"

    print(f"Data: {data_path} (rows={len(rows)})")
    print(f"Baseline: {args.baseline}")
    print(f"GT scale: [{args.gt_min}, {args.gt_max}] -> [0,1]")
    print(f"Seed: {seed}")
    print(f"Bootstrap resamples: {bootstrap_n}")
    print(f"Output: {out_dir}")

    metrics, predictions_list = run_eval(
        rows=rows,
        baseline_type=args.baseline,
        max_examples=max_examples,
        preds_path=preds_path,
        gt_min=args.gt_min,
        gt_max=args.gt_max,
        seed=seed,
        bootstrap_n=bootstrap_n,
    )

    # Write outputs
    write_summary_json(metrics, summary_json_path)
    write_summary_md(metrics, summary_md_path)
    write_run_metadata(
        run_id=run_id,
        timestamp=ts,
        data_path=data_path,
        baseline_type=args.baseline,
        seed=seed,
        bootstrap_n=bootstrap_n,
        n_total=len(rows),
        n_used=metrics["n_used"],
        n_no_ref=metrics.get("n_no_ref", 0),
        config_params={
            "max_examples": max_examples,
            "gt_min": args.gt_min,
            "gt_max": args.gt_max,
        },
        out_path=metadata_path,
    )

    print("\n" + "=" * 60)
    print("Baseline-Evaluation abgeschlossen!")
    print("=" * 60)
    print(f"\nMetriken ({args.baseline}):")
    print(f"  Pearson r:  {metrics['pearson']['value']:.4f} [{metrics['pearson']['ci_lower']:.4f}, {metrics['pearson']['ci_upper']:.4f}]")
    print(f"  Spearman ρ: {metrics['spearman']['value']:.4f} [{metrics['spearman']['ci_lower']:.4f}, {metrics['spearman']['ci_upper']:.4f}]")
    print(f"  MAE:         {metrics['mae']['value']:.4f} [{metrics['mae']['ci_lower']:.4f}, {metrics['mae']['ci_upper']:.4f}]")
    print(f"  RMSE:        {metrics['rmse']['value']:.4f} [{metrics['rmse']['ci_lower']:.4f}, {metrics['rmse']['ci_upper']:.4f}]")
    print(f"  R²:          {metrics['r_squared']:.4f}")
    print(f"\nArtefakte gespeichert in: {out_dir}")


if __name__ == "__main__":
    main()

