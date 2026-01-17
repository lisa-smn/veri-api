"""
Berechnet Baselines (ROUGE-L, BERTScore) für Factuality-Evaluation auf FRANK.

Input:
- data/frank/benchmark_data.json (Original-FRANK mit Referenzen)
  ODER data/frank/frank_clean.jsonl (falls Referenzen nachgeladen werden können)

Schema (benchmark_data.json):
- article: Artikeltext
- summary: System-Summary (generated)
- reference: Referenz-Summary (Gold)
- hash, model_name: Identifikation
- (Gold-Label kommt aus human_annotations.json, wird via hash+model_name gematcht)

Wichtig:
- Gold-Label: has_error (bool) aus human_annotations.json
- Skalierung: has_error=True → 0.0, has_error=False → 1.0 (für Korrelation mit Scores)
- Baselines: ROUGE-L und BERTScore (summary vs reference)
- Gleiche Metriken wie Agent: Pearson, Spearman, MAE, RMSE, Bootstrap-CIs

Output:
- results/evaluation/factuality_baselines/<run_id>/
  - predictions.jsonl (pro Beispiel)
  - summary.json (alle Metriken + CIs + Metadaten)
  - summary.md (human-readable)
  - run_metadata.json (timestamp, git_commit, python version, seed, etc.)
"""

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
from typing import Any

from dotenv import load_dotenv

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def check_dependencies(baseline_type: str) -> tuple[bool, list[str]]:
    """
    Prüft, ob alle benötigten Dependencies vorhanden sind.

    Returns:
        (dependencies_ok, missing_packages)
    """
    missing = []

    if baseline_type == "rouge_l":
        if not HAS_ROUGE:
            missing.append("rouge-score")
    elif baseline_type == "bertscore":
        if not HAS_BERTSCORE:
            missing.append("bert-score")
            missing.append("transformers")
            missing.append("torch")

    return len(missing) == 0, missing


# ---------------------------
# IO helpers
# ---------------------------


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def load_manifest(manifest_path: Path) -> tuple[list[dict[str, Any]], str]:
    """
    Lädt Manifest-JSONL.

    Returns:
        (examples_list, dataset_signature)
    """
    examples = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                examples.append(
                    {
                        "article": data.get("article_text", "").strip(),
                        "summary": data.get("summary_text", "").strip(),
                        "reference": data.get("reference_text", "").strip(),  # reference_text
                        "has_error": bool(data.get("gold_has_error", False)),  # gold_has_error
                        "gold_score": float(
                            data.get("gold_score", 0.0)
                        ),  # gold_score direkt aus Manifest
                        "meta": {
                            "id": data.get("id", ""),
                            "hash": data.get("hash", ""),
                            "model_name": data.get("model_name", ""),
                            "factuality": data.get("meta", {}).get("factuality", 1.0),
                        },
                    }
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

    # Fallback: compute from ids
    if dataset_signature is None:
        ids = sorted(
            [
                ex["meta"].get("id", f"{ex['meta']['hash']}_{ex['meta']['model_name']}")
                for ex in examples
            ]
        )
        content = "\n".join(ids)
        dataset_signature = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return examples, dataset_signature


def load_frank_benchmark(
    benchmark_path: Path, annotations_path: Path
) -> tuple[list[dict[str, Any]], str | None]:
    """
    Lädt FRANK benchmark_data.json und matched human_annotations.json.

    Returns:
        (examples_list, None) - Keine Manifest-Hash für Legacy-Modus
    """
    benchmark = load_json(benchmark_path)
    if not isinstance(benchmark, list):
        raise ValueError(f"benchmark_data.json muss eine Liste sein, erhalten: {type(benchmark)}")

    annotations = load_json(annotations_path)
    if not isinstance(annotations, list):
        raise ValueError(
            f"human_annotations.json muss eine Liste sein, erhalten: {type(annotations)}"
        )

    # Index: (hash, model_name) -> annotation
    ann_index: dict[tuple[str, str], dict[str, Any]] = {}
    for ann in annotations:
        if not isinstance(ann, dict):
            continue
        h = ann.get("hash")
        model = ann.get("model_name")
        if h and model:
            ann_index[(str(h), str(model))] = ann

    examples = []
    for item in benchmark:
        if not isinstance(item, dict):
            continue

        article = item.get("article", "").strip()
        summary = item.get("summary", "").strip()
        reference = item.get("reference", "").strip()
        h = item.get("hash")
        model = item.get("model_name")

        if not article or not summary:
            continue

        if not reference:
            continue  # Skip wenn keine Referenz

        # Match annotation
        key = (str(h), str(model)) if h and model else None
        ann = ann_index.get(key) if key else None

        if ann is None:
            continue  # Skip wenn keine Annotation

        factuality = ann.get("Factuality")
        try:
            factuality_f = float(factuality) if factuality is not None else 1.0
        except (ValueError, TypeError):
            factuality_f = 1.0

        # has_error: factuality < 1.0
        has_error = factuality_f < 1.0

        examples.append(
            {
                "article": article,
                "summary": summary,
                "reference": reference,
                "has_error": has_error,
                "meta": {
                    "hash": str(h) if h else None,
                    "model_name": str(model) if model else None,
                    "factuality": factuality_f,
                },
            }
        )

    return examples, None


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_git_commit() -> str | None:
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


# ---------------------------
# Metrics (gleiche wie im Agent-Script)
# ---------------------------


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (denx * deny) if denx > 0 and deny > 0 else 0.0


def _rank(values: list[float]) -> list[float]:
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


def spearman(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return 0.0
    return pearson(_rank(xs), _rank(ys))


def mae(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return 0.0
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def rmse(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return 0.0
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(xs, ys)) / len(xs))


def r_squared(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return 0.0
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    ss_tot = sum((y - my) ** 2 for y in ys)
    ss_res = sum((y - x) ** 2 for x, y in zip(xs, ys))
    return 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0


def compute_auroc(scores: list[float], labels: list[bool]) -> float:
    """
    Berechnet AUROC (Area Under ROC Curve).

    Args:
        scores: Continuous predictions [0,1] (höher = besser = no_error)
        labels: Ground-Truth (True = has_error, False = no_error)

    Note: Bei Baselines: höherer Score = besser = sollte mit no_error (False) korrelieren.
    Daher invertieren wir: lower score = error (True)
    """
    if not scores or not labels or len(scores) != len(labels):
        return 0.0

    # Invertiere Scores: lower score = error (für AUROC)
    inverted_scores = [1.0 - s for s in scores]

    # Sortiere nach Score (absteigend)
    pairs = sorted(zip(inverted_scores, labels), reverse=True)

    # Berechne AUROC (vereinfachte Trapez-Regel)
    tp = sum(labels)  # True = has_error
    fp = len(labels) - tp
    if tp == 0 or fp == 0:
        return 0.0

    auc = 0.0
    tp_prev = 0
    fp_prev = 0

    for score, label in pairs:
        if label:  # has_error = True
            tp_prev += 1
        else:  # no_error = False
            fp_prev += 1
            auc += tp_prev

    return auc / (tp * fp)


def compute_f1_at_threshold(preds: list[float], gt_binary: list[bool], threshold: float) -> float:
    """
    Berechnet F1-Score bei gegebenem Threshold.

    Args:
        preds: Continuous predictions [0,1]
        gt_binary: Ground-Truth (True = has_error, False = no_error)
        threshold: Threshold für binäre Klassifikation (pred >= threshold → no_error)

    Returns:
        F1-Score
    """
    if not preds or not gt_binary or len(preds) != len(gt_binary):
        return 0.0

    tp = 0  # pred >= threshold AND gt = no_error (False)
    fp = 0  # pred >= threshold AND gt = has_error (True)
    fn = 0  # pred < threshold AND gt = no_error (False)

    for pred, gt_has_error in zip(preds, gt_binary):
        pred_no_error = pred >= threshold
        if pred_no_error and not gt_has_error:
            tp += 1
        elif pred_no_error and gt_has_error:
            fp += 1
        elif not pred_no_error and not gt_has_error:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


def find_best_f1_threshold(
    preds: list[float], gt_binary: list[bool], n_steps: int = 100
) -> tuple[float, float]:
    """
    Findet den besten Threshold für F1-Score über Grid Search.

    Args:
        preds: Continuous predictions [0,1]
        gt_binary: Ground-Truth (True = has_error, False = no_error)
        n_steps: Anzahl Schritte für Grid Search (default: 100)

    Returns:
        (best_f1, best_threshold)
    """
    if not preds or not gt_binary:
        return 0.0, 0.5

    best_f1 = 0.0
    best_threshold = 0.5

    for i in range(n_steps + 1):
        threshold = i / n_steps  # 0.0 bis 1.0
        f1 = compute_f1_at_threshold(preds, gt_binary, threshold)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_f1, best_threshold


# ---------------------------
# Bootstrap CIs
# ---------------------------


def bootstrap_ci(
    metric_func,
    xs: list[float],
    ys: list[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    if not xs or not ys or len(xs) != len(ys):
        return {"median": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    rng = random.Random(seed)
    n = len(xs)
    resamples: list[float] = []

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
# Baseline scores
# ---------------------------


def compute_rouge_l(summary: str, reference: str) -> float:
    """Berechnet ROUGE-L F1-Score."""
    if not HAS_ROUGE:
        raise RuntimeError(
            "rouge-score ist nicht installiert. Installiere mit: pip install rouge-score"
        )
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = scorer.score(reference, summary)
    return scores["rougeL"].fmeasure  # F1-Score in [0,1]


def compute_bertscore(summary: str, reference: str) -> float:
    """Berechnet BERTScore F1-Score."""
    if not HAS_BERTSCORE:
        raise RuntimeError(
            "bert-score ist nicht installiert. Installiere mit: pip install bert-score transformers torch\n"
            "Hinweis: BERTScore benötigt Model-Download beim ersten Lauf."
        )
    try:
        # bert_score gibt (P, R, F1) zurück, wir nehmen F1
        P, R, F1 = bert_score([summary], [reference], lang="en", verbose=False)
        return float(F1[0].item())  # F1 in [0,1]
    except Exception as e:
        raise RuntimeError(f"BERTScore-Fehler: {e}") from e


# ---------------------------
# Core eval
# ---------------------------


def run_eval(
    examples: list[dict[str, Any]],
    baseline_type: str,  # "rouge_l" oder "bertscore"
    max_examples: int | None,
    preds_path: Path,
    seed: int | None,
    bootstrap_n: int,
    dataset_signature: str | None = None,
    allow_dummy: bool = False,  # Erlaube 0.0 bei fehlenden Dependencies
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Führt die Baseline-Evaluation durch.

    Returns:
        (metrics_dict, predictions_list)
    """
    if seed is not None:
        random.seed(seed)

    # Gold: has_error (bool) → 0.0 (error) oder 1.0 (no error)
    gt_scores: list[float] = []
    preds: list[float] = []
    predictions_list: list[dict[str, Any]] = []
    n_seen = 0
    n_used = 0
    n_skipped = 0

    if preds_path.exists():
        preds_path.unlink()

    for example in examples:
        if max_examples is not None and n_used >= max_examples:
            break

        n_seen += 1

        summary = example.get("summary", "").strip()
        reference = example.get("reference", "").strip()
        has_error = example.get("has_error")
        meta = example.get("meta", {})
        example_id = f"frank_{meta.get('hash', 'unknown')}_{meta.get('model_name', 'unknown')}"

        if not summary or not reference:
            n_skipped += 1
            continue

        if has_error is None:
            n_skipped += 1
            continue

        # Gold: Nutze gold_score direkt aus Manifest (falls vorhanden), sonst berechnen
        gt_score = example.get("gold_score")
        if gt_score is None:
            # Fallback: has_error=False → 1.0 (gut), has_error=True → 0.0 (schlecht)
            gt_score = 1.0 if not has_error else 0.0

        # Berechne Baseline-Score
        try:
            if baseline_type == "rouge_l":
                pred_score = compute_rouge_l(summary, reference)
            elif baseline_type == "bertscore":
                pred_score = compute_bertscore(summary, reference)
            else:
                raise ValueError(f"Unbekannter Baseline-Typ: {baseline_type}")
        except RuntimeError as e:
            if allow_dummy:
                print(f"Warnung: {e}")
                print("Verwende 0.0 als Dummy-Wert (--allow_dummy_baseline aktiviert)")
                pred_score = 0.0
            else:
                raise

        # Clamp
        if pred_score < 0.0:
            pred_score = 0.0
        if pred_score > 1.0:
            pred_score = 1.0

        gt_scores.append(gt_score)
        preds.append(pred_score)
        n_used += 1

        rec = {
            "example_id": example_id,
            "baseline": baseline_type,
            "pred": pred_score,
            "gold": gt_score,  # Standardisiert: "gold" statt "gt"
            "gt": gt_score,  # Backward-Compat
            "has_error": has_error,
            "meta": meta,
        }
        predictions_list.append(rec)

        with preds_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if n_used % 25 == 0:
            print(
                f"[{n_used}] gt={gt_score:.3f} pred={pred_score:.3f} (seen={n_seen}, skipped={n_skipped})"
            )

    # Calculate metrics
    pearson_r = pearson(preds, gt_scores)
    spearman_rho = spearman(preds, gt_scores)
    mae_val = mae(preds, gt_scores)
    rmse_val = rmse(preds, gt_scores)
    r2_val = r_squared(preds, gt_scores)

    # Binary metrics: AUROC und Best F1
    # Convert gt_scores to binary labels: 1.0 = no_error (False), 0.0 = has_error (True)
    gt_binary = [score < 0.5 for score in gt_scores]  # True = has_error, False = no_error

    print("Berechne AUROC und Best F1...")
    auroc_val = compute_auroc(preds, gt_binary)
    best_f1, best_threshold = find_best_f1_threshold(preds, gt_binary)

    # Bootstrap CIs
    print(f"Berechne Bootstrap-CIs (n={bootstrap_n})...")
    pearson_ci = bootstrap_ci(pearson, preds, gt_scores, n_resamples=bootstrap_n, seed=seed)
    spearman_ci = bootstrap_ci(spearman, preds, gt_scores, n_resamples=bootstrap_n, seed=seed)
    mae_ci = bootstrap_ci(mae, preds, gt_scores, n_resamples=bootstrap_n, seed=seed)
    rmse_ci = bootstrap_ci(rmse, preds, gt_scores, n_resamples=bootstrap_n, seed=seed)

    metrics = {
        "n_total": len(examples),
        "n_seen": n_seen,
        "n_used": len(gt_scores),
        "n_skipped": n_skipped,
        "baseline_type": baseline_type,
        "dataset_signature": dataset_signature,
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
        "auroc": auroc_val,
        "best_f1": best_f1,
        "best_f1_threshold": best_threshold,
    }
    return metrics, predictions_list


# ---------------------------
# Output generation
# ---------------------------


def write_summary_json(metrics: dict[str, Any], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def write_summary_md(
    metrics: dict[str, Any],
    out_path: Path,
    dependencies_ok: bool = True,
    missing_packages: list[str] | None = None,
    allow_dummy: bool = False,
) -> None:
    baseline_name = metrics["baseline_type"].upper().replace("_", "-")
    lines = [
        f"# Factuality Baseline Evaluation Summary ({baseline_name})",
        "",
    ]

    # Warnung bei invalid/dummy runs
    if not dependencies_ok or allow_dummy:
        missing_str = ", ".join(missing_packages) if missing_packages else "unknown"
        lines.extend(
            [
                "",
                "## ⚠️ DUMMY/INVALID RUN",
                "",
                "**Dieser Run ist ungültig und nicht für Thesis-Evaluation nutzbar!**",
                "",
                f"- **Fehlende Dependencies:** {missing_str}",
                f"- **Allow Dummy:** {allow_dummy}",
                "",
                "**Ergebnisse sind nicht vergleichbar.**",
                "",
                "---",
                "",
            ]
        )
    elif dependencies_ok:
        lines.append("**Dependencies:** ✅ OK")
        lines.append("")

    lines.extend(
        [
            "**Dataset:** FRANK",
            f"**Baseline:** {baseline_name}",
            f"**Examples used:** {metrics['n_used']}",
            f"**Examples skipped:** {metrics['n_skipped']}",
            "",
            "## Gold Label Semantik",
            "",
            "**predictions.jsonl verwendet `gold` (und `gt` für Backward-Compat):**",
            "",
            "- **gold = 1.0** → no_error (gut)",
            "- **gold = 0.0** → has_error (schlecht)",
            "- **Baseline-Score (pred):** [0,1] (ROUGE-L/BERTScore F1, höher = besser)",
            "",
            "---",
            "",
            "## Quick Comparison",
            "",
            "| Metric | Value | 95% CI |",
            "|--------|-------|--------|",
            f"| Pearson r | {metrics['pearson']['value']:.4f} | [{metrics['pearson']['ci_lower']:.4f}, {metrics['pearson']['ci_upper']:.4f}] |",
            f"| Spearman ρ | {metrics['spearman']['value']:.4f} | [{metrics['spearman']['ci_lower']:.4f}, {metrics['spearman']['ci_upper']:.4f}] |",
            f"| MAE | {metrics['mae']['value']:.4f} | [{metrics['mae']['ci_lower']:.4f}, {metrics['mae']['ci_upper']:.4f}] |",
            f"| RMSE | {metrics['rmse']['value']:.4f} | [{metrics['rmse']['ci_lower']:.4f}, {metrics['rmse']['ci_upper']:.4f}] |",
            f"| R² | {metrics['r_squared']:.4f} | - |",
            f"| AUROC | {metrics.get('auroc', 0.0):.4f} | - |",
            f"| Best F1 | {metrics.get('best_f1', 0.0):.4f} | - |",
            f"| Best F1 Threshold | {metrics.get('best_f1_threshold', 0.5):.4f} | - |",
            "",
            "---",
            "",
            "## Detailed Metrics",
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
            "### Binary Classification Metrics",
            "",
            f"- **AUROC:** {metrics.get('auroc', 0.0):.4f} (continuous pred vs binary gold)",
            f"- **Best F1:** {metrics.get('best_f1', 0.0):.4f} (bei Threshold = {metrics.get('best_f1_threshold', 0.5):.4f})",
            "",
            "### Gold Label",
            "",
            "- **gold = 1.0** → no_error (gut)",
            "- **gold = 0.0** → has_error (schlecht)",
            "- **Baseline-Score (pred):** [0,1] (ROUGE-L/BERTScore F1)",
        ]
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_run_metadata(
    run_id: str,
    timestamp: str,
    data_source: str,  # "manifest" oder "benchmark+annotations"
    data_path: str,  # Pfad zum Manifest oder benchmark
    annotations_path: str | None,  # Nur bei Legacy
    baseline_type: str,
    seed: int | None,
    bootstrap_n: int,
    n_total: int,
    n_used: int,
    n_skipped: int,
    dataset_signature: str | None,
    config_params: dict[str, Any],
    dependencies_ok: bool,
    missing_packages: list[str],
    out_path: Path,
) -> None:
    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "seed": seed,
        "data_source": data_source,
        "data_path": data_path,
        "annotations_path": annotations_path,
        "baseline_type": baseline_type,
        "dataset_signature": dataset_signature,
        "n_total": n_total,
        "n_used": n_used,
        "n_skipped": n_skipped,
        "dependencies_ok": dependencies_ok,
        "missing_packages": missing_packages,
        "config": {
            "bootstrap_n": bootstrap_n,
            **config_params,
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Berechnet Baselines (ROUGE-L, BERTScore) für Factuality-Evaluation"
    )
    ap.add_argument("--manifest", type=str, help="Pfad zum Manifest-JSONL (für fairen Vergleich)")
    ap.add_argument(
        "--benchmark",
        type=str,
        default="data/frank/benchmark_data.json",
        help="Pfad zu benchmark_data.json (Legacy)",
    )
    ap.add_argument(
        "--annotations",
        type=str,
        default="data/frank/human_annotations.json",
        help="Pfad zu human_annotations.json (Legacy)",
    )
    ap.add_argument(
        "--baseline", type=str, choices=["rouge_l", "bertscore"], required=True, help="Baseline-Typ"
    )
    ap.add_argument("--max_examples", type=int, help="Maximale Anzahl Beispiele")
    ap.add_argument("--max", type=int, help="Alias für --max_examples")
    ap.add_argument("--seed", type=int, help="Random seed für Reproduzierbarkeit")
    ap.add_argument("--bootstrap_n", type=int, default=2000, help="Anzahl Bootstrap-Resamples")
    ap.add_argument(
        "--out_dir",
        type=str,
        help="Output-Verzeichnis (default: results/evaluation/factuality_baselines)",
    )
    ap.add_argument(
        "--allow_dummy_baseline",
        action="store_true",
        help="Erlaube 0.0 als Dummy-Wert bei fehlenden Dependencies (Default: False, Script bricht ab)",
    )

    args = ap.parse_args()

    # Prüfe Dependencies früh
    dependencies_ok, missing_packages = check_dependencies(args.baseline)

    if not dependencies_ok:
        if args.allow_dummy_baseline:
            print(f"⚠️  WARNUNG: Fehlende Dependencies: {', '.join(missing_packages)}")
            print("⚠️  --allow_dummy_baseline aktiviert: Verwende 0.0 als Dummy-Wert")
        else:
            print("=" * 60)
            print("FEHLER: Fehlende Dependencies!")
            print("=" * 60)
            if args.baseline == "rouge_l":
                print("Für ROUGE-L benötigt:")
                print("  pip install rouge-score")
            elif args.baseline == "bertscore":
                print("Für BERTScore benötigt:")
                print("  pip install bert-score transformers torch")
                print("")
                print("Hinweis: BERTScore benötigt Model-Download beim ersten Lauf.")
            print("=" * 60)
            print("")
            print(
                "Alternative: Verwende --allow_dummy_baseline, um 0.0 als Dummy-Wert zu verwenden."
            )
            print("(Nicht empfohlen für Thesis-Evaluation!)")
            sys.exit(1)

    max_examples = args.max_examples or args.max
    seed = args.seed
    bootstrap_n = args.bootstrap_n
    dataset_signature = None

    # Load from manifest (preferred) or benchmark+annotations (legacy)
    if args.manifest:
        manifest_path = Path(args.manifest)
        if not manifest_path.exists():
            ap.error(f"Manifest nicht gefunden: {manifest_path}")
        print(f"Lade Manifest: {manifest_path}")
        examples, dataset_signature = load_manifest(manifest_path)
        print(f"Geladen: {len(examples)} Beispiele (Manifest-Hash: {dataset_signature})")
    else:
        benchmark_path = Path(args.benchmark)
        annotations_path = Path(args.annotations)

        if not benchmark_path.exists():
            ap.error(f"Datei nicht gefunden: {benchmark_path}")
        if not annotations_path.exists():
            ap.error(f"Datei nicht gefunden: {annotations_path}")

        print("Lade FRANK benchmark (Legacy-Modus)...")
        examples, dataset_signature = load_frank_benchmark(benchmark_path, annotations_path)
        print(f"Geladen: {len(examples)} Beispiele mit Referenzen und Annotations")

    # Run ID
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"factuality_{args.baseline}_{ts}"
    if seed is not None:
        run_id += f"_seed{seed}"

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir) / run_id
    else:
        out_dir = Path("results") / "evaluation" / "factuality_baselines" / run_id
    ensure_dir(out_dir)

    preds_path = out_dir / "predictions.jsonl"
    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    metadata_path = out_dir / "run_metadata.json"

    if args.manifest:
        print(f"Manifest: {Path(args.manifest)} (examples={len(examples)})")
    else:
        print(f"Benchmark: {benchmark_path} (examples={len(examples)})")
    print(f"Baseline: {args.baseline}")
    print(f"Seed: {seed}")
    print(f"Bootstrap resamples: {bootstrap_n}")
    print(f"Output: {out_dir}")

    metrics, predictions_list = run_eval(
        examples=examples,
        baseline_type=args.baseline,
        max_examples=max_examples,
        preds_path=preds_path,
        seed=seed,
        bootstrap_n=bootstrap_n,
        dataset_signature=dataset_signature,
        allow_dummy=args.allow_dummy_baseline,
    )

    # Write outputs
    write_summary_json(metrics, summary_json_path)
    write_summary_md(
        metrics, summary_md_path, dependencies_ok, missing_packages, args.allow_dummy_baseline
    )
    # Determine data source
    if args.manifest:
        data_source = "manifest"
        data_path = str(Path(args.manifest))
        annotations_path_meta = None
    else:
        data_source = "benchmark+annotations"
        data_path = str(Path(args.benchmark))
        annotations_path_meta = str(Path(args.annotations))

    write_run_metadata(
        run_id=run_id,
        timestamp=ts,
        data_source=data_source,
        data_path=data_path,
        annotations_path=annotations_path_meta,
        baseline_type=args.baseline,
        seed=seed,
        bootstrap_n=bootstrap_n,
        n_total=len(examples),
        n_used=metrics["n_used"],
        n_skipped=metrics["n_skipped"],
        dataset_signature=dataset_signature,
        config_params={
            "max_examples": max_examples,
        },
        dependencies_ok=dependencies_ok,
        missing_packages=missing_packages,
        out_path=metadata_path,
    )

    print("\n" + "=" * 60)
    print("Baseline-Evaluation abgeschlossen!")
    print("=" * 60)
    print(f"\nMetriken ({args.baseline}):")
    print(
        f"  Pearson r:  {metrics['pearson']['value']:.4f} [{metrics['pearson']['ci_lower']:.4f}, {metrics['pearson']['ci_upper']:.4f}]"
    )
    print(
        f"  Spearman ρ: {metrics['spearman']['value']:.4f} [{metrics['spearman']['ci_lower']:.4f}, {metrics['spearman']['ci_upper']:.4f}]"
    )
    print(
        f"  MAE:         {metrics['mae']['value']:.4f} [{metrics['mae']['ci_lower']:.4f}, {metrics['mae']['ci_upper']:.4f}]"
    )
    print(
        f"  RMSE:        {metrics['rmse']['value']:.4f} [{metrics['rmse']['ci_lower']:.4f}, {metrics['rmse']['ci_upper']:.4f}]"
    )
    print(f"  R²:          {metrics['r_squared']:.4f}")
    print(f"  AUROC:       {metrics.get('auroc', 0.0):.4f}")
    print(
        f"  Best F1:     {metrics.get('best_f1', 0.0):.4f} (Threshold: {metrics.get('best_f1_threshold', 0.5):.4f})"
    )
    print(f"\nArtefakte gespeichert in: {out_dir}")


if __name__ == "__main__":
    main()
