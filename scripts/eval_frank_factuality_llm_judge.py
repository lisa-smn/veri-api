"""
LLM-as-a-Judge Baseline für Factuality-Evaluation auf FRANK/FineSumFact.

Input:
- data/frank/frank_subset_manifest.jsonl (oder FineSumFact JSONL)
  Format: article_text, summary_text, gold_has_error (bool)

Judge-Ansatz:
- Binary verdict: error_present (true/false) mit striktem JSON-Output
- Temperature=0 für Determinismus
- Mehrfachurteile (JUDGE_N) für Robustheit
- Majority-Aggregation über binary verdicts

Output:
- results/evaluation/factuality/judge_factuality_<timestamp>_<model>_v2_binary_seed<seed>/
  - predictions.jsonl (per example: id, gold_has_error, judge_verdict, confidence)
  - summary.json (Metrics + CIs wenn bootstrap)
  - summary.md (human-readable)
  - run_metadata.json
  - cache.jsonl (optional)

Metriken:
- Binary: TP/FP/TN/FN, Precision, Recall, F1, Balanced Accuracy, Specificity, MCC, AUROC
- Bootstrap-CIs für alle Metriken
- n_used, n_skipped, n_failed
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
import statistics
import subprocess
import sys
import time
from typing import Any

from dotenv import load_dotenv

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.openai_client import OpenAIClient
from app.services.judges.llm_judge import LLMJudge

load_dotenv()


# ---------------------------
# Data loading
# ---------------------------


@dataclass
class FactualityExample:
    id: str
    article_text: str
    summary_text: str
    gold_has_error: bool
    meta: dict[str, Any]


def _parse_has_error(raw_label: Any) -> bool | None:
    """Robust gegen bool/int/str Labels."""
    if raw_label is None:
        return None
    if isinstance(raw_label, bool):
        return raw_label
    if isinstance(raw_label, (int, float)):
        if raw_label == 1:
            return True
        if raw_label == 0:
            return False
        return None
    if isinstance(raw_label, str):
        s = raw_label.strip().lower()
        if s in ("1", "true", "yes", "y", "t"):
            return True
        if s in ("0", "false", "no", "n", "f"):
            return False
        return None
    return None


def load_jsonl(path: Path) -> list[FactualityExample]:
    """Lädt JSONL mit article, summary, has_error."""
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                article = (data.get("article_text") or data.get("article") or "").strip()
                summary = (data.get("summary_text") or data.get("summary") or "").strip()
                raw_label = data.get("gold_has_error") or data.get("has_error")
                has_error = _parse_has_error(raw_label)

                if not article or not summary or has_error is None:
                    continue

                example_id = data.get("id") or data.get("hash") or f"example_{line_no}"
                examples.append(
                    FactualityExample(
                        id=example_id,
                        article_text=article,
                        summary_text=summary,
                        gold_has_error=has_error,
                        meta=data.get("meta", {}),
                    )
                )
            except json.JSONDecodeError as e:
                raise ValueError(f"Ungültiges JSONL in {path} @ Zeile {line_no}: {e}") from e
    return examples


# ---------------------------
# IO helpers
# ---------------------------


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
# Binary Metrics
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


def compute_auroc(scores: list[float], labels: list[bool]) -> tuple[float, int]:
    """
    Berechnet AUROC (Area Under ROC Curve) mit Edge-Case-Handling.

    Args:
        scores: Kontinuierliche Scores (höher = mehr error signal)
        labels: Ground-Truth (True = has_error, False = no_error)

    Returns:
        (auroc_value, skipped_resamples): auroc_value ist 0.0 wenn undefiniert, skipped_resamples zählt single-class resamples
    """
    if not scores or not labels or len(scores) != len(labels):
        return 0.0, 0

    # Sortiere nach Score (absteigend)
    pairs = sorted(zip(scores, labels), reverse=True)

    # Berechne AUROC (vereinfachte Trapez-Regel)
    tp = sum(labels)
    fp = len(labels) - tp
    if tp == 0 or fp == 0:
        # Single-class: AUROC undefiniert
        return 0.0, 1

    auc = 0.0
    tp_prev = 0
    fp_prev = 0

    for score, label in pairs:
        if label:
            tp_prev += 1
        else:
            fp_prev += 1
            auc += tp_prev

    return auc / (tp * fp), 0


# ---------------------------
# Bootstrap CIs
# ---------------------------


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


# ---------------------------
# Caching
# ---------------------------


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_key(
    article: str, summary: str, model: str, prompt_version: str, judgment_idx: int
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt_version": prompt_version,
            "article": article,
            "summary": summary,
            "judgment_idx": judgment_idx,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return _sha256(payload)


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    cache: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Ungültiges Cache-JSONL in {path} @ Zeile {line_no}: {e}") from e
            k = rec.get("key")
            v = rec.get("value")
            if isinstance(k, str) and isinstance(v, dict):
                cache[k] = v
    return cache


def append_cache(path: Path, key: str, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


# ---------------------------
# Evaluation
# ---------------------------


def run_eval(
    examples: list[FactualityExample],
    llm_model: str,
    max_examples: int | None,
    preds_path: Path,
    judge_n: int = 3,
    judge_temperature: float = 0.0,
    judge_aggregation: str = "majority",
    prompt_version: str = "v2_binary",
    retries: int = 1,
    sleep_s: float = 1.0,
    use_cache: bool = True,
    cache_path: Path = None,
    seed: int | None = None,
    bootstrap_n: int = 2000,
    cache_mode: str = "write",  # "off", "read", "write"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Führt Judge-Evaluation durch.

    Returns:
        (metrics_dict, predictions_list)
    """
    if seed is not None:
        random.seed(seed)

    llm = OpenAIClient(model_name=llm_model)
    judge = LLMJudge(
        llm_client=llm,
        default_model=llm_model,
        default_prompt_version=prompt_version,
        default_n=judge_n,
        default_temperature=judge_temperature,
        default_aggregation=judge_aggregation,
    )

    # Cache-Mode Logik
    cache_read_enabled = (cache_mode == "read" or cache_mode == "write") and use_cache
    cache_write_enabled = cache_mode == "write" and use_cache
    cache_by_key = load_cache(cache_path) if cache_read_enabled else {}

    cache_hits = 0
    cache_misses = 0

    # Label-Distribution prüfen (Dataset-Level, vor Sampling)
    dataset_pos_count = sum(1 for ex in examples if ex.gold_has_error)
    dataset_neg_count = sum(1 for ex in examples if not ex.gold_has_error)
    dataset_total = len(examples)
    dataset_label_distribution = {
        "positives": dataset_pos_count,
        "negatives": dataset_neg_count,
        "total": dataset_total,
    }

    print(
        f"\nDataset label distribution: pos={dataset_pos_count}, neg={dataset_neg_count}, total={dataset_total}"
    )
    if dataset_pos_count == 0 or dataset_neg_count == 0:
        print(
            "⚠️  WARNING: Single-class dataset detected. Some metrics (AUROC, Balanced Accuracy, MCC) will be N/A."
        )

    metrics = BinaryMetrics()
    predictions_list: list[dict[str, Any]] = []
    pred_scores: list[float] = []  # Für AUROC (nur wenn confidence vorhanden)
    pred_labels: list[bool] = []  # Für AUROC (nur wenn confidence vorhanden)

    # Parse-Statistiken (judgment-level, nicht example-level)
    parse_stats = {
        "parsed_json_ok": 0,
        "parsed_regex_fallback": 0,
        "parse_failed": 0,
    }
    total_judgments = 0  # Wird während der Evaluation gezählt

    if preds_path.exists():
        preds_path.unlink()

    for idx, ex in enumerate(examples):
        if max_examples is not None and idx >= max_examples:
            break

        if not ex.article_text or not ex.summary_text:
            continue

        # Cache-Lookup
        cached = None
        if cache_read_enabled:
            # Versuche Cache für alle Judgements
            all_cached = True
            cached_judgements = []
            for j_idx in range(judge_n):
                key = cache_key(ex.article_text, ex.summary_text, llm_model, prompt_version, j_idx)
                cached_item = cache_by_key.get(key)
                if cached_item:
                    cached_judgements.append(cached_item)
                else:
                    all_cached = False
                    break

            if all_cached and len(cached_judgements) == judge_n:
                cached = cached_judgements
                cache_hits += 1
            else:
                cache_misses += 1
        else:
            cache_misses += 1

        judge_verdict: bool | None = None
        judge_confidence: float | None = None
        judge_score_norm: float | None = None
        judge_result_data: dict[str, Any] | None = None
        judge_result = None

        if cached is not None:
            # Verwende Cache
            # Aggregiere cached judgements (majority vote)
            error_present_votes = [c.get("error_present", False) for c in cached]
            error_present_count = sum(error_present_votes)
            judge_verdict = error_present_count > (judge_n / 2)  # Majority
            # Confidence aus Cache: nur wenn vorhanden
            cached_confidences = [
                c.get("confidence") for c in cached if c.get("confidence") is not None
            ]
            judge_confidence = statistics.mean(cached_confidences) if cached_confidences else None
            # Validiere confidence im Range [0, 1]
            if judge_confidence is not None:
                judge_confidence = max(0.0, min(1.0, judge_confidence))
            judge_score_norm = 0.0 if judge_verdict else 1.0  # 0.0 = error, 1.0 = no error
            judge_result_data = {
                "cached": True,
                "judgements": cached,
            }
            # Parse-Statistiken: cached judgements zählen als JSON-OK (wurden bereits erfolgreich geparst)
            parse_stats["parsed_json_ok"] += len(cached)
            total_judgments += len(cached)
        else:
            # Führe Judge aus
            last_err: str | None = None
            for attempt in range(retries + 1):
                try:
                    judge_result = judge.judge(
                        dimension="factuality",
                        article_text=ex.article_text,
                        summary_text=ex.summary_text,
                        model=llm_model,
                        prompt_version=prompt_version,
                        n=judge_n,
                        temperature=judge_temperature,
                        aggregation=judge_aggregation,
                        retries=1,
                    )

                    # Extrahiere binary verdict aus JudgeResult
                    # score_norm: 0.0 = error, 1.0 = no error
                    # Für majority: zähle error_present votes
                    error_present_votes = []
                    confidences = []
                    for output in judge_result.outputs:
                        # score_norm: 0.0 = error, 1.0 = no error
                        # error_present = True wenn score_norm < 0.5
                        error_present = output.score_norm < 0.5
                        error_present_votes.append(error_present)
                        if output.confidence is not None:
                            confidences.append(output.confidence)

                        # Parse-Statistiken sammeln (pro Output/judgment)
                        total_judgments += 1
                        if "parse_fallback" in output.flags:
                            parse_stats["parsed_regex_fallback"] += 1
                        elif output.raw_json is not None:
                            parse_stats["parsed_json_ok"] += 1
                        else:
                            parse_stats["parse_failed"] += 1

                    # Majority vote
                    error_present_count = sum(error_present_votes)
                    judge_verdict = error_present_count > (judge_n / 2)
                    # Confidence: nur wenn mindestens ein Output confidence hat
                    judge_confidence = statistics.mean(confidences) if confidences else None
                    # Validiere confidence im Range [0, 1]
                    if judge_confidence is not None:
                        judge_confidence = max(0.0, min(1.0, judge_confidence))
                    judge_score_norm = judge_result.final_score_norm

                    judge_result_data = judge_result.model_dump()

                    # Cache schreiben
                    if cache_write_enabled:
                        for j_idx, output in enumerate(judge_result.outputs):
                            key = cache_key(
                                ex.article_text, ex.summary_text, llm_model, prompt_version, j_idx
                            )
                            cache_value = {
                                "error_present": output.score_norm < 0.5,
                                "confidence": output.confidence or 0.5,
                                "rationale": output.rationale,
                            }
                            append_cache(cache_path, key, cache_value)
                            cache_by_key[key] = cache_value

                    break
                except Exception as e:
                    last_err = str(e)
                    if attempt < retries:
                        time.sleep(sleep_s)
                    else:
                        metrics.n_failed += 1
                        # Bei Fehler: zähle judge_n failed judgments
                        total_judgments += judge_n
                        parse_stats["parse_failed"] += judge_n
                        judge_verdict = None
                        judge_result_data = {"error": last_err}

        if judge_verdict is None:
            continue

        # Update metrics
        gt_has_error = ex.gold_has_error
        if judge_verdict and gt_has_error:
            metrics.tp += 1
        elif judge_verdict and not gt_has_error:
            metrics.fp += 1
        elif not judge_verdict and not gt_has_error:
            metrics.tn += 1
        else:
            metrics.fn += 1

        # Score für AUROC: verwende confidence falls vorhanden, sonst None
        # AUROC benötigt kontinuierlichen Score (confidence), nicht binary verdict
        # confidence: höher = mehr confidence in error_present verdict
        # Für AUROC: invertiere confidence wenn error_present=False (höherer Score = error)
        if judge_confidence is not None and 0.0 <= judge_confidence <= 1.0:
            # confidence ist kontinuierlich: verwende für AUROC
            # Wenn error_present=True: confidence direkt verwenden (höher = error)
            # Wenn error_present=False: invertiere (1.0 - confidence) (höher = error)
            auroc_score = judge_confidence if judge_verdict else (1.0 - judge_confidence)
        else:
            # Kein kontinuierlicher Score verfügbar
            auroc_score = None

        rec = {
            "example_id": ex.id,
            "gt_has_error": gt_has_error,
            "judge_verdict": judge_verdict,
            "judge_confidence": judge_confidence,
            "judge_score_norm": judge_score_norm,
            "judge_result": judge_result_data,
            "meta": ex.meta,
        }
        predictions_list.append(rec)

        # Sammle AUROC-Scores (nur wenn confidence vorhanden)
        if auroc_score is not None:
            pred_scores.append(auroc_score)
            pred_labels.append(gt_has_error)

        with preds_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if (idx + 1) % 25 == 0:
            conf_str = f"{judge_confidence:.2f}" if judge_confidence is not None else "N/A"
            print(
                f"[{idx + 1}] gt={gt_has_error} judge={judge_verdict} conf={conf_str} (tp={metrics.tp}, fp={metrics.fp}, tn={metrics.tn}, fn={metrics.fn})"
            )

    # Label-Distribution im Sample (nach Sampling auf max_examples)
    sample_pos_count = sum(1 for p in predictions_list if p["gt_has_error"])
    sample_neg_count = sum(1 for p in predictions_list if not p["gt_has_error"])
    sample_total = len(predictions_list)
    sample_label_distribution = {
        "positives": sample_pos_count,
        "negatives": sample_neg_count,
        "total": sample_total,
    }

    print(
        f"\nSample label distribution (n_used={sample_total}): pos={sample_pos_count}, neg={sample_neg_count}"
    )
    if sample_pos_count == 0 or sample_neg_count == 0:
        print(
            "⚠️  WARNING: Single-class sample detected. Some metrics (AUROC, Balanced Accuracy, MCC) will be N/A."
        )

    # Berechne AUROC nur wenn kontinuierliche Scores vorhanden UND beide Klassen vorhanden
    auroc_bootstrap_skipped = 0
    auroc_available = False
    auroc = None

    # Prüfe: beide Klassen vorhanden? (im Sample)
    has_both_classes = sample_pos_count > 0 and sample_neg_count > 0

    if pred_scores and len(pred_scores) == len(pred_labels) and has_both_classes:
        # Prüfe: confidence nicht konstant?
        if len(set(pred_scores)) > 1:
            auroc, _ = compute_auroc(pred_scores, pred_labels)
            auroc_available = True
        else:
            # Alle Scores gleich -> AUROC undefiniert
            auroc_available = False
            auroc = None
    elif not has_both_classes:
        # Single-class: AUROC undefiniert
        auroc_available = False
        auroc = None
    else:
        # Keine confidence vorhanden
        auroc_available = False
        auroc = None

    # Bootstrap CIs
    print(f"Berechne Bootstrap-CIs (n={bootstrap_n})...")
    predictions_binary = [p["judge_verdict"] for p in predictions_list]
    ground_truths_binary = [p["gt_has_error"] for p in predictions_list]

    precision_ci = bootstrap_ci_binary(
        lambda tp, fp, tn, fn: tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        predictions_binary,
        ground_truths_binary,
        n_resamples=bootstrap_n,
        seed=seed,
    )
    recall_ci = bootstrap_ci_binary(
        lambda tp, fp, tn, fn: tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        predictions_binary,
        ground_truths_binary,
        n_resamples=bootstrap_n,
        seed=seed,
    )
    f1_ci = bootstrap_ci_binary(
        lambda tp, fp, tn, fn: 2
        * (tp / (tp + fp))
        * (tp / (tp + fn))
        / ((tp / (tp + fp)) + (tp / (tp + fn)))
        if (tp + fp) > 0 and (tp + fn) > 0
        else 0.0,
        predictions_binary,
        ground_truths_binary,
        n_resamples=bootstrap_n,
        seed=seed,
    )
    # Balanced Accuracy: nur wenn beide Klassen vorhanden
    if has_both_classes:
        ba_ci = bootstrap_ci_binary(
            lambda tp, fp, tn, fn: ((tp / (tp + fn)) + (tn / (tn + fp))) / 2.0
            if (tp + fn) > 0 and (tn + fp) > 0
            else 0.0,
            predictions_binary,
            ground_truths_binary,
            n_resamples=bootstrap_n,
            seed=seed,
        )
        ba_available = True
    else:
        ba_ci = {"median": None, "ci_lower": None, "ci_upper": None}
        ba_available = False

    # MCC: nur wenn beide Klassen vorhanden
    if has_both_classes:
        mcc_value = compute_mcc(metrics.tp, metrics.fp, metrics.tn, metrics.fn)
        mcc_available = True
    else:
        mcc_value = None
        mcc_available = False

    metrics_dict = {
        "n_used": len(predictions_list),
        "n_failed": metrics.n_failed,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "tn": metrics.tn,
        "fn": metrics.fn,
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
        "balanced_accuracy": {
            "value": metrics.balanced_accuracy if ba_available else None,
            "ci_lower": ba_ci["ci_lower"],
            "ci_upper": ba_ci["ci_upper"],
            "available": ba_available,
        },
        "specificity": metrics.specificity,
        "accuracy": metrics.accuracy,
        "mcc": mcc_value,
        "mcc_available": mcc_available,
        "label_distribution": {
            "dataset": dataset_label_distribution,
            "sample": sample_label_distribution,
        },
        "auroc": auroc if auroc_available else None,
        "auroc_available": auroc_available,
        "auroc_bootstrap_skipped_resamples": auroc_bootstrap_skipped if auroc_available else None,
        "parse_stats": {
            **parse_stats,
            "total_judgments": total_judgments,
        },
        "cache_stats": {
            "cache_mode": cache_mode,
            "cache_source": str(cache_path),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        },
    }

    return metrics_dict, predictions_list


# ---------------------------
# Output generation
# ---------------------------


def write_summary_json(metrics: dict[str, Any], out_path: Path) -> None:
    """Schreibt summary.json."""
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def write_summary_md(
    metrics: dict[str, Any],
    predictions_list: list[dict[str, Any]],
    out_path: Path,
    bootstrap_n: int,
) -> None:
    """Schreibt human-readable summary.md."""
    lines = [
        "# Factuality LLM-as-a-Judge Evaluation Summary",
        "",
        f"**Examples used:** {metrics['n_used']}",
        f"**Examples failed:** {metrics['n_failed']}",
        "",
        "## Metrics",
        "",
        "### Binary Classification",
        "",
        f"- **True Positives (TP):** {metrics['tp']}",
        f"- **False Positives (FP):** {metrics['fp']}",
        f"- **True Negatives (TN):** {metrics['tn']}",
        f"- **False Negatives (FN):** {metrics['fn']}",
        "",
        "### Performance Metrics (95% CI, Bootstrap n={bootstrap_n})",
        "",
        f"- **Precision:** {metrics['precision']['value']:.4f} [{metrics['precision']['ci_lower']:.4f}, {metrics['precision']['ci_upper']:.4f}]",
        f"- **Recall:** {metrics['recall']['value']:.4f} [{metrics['recall']['ci_lower']:.4f}, {metrics['recall']['ci_upper']:.4f}]",
        f"- **F1:** {metrics['f1']['value']:.4f} [{metrics['f1']['ci_lower']:.4f}, {metrics['f1']['ci_upper']:.4f}]",
    ]

    # Balanced Accuracy: nur wenn beide Klassen vorhanden
    if metrics.get("balanced_accuracy", {}).get("available", True):
        ba_value = metrics["balanced_accuracy"]["value"]
        if ba_value is not None:
            lines.append(
                f"- **Balanced Accuracy:** {ba_value:.4f} [{metrics['balanced_accuracy']['ci_lower']:.4f}, {metrics['balanced_accuracy']['ci_upper']:.4f}]"
            )
        else:
            lines.append("- **Balanced Accuracy:** N/A (single-class labels)")
    else:
        lines.append("- **Balanced Accuracy:** N/A (single-class labels)")

    lines.extend(
        [
            f"- **Specificity:** {metrics['specificity']:.4f}",
            f"- **Accuracy:** {metrics['accuracy']:.4f}",
        ]
    )

    # MCC: nur wenn beide Klassen vorhanden
    if metrics.get("mcc_available", True) and metrics.get("mcc") is not None:
        lines.append(f"- **MCC:** {metrics['mcc']:.4f}")
    else:
        lines.append("- **MCC:** N/A (single-class labels)")

    # AUROC: nur wenn verfügbar und beide Klassen vorhanden
    if metrics.get("auroc_available") and metrics.get("auroc") is not None:
        auroc_line = f"- **AUROC:** {metrics['auroc']:.4f} (computed on confidence scores)"
        if metrics.get("auroc_bootstrap_skipped_resamples"):
            auroc_line += f" (bootstrap skipped {metrics['auroc_bootstrap_skipped_resamples']} single-class resamples)"
        lines.append(auroc_line)
    else:
        sample_dist = metrics.get("label_distribution", {}).get("sample", {})
        if not sample_dist.get("positives", 0) or not sample_dist.get("negatives", 0):
            lines.append("- **AUROC:** N/A (single-class labels)")
        else:
            lines.append(
                "- **AUROC:** N/A (no continuous score available; requires confidence from judge outputs)"
            )

    # Label Distribution (Dataset vs Sample)
    dataset_dist = metrics.get("label_distribution", {}).get("dataset", {})
    sample_dist = metrics.get("label_distribution", {}).get("sample", {})
    lines.extend(
        [
            "",
            "## Label Distribution",
            "",
            "### Dataset (before sampling):",
            "",
            f"- **Positives (has_error=True):** {dataset_dist.get('positives', 0)}",
            f"- **Negatives (has_error=False):** {dataset_dist.get('negatives', 0)}",
            f"- **Total:** {dataset_dist.get('total', 0)}",
            "",
            "### Sample (used in evaluation, n_used={}):".format(metrics.get("n_used", 0)),
            "",
            f"- **Positives (has_error=True):** {sample_dist.get('positives', 0)}",
            f"- **Negatives (has_error=False):** {sample_dist.get('negatives', 0)}",
            f"- **Total:** {sample_dist.get('total', 0)}",
            "",
            "## Parse Statistics",
            "",
        ]
    )

    # Parse Stats: Prozent basierend auf total_judgments (nicht n_used)
    total_judgments = metrics.get("parse_stats", {}).get("total_judgments", 0)
    if total_judgments > 0:
        json_ok_pct = 100.0 * metrics["parse_stats"]["parsed_json_ok"] / total_judgments
        regex_pct = 100.0 * metrics["parse_stats"]["parsed_regex_fallback"] / total_judgments
        failed_pct = 100.0 * metrics["parse_stats"]["parse_failed"] / total_judgments
        lines.extend(
            [
                f"- **Parsed JSON (OK):** {metrics['parse_stats']['parsed_json_ok']} ({json_ok_pct:.1f}% of {total_judgments} judgments)",
                f"- **Parsed Regex (Fallback):** {metrics['parse_stats']['parsed_regex_fallback']} ({regex_pct:.1f}% of {total_judgments} judgments)",
                f"- **Parse Failed:** {metrics['parse_stats']['parse_failed']} ({failed_pct:.1f}% of {total_judgments} judgments)",
            ]
        )
    else:
        lines.extend(
            [
                f"- **Parsed JSON (OK):** {metrics['parse_stats']['parsed_json_ok']}",
                f"- **Parsed Regex (Fallback):** {metrics['parse_stats']['parsed_regex_fallback']}",
                f"- **Parse Failed:** {metrics['parse_stats']['parse_failed']}",
            ]
        )

    lines.extend(
        [
            "",
            "**Note:** JSON parsing is primary; regex extraction is fallback only.",
            "",
            "## Cache Statistics",
            "",
            f"- **Cache Mode:** {metrics['cache_stats']['cache_mode']}",
            f"- **Cache Hits:** {metrics['cache_stats']['cache_hits']}",
            f"- **Cache Misses:** {metrics['cache_stats']['cache_misses']}",
            "",
        ]
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_run_metadata(
    run_id: str,
    timestamp: str,
    data_path: Path,
    llm_model: str,
    prompt_version: str,
    judge_n: int,
    judge_temperature: float,
    judge_aggregation: str,
    seed: int | None,
    bootstrap_n: int,
    n_total: int,
    n_used: int,
    n_failed: int,
    config_params: dict[str, Any],
    out_path: Path,
) -> None:
    """Schreibt run_metadata.json."""
    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "seed": seed,
        "dataset_path": str(data_path),
        "n_total": n_total,
        "n_used": n_used,
        "n_failed": n_failed,
        "config": {
            "llm_model": llm_model,
            "prompt_version": prompt_version,
            "judge_n": judge_n,
            "judge_temperature": judge_temperature,
            "judge_aggregation": judge_aggregation,
            "bootstrap_n": bootstrap_n,
            **config_params,
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluiert LLM-as-a-Judge auf Factuality (FRANK/FineSumFact)"
    )
    ap.add_argument(
        "--data",
        type=str,
        required=True,
        help="Pfad zur JSONL-Datei (article_text, summary_text, gold_has_error)",
    )
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM-Modell")
    ap.add_argument(
        "--prompt_version",
        type=str,
        default="v2_binary",
        help="Prompt-Version (v2_binary für binary verdict)",
    )
    ap.add_argument("--max_examples", type=int, help="Maximale Anzahl Beispiele")
    ap.add_argument("--seed", type=int, help="Random seed für Reproduzierbarkeit")
    ap.add_argument("--bootstrap_n", type=int, default=2000, help="Anzahl Bootstrap-Resamples")
    ap.add_argument(
        "--out_dir", type=str, help="Output-Verzeichnis (default: results/evaluation/factuality)"
    )
    ap.add_argument("--retries", type=int, default=1, help="Anzahl Retries bei Fehlern")
    ap.add_argument("--sleep-s", type=float, default=1.0, help="Sleep zwischen Retries")
    ap.add_argument(
        "--cache_mode",
        type=str,
        choices=["off", "read", "write"],
        default="write",
        help="Cache-Modus",
    )
    ap.add_argument(
        "--cache_path",
        type=str,
        help="Optional: expliziter Cache-Pfad (überschreibt run-local cache.jsonl)",
    )
    ap.add_argument("--judge_n", type=int, help="Anzahl Judgements (default: ENV JUDGE_N oder 3)")
    ap.add_argument(
        "--judge_temperature",
        type=float,
        help="Judge-Temperatur (default: ENV JUDGE_TEMPERATURE oder 0.0)",
    )
    ap.add_argument(
        "--judge_aggregation",
        type=str,
        choices=["mean", "median", "majority"],
        help="Aggregationsmethode (default: ENV JUDGE_AGGREGATION oder majority)",
    )

    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        ap.error(f"Datei nicht gefunden: {data_path}")

    llm_model = args.model
    prompt_version = args.prompt_version
    max_examples = args.max_examples
    seed = args.seed
    bootstrap_n = args.bootstrap_n

    # Judge-Parameter (ENV oder CLI)
    judge_n = args.judge_n or int(os.getenv("JUDGE_N", "3"))
    judge_temperature = (
        args.judge_temperature
        if args.judge_temperature is not None
        else float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
    )
    judge_aggregation = args.judge_aggregation or os.getenv("JUDGE_AGGREGATION", "majority")

    # Cache-Mode
    use_cache = args.cache_mode != "off"
    cache_mode = args.cache_mode

    examples = load_jsonl(data_path)

    # Run ID
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"judge_factuality_{ts}_{llm_model}_{prompt_version}"
    if seed is not None:
        run_id += f"_seed{seed}"

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir) / run_id
    else:
        out_dir = Path("results") / "evaluation" / "factuality" / run_id
    ensure_dir(out_dir)

    preds_path = out_dir / "predictions.jsonl"
    # Cache-Pfad: explizit angegeben oder run-local
    if args.cache_path:
        cache_path = Path(args.cache_path)
        cache_source = str(cache_path)
    else:
        cache_path = out_dir / "cache.jsonl"
        cache_source = str(cache_path)
    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    metadata_path = out_dir / "run_metadata.json"

    print(f"Data: {data_path} (examples={len(examples)})")
    print(f"Model: {llm_model} | Prompt: {prompt_version}")
    print(f"Judge: n={judge_n}, temperature={judge_temperature}, aggregation={judge_aggregation}")
    print(f"Seed: {seed}")
    print(f"Bootstrap resamples: {bootstrap_n}")
    print(f"Cache: {cache_mode} ({cache_path})")
    if args.cache_path:
        print(f"Cache Source: {cache_path} (explicit)")
    print(f"Output: {out_dir}")

    metrics, predictions_list = run_eval(
        examples=examples,
        llm_model=llm_model,
        max_examples=max_examples,
        preds_path=preds_path,
        judge_n=judge_n,
        judge_temperature=judge_temperature,
        judge_aggregation=judge_aggregation,
        prompt_version=prompt_version,
        retries=args.retries,
        sleep_s=args.sleep_s,
        use_cache=use_cache,
        cache_path=cache_path,
        seed=seed,
        bootstrap_n=bootstrap_n,
        cache_mode=cache_mode,
    )

    # Write outputs
    write_summary_json(metrics, summary_json_path)
    write_summary_md(metrics, predictions_list, summary_md_path, bootstrap_n)
    write_run_metadata(
        run_id=run_id,
        timestamp=ts,
        data_path=data_path,
        llm_model=llm_model,
        prompt_version=prompt_version,
        judge_n=judge_n,
        judge_temperature=judge_temperature,
        judge_aggregation=judge_aggregation,
        seed=seed,
        bootstrap_n=bootstrap_n,
        n_total=len(examples),
        n_used=metrics["n_used"],
        n_failed=metrics["n_failed"],
        config_params={
            "max_examples": max_examples,
            "retries": args.retries,
            "sleep_s": args.sleep_s,
            "cache": use_cache,
            "cache_mode": cache_mode,
        },
        out_path=metadata_path,
    )

    print("\n" + "=" * 60)
    print("Evaluation abgeschlossen!")
    print("=" * 60)
    print("\nMetriken:")
    print(
        f"  Precision:  {metrics['precision']['value']:.4f} [{metrics['precision']['ci_lower']:.4f}, {metrics['precision']['ci_upper']:.4f}]"
    )
    print(
        f"  Recall:     {metrics['recall']['value']:.4f} [{metrics['recall']['ci_lower']:.4f}, {metrics['recall']['ci_upper']:.4f}]"
    )
    print(
        f"  F1:         {metrics['f1']['value']:.4f} [{metrics['f1']['ci_lower']:.4f}, {metrics['f1']['ci_upper']:.4f}]"
    )

    # Balanced Accuracy
    ba_available = metrics.get("balanced_accuracy", {}).get("available", True)
    ba_value = metrics.get("balanced_accuracy", {}).get("value")
    if ba_available and ba_value is not None:
        ba_ci_lower = metrics.get("balanced_accuracy", {}).get("ci_lower")
        ba_ci_upper = metrics.get("balanced_accuracy", {}).get("ci_upper")
        if ba_ci_lower is not None and ba_ci_upper is not None:
            print(f"  Balanced Accuracy: {ba_value:.4f} [{ba_ci_lower:.4f}, {ba_ci_upper:.4f}]")
        else:
            print(f"  Balanced Accuracy: {ba_value:.4f}")
    else:
        print("  Balanced Accuracy: N/A (single-class labels)")

    # MCC
    mcc_available = metrics.get("mcc_available", True)
    mcc_value = metrics.get("mcc")
    if mcc_available and mcc_value is not None:
        print(f"  MCC:        {mcc_value:.4f}")
    else:
        print("  MCC:        N/A (single-class labels)")

    # AUROC
    if metrics.get("auroc_available") and metrics.get("auroc") is not None:
        auroc_msg = f"  AUROC:      {metrics['auroc']:.4f} (computed on confidence)"
        if metrics.get("auroc_bootstrap_skipped_resamples"):
            auroc_msg += (
                f" (bootstrap skipped {metrics['auroc_bootstrap_skipped_resamples']} resamples)"
            )
        print(auroc_msg)
    else:
        sample_dist = metrics.get("label_distribution", {}).get("sample", {})
        if not sample_dist.get("positives", 0) or not sample_dist.get("negatives", 0):
            print("  AUROC:      N/A (single-class labels)")
        else:
            print("  AUROC:      N/A (no continuous score available)")

    # Parse Stats: Prozent basierend auf total_judgments
    total_judgments = metrics.get("parse_stats", {}).get("total_judgments", 0)
    print("\nParse Stats:")
    if total_judgments > 0:
        json_ok_pct = 100.0 * metrics["parse_stats"]["parsed_json_ok"] / total_judgments
        regex_pct = 100.0 * metrics["parse_stats"]["parsed_regex_fallback"] / total_judgments
        failed_pct = 100.0 * metrics["parse_stats"]["parse_failed"] / total_judgments
        print(
            f"  JSON OK:    {metrics['parse_stats']['parsed_json_ok']} ({json_ok_pct:.1f}% of {total_judgments} judgments)"
        )
        print(
            f"  Regex Fallback: {metrics['parse_stats']['parsed_regex_fallback']} ({regex_pct:.1f}% of {total_judgments} judgments)"
        )
        print(
            f"  Failed:     {metrics['parse_stats']['parse_failed']} ({failed_pct:.1f}% of {total_judgments} judgments)"
        )
    else:
        print(f"  JSON OK:    {metrics['parse_stats']['parsed_json_ok']}")
        print(f"  Regex Fallback: {metrics['parse_stats']['parsed_regex_fallback']}")
        print(f"  Failed:     {metrics['parse_stats']['parse_failed']}")
    print(f"\nArtefakte gespeichert in: {out_dir}")


if __name__ == "__main__":
    main()
