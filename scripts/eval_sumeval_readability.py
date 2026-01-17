"""
Evaluiert den ReadabilityAgent auf SumEval-Readability-Ratings.

Input:
- data/sumeval/sumeval_clean.jsonl (pro Zeile z.B.):
  {
    "article": "...",
    "summary": "...",
    "gt": {"readability": 3.7},
    "meta": {...}
  }

Wichtig:
- Der Agent-Score bleibt in [0, 1].
- Die Ground-Truth Readability-Skala (typisch 1..5) wird auf [0, 1] normalisiert:
    gt_norm = (gt_raw - 1) / 4
  Standard: gt_min=1.0, gt_max=5.0 (konfigurierbar).

Output:
- results/evaluation/readability/<run_id>/
  - predictions.jsonl (pro Beispiel)
  - summary.json (alle Metriken + CIs + Metadaten)
  - summary.md (human-readable)
  - run_metadata.json (timestamp, git_commit, python version, seed, etc.)

Metriken:
- Pearson r, Spearman ρ (mit Bootstrap-CIs)
- MAE, RMSE (mit Bootstrap-CIs)
- Optional: R²
- n_used, n_failed
"""

import argparse
from collections import Counter
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
from app.services.agents.readability.calibration import (
    apply_calibration,
    apply_isotonic_calibration,
    load_calibration_params,
)
from app.services.agents.readability.readability_agent import ReadabilityAgent

load_dotenv()


# ---------------------------
# IO helpers
# ---------------------------


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
# Metrics
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
    # average rank for ties
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
    """R² (coefficient of determination)."""
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
    xs: list[float],
    ys: list[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> dict[str, float]:
    """
    Berechnet Bootstrap-Konfidenzintervalle für eine Metrik.

    Returns:
        {
            "median": float,
            "ci_lower": float (2.5th percentile),
            "ci_upper": float (97.5th percentile),
        }
    """
    if not xs or not ys or len(xs) != len(ys):
        return {"median": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    rng = random.Random(seed)
    n = len(xs)
    resamples: list[float] = []

    for _ in range(n_resamples):
        # Resample with replacement
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
    # clamp then normalize
    if x < min_v:
        x = min_v
    if x > max_v:
        x = max_v
    return (x - min_v) / (max_v - min_v)


# ---------------------------
# Caching
# ---------------------------


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_key(article: str, summary: str, model: str, prompt_version: str) -> str:
    # Stabiler Key: hash über Inhalte + settings
    payload = json.dumps(
        {"model": model, "prompt_version": prompt_version, "article": article, "summary": summary},
        ensure_ascii=False,
        sort_keys=True,
    )
    return _sha256(payload)


def load_cache(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Lädt Cache und gibt zwei Dicts zurück:
    - cache_by_key: Lookup über cache_key (SHA256)
    - cache_by_id: Lookup über example_id
    """
    cache_by_key: dict[str, dict[str, Any]] = {}
    cache_by_id: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return cache_by_key, cache_by_id
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
                cache_by_key[k] = v
                # Auch nach example_id indexieren (falls vorhanden)
                example_id = v.get("example_id")
                if example_id:
                    cache_by_id[example_id] = v
    return cache_by_key, cache_by_id


def append_cache(path: Path, key: str, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


# ---------------------------
# Issue type extraction
# ---------------------------


def extract_issue_types_counts(issue_spans: list[dict[str, Any]]) -> dict[str, int]:
    """Extrahiert Issue-Type-Counts aus issue_spans."""
    types = []
    for span in issue_spans:
        issue_type = span.get("issue_type")
        if issue_type:
            types.append(issue_type)
    return dict(Counter(types))


def get_max_severity(issue_spans: list[dict[str, Any]]) -> str | None:
    """Gibt die höchste Severity zurück (high > medium > low)."""
    severity_order = {"high": 3, "medium": 2, "low": 1}
    max_sev = None
    max_val = 0
    for span in issue_spans:
        sev = span.get("severity")
        if sev and severity_order.get(sev, 0) > max_val:
            max_val = severity_order[sev]
            max_sev = sev
    return max_sev


def get_top_issues(issue_spans: list[dict[str, Any]], max_n: int = 3) -> list[dict[str, Any]]:
    """Gibt die Top-N Issues zurück (sortiert nach Severity)."""
    severity_order = {"high": 3, "medium": 2, "low": 1}
    sorted_spans = sorted(
        issue_spans,
        key=lambda s: severity_order.get(s.get("severity", "low"), 0),
        reverse=True,
    )
    return [
        {
            "type": s.get("issue_type"),
            "severity": s.get("severity"),
            "span_indices": [s.get("start_char"), s.get("end_char")],
        }
        for s in sorted_spans[:max_n]
    ]


# ---------------------------
# Core eval
# ---------------------------


def run_eval(
    rows: list[dict[str, Any]],
    llm_model: str,
    max_examples: int | None,
    preds_path: Path,
    gt_min: float,
    gt_max: float,
    retries: int,
    sleep_s: float,
    use_cache: bool,
    cache_path: Path,
    prompt_version: str,
    seed: int | None,
    bootstrap_n: int,
    calibration_path: Path | None = None,
    score_source: str = "agent",  # "agent" oder "judge"
    cache_mode: str = "write",  # "off", "read", "write"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """
    Führt die Evaluation durch.

    Returns:
        (metrics_dict, predictions_list)
    """
    if seed is not None:
        random.seed(seed)

    llm = OpenAIClient(model_name=llm_model)
    agent = ReadabilityAgent(llm, prompt_version=prompt_version)

    # Cache-Mode Logik
    cache_read_enabled = (cache_mode == "read" or cache_mode == "write") and use_cache
    cache_write_enabled = cache_mode == "write" and use_cache
    cache_by_key, cache_by_id = load_cache(cache_path) if cache_read_enabled else ({}, {})

    cache_hits = 0
    cache_misses = 0

    # Load calibration if provided
    calibration_method = None
    calibration_a = None
    calibration_b = None
    calibration_isotonic = None
    if calibration_path and calibration_path.exists():
        print(f"Lade Kalibrierung aus: {calibration_path}")
        calibration_method, calibration_params = load_calibration_params(calibration_path)
        if calibration_method == "linear":
            calibration_a, calibration_b = calibration_params
            print(
                f"  Kalibrierung (linear): pred_cal = {calibration_a:.6f} * pred + {calibration_b:.6f}"
            )
        elif calibration_method == "isotonic":
            calibration_isotonic = calibration_params
            print("  Kalibrierung (isotonic): Monotone Funktion")
        else:
            raise ValueError(f"Unbekannte Kalibrierungsmethode: {calibration_method}")
    elif calibration_path:
        print(
            f"Warnung: Kalibrierungspfad angegeben, aber Datei nicht gefunden: {calibration_path}"
        )

    gt_norms: list[float] = []
    preds: list[float] = []
    predictions_list: list[dict[str, Any]] = []
    n_seen = 0
    n_used = 0
    n_failed = 0
    n_skipped = 0

    if preds_path.exists():
        preds_path.unlink()

    for row in rows:
        if max_examples is not None and n_used >= max_examples:
            break

        n_seen += 1

        gt_raw = row.get("gt", {}).get("readability")  # <- Readability statt Coherence
        article = row.get("article")
        summary = row.get("summary")
        meta = row.get("meta", {})
        # Erzeuge eindeutige example_id: doc_id + Hash der Summary (falls mehrere Summaries pro Artikel)
        base_id = meta.get("doc_id") or meta.get("id") or f"example_{n_seen}"
        if summary:
            import hashlib

            summary_hash = hashlib.sha256(summary.encode("utf-8")).hexdigest()[:8]
            example_id = f"{base_id}_{summary_hash}"
        else:
            example_id = f"{base_id}_{n_seen}"

        if (
            gt_raw is None
            or not isinstance(article, str)
            or not article.strip()
            or not isinstance(summary, str)
            or not summary.strip()
        ):
            n_skipped += 1
            continue

        try:
            gt_raw_f = float(gt_raw)
        except Exception:
            n_skipped += 1
            continue

        gt_norm = normalize_to_0_1(gt_raw_f, min_v=gt_min, max_v=gt_max)

        # Cache-Lookup: zuerst nach example_id, dann nach cache_key
        cached = None
        if cache_read_enabled:
            cached = cache_by_id.get(example_id)
            if not cached:
                key = cache_key(article, summary, llm_model, prompt_version)
                cached = cache_by_key.get(key)
            if cached:
                cache_hits += 1
            else:
                cache_misses += 1
        else:
            cache_misses += 1

        pred_score: float | None = None
        payload: dict[str, Any] | None = None
        issue_spans: list[dict[str, Any]] = []

        if cached is not None:
            pred_score = float(cached.get("pred_score", 0.0))
            payload = cached
            issue_spans = payload.get("issue_spans", [])
            # Cache-Hit: kein LLM-Call nötig
        else:
            last_err: str | None = None
            for attempt in range(retries + 1):
                try:
                    res = agent.run(article_text=article, summary_text=summary, meta=meta)
                    pred_score = float(res.score)  # expected 0..1
                    issue_spans = [s.model_dump() for s in getattr(res, "issue_spans", [])]
                    payload = {
                        "example_id": example_id,  # Speichere example_id für Lookup
                        "pred_score": pred_score,
                        "issue_spans": issue_spans,
                        "details": res.details,
                    }
                    if cache_write_enabled:
                        key = cache_key(article, summary, llm_model, prompt_version)
                        append_cache(cache_path, key, payload)
                        cache_by_key[key] = payload
                        cache_by_id[example_id] = payload  # Auch nach ID indexieren
                    break
                except Exception as e:
                    last_err = str(e)
                    if attempt < retries:
                        time.sleep(sleep_s)
                    else:
                        n_failed += 1
                        pred_score = None
                        payload = None
                        issue_spans = []

                        # Write failed prediction
                        rec = {
                            "example_id": example_id,
                            "model": llm_model,
                            "prompt_version": prompt_version,
                            "pred": None,
                            "pred_1_5": None,
                            "gt_raw": gt_raw_f,
                            "gt_norm": gt_norm,
                            "num_issues": 0,
                            "max_severity": None,
                            "issue_types_counts": {},
                            "failed": True,
                            "error": last_err,
                        }
                        predictions_list.append(rec)
                        with preds_path.open("a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        continue

        if pred_score is None:
            continue

        # Extract Judge-Score aus details (falls vorhanden)
        judge_score = None
        judge_result_data = None
        if payload and payload.get("details"):
            details = payload.get("details", {})
            if "judge" in details:
                judge_result_data = details["judge"]
                if isinstance(judge_result_data, dict):
                    judge_score = judge_result_data.get("final_score_norm")
            # Fallback: judge_score direkt in details
            if judge_score is None:
                judge_score = details.get("judge_score")

        # clamp pred into [0,1] defensively (should already be true)
        if pred_score < 0.0:
            pred_score = 0.0
        if pred_score > 1.0:
            pred_score = 1.0

        # Apply calibration if provided
        pred_raw = pred_score
        if (
            calibration_method == "linear"
            and calibration_a is not None
            and calibration_b is not None
        ):
            pred_score = apply_calibration(pred_score, calibration_a, calibration_b)
        elif calibration_method == "isotonic" and calibration_isotonic is not None:
            pred_score = apply_isotonic_calibration(pred_score, calibration_isotonic)

        # Score-Auswahl basierend auf score_source
        final_pred = pred_score
        if score_source == "judge" and judge_score is not None:
            final_pred = judge_score
        elif score_source == "judge" and judge_score is None:
            # Warnung: Judge-Score nicht verfügbar, verwende Agent-Score
            print(f"Warnung: Judge-Score nicht verfügbar für {example_id}, verwende Agent-Score")

        pred_1_5 = 1.0 + 4.0 * final_pred  # Map [0,1] -> [1,5]

        gt_norms.append(gt_norm)
        preds.append(final_pred)
        n_used += 1

        # Extract issue info
        issue_types_counts = extract_issue_types_counts(issue_spans)
        max_severity = get_max_severity(issue_spans)
        top_issues = get_top_issues(issue_spans, max_n=3)

        rec = {
            "example_id": example_id,
            "model": llm_model,
            "prompt_version": prompt_version,
            "pred": final_pred,
            "pred_agent": pred_score,  # Immer speichern
            "pred_judge": judge_score,  # Falls vorhanden
            "pred_raw": pred_raw
            if calibration_method is not None
            else None,  # Store raw if calibrated
            "pred_1_5": pred_1_5,
            "gt_raw": gt_raw_f,
            "gt_norm": gt_norm,
            "num_issues": len(issue_spans),
            "max_severity": max_severity,
            "issue_types_counts": issue_types_counts,
            "top_issues": top_issues,
            "score_source": score_source,
        }

        # Judge-Metadaten hinzufügen (falls vorhanden)
        if judge_result_data:
            rec["judge_committee"] = judge_result_data.get("committee")
            rec["judge_aggregation"] = judge_result_data.get("aggregation")
            rec["judge_outputs_count"] = len(judge_result_data.get("outputs", []))
        predictions_list.append(rec)

        with preds_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if n_used % 25 == 0:
            # Erweiterte Progress-Zeile mit agent/judge Scores
            agent_str = f"agent={pred_score:.3f}"
            judge_str = f"judge={judge_score:.3f}" if judge_score is not None else "judge=None"
            used_str = f"used={final_pred:.3f}({score_source})"
            print(
                f"[{n_used}] gt_norm={gt_norm:.3f} {agent_str} {judge_str} {used_str} (seen={n_seen}, skipped={n_skipped}, failed={n_failed})"
            )

    # Collapse-Detector: Wenn >80% der predictions im selben Bucket
    collapse_detected = False
    collapse_bucket = None
    collapse_percentage = 0.0
    if preds:
        from collections import Counter

        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

        def bin_value(v: float) -> str:
            for i in range(len(bins) - 1):
                if bins[i] <= v < bins[i + 1]:
                    return f"[{bins[i]:.1f}, {bins[i + 1]:.1f})"
            if v >= bins[-1]:
                return f"[{bins[-1]:.1f}, 1.0]"
            return "[0.0, 0.2)"

        pred_bins = Counter([bin_value(p) for p in preds])
        max_bin_count = max(pred_bins.values()) if pred_bins else 0
        collapse_detected = max_bin_count > 0.8 * len(preds)
        if collapse_detected:
            collapse_bucket = max(pred_bins.items(), key=lambda x: x[1])[0]
            collapse_percentage = (max_bin_count / len(preds)) * 100

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
        "n_failed": n_failed,
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
        "collapse_detected": collapse_detected,
        "collapse_bucket": collapse_bucket,
        "collapse_percentage": collapse_percentage,
        "cache_stats": {
            "cache_mode": cache_mode,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
        },
    }
    return metrics, predictions_list


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
    max_examples: int | None = None,
) -> None:
    """Schreibt human-readable summary.md."""
    lines = [
        "# Readability Evaluation Summary",
        "",
        "**Dataset:** SummEval",
        f"**Examples used:** {metrics['n_used']}",
        f"**Examples skipped:** {metrics['n_skipped']}",
        f"**Examples failed:** {metrics['n_failed']}",
        "",
    ]

    # Collapse-Warnung
    if metrics.get("collapse_detected", False):
        collapse_pct = metrics.get("collapse_percentage", 0.0)
        collapse_bucket = metrics.get("collapse_bucket", "unknown")
        lines.extend(
            [
                "## ⚠️ OUTPUT-COLLAPSE ERKANNT",
                "",
                f"**Warnung:** {collapse_pct:.1f}% der Predictions landen im selben Bucket ({collapse_bucket}).",
                "",
                "**Ursache:** LLM-Output-Collapse oder Kalibrierungsproblem.",
                "**Auswirkung:** Ranking-Korrelation (Spearman) ist vermutlich schlecht, auch wenn MAE niedrig ist.",
                "",
                "---",
                "",
            ]
        )

    # Quick-Run Info
    is_quick = bootstrap_n < 1000 or (max_examples is not None and max_examples < 150)
    if is_quick:
        n_str = str(max_examples) if max_examples is not None else "all"
        lines.append(f"**Quick run:** n={n_str}, bootstrap_n={bootstrap_n} (reduced)")
        lines.append("")

    # Cache-Statistiken
    cache_stats = metrics.get("cache_stats", {})
    cache_mode = cache_stats.get("cache_mode", "unknown")
    cache_hits = cache_stats.get("cache_hits", 0)
    cache_misses = cache_stats.get("cache_misses", 0)
    lines.extend(
        [
            "## Cache",
            "",
            f"- **Cache Mode:** {cache_mode}",
            f"- **Cache Hits:** {cache_hits}",
            f"- **Cache Misses:** {cache_misses}",
            "",
        ]
    )

    lines.extend(
        [
            "## Setup",
            "",
            "- **Label-Definition:** `gt.readability` (1-5 Likert-Skala)",
            f"- **Normalisierung:** `gt_norm = (gt_raw - {metrics['gt_normalization']['raw_min']}) / ({metrics['gt_normalization']['raw_max']} - {metrics['gt_normalization']['raw_min']})`",
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
            "",
            "## Distributions",
            "",
        ]
    )

    # Berechne Verteilungen
    from collections import Counter

    valid_preds = [
        p
        for p in predictions_list
        if not p.get("failed", False) and p.get("gt_norm") is not None and p.get("pred") is not None
    ]

    # Bins: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def bin_value(v: float) -> str:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                return f"[{bins[i]:.1f}, {bins[i + 1]:.1f})"
        if v >= bins[-1]:
            return f"[{bins[-1]:.1f}, 1.0]"
        return "[0.0, 0.2)"

    gt_bins = Counter([bin_value(p.get("gt_norm", 0.0)) for p in valid_preds])
    pred_bins = Counter([bin_value(p.get("pred", 0.0)) for p in valid_preds])

    lines.append("### GT (gt_norm) Distribution")
    lines.append("")
    for bin_name in sorted(gt_bins.keys()):
        count = gt_bins[bin_name]
        lines.append(f"- {bin_name}: {count}")

    lines.extend(
        [
            "",
            "### Predictions (pred) Distribution",
            "",
            "Prediction distribution (counts per bucket/value):",
            "",
        ]
    )

    for bin_name in sorted(pred_bins.keys()):
        count = pred_bins[bin_name]
        lines.append(f"- {bin_name}: {count}")

    lines.extend(
        [
            "",
            "---",
            "",
        ]
    )

    # Top 5 qualitative examples (highest severity issues)
    high_severity_examples = [
        p
        for p in predictions_list
        if not p.get("failed", False) and p.get("max_severity") == "high"
    ]
    if high_severity_examples:
        lines.extend(
            [
                "## Qualitative Examples (Top 5 High-Severity Issues)",
                "",
            ]
        )
        for i, ex in enumerate(high_severity_examples[:5], 1):
            top_issues_str = ", ".join(
                [
                    f"{iss.get('type', 'unknown')} ({iss.get('severity', 'unknown')})"
                    for iss in ex.get("top_issues", [])[:2]
                ]
            )
            lines.extend(
                [
                    f"### Example {i}",
                    f"- **ID:** {ex.get('example_id', 'unknown')}",
                    f"- **GT:** {ex.get('gt_raw', 0):.2f} (norm: {ex.get('gt_norm', 0):.3f})",
                    f"- **Pred:** {ex.get('pred', 0):.3f}",
                    f"- **Issues:** {ex.get('num_issues', 0)} (max severity: {ex.get('max_severity', 'none')})",
                    f"- **Top Issues:** {top_issues_str or 'none'}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Limitations",
            "",
            "- SummEval Readability-Ratings haben intrinsische Varianz (Rater-Noise), was die Unsicherheit erhöht.",
            "- Die Bootstrap-CIs quantifizieren diese Unsicherheit, aber größere Samples würden engere CIs liefern.",
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
            "bootstrap_n": bootstrap_n,
            **config_params,
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluiert ReadabilityAgent auf SummEval")
    ap.add_argument("--data", type=str, required=True, help="Pfad zur JSONL-Datei")
    ap.add_argument("--data1", type=str, help="Alias für --data (Backward-Compat)")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM-Modell")
    ap.add_argument("--llm-model", type=str, help="Alias für --model")
    ap.add_argument("--prompt_version", type=str, help="Prompt-Version (default: ENV oder v1)")
    ap.add_argument("--max_examples", type=int, help="Maximale Anzahl Beispiele")
    ap.add_argument("--max", type=int, help="Alias für --max_examples")
    ap.add_argument("--seed", type=int, help="Random seed für Reproduzierbarkeit")
    ap.add_argument(
        "--bootstrap_n",
        type=int,
        default=2000,
        help="Anzahl Bootstrap-Resamples (Quick: 300-500, Final: 2000)",
    )
    ap.add_argument(
        "--out_dir", type=str, help="Output-Verzeichnis (default: results/evaluation/readability)"
    )
    ap.add_argument("--gt-min", type=float, default=1.0, help="GT-Minimum (default: 1.0)")
    ap.add_argument("--gt-max", type=float, default=5.0, help="GT-Maximum (default: 5.0)")
    ap.add_argument("--retries", type=int, default=1, help="Anzahl Retries bei Fehlern")
    ap.add_argument("--sleep-s", type=float, default=1.0, help="Sleep zwischen Retries (Sekunden)")
    ap.add_argument(
        "--cache",
        action="store_true",
        default=True,
        help="Aktiviere Caching (default: ON, deprecated: use --cache_mode)",
    )
    ap.add_argument(
        "--no-cache",
        dest="cache",
        action="store_false",
        help="Deaktiviere Caching (deprecated: use --cache_mode)",
    )
    ap.add_argument(
        "--cache_mode",
        type=str,
        choices=["off", "read", "write"],
        help="Cache-Modus: 'off' (kein Cache), 'read' (nur lesen), 'write' (lesen+schreiben, default)",
    )
    ap.add_argument(
        "--calibration_path", type=str, help="Pfad zu calibration_params.json (optional)"
    )
    ap.add_argument(
        "--score_source",
        type=str,
        default="agent",
        choices=["agent", "judge"],
        help="Score-Quelle: 'agent' (Standard) oder 'judge' (LLM-as-a-Judge)",
    )

    args = ap.parse_args()

    # CLI-Fix: --data hat Priorität, --data1 als Fallback
    data_path_str = args.data or args.data1
    if not data_path_str:
        ap.error("--data oder --data1 muss angegeben werden")

    data_path = Path(data_path_str)
    if not data_path.exists():
        ap.error(f"Datei nicht gefunden: {data_path}")

    # Model: --model hat Priorität, --llm-model als Fallback
    llm_model = args.model or args.llm_model or "gpt-4o-mini"

    # Max examples
    max_examples = args.max_examples or args.max

    # Prompt version
    prompt_version = args.prompt_version or os.getenv("READABILITY_PROMPT_VERSION", "v1")

    # Seed
    seed = args.seed

    # Bootstrap
    bootstrap_n = args.bootstrap_n

    # Cache-Mode: --cache_mode hat Priorität, sonst --cache/--no-cache
    if args.cache_mode:
        cache_mode = args.cache_mode
        use_cache = cache_mode != "off"
    else:
        # Legacy: --cache/--no-cache
        use_cache = args.cache
        cache_mode = "write" if use_cache else "off"

    rows = load_jsonl(data_path)

    # Run ID: timestamp-basiert
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"readability_{ts}_{llm_model}_{prompt_version}"
    if seed is not None:
        run_id += f"_seed{seed}"

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir) / run_id
    else:
        out_dir = Path("results") / "evaluation" / "readability" / run_id
    ensure_dir(out_dir)

    preds_path = out_dir / "predictions.jsonl"
    cache_path = out_dir / "cache.jsonl"
    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    metadata_path = out_dir / "run_metadata.json"

    print(f"Data: {data_path} (rows={len(rows)})")
    print(f"Model: {llm_model} | Prompt: {prompt_version}")
    print(f"GT scale: [{args.gt_min}, {args.gt_max}] -> [0,1]")
    print(f"Seed: {seed}")
    print(f"Bootstrap resamples: {bootstrap_n}")
    print(f"Cache: {cache_mode} ({cache_path})")
    calibration_path = Path(args.calibration_path) if args.calibration_path else None
    if calibration_path:
        print(f"Calibration: {calibration_path}")
    print(f"Output: {out_dir}")

    metrics, predictions_list = run_eval(
        rows=rows,
        llm_model=llm_model,
        max_examples=max_examples,
        preds_path=preds_path,
        gt_min=args.gt_min,
        gt_max=args.gt_max,
        retries=args.retries,
        sleep_s=args.sleep_s,
        use_cache=use_cache,
        cache_path=cache_path,
        prompt_version=prompt_version,
        seed=seed,
        bootstrap_n=bootstrap_n,
        calibration_path=calibration_path,
        score_source=args.score_source,
        cache_mode=cache_mode,
    )

    # Write outputs
    write_summary_json(metrics, summary_json_path)
    write_summary_md(metrics, predictions_list, summary_md_path, bootstrap_n, max_examples)
    write_run_metadata(
        run_id=run_id,
        timestamp=ts,
        data_path=data_path,
        llm_model=llm_model,
        prompt_version=prompt_version,
        seed=seed,
        bootstrap_n=bootstrap_n,
        n_total=len(rows),
        n_used=metrics["n_used"],
        n_failed=metrics["n_failed"],
        config_params={
            "max_examples": max_examples,
            "gt_min": args.gt_min,
            "gt_max": args.gt_max,
            "retries": args.retries,
            "sleep_s": args.sleep_s,
            "cache": use_cache,
            "cache_mode": cache_mode,
            "cache_hits": metrics.get("cache_stats", {}).get("cache_hits", 0),
            "cache_misses": metrics.get("cache_stats", {}).get("cache_misses", 0),
            "calibration_path": str(calibration_path) if calibration_path else None,
        },
        out_path=metadata_path,
    )

    print("\n" + "=" * 60)
    print("Evaluation abgeschlossen!")
    print("=" * 60)
    print("\nMetriken:")
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
    print(f"\nArtefakte gespeichert in: {out_dir}")


if __name__ == "__main__":
    main()
