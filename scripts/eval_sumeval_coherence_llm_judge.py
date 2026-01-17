"""
LLM-as-a-Judge Baseline für Coherence-Evaluation auf SummEval.

Input:
- data/sumeval/sumeval_clean.jsonl (gleiche Struktur wie eval_sumeval_coherence.py)

Judge-Ansatz:
- Rubrik-basierte Bewertung (1-5) mit striktem JSON-Output
- Temperature=0 für Determinismus
- Mehrfachurteile (n_judgments) für Self-Consistency
- Mean-Aggregation über normalisierte Scores

Output:
- results/evaluation/coherence_judge/<run_id>/
  - predictions.jsonl
  - summary.json
  - summary.md
  - run_metadata.json
  - cache.jsonl (optional)

Metriken:
- Pearson r, Spearman ρ, MAE, RMSE, R² (mit Bootstrap-CIs)
- n_used, n_skipped, n_failed
"""

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any

from dotenv import load_dotenv

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.openai_client import OpenAIClient

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
# Metrics (reused from eval_sumeval_coherence.py)
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
# Caching
# -------------------


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def cache_key(summary: str, judge_model: str, rubric_version: str, judgment_idx: int) -> str:
    payload = json.dumps(
        {
            "model": judge_model,
            "rubric_version": rubric_version,
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
# Judge Prompt & Rubric
# ---------------------------


def get_judge_prompt_v1(summary: str) -> str:
    """Rubrik-basierter Prompt für Coherence-Bewertung (v1)."""
    return f"""Du bist ein Experte für Textkohärenz. Bewerte die folgende Summary auf einer Skala von 1-5 hinsichtlich ihrer Kohärenz.

**Rubrik für Kohärenz (1-5):**

1 = Sehr inkohärent:
- Kein logischer Fluss, abrupte Sprünge
- Widersprüche innerhalb der Summary
- Unklare Referenzen (Pronomen/Bezüge unklar)
- Fehlende Verknüpfungen zwischen Sätzen

2 = Inkohärent:
- Schwacher logischer Fluss, mehrere Probleme
- Einige Widersprüche oder unklare Referenzen
- Schlechte Übergänge zwischen Sätzen

3 = Teilweise kohärent:
- Grundlegender logischer Fluss vorhanden
- Einzelne Probleme mit Übergängen oder Referenzen
- Keine schwerwiegenden Widersprüche

4 = Kohärent:
- Guter logischer Fluss
- Klare Referenzen und Übergänge
- Keine nennenswerten Probleme

5 = Sehr kohärent:
- Exzellenter logischer Fluss
- Perfekte Übergänge und klare Referenzen
- Keine Widersprüche, sehr gut strukturiert

**Kriterien:**
a) Logischer Fluss/Übergänge zwischen Sätzen
b) Keine Widersprüche innerhalb der Summary
c) Klare Referenzen (Pronomen/Bezüge eindeutig)
d) Keine abrupten Sprünge/fehlende Verknüpfungen

**Summary:**
{summary}

**Aufgabe:**
Bewerte die Kohärenz der Summary und gib deine Antwort AUSSCHLIESSLICH als gültiges JSON im folgenden Format zurück (keine zusätzlichen Erklärungen, kein Markdown):

{{
  "coherence_score_1_to_5": <integer 1-5>,
  "main_issue": "<none|missing_link|contradiction|jumpy_order|unclear_reference|other>",
  "explanation_short": "<max 2 Sätze, Deutsch oder Englisch>",
  "confidence_0_to_1": <float 0.0-1.0>
}}

Wichtig: Gib NUR das JSON zurück, keine zusätzlichen Zeichen oder Erklärungen."""


def get_repair_prompt(response: str) -> str:
    """Repair-Prompt für fehlerhaftes JSON."""
    return f"""Die folgende Antwort enthält kein gültiges JSON. Bitte extrahiere die relevanten Informationen und gib NUR ein gültiges JSON im folgenden Format zurück:

{{
  "coherence_score_1_to_5": <integer 1-5>,
  "main_issue": "<none|missing_link|contradiction|jumpy_order|unclear_reference|other>",
  "explanation_short": "<max 2 Sätze>",
  "confidence_0_to_1": <float 0.0-1.0>
}}

**Ursprüngliche Antwort:**
{response}

**Korrigierte JSON-Antwort (NUR JSON, keine Erklärungen):**"""


# ---------------------------
# JSON Parsing
# ---------------------------


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Versucht JSON aus Text zu extrahieren (auch wenn Markdown-Code-Blöcke vorhanden)."""
    # Entferne Markdown-Code-Blöcke
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Versuche direktes Parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Versuche JSON in geschweiften Klammern zu finden
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_judge_response(
    response: str, llm: OpenAIClient, judge_model: str, temperature: float
) -> dict[str, Any] | None:
    """Parst Judge-Response, versucht Repair bei Fehlern."""
    parsed = extract_json_from_text(response)

    if parsed is not None:
        # Validiere Schema
        score = parsed.get("coherence_score_1_to_5")
        if isinstance(score, (int, float)) and 1 <= score <= 5:
            return parsed

    # Repair-Versuch
    repair_prompt = get_repair_prompt(response)
    try:
        repair_response = llm.complete(
            repair_prompt, model=judge_model, temperature=temperature, max_tokens=200
        )
        parsed = extract_json_from_text(repair_response)
        if parsed is not None:
            score = parsed.get("coherence_score_1_to_5")
            if isinstance(score, (int, float)) and 1 <= score <= 5:
                return parsed
    except Exception:
        pass

    return None


# ---------------------------
# Judge Evaluation
# ---------------------------


def judge_coherence(
    summary: str,
    llm: OpenAIClient,
    judge_model: str,
    rubric_version: str,
    temperature: float,
    judgment_idx: int,
    use_cache: bool,
    cache: dict[str, dict[str, Any]],
    cache_path: Path,
) -> dict[str, Any] | None:
    """Führt eine einzelne Judge-Bewertung durch."""
    key = cache_key(summary, judge_model, rubric_version, judgment_idx)

    if use_cache and key in cache:
        return cache[key]

    prompt = get_judge_prompt_v1(summary)

    try:
        response = llm.complete(prompt, model=judge_model, temperature=temperature, max_tokens=300)
        parsed = parse_judge_response(response, llm, judge_model, temperature)

        if parsed is None:
            return None

        result = {
            "coherence_score_1_to_5": int(parsed.get("coherence_score_1_to_5", 3)),
            "main_issue": parsed.get("main_issue", "none"),
            "explanation_short": parsed.get("explanation_short", ""),
            "confidence_0_to_1": float(parsed.get("confidence_0_to_1", 0.5)),
        }

        if use_cache:
            append_cache(cache_path, key, result)
            cache[key] = result

        return result
    except Exception as e:
        print(f"Fehler bei Judge-Call (judgment_idx={judgment_idx}): {e}")
        return None


# ---------------------------
# Core eval
# ---------------------------


def run_eval(
    rows: list[dict[str, Any]],
    judge_model: str,
    rubric_version: str,
    n_judgments: int,
    temperature: float,
    max_examples: int | None,
    preds_path: Path,
    gt_min: float,
    gt_max: float,
    use_cache: bool,
    cache_path: Path,
    seed: int | None,
    bootstrap_n: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Führt die Judge-Evaluation durch."""
    if seed is not None:
        random.seed(seed)

    llm = OpenAIClient(model_name=judge_model)
    cache = load_cache(cache_path) if use_cache else {}

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

        gt_raw = row.get("gt", {}).get("coherence")
        summary = row.get("summary")
        meta = row.get("meta", {})
        example_id = meta.get("doc_id") or meta.get("id") or f"example_{n_seen}"

        if gt_raw is None or not isinstance(summary, str) or not summary.strip():
            n_skipped += 1
            continue

        try:
            gt_raw_f = float(gt_raw)
        except Exception:
            n_skipped += 1
            continue

        gt_norm = normalize_to_0_1(gt_raw_f, min_v=gt_min, max_v=gt_max)

        # Mehrfachurteile
        judgments: list[dict[str, Any]] = []
        for j_idx in range(n_judgments):
            judgment = judge_coherence(
                summary=summary,
                llm=llm,
                judge_model=judge_model,
                rubric_version=rubric_version,
                temperature=temperature,
                judgment_idx=j_idx,
                use_cache=use_cache,
                cache=cache,
                cache_path=cache_path,
            )
            if judgment is None:
                n_failed += 1
                break
            judgments.append(judgment)

        if len(judgments) < n_judgments:
            # Nicht genug erfolgreiche Urteile
            rec = {
                "example_id": example_id,
                "gt": gt_raw_f,
                "gt_norm": gt_norm,
                "pred": None,
                "pred_raw_scores_1_to_5": [j["coherence_score_1_to_5"] for j in judgments],
                "pred_norm_per_judge": None,
                "main_issue_mode": None,
                "confidence_mean": None,
                "judge_model": judge_model,
                "rubric_version": rubric_version,
                "n_judgments": len(judgments),
                "n_judgments_expected": n_judgments,
                "failed": True,
                "meta": meta,
            }
            predictions_list.append(rec)
            with preds_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            continue

        # Aggregation: Mean über normalisierte Scores
        raw_scores = [j["coherence_score_1_to_5"] for j in judgments]
        norm_scores = [normalize_to_0_1(score, min_v=1.0, max_v=5.0) for score in raw_scores]
        pred_score = sum(norm_scores) / len(norm_scores)

        # Clamp
        if pred_score < 0.0:
            pred_score = 0.0
        if pred_score > 1.0:
            pred_score = 1.0

        # Main issue (Mode)
        main_issues = [j["main_issue"] for j in judgments]
        main_issue_mode = Counter(main_issues).most_common(1)[0][0] if main_issues else "none"

        # Confidence mean
        confidences = [j["confidence_0_to_1"] for j in judgments]
        confidence_mean = sum(confidences) / len(confidences) if confidences else 0.0

        gt_norms.append(gt_norm)
        preds.append(pred_score)
        n_used += 1

        rec = {
            "example_id": example_id,
            "gt": gt_raw_f,
            "gt_norm": gt_norm,
            "pred": pred_score,
            "pred_raw_scores_1_to_5": raw_scores,
            "pred_norm_per_judge": norm_scores,
            "main_issue_mode": main_issue_mode,
            "confidence_mean": confidence_mean,
            "judge_model": judge_model,
            "rubric_version": rubric_version,
            "n_judgments": n_judgments,
            "meta": meta,
        }
        predictions_list.append(rec)

        with preds_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if n_used % 25 == 0:
            print(
                f"[{n_used}] gt_norm={gt_norm:.3f} pred={pred_score:.3f} (seen={n_seen}, skipped={n_skipped}, failed={n_failed})"
            )

    # Calculate metrics (nur valide Beispiele)
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
        "method": "llm_judge",
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
        "judge_config": {
            "judge_model": judge_model,
            "rubric_version": rubric_version,
            "n_judgments": n_judgments,
            "temperature": temperature,
        },
    }
    return metrics, predictions_list


# ---------------------------
# Output generation
# ---------------------------


def write_summary_json(metrics: dict[str, Any], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def write_summary_md(metrics: dict[str, Any], out_path: Path) -> None:
    judge_config = metrics.get("judge_config", {})
    lines = [
        "# Coherence LLM-Judge Evaluation Summary",
        "",
        "**Dataset:** SummEval",
        "**Method:** LLM-as-a-Judge Baseline",
        f"**Examples used:** {metrics['n_used']}",
        f"**Examples skipped:** {metrics['n_skipped']}",
        f"**Examples failed:** {metrics['n_failed']}",
        "",
        "## Judge Configuration",
        "",
        f"- **Judge Model:** {judge_config.get('judge_model', 'unknown')}",
        f"- **Rubric Version:** {judge_config.get('rubric_version', 'unknown')}",
        f"- **N Judgments:** {judge_config.get('n_judgments', 'unknown')}",
        f"- **Temperature:** {judge_config.get('temperature', 'unknown')}",
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
        "### Interpretation",
        "",
        "Judge baseline: rubric-based, temp=0, n_judgments={}, mean aggregation".format(
            judge_config.get("n_judgments", "unknown")
        ),
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
    judge_model: str,
    rubric_version: str,
    n_judgments: int,
    temperature: float,
    seed: int | None,
    bootstrap_n: int,
    n_total: int,
    n_used: int,
    n_failed: int,
    config_params: dict[str, Any],
    out_path: Path,
) -> None:
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
            "judge_model": judge_model,
            "rubric_version": rubric_version,
            "n_judgments": n_judgments,
            "temperature": temperature,
            "bootstrap_n": bootstrap_n,
            **config_params,
        },
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-as-a-Judge Baseline für Coherence auf SummEval")
    ap.add_argument("--data", type=str, required=True, help="Pfad zur JSONL-Datei")
    ap.add_argument("--data1", type=str, help="Alias für --data (Backward-Compat)")
    ap.add_argument("--judge_model", type=str, default="gpt-4o-mini", help="Judge LLM-Modell")
    ap.add_argument("--rubric_version", type=str, default="v1", help="Rubrik-Version")
    ap.add_argument(
        "--n_judgments", type=int, default=3, help="Anzahl Urteile pro Beispiel (Self-Consistency)"
    )
    ap.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature für Judge (default: 0)"
    )
    ap.add_argument("--max_examples", type=int, help="Maximale Anzahl Beispiele")
    ap.add_argument("--max", type=int, help="Alias für --max_examples")
    ap.add_argument("--seed", type=int, help="Random seed für Reproduzierbarkeit")
    ap.add_argument("--bootstrap_n", type=int, default=2000, help="Anzahl Bootstrap-Resamples")
    ap.add_argument(
        "--out_dir",
        type=str,
        help="Output-Verzeichnis (default: results/evaluation/coherence_judge)",
    )
    ap.add_argument("--gt-min", type=float, default=1.0, help="GT-Minimum (default: 1.0)")
    ap.add_argument("--gt-max", type=float, default=5.0, help="GT-Maximum (default: 5.0)")
    ap.add_argument(
        "--cache", action="store_true", default=True, help="Aktiviere Caching (default: ON)"
    )
    ap.add_argument("--no-cache", dest="cache", action="store_false", help="Deaktiviere Caching")

    args = ap.parse_args()

    data_path_str = args.data or args.data1
    if not data_path_str:
        ap.error("--data oder --data1 muss angegeben werden")

    data_path = Path(data_path_str)
    if not data_path.exists():
        ap.error(f"Datei nicht gefunden: {data_path}")

    max_examples = args.max_examples or args.max
    seed = args.seed
    bootstrap_n = args.bootstrap_n

    rows = load_jsonl(data_path)

    # Run ID
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"coherence_judge_{ts}_{args.judge_model}_{args.rubric_version}_n{args.n_judgments}"
    if seed is not None:
        run_id += f"_seed{seed}"

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir) / run_id
    else:
        out_dir = Path("results") / "evaluation" / "coherence_judge" / run_id
    ensure_dir(out_dir)

    preds_path = out_dir / "predictions.jsonl"
    cache_path = out_dir / "cache.jsonl"
    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    metadata_path = out_dir / "run_metadata.json"

    print(f"Data: {data_path} (rows={len(rows)})")
    print(f"Judge Model: {args.judge_model} | Rubric: {args.rubric_version}")
    print(f"N Judgments: {args.n_judgments} | Temperature: {args.temperature}")
    print(f"GT scale: [{args.gt_min}, {args.gt_max}] -> [0,1]")
    print(f"Seed: {seed}")
    print(f"Bootstrap resamples: {bootstrap_n}")
    print(f"Cache: {'ON' if args.cache else 'OFF'} ({cache_path})")
    print(f"Output: {out_dir}")

    metrics, predictions_list = run_eval(
        rows=rows,
        judge_model=args.judge_model,
        rubric_version=args.rubric_version,
        n_judgments=args.n_judgments,
        temperature=args.temperature,
        max_examples=max_examples,
        preds_path=preds_path,
        gt_min=args.gt_min,
        gt_max=args.gt_max,
        use_cache=args.cache,
        cache_path=cache_path,
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
        judge_model=args.judge_model,
        rubric_version=args.rubric_version,
        n_judgments=args.n_judgments,
        temperature=args.temperature,
        seed=seed,
        bootstrap_n=bootstrap_n,
        n_total=len(rows),
        n_used=metrics["n_used"],
        n_failed=metrics["n_failed"],
        config_params={
            "max_examples": max_examples,
            "gt_min": args.gt_min,
            "gt_max": args.gt_max,
            "cache": args.cache,
        },
        out_path=metadata_path,
    )

    print("\n" + "=" * 60)
    print("Judge-Evaluation abgeschlossen!")
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
