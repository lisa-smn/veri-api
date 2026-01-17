"""
M10 Factuality Evaluation Runner.

Führt alle 6 Runs aus der YAML-Config aus:
1. FRANK Baseline
2. FRANK Tuned (nach Baseline-Analyse)
3. FRANK Ablation
4. FineSumFact Final
5. FineSumFact Ablation
6. Combined Final

Jeder Run wird automatisch dokumentiert mit standardisierter Markdown-Dokumentation.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.analysis.metrics import BinaryMetrics, compute_auroc
from app.services.run_manager import RunManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict[str, Any]:
    """Lädt YAML-Config."""
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_commit_hash() -> str | None:
    """Versucht Git-Commit-Hash zu ermitteln."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]
    except Exception:
        pass
    return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lädt JSONL-Datei."""
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def parse_has_error(raw_label: Any) -> bool | None:
    """Robustes Parsing von has_error."""
    if raw_label is None:
        return None
    if isinstance(raw_label, bool):
        return raw_label
    if isinstance(raw_label, (int, float)):
        return bool(raw_label)
    if isinstance(raw_label, str):
        s = raw_label.strip().lower()
        if s in ("1", "true", "yes", "y", "t"):
            return True
        if s in ("0", "false", "no", "n", "f"):
            return False
    return None


def load_examples(path: Path, max_examples: int | None = None) -> list[dict[str, Any]]:
    """Lädt Beispiele aus JSONL."""
    rows = load_jsonl(path)
    examples = []
    for idx, row in enumerate(rows):
        if max_examples and len(examples) >= max_examples:
            break

        article = (row.get("article") or "").strip()
        summary = (row.get("summary") or "").strip()
        has_error = parse_has_error(row.get("has_error"))

        if has_error is None:
            continue
        if not article or not summary:
            continue

        examples.append(
            {
                "example_id": row.get("id") or f"ex_{idx}",
                "article": article,
                "summary": summary,
                "ground_truth": has_error,
                "meta": row.get("meta"),
            }
        )
    return examples


# ----------------------------- Cache ----------------------------- #


def cache_key(article: str, summary: str, model: str, prompt_version: str) -> str:
    """Erstellt deterministischen Cache-Key (kompatibel mit eval_factuality_binary_v2)."""
    data = f"{model}||{prompt_version}||{article}||{summary}"
    key_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()
    # Cache verwendet "sha256:" Prefix
    return f"sha256:{key_hash}"


def load_cache(path: Path) -> dict[str, Any]:
    """Lädt Cache-Datei."""
    if not path.exists():
        return {}
    cache = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                cache[entry["key"]] = entry["value"]
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


def append_cache(path: Path, key: str, value: Any) -> None:
    """Fügt Cache-Eintrag hinzu."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


def run_single_evaluation(
    run_config: dict[str, Any],
    run_manager: RunManager,
) -> dict[str, Any]:
    """Führt einen einzelnen Run aus."""
    run_id = run_config["run_id"]
    logger.info(f"Starting run: {run_id}")

    # Import here to avoid circular imports
    from app.llm.openai_client import OpenAIClient
    from app.services.agents.factuality.factuality_agent import FactualityAgent

    # Setup LLM
    llm_client = OpenAIClient(model_name=run_config["llm_model"])

    # Setup Agent mit Ablation-Flags
    agent = FactualityAgent(
        llm_client,
        use_claim_extraction=run_config.get("use_claim_extraction", True),
        use_claim_verification=run_config.get("use_claim_verification", True),
        use_spans=run_config.get("use_spans", True),
    )

    # Load examples
    if run_config["dataset"] == "combined":
        examples = []
        for ds_config in run_config["dataset_paths"]:
            ds_examples = load_examples(
                ROOT / ds_config["path"],
                max_examples=ds_config.get("max_examples"),
            )
            examples.extend(ds_examples)
    else:
        examples = load_examples(
            ROOT / run_config["dataset_path"],
            max_examples=run_config.get("max_examples"),
        )

    logger.info(f"Loaded {len(examples)} examples")

    # Setup Cache
    cache_enabled = run_config.get("cache_enabled", True)
    prompt_version = run_config.get("prompt_version", "v3_uncertain_spans")

    # Cache-Pfad basierend auf Dataset
    if run_config["dataset"] == "finesumfact":
        cache_path = (
            ROOT
            / "results"
            / "evaluation"
            / "factuality"
            / f"cache_finesumfact_{run_config['llm_model']}_{prompt_version}.jsonl"
        )
    else:
        cache_path = (
            ROOT
            / "results"
            / "evaluation"
            / "factuality"
            / f"cache_{run_config['llm_model']}_{prompt_version}.jsonl"
        )

    cache = load_cache(cache_path) if cache_enabled else {}
    cache_hits = 0
    cache_misses = 0

    logger.info(
        f"Cache: {len(cache)} entries loaded from {cache_path.name if cache_enabled else 'disabled'}"
    )

    # Evaluate
    predictions = []
    ground_truths = []
    pred_scores = []
    results_examples = []

    # Decision Logic Parameter
    error_threshold = run_config.get("error_threshold", 1)  # Legacy: integer threshold
    decision_threshold_float = run_config.get(
        "decision_threshold_float"
    )  # Neu: float threshold für gewichtete Aggregation
    severity_min = run_config.get("severity_min", "low")
    ignore_issue_types = set(run_config.get("ignore_issue_types", []))
    uncertainty_policy = run_config.get("uncertainty_policy", "count_as_error")
    decision_mode = run_config.get("decision_mode", "issues")
    score_cutoff = run_config.get("score_cutoff")
    confidence_min = run_config.get("confidence_min", 0.0)  # Optional: minimale Confidence

    # Severity weights (konfigurierbar)
    severity_weights = run_config.get("severity_weights", {"low": 1.0, "medium": 1.5, "high": 2.0})

    # Severity ranking
    SEVERITY_RANK = {"low": 0, "medium": 1, "high": 2}
    min_severity_rank = SEVERITY_RANK.get(severity_min.lower(), 0)

    def compute_weighted_issue_score(
        issue_spans, uncertainty_policy: str, severity_weights: dict, confidence_min: float
    ) -> float:
        """
        Berechnet gewichteten Issue-Score basierend auf:
        - severity_weight (low/medium/high)
        - confidence_weight (confidence des Claims)
        - uncertainty_weight (uncertainty_policy)
        - evidence_found (downgrade wenn keine Evidence)
        """
        if not issue_spans:
            return 0.0

        total_weight = 0.0
        for span in issue_spans:
            # Filter by severity_min
            span_severity = (span.severity or "low").lower()
            span_rank = SEVERITY_RANK.get(span_severity, 0)
            if span_rank < min_severity_rank:
                continue

            # Filter by ignore_issue_types
            issue_type = (span.issue_type or "OTHER").upper()
            if issue_type in ignore_issue_types:
                continue

            # Filter by confidence_min
            span_confidence = float(span.confidence or 0.5)
            if span_confidence < confidence_min:
                continue

            # Severity weight
            severity_weight = severity_weights.get(span_severity, 1.0)

            # Confidence weight (clamp 0.0..1.0)
            confidence_weight = max(0.0, min(1.0, span_confidence))

            # Uncertainty weight
            is_uncertain = (
                "uncertain" in (span.message or "").lower()
                or "nicht sicher verifizierbar" in (span.message or "").lower()
            )
            if uncertainty_policy == "non_error" and is_uncertain:
                uncertainty_weight = 0.0  # Skip uncertain
            elif uncertainty_policy == "weight_0.5" and is_uncertain:
                uncertainty_weight = 0.5  # Count as half
            else:
                uncertainty_weight = 1.0  # Count as full (count_as_error or incorrect)

            # Evidence penalty: incorrect ohne Evidence => downgrade
            evidence_found = span.evidence_found if span.evidence_found is not None else True
            if not is_uncertain and not evidence_found:
                # Incorrect ohne Evidence: downgrade zu uncertain weight
                uncertainty_weight = min(uncertainty_weight, 0.5)

            # Gesamtgewicht
            weight = severity_weight * confidence_weight * uncertainty_weight
            total_weight += weight

        return total_weight

    def count_effective_issues(issue_spans, uncertainty_policy: str) -> float:
        """Legacy: Zählt effektive Issues (für backward compatibility)."""
        if not issue_spans:
            return 0.0

        count = 0.0
        for span in issue_spans:
            span_severity = (span.severity or "low").lower()
            span_rank = SEVERITY_RANK.get(span_severity, 0)
            if span_rank < min_severity_rank:
                continue

            issue_type = (span.issue_type or "OTHER").upper()
            if issue_type in ignore_issue_types:
                continue

            is_uncertain = (
                "uncertain" in (span.message or "").lower()
                or "nicht sicher verifizierbar" in (span.message or "").lower()
            )

            if uncertainty_policy == "non_error" and is_uncertain:
                continue
            if uncertainty_policy == "weight_0.5" and is_uncertain:
                count += 0.5
            else:
                count += 1.0

        return count

    for ex in examples:
        try:
            # Check cache first
            key = cache_key(ex["article"], ex["summary"], run_config["llm_model"], prompt_version)
            cached_result = cache.get(key) if cache_enabled else None

            if cached_result:
                # Reconstruct AgentResult from cache
                cache_hits += 1
                from app.models.pydantic import AgentResult, IssueSpan

                # Reconstruct issue_spans from cache
                issue_spans = []
                for span_data in cached_result.get("issue_spans", []):
                    if isinstance(span_data, dict):
                        # Ensure all fields are present
                        span_dict = {
                            "start_char": span_data.get("start_char"),
                            "end_char": span_data.get("end_char"),
                            "message": span_data.get("message", ""),
                            "severity": span_data.get("severity"),
                            "issue_type": span_data.get("issue_type"),
                            "confidence": span_data.get("confidence"),
                            "mapping_confidence": span_data.get("mapping_confidence"),
                            "evidence_found": span_data.get("evidence_found"),
                        }
                        issue_spans.append(IssueSpan(**span_dict))
                    else:
                        issue_spans.append(span_data)

                # Reconstruct explanation from details if not in cache
                explanation = cached_result.get("explanation")
                if not explanation:
                    # Fallback: generate explanation from score
                    score = cached_result.get("score", 0.5)
                    explanation = f"Score {score:.2f}."

                agent_result = AgentResult(
                    name="factuality",
                    score=cached_result.get("score", 0.5),
                    explanation=explanation,
                    issue_spans=issue_spans,
                    details=cached_result.get("details"),
                )
            else:
                # Cache miss: Run agent
                cache_misses += 1
                agent_result = agent.run(
                    ex["article"],
                    ex["summary"],
                    meta={
                        "source": "m10_evaluation",
                        "run_id": run_id,
                        "example_id": ex["example_id"],
                    },
                )

                # Save to cache
                if cache_enabled:
                    cache_entry = {
                        "score": agent_result.score,
                        "explanation": agent_result.explanation,
                        "issue_spans": [s.model_dump() for s in agent_result.issue_spans],
                        "details": agent_result.details,
                    }
                    append_cache(cache_path, key, cache_entry)
                    cache[key] = cache_entry  # Update in-memory cache

            # Gewichtete Issue-Aggregation (wenn decision_threshold_float gesetzt)
            if decision_threshold_float is not None:
                weighted_score = compute_weighted_issue_score(
                    agent_result.issue_spans, uncertainty_policy, severity_weights, confidence_min
                )
                # Bei gewichteter Aggregation: nur weighted_score verwenden
                pred_has_error = weighted_score >= decision_threshold_float
            else:
                # Legacy: Count-based (für backward compatibility)
                effective_issues = count_effective_issues(
                    agent_result.issue_spans, uncertainty_policy
                )

                # Binary prediction based on decision_mode
                if decision_mode == "issues":
                    pred_has_error = effective_issues >= error_threshold
                elif decision_mode == "score":
                    pred_has_error = (score_cutoff is not None) and (
                        agent_result.score < score_cutoff
                    )
                elif decision_mode == "either":
                    by_issues = effective_issues >= error_threshold
                    by_score = (score_cutoff is not None) and (agent_result.score < score_cutoff)
                    pred_has_error = by_issues or by_score
                elif decision_mode == "both":
                    by_issues = effective_issues >= error_threshold
                    by_score = (score_cutoff is not None) and (agent_result.score < score_cutoff)
                    pred_has_error = by_issues and by_score
                else:
                    # Default: issues
                    pred_has_error = effective_issues >= error_threshold

            gt_has_error = ex["ground_truth"]

            predictions.append(pred_has_error)
            ground_truths.append(gt_has_error)
            pred_scores.append(agent_result.score)

            # Berechne weighted_score für Dokumentation
            weighted_score = (
                compute_weighted_issue_score(
                    agent_result.issue_spans, uncertainty_policy, severity_weights, confidence_min
                )
                if decision_threshold_float is not None
                else None
            )

            effective_issues = (
                count_effective_issues(agent_result.issue_spans, uncertainty_policy)
                if decision_threshold_float is None
                else None
            )

            results_examples.append(
                {
                    "example_id": ex["example_id"],
                    "ground_truth": gt_has_error,
                    "prediction": pred_has_error,
                    "score": agent_result.score,
                    "num_issues": len(agent_result.issue_spans),  # Raw count
                    "effective_issues": effective_issues,  # After filtering (legacy)
                    "weighted_score": weighted_score,  # Weighted aggregation (neu)
                    "issue_spans": [s.model_dump() for s in agent_result.issue_spans],
                    "summary": ex.get("summary", ""),  # Für Dokumentation (Top FP/FN)
                    "meta": ex.get("meta"),
                }
            )
        except Exception as e:
            logger.error(f"Error processing example {ex['example_id']}: {e}")
            continue

    # Calculate metrics
    metrics_obj = BinaryMetrics()
    for pred, gt in zip(predictions, ground_truths):
        if pred and gt:
            metrics_obj.tp += 1
        elif pred and not gt:
            metrics_obj.fp += 1
        elif not pred and not gt:
            metrics_obj.tn += 1
        else:
            metrics_obj.fn += 1

    metrics = metrics_obj.to_dict()

    # Add AUROC
    if pred_scores:
        auroc = compute_auroc([1.0 - s for s in pred_scores], ground_truths)
        metrics["auroc"] = auroc

    # Add balanced accuracy
    metrics["balanced_accuracy"] = (metrics_obj.recall + metrics_obj.specificity) / 2.0

    # Calculate label distribution
    pos_count = sum(ground_truths)
    neg_count = len(ground_truths) - pos_count
    pos_rate = pos_count / len(ground_truths) if ground_truths else 0.0

    # Log cache statistics
    if cache_enabled:
        logger.info(
            f"Cache: {cache_hits} hits, {cache_misses} misses ({cache_hits / (cache_hits + cache_misses) * 100:.1f}% hit rate)"
        )

    return {
        "run_id": run_id,
        "metrics": metrics,
        "examples": results_examples,
        "n": len(examples),
        "pos_count": pos_count,
        "neg_count": neg_count,
        "pos_rate": pos_rate,
        "config": run_config,
        "cache_stats": {
            "hits": cache_hits,
            "misses": cache_misses,
            "hit_rate": cache_hits / (cache_hits + cache_misses)
            if (cache_hits + cache_misses) > 0
            else 0.0,
        }
        if cache_enabled
        else None,
    }


def analyze_failure_patterns(examples: list[dict[str, Any]]) -> str:
    """Analysiert Failure-Patterns für Dokumentation."""
    fp_examples = [ex for ex in examples if ex["prediction"] and not ex["ground_truth"]]
    fn_examples = [ex for ex in examples if not ex["prediction"] and ex["ground_truth"]]

    # Analyze issue types in FP
    fp_issue_types = {}
    for ex in fp_examples:
        for span in ex.get("issue_spans", []):
            issue_type = span.get("issue_type", "OTHER")
            fp_issue_types[issue_type] = fp_issue_types.get(issue_type, 0) + 1

    # Analyze issue types in FN
    fn_issue_types = {}
    for ex in fn_examples:
        for span in ex.get("issue_spans", []):
            issue_type = span.get("issue_type", "OTHER")
            fn_issue_types[issue_type] = fn_issue_types.get(issue_type, 0) + 1

    lines = []
    if fp_examples:
        lines.append(f"**False Positives ({len(fp_examples)}):**")
        if fp_issue_types:
            top_types = sorted(fp_issue_types.items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append(
                f"  - Häufigste Issue-Types: {', '.join(f'{t}({c})' for t, c in top_types)}"
            )
        lines.append("")

    if fn_examples:
        lines.append(f"**False Negatives ({len(fn_examples)}):**")
        if fn_issue_types:
            top_types = sorted(fn_issue_types.items(), key=lambda x: x[1], reverse=True)[:3]
            lines.append(
                f"  - Häufigste Issue-Types: {', '.join(f'{t}({c})' for t, c in top_types)}"
            )
        lines.append("")

    return "\n".join(lines)


def generate_run_documentation(
    run_result: dict[str, Any],
    commit_hash: str | None,
) -> str:
    """Generiert standardisierte Markdown-Dokumentation für einen Run."""
    run_id = run_result["run_id"]
    config = run_result["config"]
    metrics = run_result["metrics"]
    examples = run_result["examples"]
    n = run_result["n"]
    pos_rate = run_result["pos_rate"]

    # Get top FP/FN
    fp_examples = [ex for ex in examples if ex["prediction"] and not ex["ground_truth"]]
    fn_examples = [ex for ex in examples if not ex["prediction"] and ex["ground_truth"]]

    fp_examples.sort(
        key=lambda x: x.get("score", 1.0), reverse=True
    )  # Highest scores = most confident FPs
    fn_examples.sort(key=lambda x: x.get("score", 0.0))  # Lowest scores = most confident FNs

    lines = [
        f"# Run Documentation: {run_id}",
        "",
        f"**Description:** {config.get('description', 'N/A')}",
        f"**Created:** {datetime.now().isoformat()}",
        "",
        "## Configuration",
        "",
        "```yaml",
        yaml.dump(config, default_flow_style=False, sort_keys=False),
        "```",
        "",
        "### Tuning Parameters",
        "",
        f"- **Decision Mode:** {config.get('decision_mode', 'N/A')}",
        f"- **Error Threshold (Legacy):** {config.get('error_threshold', 'N/A')}",
        f"- **Decision Threshold (Float):** {config.get('decision_threshold_float', 'N/A')}",
        f"- **Severity Min:** {config.get('severity_min', 'N/A')}",
        f"- **Severity Weights:** {config.get('severity_weights', {})}",
        f"- **Ignore Issue Types:** {config.get('ignore_issue_types', [])}",
        f"- **Uncertainty Policy:** {config.get('uncertainty_policy', 'N/A')}",
        f"- **Confidence Min:** {config.get('confidence_min', 'N/A')}",
        f"- **Score Cutoff:** {config.get('score_cutoff', 'N/A')}",
        "",
        "## Dataset",
        "",
        f"- **Dataset:** {config['dataset']}",
        f"- **N:** {n}",
        f"- **Positive Rate:** {pos_rate:.3f} ({run_result['pos_count']} pos, {run_result['neg_count']} neg)",
        "",
        "## Results",
        "",
        "### Confusion Matrix",
        "",
        f"- **TP:** {metrics['tp']}",
        f"- **FP:** {metrics['fp']}",
        f"- **TN:** {metrics['tn']}",
        f"- **FN:** {metrics['fn']}",
        "",
        "### Metrics",
        "",
        f"- **Accuracy:** {metrics['accuracy']:.3f}",
        f"- **Precision:** {metrics['precision']:.3f}",
        f"- **Recall:** {metrics['recall']:.3f}",
        f"- **F1:** {metrics['f1']:.3f}",
        f"- **Specificity:** {metrics['specificity']:.3f}",
        f"- **Balanced Accuracy:** {metrics['balanced_accuracy']:.3f}",
        f"- **AUROC:** {metrics.get('auroc', 0.0):.3f}",
        "",
    ]

    if fp_examples:
        lines.extend(
            [
                "### Top 5 False Positives",
                "",
            ]
        )
        for i, ex in enumerate(fp_examples[:5], 1):
            # Get summary from examples if available, otherwise use meta
            summary_text = ex.get("summary") or ex.get("meta", {}).get("summary", "") or "N/A"
            summary_snippet = (
                (summary_text[:100] + "...") if len(summary_text) > 100 else summary_text
            )
            lines.append(
                f"{i}. **ID:** {ex['example_id']} | **Score:** {ex['score']:.3f} | **Issues:** {ex['num_issues']}"
            )
            lines.append(f"   Summary: {summary_snippet}")
            lines.append("")

    if fn_examples:
        lines.extend(
            [
                "### Top 5 False Negatives",
                "",
            ]
        )
        for i, ex in enumerate(fn_examples[:5], 1):
            # Get summary from examples if available, otherwise use meta
            summary_text = ex.get("summary") or ex.get("meta", {}).get("summary", "") or "N/A"
            summary_snippet = (
                (summary_text[:100] + "...") if len(summary_text) > 100 else summary_text
            )
            lines.append(
                f"{i}. **ID:** {ex['example_id']} | **Score:** {ex['score']:.3f} | **Issues:** {ex['num_issues']}"
            )
            lines.append(f"   Summary: {summary_snippet}")
            lines.append("")

    # Failure pattern analysis
    lines.extend(
        [
            "## Failure Pattern Analysis",
            "",
        ]
    )

    pattern_analysis = analyze_failure_patterns(examples)
    if pattern_analysis:
        lines.append(pattern_analysis)

    if metrics["fp"] > metrics["fn"]:
        lines.append(
            "**Dominantes Pattern:** False Positives überwiegen. System ist zu konservativ und markiert korrekte Summaries als fehlerhaft."
        )
    elif metrics["fn"] > metrics["fp"]:
        lines.append(
            "**Dominantes Pattern:** False Negatives überwiegen. System übersieht Fehler in Summaries."
        )
    else:
        lines.append("**Dominantes Pattern:** Ausgewogene Fehlerverteilung zwischen FP und FN.")

    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            f"- **Commit Hash:** {commit_hash or 'N/A'}",
            f"- **Model:** {config['llm_model']}",
            f"- **Prompt Version:** {config['prompt_version']}",
            f"- **Temperature:** {config['llm_temperature']}",
            f"- **Seed:** {config.get('llm_seed', 'N/A')}",
            "",
        ]
    )

    return "\n".join(lines)


def save_run_results(
    run_result: dict[str, Any],
    run_manager: RunManager,
    commit_hash: str | None,
):
    """Speichert Run-Ergebnisse und Dokumentation."""
    run_id = run_result["run_id"]

    # Save metrics JSON
    metrics_path = ROOT / "results" / "evaluation" / "runs" / "results" / f"{run_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "run_id": run_id,
                "metrics": run_result["metrics"],
                "n": run_result["n"],
                "pos_count": run_result["pos_count"],
                "neg_count": run_result["neg_count"],
                "pos_rate": run_result["pos_rate"],
                "config": run_result["config"],
                "commit_hash": commit_hash,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Save examples JSONL (optional, kann groß werden)
    examples_path = (
        ROOT / "results" / "evaluation" / "runs" / "results" / f"{run_id}_examples.jsonl"
    )
    with examples_path.open("w", encoding="utf-8") as f:
        for ex in run_result["examples"]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Generate and save documentation
    doc = generate_run_documentation(run_result, commit_hash)
    doc_path = ROOT / "results" / "evaluation" / "runs" / "docs" / f"{run_id}.md"
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    with doc_path.open("w", encoding="utf-8") as f:
        f.write(doc)

    logger.info(f"Saved results for {run_id}")


def main():
    parser = argparse.ArgumentParser(description="M10 Factuality Evaluation Runner")
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="configs/m10_factuality_runs.yaml",
        help="Path to YAML config file (optional, defaults to configs/m10_factuality_runs.yaml)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run only specific run_id (for testing)",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip baseline run (if already done)",
    )
    args = parser.parse_args()

    # Load config
    config_path = ROOT / args.config
    if not config_path.exists():
        raise SystemExit(f"Config not found: {config_path}")

    config = load_config(config_path)
    runs = config["runs"]

    # Filter by run_id if specified
    if args.run_id:
        runs = [r for r in runs if r["run_id"] == args.run_id]
        if not runs:
            raise SystemExit(f"Run {args.run_id} not found in config")

    # Skip baseline if requested
    if args.skip_baseline:
        runs = [r for r in runs if "baseline" not in r["run_id"]]

    logger.info(f"Found {len(runs)} runs to execute")

    # Get commit hash
    commit_hash = get_commit_hash()
    if commit_hash:
        logger.info(f"Git commit: {commit_hash}")

    # Initialize run manager
    run_manager = RunManager(ROOT / "results" / "evaluation")

    # Execute runs
    all_results = []
    for i, run_config in enumerate(runs, 1):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Run {i}/{len(runs)}: {run_config['run_id']}")
        logger.info(f"{'=' * 80}\n")

        try:
            result = run_single_evaluation(run_config, run_manager)
            save_run_results(result, run_manager, commit_hash)
            all_results.append(result)
            logger.info(f"✅ Completed: {run_config['run_id']}")
        except Exception as e:
            logger.error(f"❌ Failed: {run_config['run_id']}: {e}", exc_info=True)
            continue

    logger.info(f"\n{'=' * 80}")
    logger.info(f"Completed {len(all_results)}/{len(runs)} runs")
    logger.info(f"{'=' * 80}")

    # Run aggregator
    logger.info("\nRunning aggregator...")
    try:
        from scripts.aggregate_m10_results import aggregate_results

        aggregate_results(all_results, commit_hash)
        logger.info("✅ Aggregation completed")
    except Exception as e:
        logger.warning(f"⚠️  Aggregation failed: {e}")


if __name__ == "__main__":
    main()
