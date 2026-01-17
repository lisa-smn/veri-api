"""
Strukturierte Factuality-Evaluation mit automatischer Run-Dokumentation.

Workflow:
1. Run definieren (Freeze) - Config erstellen
2. Run ausführen (Agenten + Explainability + Baselines)
3. Ergebnisse speichern (Example-Level + Run-Summary + Log)
4. Auswertung berechnen (Quant + Robustheit + Subsets)
5. Interpretieren und dokumentieren (Fallstudien + Template)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.openai_client import OpenAIClient
from app.pipeline.verification_pipeline import VerificationPipeline
from app.services.agents.factuality.factuality_agent import FactualityAgent
from app.services.analysis.metrics import (
    BinaryMetrics,
    analyze_error_patterns,
    analyze_subsets,
    compute_auroc,
    compute_threshold_sweep,
)
from app.services.run_manager import (
    RunAnalysis,
    RunDefinition,
    RunDocumentation,
    RunExecution,
    RunManager,
    RunResults,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ----------------------------- Data Loading ----------------------------- #


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


def load_frank_examples(path: Path) -> list[dict[str, Any]]:
    """Lädt FRANK/FineSumFact-Format."""
    rows = load_jsonl(path)
    examples = []
    for idx, row in enumerate(rows):
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
    """Erstellt deterministischen Cache-Key."""
    data = f"{model}||{prompt_version}||{article}||{summary}"
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


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
            entry = json.loads(line)
            cache[entry["key"]] = entry["value"]
    return cache


def append_cache(path: Path, key: str, value: Any) -> None:
    """Fügt Cache-Eintrag hinzu."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


# ----------------------------- Evaluation ----------------------------- #


def evaluate_factuality(
    examples: list[dict[str, Any]],
    config: dict[str, Any],
    run_manager: RunManager,
    execution: RunExecution,
    cache_path: Path,
) -> RunResults:
    """Führt Factuality-Evaluation durch."""
    llm_client = OpenAIClient(model_name=config["llm_model"])
    agent = FactualityAgent(llm_client)

    # Optional: Full pipeline for explainability
    use_explainability = config.get("include_explainability", False)
    pipeline = None
    if use_explainability:
        pipeline = VerificationPipeline(model_name=config["llm_model"])

    cache = load_cache(cache_path) if config.get("cache_enabled", True) else {}
    prompt_version = config["prompt_versions"]["factuality"]
    thresholds = config.get("thresholds", {})
    error_threshold = thresholds.get("error_threshold", 1)

    results_examples = []
    predictions = []
    ground_truths = []
    pred_scores = []

    for idx, ex in enumerate(examples):
        if config.get("max_examples") and idx >= config["max_examples"]:
            break

        execution.num_processed += 1

        key = cache_key(ex["article"], ex["summary"], config["llm_model"], prompt_version)
        cached = cache.get(key) if cache else None

        if cached:
            agent_result_data = cached
            execution.num_cached += 1
        else:
            try:
                agent_result = agent.run(
                    ex["article"],
                    ex["summary"],
                    meta={"source": "eval_factuality_structured", "example_id": ex["example_id"]},
                )
                agent_result_data = {
                    "score": agent_result.score,
                    "num_issues": len(agent_result.issue_spans),
                    "issue_spans": [s.model_dump() for s in agent_result.issue_spans],
                    "details": agent_result.details,
                }
                if config.get("cache_enabled", True):
                    append_cache(cache_path, key, agent_result_data)
            except Exception as e:
                logger.error(f"Error processing example {ex['example_id']}: {e}")
                execution.num_failed += 1
                execution.errors.append(f"Example {ex['example_id']}: {e!s}")
                continue

        # Binary prediction
        pred_has_error = agent_result_data["num_issues"] >= error_threshold
        gt_has_error = ex["ground_truth"]

        predictions.append(pred_has_error)
        ground_truths.append(gt_has_error)
        pred_scores.append(agent_result_data["score"])

        # Get explainability if requested
        explainability_data = None
        if use_explainability and pipeline:
            try:
                pipeline_result = pipeline.run(
                    ex["article"],
                    ex["summary"],
                    meta={"source": "eval_factuality_structured", "example_id": ex["example_id"]},
                )
                if pipeline_result.explainability:
                    explainability_data = pipeline_result.explainability.model_dump()
            except Exception as e:
                logger.warning(f"Explainability failed for {ex['example_id']}: {e}")

        result_example = {
            "example_id": ex["example_id"],
            "ground_truth": gt_has_error,
            "prediction": pred_has_error,
            "score": agent_result_data["score"],
            "num_issues": agent_result_data["num_issues"],
            "issue_spans": agent_result_data["issue_spans"],
            "explainability": explainability_data,
            "meta": ex.get("meta"),
        }
        results_examples.append(result_example)

        # Update execution progress
        if (idx + 1) % 10 == 0:
            run_manager.update_execution(execution)
            logger.info(f"Processed {idx + 1}/{len(examples)} examples")

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
        auroc = compute_auroc(
            [1.0 - s for s in pred_scores], ground_truths
        )  # Invert: lower score = error
        metrics["auroc"] = auroc

    return RunResults(
        run_id=execution.run_id,
        examples=results_examples,
        metrics=metrics,
        explainability_stats=None,  # Can be aggregated later
    )


def compute_analysis(
    results: RunResults,
    config: dict[str, Any],
) -> RunAnalysis:
    """Berechnet quantitative Auswertung."""
    examples = results.examples
    predictions = [ex["prediction"] for ex in examples]
    ground_truths = [ex["ground_truth"] for ex in examples]
    pred_scores = [ex["score"] for ex in examples]

    # Primary metrics (already in results.metrics)
    primary_metrics = results.metrics.copy()

    # Robustness: Threshold sweep
    thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    sweep_results = compute_threshold_sweep(
        [1.0 - s for s in pred_scores],  # Invert scores
        ground_truths,
        thresholds,
    )

    # Find best threshold by F1
    best_threshold = max(sweep_results, key=lambda x: x.get("f1", 0.0))

    robustness = {
        "threshold_sweep": sweep_results,
        "best_threshold": best_threshold,
    }

    # Error analysis
    error_analysis = analyze_error_patterns(examples, predictions, ground_truths)

    # Subset analysis (if meta available)
    subsets = {}
    if examples and examples[0].get("meta"):
        subsets = analyze_subsets(examples, predictions, ground_truths)

    return RunAnalysis(
        run_id=results.run_id,
        primary_metrics=primary_metrics,
        robustness=robustness,
        subsets=subsets,
        error_analysis=error_analysis,
    )


def generate_interpretation(
    definition: RunDefinition,
    execution: RunExecution,
    results: RunResults,
    analysis: RunAnalysis,
) -> str:
    """Generiert Interpretation-Text."""
    lines = [
        f"## Run {definition.run_id} - Interpretation",
        "",
        f"**Dataset:** {definition.dataset} ({definition.split})",
        f"**Model:** {definition.llm_model}",
        f"**Prompt Version:** {definition.prompt_versions.get('factuality', 'N/A')}",
        "",
        "### Ergebnisse",
        "",
        f"Der Factuality-Agent erreicht auf {definition.dataset} folgende Metriken:",
        "",
        f"- **Accuracy:** {results.metrics.get('accuracy', 0):.3f}",
        f"- **Precision:** {results.metrics.get('precision', 0):.3f}",
        f"- **Recall:** {results.metrics.get('recall', 0):.3f}",
        f"- **F1:** {results.metrics.get('f1', 0):.3f}",
        "",
    ]

    if "auroc" in results.metrics:
        lines.append(f"- **AUROC:** {results.metrics['auroc']:.3f}")
        lines.append("")

    lines.extend(
        [
            "### Robustheit",
            "",
            f"Bester Threshold (nach F1): {analysis.robustness.get('best_threshold', {}).get('threshold', 'N/A')}",
            "",
            "### Fehleranalyse",
            "",
            f"- **False Positives:** {analysis.error_analysis.get('num_fp', 0)}",
            f"- **False Negatives:** {analysis.error_analysis.get('num_fn', 0)}",
            "",
        ]
    )

    if analysis.error_analysis.get("fp_issue_types"):
        lines.extend(
            [
                "Häufigste Issue-Types bei False Positives:",
                "",
            ]
        )
        for issue_type, count in sorted(
            analysis.error_analysis["fp_issue_types"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]:
            lines.append(f"- {issue_type}: {count}")
        lines.append("")

    if analysis.error_analysis.get("fn_issue_types"):
        lines.extend(
            [
                "Häufigste Issue-Types bei False Negatives:",
                "",
            ]
        )
        for issue_type, count in sorted(
            analysis.error_analysis["fn_issue_types"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]:
            lines.append(f"- {issue_type}: {count}")
        lines.append("")

    return "\n".join(lines)


# ----------------------------- Main ----------------------------- #


def main():
    parser = argparse.ArgumentParser(description="Structured Factuality Evaluation")
    parser.add_argument("config", type=str, help="Path to evaluation config JSON")
    parser.add_argument("--dataset-path", type=str, help="Override dataset path")
    parser.add_argument(
        "--include-explainability", action="store_true", help="Include explainability"
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = ROOT / "evaluation_configs" / args.config
    if not config_path.exists():
        raise SystemExit(f"Config not found: {args.config}")

    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    # Create run definition
    definition = RunDefinition(
        run_id=config["run_id"],
        dimension=config["dimension"],
        dataset=config["dataset"],
        split=config.get("split", "test"),
        llm_model=config["llm_model"],
        llm_temperature=config.get("llm_temperature", 0.0),
        llm_seed=config.get("llm_seed"),
        prompt_versions=config["prompt_versions"],
        explainability_version=config.get("explainability_version", "m9_v1"),
        thresholds=config.get("thresholds", {}),
        max_examples=config.get("max_examples"),
        cache_enabled=config.get("cache_enabled", True),
        description=config.get("description", ""),
    )

    # Initialize run manager
    run_manager = RunManager(ROOT / "results" / "evaluation")

    # Create run
    execution = run_manager.create_run(definition)
    logger.info(f"Created run: {definition.run_id}")

    # Load dataset
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        dataset_path = ROOT / "data" / config["dataset"] / f"{config['dataset']}_clean.jsonl"
        if not dataset_path.exists():
            dataset_path = ROOT / "data" / config["dataset"] / f"{config['dataset']}.jsonl"

    if not dataset_path.exists():
        execution.finish("failed")
        execution.errors.append(f"Dataset not found: {dataset_path}")
        run_manager.update_execution(execution)
        raise SystemExit(f"Dataset not found: {dataset_path}")

    logger.info(f"Loading examples from {dataset_path}...")
    examples = load_frank_examples(dataset_path)
    execution.num_examples = len(examples)

    if config.get("max_examples"):
        examples = examples[: config["max_examples"]]
        execution.num_examples = len(examples)

    logger.info(f"Loaded {len(examples)} examples")

    # Setup paths
    prompt_version = config["prompt_versions"]["factuality"]
    output_dir = ROOT / "results" / "evaluation" / "factuality"
    cache_path = output_dir / f"cache_{config['llm_model']}_{prompt_version}.jsonl"

    # Update config for evaluation
    config["include_explainability"] = args.include_explainability

    # Run evaluation
    logger.info("Starting evaluation...")
    try:
        results = evaluate_factuality(examples, config, run_manager, execution, cache_path)
        execution.finish("success")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        execution.finish("failed")
        execution.errors.append(str(e))
        run_manager.update_execution(execution)
        raise

    # Save results
    run_manager.save_results(results)

    # Compute analysis
    logger.info("Computing analysis...")
    analysis = compute_analysis(results, config)
    run_manager.save_analysis(analysis)

    # Generate interpretation
    interpretation = generate_interpretation(definition, execution, results, analysis)

    # Create documentation
    doc = RunDocumentation(
        run_id=definition.run_id,
        definition=definition,
        execution=execution,
        results=results,
        analysis=analysis,
        interpretation=interpretation,
    )

    run_manager.save_documentation(doc)

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Run ID: {definition.run_id}")
    print(f"Status: {execution.status}")
    print(f"Examples: {execution.num_processed}/{execution.num_examples}")
    print(f"Failed: {execution.num_failed}")
    print(f"Cached: {execution.num_cached}")
    print("\nMetrics:")
    for key, value in results.metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print(f"\nDocumentation: results/evaluation/runs/docs/{definition.run_id}.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
