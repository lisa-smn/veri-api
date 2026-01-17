"""
Einheitliches Evaluationsskript für alle Dimensionen (M10+).

Unterstützt:
- Factuality (binär, FRANK/FineSumFact)
- Coherence (kontinuierlich, SummEval)
- Readability (kontinuierlich, SummEval)
- Explainability (vollständiges System)

Verwendet Run-Configs aus evaluation_configs/ für reproduzierbare Runs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

from dotenv import load_dotenv

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.openai_client import OpenAIClient
from app.pipeline.verification_pipeline import VerificationPipeline
from app.services.agents.coherence.coherence_agent import CoherenceAgent
from app.services.agents.factuality.factuality_agent import FactualityAgent
from app.services.agents.readability.readability_agent import ReadabilityAgent

load_dotenv()


# ----------------------------- Data Models ----------------------------- #


@dataclass
class EvaluationExample:
    """Einheitliches Format für Evaluation-Beispiele."""

    example_id: str | None
    article: str
    summary: str
    ground_truth: dict[str, Any]  # dimension -> value
    meta: dict[str, Any] | None = None


@dataclass
class Metrics:
    """Metriken für binäre Klassifikation."""

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


@dataclass
class CorrelationMetrics:
    """Metriken für kontinuierliche Scores."""

    pearson_r: float = 0.0
    spearman_rho: float = 0.0
    mae: float = 0.0
    rmse: float = 0.0
    n: int = 0


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


def load_frank_examples(path: Path) -> list[EvaluationExample]:
    """Lädt FRANK/FineSumFact-Format (binär, has_error)."""
    rows = load_jsonl(path)
    examples = []
    for idx, row in enumerate(rows):
        article = (row.get("article") or "").strip()
        summary = (row.get("summary") or "").strip()
        has_error = row.get("has_error")

        # Parse has_error robust
        if has_error is None:
            continue
        if isinstance(has_error, bool):
            gt_bool = has_error
        elif isinstance(has_error, (int, float)):
            gt_bool = bool(has_error)
        elif isinstance(has_error, str):
            s = has_error.strip().lower()
            gt_bool = s in ("1", "true", "yes", "y", "t")
        else:
            continue

        if not article or not summary:
            continue

        examples.append(
            EvaluationExample(
                example_id=row.get("id") or f"ex_{idx}",
                article=article,
                summary=summary,
                ground_truth={"factuality": gt_bool},
                meta=row.get("meta"),
            )
        )
    return examples


def load_sumeval_examples(path: Path, dimension: str) -> list[EvaluationExample]:
    """Lädt SummEval-Format (kontinuierlich, gt[dimension])."""
    rows = load_jsonl(path)
    examples = []
    for idx, row in enumerate(rows):
        article = (row.get("article") or "").strip()
        summary = (row.get("summary") or "").strip()
        gt_dict = row.get("gt", {})
        gt_value = gt_dict.get(dimension)

        if gt_value is None:
            continue
        try:
            gt_float = float(gt_value)
        except (ValueError, TypeError):
            continue

        if not article or not summary:
            continue

        examples.append(
            EvaluationExample(
                example_id=row.get("id") or f"ex_{idx}",
                article=article,
                summary=summary,
                ground_truth={dimension: gt_float},
                meta=row.get("meta"),
            )
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


# ----------------------------- Metrics Calculation ----------------------------- #


def compute_correlation_metrics(
    predictions: list[float], ground_truths: list[float]
) -> CorrelationMetrics:
    """Berechnet Korrelationsmetriken."""
    n = len(predictions)
    if n == 0:
        return CorrelationMetrics()

    # Pearson correlation
    pred_mean = sum(predictions) / n
    gt_mean = sum(ground_truths) / n

    num = sum((p - pred_mean) * (g - gt_mean) for p, g in zip(predictions, ground_truths))
    pred_var = sum((p - pred_mean) ** 2 for p in predictions)
    gt_var = sum((g - gt_mean) ** 2 for g in ground_truths)

    pearson_r = num / math.sqrt(pred_var * gt_var) if (pred_var * gt_var) > 0 else 0.0

    # Spearman (vereinfacht: rank correlation)
    pred_ranks = sorted(range(n), key=lambda i: predictions[i])
    gt_ranks = sorted(range(n), key=lambda i: ground_truths[i])
    rank_pred = [0] * n
    rank_gt = [0] * n
    for rank, idx in enumerate(pred_ranks):
        rank_pred[idx] = rank
    for rank, idx in enumerate(gt_ranks):
        rank_gt[idx] = rank

    rank_pred_mean = sum(rank_pred) / n
    rank_gt_mean = sum(rank_gt) / n
    rank_num = sum((r - rank_pred_mean) * (g - rank_gt_mean) for r, g in zip(rank_pred, rank_gt))
    rank_pred_var = sum((r - rank_pred_mean) ** 2 for r in rank_pred)
    rank_gt_var = sum((g - rank_gt_mean) ** 2 for g in rank_gt)
    spearman_rho = (
        rank_num / math.sqrt(rank_pred_var * rank_gt_var)
        if (rank_pred_var * rank_gt_var) > 0
        else 0.0
    )

    # MAE & RMSE
    mae = sum(abs(p - g) for p, g in zip(predictions, ground_truths)) / n
    rmse = math.sqrt(sum((p - g) ** 2 for p, g in zip(predictions, ground_truths)) / n)

    return CorrelationMetrics(
        pearson_r=pearson_r,
        spearman_rho=spearman_rho,
        mae=mae,
        rmse=rmse,
        n=n,
    )


def normalize_to_0_1(value: float, min_v: float, max_v: float) -> float:
    """Normalisiert Wert auf [0, 1]."""
    if max_v == min_v:
        return 0.5
    return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))


# ----------------------------- Evaluation Functions ----------------------------- #


def evaluate_factuality(
    examples: list[EvaluationExample],
    config: dict[str, Any],
    cache_path: Path,
    predictions_path: Path,
) -> tuple[Metrics, list[dict[str, Any]]]:
    """Evaluiert Factuality-Agent (binär)."""
    llm_client = OpenAIClient(model_name=config["llm_model"])
    agent = FactualityAgent(llm_client)

    cache = load_cache(cache_path) if config.get("cache_enabled", True) else {}
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        predictions_path.unlink()

    metrics = Metrics()
    rows = []
    prompt_version = config["prompt_versions"]["factuality"]
    thresholds = config.get("thresholds", {})
    error_threshold = thresholds.get("error_threshold", 1)

    for ex in examples:
        key = cache_key(ex.article, ex.summary, config["llm_model"], prompt_version)
        cached = cache.get(key) if cache else None

        if cached:
            result = cached
        else:
            try:
                agent_result = agent.run(ex.article, ex.summary, meta={"source": "eval_unified"})
                result = {
                    "score": agent_result.score,
                    "num_issues": len(agent_result.issue_spans),
                    "issue_spans": [s.model_dump() for s in agent_result.issue_spans],
                }
                if config.get("cache_enabled", True):
                    append_cache(cache_path, key, result)
            except Exception as e:
                print(f"Error processing example {ex.example_id}: {e}")
                continue

        # Binary prediction: has_error if num_issues >= threshold
        pred_has_error = result["num_issues"] >= error_threshold
        gt_has_error = ex.ground_truth["factuality"]

        # Update metrics
        if pred_has_error and gt_has_error:
            metrics.tp += 1
        elif pred_has_error and not gt_has_error:
            metrics.fp += 1
        elif not pred_has_error and not gt_has_error:
            metrics.tn += 1
        else:
            metrics.fn += 1

        row = {
            "example_id": ex.example_id,
            "gt_has_error": gt_has_error,
            "pred_has_error": pred_has_error,
            "score": result["score"],
            "num_issues": result["num_issues"],
            "meta": ex.meta,
        }
        rows.append(row)

        with predictions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return metrics, rows


def evaluate_continuous(
    examples: list[EvaluationExample],
    config: dict[str, Any],
    dimension: str,
    cache_path: Path,
    predictions_path: Path,
) -> tuple[CorrelationMetrics, list[dict[str, Any]]]:
    """Evaluiert Coherence oder Readability (kontinuierlich)."""
    llm_client = OpenAIClient(model_name=config["llm_model"])

    if dimension == "coherence":
        agent = CoherenceAgent(llm_client)
    elif dimension == "readability":
        agent = ReadabilityAgent(llm_client)
    else:
        raise ValueError(f"Unknown dimension: {dimension}")

    cache = load_cache(cache_path) if config.get("cache_enabled", True) else {}
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        predictions_path.unlink()

    predictions = []
    ground_truths = []
    rows = []
    prompt_version = config["prompt_versions"][dimension]
    thresholds = config.get("thresholds", {})
    gt_min = thresholds.get("gt_min", 1.0)
    gt_max = thresholds.get("gt_max", 5.0)

    for ex in examples:
        key = cache_key(ex.article, ex.summary, config["llm_model"], prompt_version)
        cached = cache.get(key) if cache else None

        if cached:
            pred_score = cached["score"]
        else:
            try:
                agent_result = agent.run(ex.article, ex.summary, meta={"source": "eval_unified"})
                pred_score = agent_result.score
                if config.get("cache_enabled", True):
                    append_cache(cache_path, key, {"score": pred_score})
            except Exception as e:
                print(f"Error processing example {ex.example_id}: {e}")
                continue

        gt_raw = ex.ground_truth[dimension]
        gt_norm = normalize_to_0_1(gt_raw, gt_min, gt_max)

        predictions.append(pred_score)
        ground_truths.append(gt_norm)

        row = {
            "example_id": ex.example_id,
            "gt_raw": gt_raw,
            "gt_norm": gt_norm,
            "pred_score": pred_score,
            "meta": ex.meta,
        }
        rows.append(row)

        with predictions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = compute_correlation_metrics(predictions, ground_truths)
    return metrics, rows


def evaluate_explainability(
    examples: list[EvaluationExample],
    config: dict[str, Any],
    cache_path: Path,
    predictions_path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluiert vollständiges System mit Explainability."""
    pipeline = VerificationPipeline(model_name=config["llm_model"])

    cache = load_cache(cache_path) if config.get("cache_enabled", True) else {}
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    if predictions_path.exists():
        predictions_path.unlink()

    rows = []

    for ex in examples:
        # Use article+summary as cache key (simplified)
        key = cache_key(ex.article, ex.summary, config["llm_model"], "full_pipeline")
        cached = cache.get(key) if cache else None

        if cached:
            result = cached
        else:
            try:
                pipeline_result = pipeline.run(
                    ex.article, ex.summary, meta={"source": "eval_unified"}
                )
                result = {
                    "overall_score": pipeline_result.overall_score,
                    "factuality": pipeline_result.factuality.model_dump(),
                    "coherence": pipeline_result.coherence.model_dump(),
                    "readability": pipeline_result.readability.model_dump(),
                    "explainability": pipeline_result.explainability.model_dump()
                    if pipeline_result.explainability
                    else None,
                }
                if config.get("cache_enabled", True):
                    append_cache(cache_path, key, result)
            except Exception as e:
                print(f"Error processing example {ex.example_id}: {e}")
                continue

        row = {
            "example_id": ex.example_id,
            "ground_truth": ex.ground_truth,
            **result,
            "meta": ex.meta,
        }
        rows.append(row)

        with predictions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Aggregate stats
    stats = {
        "num_examples": len(rows),
        "avg_overall_score": sum(r["overall_score"] for r in rows) / len(rows) if rows else 0.0,
    }

    return stats, rows


# ----------------------------- Main ----------------------------- #


def load_config(config_path: Path) -> dict[str, Any]:
    """Lädt Run-Config."""
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_run_summary(
    output_dir: Path,
    run_id: str,
    config: dict[str, Any],
    metrics: Any,
    num_examples: int,
    timestamp: str,
) -> Path:
    """Speichert Run-Summary."""
    summary = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config": config,
        "metrics": metrics.__dict__ if hasattr(metrics, "__dict__") else metrics,
        "num_examples": num_examples,
    }

    path = output_dir / f"run_{run_id}_{timestamp}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path


def main():
    parser = argparse.ArgumentParser(description="Unified Evaluation Script (M10+)")
    parser.add_argument("config", type=str, help="Path to evaluation config JSON")
    parser.add_argument("--dataset-path", type=str, help="Override dataset path from config")
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = ROOT / "evaluation_configs" / args.config
    if not config_path.exists():
        raise SystemExit(f"Config not found: {args.config}")

    config = load_config(config_path)
    run_id = config["run_id"]
    dimension = config["dimension"]
    dataset_name = config["dataset"]

    # Determine dataset path
    if args.dataset_path:
        dataset_path = Path(args.dataset_path)
    else:
        # Try to find in data/
        dataset_path = ROOT / "data" / dataset_name / f"{dataset_name}_clean.jsonl"
        if not dataset_path.exists():
            dataset_path = ROOT / "data" / dataset_name / f"{dataset_name}.jsonl"

    if not dataset_path.exists():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    # Load examples
    print(f"Loading examples from {dataset_path}...")
    if dimension == "all":
        # For explainability, try to load as factuality first (can be extended)
        if dataset_name in ("frank", "finesumfact"):
            examples = load_frank_examples(dataset_path)
        elif dataset_name == "sumeval":
            # For sumeval with "all", we need a different loader or use factuality as fallback
            examples = load_sumeval_examples(dataset_path, "coherence")  # Use coherence as default
        else:
            raise SystemExit(f"Unknown dataset for explainability: {dataset_name}")
    elif dimension == "factuality" or dataset_name in ("frank", "finesumfact"):
        examples = load_frank_examples(dataset_path)
    elif dimension in ("coherence", "readability") or dataset_name == "sumeval":
        examples = load_sumeval_examples(dataset_path, dimension)
    else:
        raise SystemExit(f"Unknown dimension/dataset combination: {dimension}/{dataset_name}")

    if config.get("max_examples"):
        examples = examples[: config["max_examples"]]

    print(f"Loaded {len(examples)} examples")

    # Setup paths
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        ROOT / "results" / "evaluation" / (dimension if dimension != "all" else "explainability")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    if dimension == "all":
        prompt_version = "full_pipeline"
    else:
        prompt_version = config["prompt_versions"].get(dimension, "v1")
    cache_path = output_dir / f"cache_{config['llm_model']}_{prompt_version}.jsonl"
    predictions_path = output_dir / f"predictions_{run_id}_{timestamp}.jsonl"

    # Run evaluation
    print(f"Running evaluation: {dimension} on {dataset_name}...")

    if dimension == "factuality":
        metrics, rows = evaluate_factuality(examples, config, cache_path, predictions_path)
        print("\nResults:")
        print(f"  TP: {metrics.tp}  FP: {metrics.fp}  TN: {metrics.tn}  FN: {metrics.fn}")
        print(f"  Accuracy:  {metrics.accuracy:.3f}")
        print(f"  Precision: {metrics.precision:.3f}")
        print(f"  Recall:    {metrics.recall:.3f}")
        print(f"  F1:        {metrics.f1:.3f}")
    elif dimension in ("coherence", "readability"):
        metrics, rows = evaluate_continuous(
            examples, config, dimension, cache_path, predictions_path
        )
        print("\nResults:")
        print(f"  Pearson r:  {metrics.pearson_r:.3f}")
        print(f"  Spearman ρ: {metrics.spearman_rho:.3f}")
        print(f"  MAE:        {metrics.mae:.3f}")
        print(f"  RMSE:       {metrics.rmse:.3f}")
        print(f"  N:          {metrics.n}")
    elif dimension == "all":
        stats, rows = evaluate_explainability(examples, config, cache_path, predictions_path)
        print("\nResults:")
        print(f"  Examples:   {stats['num_examples']}")
        print(f"  Avg Score:  {stats['avg_overall_score']:.3f}")
        metrics = stats
    else:
        raise SystemExit(f"Unknown dimension: {dimension}")

    # Save summary
    summary_path = save_run_summary(output_dir, run_id, config, metrics, len(examples), timestamp)
    print(f"\nRun summary saved: {summary_path}")
    print(f"Predictions saved: {predictions_path}")


if __name__ == "__main__":
    main()
