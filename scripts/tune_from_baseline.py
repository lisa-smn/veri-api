"""
Helper-Script: Analysiert Baseline-Ergebnisse und schlägt Tuned-Konfiguration vor.

Wird nach Baseline-Run ausgeführt, um optimale Thresholds/Decision-Regeln zu finden.
"""

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def load_baseline_results(run_id: str) -> dict[str, Any]:
    """Lädt Baseline-Ergebnisse."""
    results_path = ROOT / "results" / "evaluation" / "runs" / "results" / f"{run_id}.json"
    if not results_path.exists():
        raise SystemExit(f"Baseline results not found: {results_path}")

    with results_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze_threshold_sweep(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Analysiert verschiedene Thresholds basierend auf Example-Level Results."""
    # Group by score ranges
    score_ranges = {
        "high_confidence_error": [],  # score < 0.3, has_error=True
        "medium_confidence_error": [],  # 0.3 <= score < 0.7, has_error=True
        "low_confidence_error": [],  # score >= 0.7, has_error=True
        "high_confidence_correct": [],  # score >= 0.7, has_error=False
        "medium_confidence_correct": [],  # 0.3 <= score < 0.7, has_error=False
        "low_confidence_correct": [],  # score < 0.3, has_error=False
    }

    for ex in examples:
        score = ex.get("score", 0.5)
        gt = ex.get("ground_truth", False)

        if score < 0.3:
            key = "high_confidence_error" if gt else "low_confidence_correct"
        elif score < 0.7:
            key = "medium_confidence_error" if gt else "medium_confidence_correct"
        else:
            key = "low_confidence_error" if gt else "high_confidence_correct"

        score_ranges[key].append(ex)

    return score_ranges


def analyze_fp_issue_types(examples: list[dict[str, Any]]) -> dict[str, Any]:
    """Analysiert Issue-Types in False Positives."""
    fp_examples = [ex for ex in examples if ex.get("prediction") and not ex.get("ground_truth")]

    issue_type_counts: dict[str, int] = {}
    uncertain_count = 0
    total_fp_issues = 0

    for ex in fp_examples:
        issue_spans = ex.get("issue_spans", [])
        for span in issue_spans:
            total_fp_issues += 1
            issue_type = span.get("issue_type", "OTHER")
            issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1

            # Prüfe ob "uncertain" im Message vorkommt (heuristisch)
            message = span.get("message", "").lower()
            if "uncertain" in message or "nicht sicher verifizierbar" in message:
                uncertain_count += 1

    # Sortiere nach Häufigkeit
    top_types = sorted(issue_type_counts.items(), key=lambda x: x[1], reverse=True)

    return {
        "fp_count": len(fp_examples),
        "total_fp_issues": total_fp_issues,
        "uncertain_count": uncertain_count,
        "uncertain_ratio": uncertain_count / total_fp_issues if total_fp_issues > 0 else 0.0,
        "issue_type_counts": dict(issue_type_counts),
        "top_10_types": top_types[:10],
    }


def suggest_tuned_config(baseline_results: dict[str, Any]) -> dict[str, Any]:
    """Schlägt optimale Tuned-Konfiguration vor."""
    metrics = baseline_results["metrics"]

    # Load examples for detailed analysis
    examples_path = (
        ROOT
        / "results"
        / "evaluation"
        / "runs"
        / "results"
        / f"{baseline_results['run_id']}_examples.jsonl"
    )
    examples = []
    if examples_path.exists():
        with examples_path.open("r", encoding="utf-8") as f:
            for line in f:
                examples.append(json.loads(line))

    suggestions = {
        "current_f1": metrics["f1"],
        "current_precision": metrics["precision"],
        "current_recall": metrics["recall"],
        "current_specificity": metrics.get("specificity", 0.0),
        "current_balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
        "suggested_error_threshold": baseline_results["config"].get("error_threshold", 1),
        "suggested_decision_mode": baseline_results["config"].get("decision_mode", "issues"),
        "fp_analysis": None,
        "notes": [],
    }

    # Analyze FP/FN ratio
    fp = metrics["fp"]
    fn = metrics["fn"]

    if fp > fn * 1.5:
        # Too many FPs - increase threshold or be more strict
        suggestions["suggested_error_threshold"] = max(
            1, baseline_results["config"].get("error_threshold", 1) + 1
        )
        suggestions["notes"].append("Viele False Positives → Threshold erhöhen")
    elif fn > fp * 1.5:
        # Too many FNs - decrease threshold or be less strict
        suggestions["suggested_error_threshold"] = max(
            1, baseline_results["config"].get("error_threshold", 1) - 1
        )
        suggestions["notes"].append("Viele False Negatives → Threshold senken")

    # Analyze FP Issue Types
    if examples:
        fp_analysis = analyze_fp_issue_types(examples)
        suggestions["fp_analysis"] = fp_analysis

        if fp_analysis["uncertain_ratio"] > 0.3:
            suggestions["notes"].append(
                f"Hoher Uncertainty-Anteil in FPs ({fp_analysis['uncertain_ratio']:.1%}) → "
                "uncertainty_policy='non_error' in Betracht ziehen"
            )

        if fp_analysis["top_10_types"]:
            top_3_types = [t[0] for t in fp_analysis["top_10_types"][:3]]
            suggestions["notes"].append(
                f"Top FP Issue Types: {', '.join(top_3_types)} → "
                "ignore_issue_types in Betracht ziehen"
            )

    # Analyze score distribution if available
    if examples:
        score_analysis = analyze_threshold_sweep(examples)
        high_conf_errors = len(score_analysis["high_confidence_error"])
        if high_conf_errors > len(examples) * 0.1:
            suggestions["notes"].append(
                "Viele high-confidence Errors → Score-Cutoff in Betracht ziehen"
            )

    return suggestions


def main():
    parser = argparse.ArgumentParser(description="Analyze baseline and suggest tuned config")
    parser.add_argument("--baseline-run-id", type=str, default="factuality_frank_baseline_v1")
    parser.add_argument("--output", type=str, default=None, help="Output path for suggestions JSON")

    args = parser.parse_args()

    # Load baseline
    baseline_results = load_baseline_results(args.baseline_run_id)

    # Analyze
    suggestions = suggest_tuned_config(baseline_results)

    # Print
    print("=" * 80)
    print("Baseline Analysis & Tuning Suggestions")
    print("=" * 80)
    print(f"\nBaseline Run: {args.baseline_run_id}")
    print(f"Current F1: {suggestions['current_f1']:.3f}")
    print(f"Current Precision: {suggestions['current_precision']:.3f}")
    print(f"Current Recall: {suggestions['current_recall']:.3f}")
    print(f"Current Specificity: {suggestions['current_specificity']:.3f}")
    print(f"Current Balanced Accuracy: {suggestions['current_balanced_accuracy']:.3f}")

    if suggestions.get("fp_analysis"):
        fp_ana = suggestions["fp_analysis"]
        print("\nFalse Positive Analysis:")
        print(f"  FP Count: {fp_ana['fp_count']}")
        print(f"  Total FP Issues: {fp_ana['total_fp_issues']}")
        print(f"  Uncertain Issues: {fp_ana['uncertain_count']} ({fp_ana['uncertain_ratio']:.1%})")
        print("\n  Top 10 FP Issue Types:")
        for issue_type, count in fp_ana["top_10_types"]:
            print(f"    - {issue_type}: {count}")

    print("\nSuggested Configuration:")
    print(f"  error_threshold: {suggestions['suggested_error_threshold']}")
    print(f"  decision_mode: {suggestions['suggested_decision_mode']}")
    print("\nNotes:")
    for note in suggestions["notes"]:
        print(f"  - {note}")
    print("=" * 80)

    # Save if requested
    if args.output:
        output_path = Path(args.output)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(suggestions, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Saved suggestions to: {output_path}")

    # Update config file
    print("\n💡 Next step: Update configs/m10_factuality_runs.yaml")
    print(
        f"   Set factuality_frank_tuned_v1.error_threshold = {suggestions['suggested_error_threshold']}"
    )


if __name__ == "__main__":
    main()
