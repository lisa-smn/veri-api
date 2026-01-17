"""
Vergleicht zwei Readability-Runs (Baseline vs Improved).

Berechnet:
- Metriken-Vergleich (Agent vs Humans, Judge vs Humans)
- Delta (Improved - Baseline)
- Distribution-Vergleich
- Committee Reliability
"""

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from scipy.stats import pearsonr, spearmanr


def load_summary(summary_path: Path) -> dict[str, Any]:
    """Lädt summary.json."""
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json nicht gefunden: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """Lädt predictions.jsonl."""
    predictions = []
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                predictions.append(rec)
            except json.JSONDecodeError:
                continue
    return predictions


def analyze_distribution(
    predictions: list[dict[str, Any]], score_key: str = "pred"
) -> dict[str, int]:
    """Analysiert Score-Verteilung."""
    scores = [p.get(score_key) for p in predictions if p.get(score_key) is not None]
    if not scores:
        return {}

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def bin_value(v: float) -> str:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                return f"[{bins[i]:.1f}, {bins[i + 1]:.1f})"
        if v >= bins[-1]:
            return f"[{bins[-1]:.1f}, 1.0]"
        return "[0.0, 0.2)"

    from collections import Counter

    return Counter([bin_value(s) for s in scores])


def compute_committee_reliability(predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Berechnet Committee-Reliability."""
    committee_stds = []
    full_agreements = 0
    majority_agreements = 0
    total_with_committee = 0

    for p in predictions:
        committee = p.get("judge_committee")
        if committee and isinstance(committee, dict):
            total_with_committee += 1
            std = committee.get("std")
            if std is not None and std != 0:
                committee_stds.append(float(std))
            if std == 0.0 or std is None:
                full_agreements += 1
                majority_agreements += 1
            elif std is not None and std < 0.1:  # Sehr niedrige Varianz
                majority_agreements += 1

    if total_with_committee == 0:
        return None

    return {
        "n_with_committee": total_with_committee,
        "mean_std": statistics.mean(committee_stds) if committee_stds else 0.0,
        "median_std": statistics.median(committee_stds) if committee_stds else 0.0,
        "full_agreement_pct": (full_agreements / total_with_committee * 100.0)
        if total_with_committee > 0
        else 0.0,
        "majority_agreement_pct": (majority_agreements / total_with_committee * 100.0)
        if total_with_committee > 0
        else 0.0,
    }


def compute_agent_judge_agreement(
    predictions: list[dict[str, Any]], threshold: float = 0.7
) -> dict[str, Any] | None:
    """Berechnet Agent vs Judge Agreement innerhalb eines Runs."""
    agent_scores = []
    judge_scores = []

    for p in predictions:
        agent_score = p.get("pred_agent")
        judge_score = p.get("pred_judge")
        if agent_score is not None and judge_score is not None:
            agent_scores.append(float(agent_score))
            judge_scores.append(float(judge_score))

    if len(agent_scores) == 0:
        return None

    # Korrelation
    pearson_r, pearson_p = pearsonr(agent_scores, judge_scores)
    spearman_rho, spearman_p = spearmanr(agent_scores, judge_scores)

    # Threshold Agreement
    agent_good = [a >= threshold for a in agent_scores]
    judge_good = [j >= threshold for j in judge_scores]

    tp = sum(1 for a, j in zip(agent_good, judge_good) if a and j)
    tn = sum(1 for a, j in zip(agent_good, judge_good) if not a and not j)
    fp = sum(1 for a, j in zip(agent_good, judge_good) if a and not j)
    fn = sum(1 for a, j in zip(agent_good, judge_good) if not a and j)

    agreement_pct = (tp + tn) / len(agent_scores) * 100.0 if agent_scores else 0.0

    return {
        "n": len(agent_scores),
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
        "threshold_agreement_pct": agreement_pct,
        "threshold": threshold,
        "confusion_matrix": {
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        },
    }


def extract_metrics_for_source(
    summary: dict[str, Any], predictions: list[dict[str, Any]], source: str
) -> dict[str, Any]:
    """Extrahiert Metriken für Agent oder Judge."""
    # Prüfe, ob Judge-Daten vorhanden sind
    has_judge = any(p.get("pred_judge") is not None for p in predictions)

    if source == "judge" and not has_judge:
        return None

    # Für Judge: Berechne Metriken aus pred_judge
    if source == "judge":
        judge_scores = [p.get("pred_judge") for p in predictions if p.get("pred_judge") is not None]
        gt_scores = [
            p.get("gt_norm")
            for p in predictions
            if p.get("pred_judge") is not None and p.get("gt_norm") is not None
        ]

        if len(judge_scores) != len(gt_scores) or len(judge_scores) == 0:
            return None

        # Berechne Metriken
        pearson_r, pearson_p = pearsonr(judge_scores, gt_scores)
        spearman_rho, spearman_p = spearmanr(judge_scores, gt_scores)

        # MAE, RMSE
        mae_val = sum(abs(j - g) for j, g in zip(judge_scores, gt_scores)) / len(judge_scores)
        rmse_val = (
            sum((j - g) ** 2 for j, g in zip(judge_scores, gt_scores)) / len(judge_scores)
        ) ** 0.5

        # R²
        mean_gt = statistics.mean(gt_scores)
        ss_res = sum((g - j) ** 2 for j, g in zip(judge_scores, gt_scores))
        ss_tot = sum((g - mean_gt) ** 2 for g in gt_scores)
        r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "pearson": {"value": pearson_r},
            "spearman": {"value": spearman_rho},
            "mae": {"value": mae_val},
            "rmse": {"value": rmse_val},
            "r_squared": r2_val,
            "n": len(judge_scores),
        }
    # Agent: Nutze summary.json
    return {
        "pearson": summary.get("pearson", {}),
        "spearman": summary.get("spearman", {}),
        "mae": summary.get("mae", {}),
        "rmse": summary.get("rmse", {}),
        "r_squared": summary.get("r_squared"),
        "n": summary.get("n_used"),
    }


def write_report(
    baseline_run_dir: Path,
    improved_run_dir: Path,
    out_path: Path,
) -> None:
    """Schreibt Vergleichs-Report."""
    # Lade Daten
    baseline_summary = load_summary(baseline_run_dir / "summary.json")
    improved_summary = load_summary(improved_run_dir / "summary.json")
    baseline_preds = load_predictions(baseline_run_dir / "predictions.jsonl")
    improved_preds = load_predictions(improved_run_dir / "predictions.jsonl")

    # Prüfe n
    baseline_n = len(baseline_preds)
    improved_n = len(improved_preds)
    if baseline_n < 200 or improved_n < 200:
        print(f"WARNUNG: Baseline n={baseline_n}, Improved n={improved_n} (erwartet: 200)")

    # Extrahiere Metriken
    baseline_agent = extract_metrics_for_source(baseline_summary, baseline_preds, "agent")
    baseline_judge = extract_metrics_for_source(baseline_summary, baseline_preds, "judge")
    improved_agent = extract_metrics_for_source(improved_summary, improved_preds, "agent")
    improved_judge = extract_metrics_for_source(improved_summary, improved_preds, "judge")

    # Committee Reliability
    baseline_committee = compute_committee_reliability(baseline_preds)
    improved_committee = compute_committee_reliability(improved_preds)

    # Agent vs Judge Agreement (innerhalb jedes Runs)
    baseline_agreement = compute_agent_judge_agreement(baseline_preds, threshold=0.7)
    improved_agreement = compute_agent_judge_agreement(improved_preds, threshold=0.7)

    # Distribution
    baseline_dist_agent = analyze_distribution(baseline_preds, "pred_agent")
    baseline_dist_judge = (
        analyze_distribution(baseline_preds, "pred_judge") if baseline_judge else {}
    )
    improved_dist_agent = analyze_distribution(improved_preds, "pred_agent")
    improved_dist_judge = (
        analyze_distribution(improved_preds, "pred_judge") if improved_judge else {}
    )

    # Schreibe Report
    lines = [
        "# Readability Judge: Baseline vs Improved",
        "",
        f"**Baseline Run:** {baseline_run_dir.name}",
        f"**Improved Run:** {improved_run_dir.name}",
        "",
        "## Agent vs Humans",
        "",
        "| Metrik | Baseline | Improved | Delta |",
        "|---|---|---|---|",
    ]

    if baseline_agent and improved_agent:

        def format_metric(m, key):
            val = m.get(key, {})
            if isinstance(val, dict):
                return val.get("value", 0.0)
            return val

        def format_delta(b, i, higher_better=True):
            d = i - b
            sign = "+" if (d > 0 and higher_better) or (d < 0 and not higher_better) else ""
            return f"{sign}{d:.4f}"

        lines.extend(
            [
                f"| Spearman ρ | {format_metric(baseline_agent, 'spearman'):.4f} | {format_metric(improved_agent, 'spearman'):.4f} | {format_delta(format_metric(baseline_agent, 'spearman'), format_metric(improved_agent, 'spearman'))} |",
                f"| Pearson r | {format_metric(baseline_agent, 'pearson'):.4f} | {format_metric(improved_agent, 'pearson'):.4f} | {format_delta(format_metric(baseline_agent, 'pearson'), format_metric(improved_agent, 'pearson'))} |",
                f"| MAE | {format_metric(baseline_agent, 'mae'):.4f} | {format_metric(improved_agent, 'mae'):.4f} | {format_delta(format_metric(baseline_agent, 'mae'), format_metric(improved_agent, 'mae'), higher_better=False)} |",
                f"| RMSE | {format_metric(baseline_agent, 'rmse'):.4f} | {format_metric(improved_agent, 'rmse'):.4f} | {format_delta(format_metric(baseline_agent, 'rmse'), format_metric(improved_agent, 'rmse'), higher_better=False)} |",
                f"| R² | {format_metric(baseline_agent, 'r_squared'):.4f} | {format_metric(improved_agent, 'r_squared'):.4f} | {format_delta(format_metric(baseline_agent, 'r_squared'), format_metric(improved_agent, 'r_squared'))} |",
            ]
        )

    lines.extend(
        [
            "",
            "## Judge vs Humans",
            "",
            "| Metrik | Baseline | Improved | Delta |",
            "|---|---|---|---|",
        ]
    )

    if baseline_judge and improved_judge:
        lines.extend(
            [
                f"| Spearman ρ | {format_metric(baseline_judge, 'spearman'):.4f} | {format_metric(improved_judge, 'spearman'):.4f} | {format_delta(format_metric(baseline_judge, 'spearman'), format_metric(improved_judge, 'spearman'))} |",
                f"| Pearson r | {format_metric(baseline_judge, 'pearson'):.4f} | {format_metric(improved_judge, 'pearson'):.4f} | {format_delta(format_metric(baseline_judge, 'pearson'), format_metric(improved_judge, 'pearson'))} |",
                f"| MAE | {format_metric(baseline_judge, 'mae'):.4f} | {format_metric(improved_judge, 'mae'):.4f} | {format_delta(format_metric(baseline_judge, 'mae'), format_metric(improved_judge, 'mae'), higher_better=False)} |",
                f"| RMSE | {format_metric(baseline_judge, 'rmse'):.4f} | {format_metric(improved_judge, 'rmse'):.4f} | {format_delta(format_metric(baseline_judge, 'rmse'), format_metric(improved_judge, 'rmse'), higher_better=False)} |",
                f"| R² | {format_metric(baseline_judge, 'r_squared'):.4f} | {format_metric(improved_judge, 'r_squared'):.4f} | {format_delta(format_metric(baseline_judge, 'r_squared'), format_metric(improved_judge, 'r_squared'))} |",
            ]
        )
    else:
        lines.append("| *Keine Judge-Daten verfügbar* | | | |")

    # Distribution
    lines.extend(
        [
            "",
            "## Prediction Distribution",
            "",
            "### Baseline Agent",
            "",
        ]
    )
    for bin_name in sorted(baseline_dist_agent.keys()):
        lines.append(f"- {bin_name}: {baseline_dist_agent[bin_name]}")

    if baseline_dist_judge:
        lines.extend(
            [
                "",
                "### Baseline Judge",
                "",
            ]
        )
        for bin_name in sorted(baseline_dist_judge.keys()):
            lines.append(f"- {bin_name}: {baseline_dist_judge[bin_name]}")

    lines.extend(
        [
            "",
            "### Improved Agent",
            "",
        ]
    )
    for bin_name in sorted(improved_dist_agent.keys()):
        lines.append(f"- {bin_name}: {improved_dist_agent[bin_name]}")

    if improved_dist_judge:
        lines.extend(
            [
                "",
                "### Improved Judge",
                "",
            ]
        )
        for bin_name in sorted(improved_dist_judge.keys()):
            lines.append(f"- {bin_name}: {improved_dist_judge[bin_name]}")

    # Committee Reliability
    lines.extend(
        [
            "",
            "## Committee Reliability",
            "",
            "| Metrik | Baseline | Improved |",
            "|---|---|---|",
        ]
    )

    if baseline_committee and improved_committee:
        lines.extend(
            [
                f"| Mean Std | {baseline_committee['mean_std']:.4f} | {improved_committee['mean_std']:.4f} |",
                f"| Median Std | {baseline_committee['median_std']:.4f} | {improved_committee['median_std']:.4f} |",
                f"| Full Agreement | {baseline_committee['full_agreement_pct']:.1f}% | {improved_committee['full_agreement_pct']:.1f}% |",
                f"| Majority Agreement | {baseline_committee['majority_agreement_pct']:.1f}% | {improved_committee['majority_agreement_pct']:.1f}% |",
            ]
        )
    else:
        lines.append("| *Keine Committee-Daten verfügbar* | | |")

    # Agent vs Judge Agreement
    lines.extend(
        [
            "",
            "## Agent vs Judge Agreement (innerhalb Runs)",
            "",
        ]
    )

    if baseline_agreement and improved_agreement:
        lines.extend(
            [
                "| Metrik | Baseline | Improved |",
                "|---|---|---|",
                f"| Spearman ρ | {baseline_agreement['spearman_rho']:.4f} | {improved_agreement['spearman_rho']:.4f} |",
                f"| Pearson r | {baseline_agreement['pearson_r']:.4f} | {improved_agreement['pearson_r']:.4f} |",
                f"| Threshold Agreement (≥{baseline_agreement['threshold']:.1f}) | {baseline_agreement['threshold_agreement_pct']:.1f}% | {improved_agreement['threshold_agreement_pct']:.1f}% |",
                f"| n | {baseline_agreement['n']} | {improved_agreement['n']} |",
                "",
            ]
        )
    else:
        lines.append("*Keine Agent vs Judge Agreement-Daten verfügbar*")
        lines.append("")

    # Collapse Check
    lines.extend(
        [
            "## Collapse Detection",
            "",
        ]
    )

    baseline_collapse = baseline_summary.get("collapse_detected", False)
    improved_collapse = improved_summary.get("collapse_detected", False)

    lines.extend(
        [
            f"- **Baseline:** {'⚠️ Collapse erkannt' if baseline_collapse else '✅ Kein Collapse'}",
            f"- **Improved:** {'⚠️ Collapse erkannt' if improved_collapse else '✅ Kein Collapse'}",
        ]
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Vergleicht Baseline vs Improved Readability Runs")
    ap.add_argument("--baseline_run_dir", type=str, required=True, help="Baseline Run-Verzeichnis")
    ap.add_argument("--improved_run_dir", type=str, required=True, help="Improved Run-Verzeichnis")
    ap.add_argument("--out_md", type=str, required=True, help="Output-Pfad (.md)")

    args = ap.parse_args()

    write_report(
        Path(args.baseline_run_dir),
        Path(args.improved_run_dir),
        Path(args.out_md),
    )

    print(f"Report geschrieben: {args.out_md}")


if __name__ == "__main__":
    main()
