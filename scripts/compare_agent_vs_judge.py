"""
Vergleicht Agent-Score vs Judge-Score aus zwei Evaluation-Runs.

Berechnet:
- Spearman/Pearson Korrelation
- Threshold-Agreement (% Agreement, Confusion Matrix)
- Committee-Reliability (falls verfügbar)
"""

import argparse
import json
from pathlib import Path
import statistics
from typing import Any

from scipy.stats import pearsonr, spearmanr


def load_predictions(predictions_path: Path) -> dict[str, dict[str, Any]]:
    """Lädt predictions.jsonl und indexiert nach example_id."""
    predictions = {}
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions-Datei nicht gefunden: {predictions_path}")

    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                example_id = rec.get("example_id")
                if example_id:
                    predictions[example_id] = rec
            except json.JSONDecodeError as e:
                print(f"Warnung: Konnte Zeile nicht parsen: {e}")
                continue

    return predictions


def merge_predictions(
    agent_preds: dict[str, dict[str, Any]],
    judge_preds: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge Agent und Judge Predictions über example_id."""
    merged = []
    all_ids = set(agent_preds.keys()) | set(judge_preds.keys())

    for example_id in all_ids:
        agent_rec = agent_preds.get(example_id)
        judge_rec = judge_preds.get(example_id)

        if not agent_rec or not judge_rec:
            continue

        # Extrahiere Scores
        agent_score = agent_rec.get("pred_agent") or agent_rec.get("pred")
        judge_score = judge_rec.get("pred_judge") or judge_rec.get("pred")

        if agent_score is None or judge_score is None:
            continue

        merged.append(
            {
                "example_id": example_id,
                "agent_score": float(agent_score),
                "judge_score": float(judge_score),
                "gt_norm": agent_rec.get("gt_norm") or judge_rec.get("gt_norm"),
                "judge_committee": judge_rec.get("judge_committee"),
                "judge_outputs_count": judge_rec.get("judge_outputs_count"),
            }
        )

    return merged


def compute_correlations(merged: list[dict[str, Any]]) -> dict[str, float]:
    """Berechnet Pearson und Spearman Korrelation."""
    agent_scores = [m["agent_score"] for m in merged]
    judge_scores = [m["judge_score"] for m in merged]

    pearson_r, pearson_p = pearsonr(agent_scores, judge_scores)
    spearman_rho, spearman_p = spearmanr(agent_scores, judge_scores)

    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "n": len(merged),
    }


def compute_threshold_agreement(
    merged: list[dict[str, Any]],
    threshold: float = 0.7,
) -> dict[str, Any]:
    """
    Berechnet Threshold-Agreement (binäre Klassifikation: good/bad).

    Args:
        merged: Liste von merged predictions
        threshold: Schwellwert für "good" (>= threshold) vs "bad" (< threshold)

    Returns:
        Dict mit Agreement-Metriken und Confusion Matrix
    """
    agent_good = []
    judge_good = []

    for m in merged:
        agent_good.append(m["agent_score"] >= threshold)
        judge_good.append(m["judge_score"] >= threshold)

    # Confusion Matrix
    tp = sum(1 for a, j in zip(agent_good, judge_good) if a and j)  # beide good
    tn = sum(1 for a, j in zip(agent_good, judge_good) if not a and not j)  # beide bad
    fp = sum(1 for a, j in zip(agent_good, judge_good) if a and not j)  # agent good, judge bad
    fn = sum(1 for a, j in zip(agent_good, judge_good) if not a and j)  # agent bad, judge good

    total = len(merged)
    agreement_pct = (tp + tn) / total * 100.0 if total > 0 else 0.0

    return {
        "threshold": threshold,
        "agreement_pct": agreement_pct,
        "confusion_matrix": {
            "tp": tp,  # beide good
            "tn": tn,  # beide bad
            "fp": fp,  # agent good, judge bad
            "fn": fn,  # agent bad, judge good
        },
        "n": total,
    }


def compute_committee_reliability(merged: list[dict[str, Any]]) -> dict[str, Any] | None:
    """
    Berechnet Committee-Reliability (falls Judge Committee-Daten verfügbar).

    Returns:
        Dict mit Stats oder None falls keine Committee-Daten vorhanden
    """
    committee_stds = []
    full_agreements = 0
    total_with_committee = 0

    for m in merged:
        committee = m.get("judge_committee")
        if not committee:
            continue

        total_with_committee += 1

        if isinstance(committee, dict):
            std = committee.get("std")
            if std is not None:
                committee_stds.append(float(std))

            # Prüfe auf Full Agreement (alle Outputs identisch)
            outputs_count = m.get("judge_outputs_count", 0)
            if outputs_count > 1 and std == 0.0:
                full_agreements += 1

    if total_with_committee == 0:
        return None

    return {
        "n_with_committee": total_with_committee,
        "mean_std": statistics.mean(committee_stds) if committee_stds else None,
        "median_std": statistics.median(committee_stds) if committee_stds else None,
        "full_agreement_pct": (full_agreements / total_with_committee * 100.0)
        if total_with_committee > 0
        else 0.0,
    }


def write_report(
    correlations: dict[str, float],
    threshold_agreement: dict[str, Any],
    committee_reliability: dict[str, Any] | None,
    agent_run_id: str,
    judge_run_id: str,
    out_path: Path,
) -> None:
    """Schreibt Vergleichs-Report als Markdown."""
    lines = [
        "# Agent vs Judge Agreement Report",
        "",
        f"**Agent Run:** {agent_run_id}",
        f"**Judge Run:** {judge_run_id}",
        "",
        "## Korrelation",
        "",
        f"- **Pearson r:** {correlations['pearson_r']:.4f} (p={correlations['pearson_p']:.4f})",
        f"- **Spearman ρ:** {correlations['spearman_rho']:.4f} (p={correlations['spearman_p']:.4f})",
        f"- **n:** {correlations['n']}",
        "",
        "## Threshold-Agreement",
        "",
        f"**Schwellwert:** {threshold_agreement['threshold']:.2f}",
        f"**Agreement:** {threshold_agreement['agreement_pct']:.2f}%",
        "",
        "### Confusion Matrix",
        "",
        "| | Judge Good | Judge Bad |",
        "|---|---|---|",
        f"| **Agent Good** | {threshold_agreement['confusion_matrix']['tp']} (TP) | {threshold_agreement['confusion_matrix']['fp']} (FP) |",
        f"| **Agent Bad** | {threshold_agreement['confusion_matrix']['fn']} (FN) | {threshold_agreement['confusion_matrix']['tn']} (TN) |",
        "",
    ]

    if committee_reliability:
        lines.extend(
            [
                "## Committee-Reliability",
                "",
                f"- **Beispiele mit Committee:** {committee_reliability['n_with_committee']}",
                f"- **Durchschnittliche Std:** {committee_reliability['mean_std']:.4f}"
                if committee_reliability["mean_std"] is not None
                else "- **Durchschnittliche Std:** N/A",
                f"- **Median Std:** {committee_reliability['median_std']:.4f}"
                if committee_reliability["median_std"] is not None
                else "- **Median Std:** N/A",
                f"- **Full Agreement:** {committee_reliability['full_agreement_pct']:.2f}%",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Committee-Reliability",
                "",
                "*Keine Committee-Daten verfügbar (JUDGE_N=1 oder nicht gespeichert)*",
                "",
            ]
        )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Vergleicht Agent vs Judge Scores")
    ap.add_argument(
        "--run_dir_agent", type=str, required=True, help="Run-Verzeichnis des Agent-Runs"
    )
    ap.add_argument(
        "--run_dir_judge", type=str, required=True, help="Run-Verzeichnis des Judge-Runs"
    )
    ap.add_argument("--out", type=str, required=True, help="Output-Pfad (.md)")
    ap.add_argument(
        "--threshold", type=float, default=0.7, help="Schwellwert für Agreement (default: 0.7)"
    )
    ap.add_argument(
        "--auto_find", action="store_true", help="Automatisch neueste Runs finden (experimentell)"
    )

    args = ap.parse_args()

    # Auto-Find: Suche neueste Runs mit seed42
    if args.auto_find:
        import glob

        base_dir = Path("results/evaluation/readability")
        runs = sorted(
            [Path(p) for p in glob.glob(str(base_dir / "readability_*_seed42"))], reverse=True
        )
        if len(runs) >= 2:
            agent_dir = runs[0]  # Neuester
            judge_dir = runs[1]  # Zweitneuester
            print(f"Auto-Find: Agent={agent_dir.name}, Judge={judge_dir.name}")
        else:
            print("FEHLER: Auto-Find benötigt mindestens 2 Runs mit seed42")
            return
    else:
        agent_dir = Path(args.run_dir_agent)
        judge_dir = Path(args.run_dir_judge)

    out_path = Path(args.out)

    # Lade Run-IDs aus run_metadata.json
    agent_run_id = "unknown"
    judge_run_id = "unknown"

    agent_metadata_path = agent_dir / "run_metadata.json"
    judge_metadata_path = judge_dir / "run_metadata.json"

    if agent_metadata_path.exists():
        with agent_metadata_path.open("r", encoding="utf-8") as f:
            agent_metadata = json.load(f)
            agent_run_id = agent_metadata.get("run_id", agent_dir.name)

    if judge_metadata_path.exists():
        with judge_metadata_path.open("r", encoding="utf-8") as f:
            judge_metadata = json.load(f)
            judge_run_id = judge_metadata.get("run_id", judge_dir.name)

    # Lade Predictions
    agent_preds_path = agent_dir / "predictions.jsonl"
    judge_preds_path = judge_dir / "predictions.jsonl"

    print(f"Lade Agent-Predictions: {agent_preds_path}")
    agent_preds = load_predictions(agent_preds_path)
    print(f"  {len(agent_preds)} Beispiele gefunden")

    print(f"Lade Judge-Predictions: {judge_preds_path}")
    judge_preds = load_predictions(judge_preds_path)
    print(f"  {len(judge_preds)} Beispiele gefunden")

    # Merge
    merged = merge_predictions(agent_preds, judge_preds)
    print(f"  {len(merged)} übereinstimmende Beispiele")

    if len(merged) == 0:
        print("FEHLER: Keine übereinstimmenden Beispiele gefunden!")
        return

    # Berechne Metriken
    correlations = compute_correlations(merged)
    threshold_agreement = compute_threshold_agreement(merged, threshold=args.threshold)
    committee_reliability = compute_committee_reliability(merged)

    # Schreibe Report
    write_report(
        correlations,
        threshold_agreement,
        committee_reliability,
        agent_run_id,
        judge_run_id,
        out_path,
    )

    print(f"\nReport geschrieben: {out_path}")
    print(f"  Pearson r: {correlations['pearson_r']:.4f}")
    print(f"  Spearman ρ: {correlations['spearman_rho']:.4f}")
    print(f"  Agreement: {threshold_agreement['agreement_pct']:.2f}%")


if __name__ == "__main__":
    main()
