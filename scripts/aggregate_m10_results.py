"""
Aggregiert M10-Evaluationsergebnisse zu Summary-Matrix und Summary-MD.
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def aggregate_results(
    run_results: list[dict[str, Any]],
    commit_hash: str | None = None,
):
    """Aggregiert Run-Ergebnisse zu Summary-Matrix und Summary-MD."""

    # Build summary matrix
    matrix_rows = []
    for result in run_results:
        metrics = result["metrics"]
        config = result["config"]

        row = {
            "run_id": result["run_id"],
            "dataset": config["dataset"],
            "config_name": config.get("description", "").split(" - ")[0]
            if config.get("description")
            else "",
            "n": result["n"],
            "pos_rate": result["pos_rate"],
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "tn": metrics["tn"],
            "fn": metrics["fn"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "specificity": metrics["specificity"],
            "accuracy": metrics["accuracy"],
            "auroc": metrics.get("auroc", 0.0),
            "balanced_accuracy": metrics.get("balanced_accuracy", 0.0),
        }
        matrix_rows.append(row)

    # Save CSV
    csv_path = ROOT / "results" / "evaluation" / "summary_matrix.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    if matrix_rows:
        fieldnames = list(matrix_rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(matrix_rows)
        print(f"✅ Saved summary matrix: {csv_path}")

    # Generate summary.md
    generate_summary_md(matrix_rows, commit_hash)


def generate_summary_md(
    matrix_rows: list[dict[str, Any]],
    commit_hash: str | None = None,
):
    """Generiert Summary-Markdown."""

    lines = [
        "# M10 Factuality Evaluation - Summary",
        "",
        f"**Generated:** {datetime.now().isoformat()}",
        f"**Commit Hash:** {commit_hash or 'N/A'}",
        "",
        "## Results Matrix",
        "",
        "| Run ID | Dataset | N | Pos Rate | TP | FP | TN | FN | Precision | Recall | F1 | Specificity | Accuracy | AUROC | Balanced Acc |",
        "|--------|---------|---|----------|----|----|----|----|-----------|--------|----|--------------|----------|-------|--------------|",
    ]

    for row in matrix_rows:
        lines.append(
            f"| {row['run_id']} | {row['dataset']} | {row['n']} | {row['pos_rate']:.3f} | "
            f"{row['tp']} | {row['fp']} | {row['tn']} | {row['fn']} | "
            f"{row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | "
            f"{row['specificity']:.3f} | {row['accuracy']:.3f} | {row['auroc']:.3f} | "
            f"{row['balanced_accuracy']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "### Evaluationsprinzipien",
            "",
            "- **Optimierungsmetrik:** Wir optimieren auf Balanced Accuracy, da die Klassen unbalanced sind und FP/FN asymmetrisch relevant sind.",
            "- **FineSumFact:** FineSumFact ist ein reines Testset; es werden keine Parameteränderungen nach FRANK vorgenommen.",
            "",
        ]
    )

    # Analyze results
    frank_runs = [r for r in matrix_rows if r["dataset"] == "frank"]
    finesumfact_runs = [r for r in matrix_rows if r["dataset"] == "finesumfact"]
    combined_runs = [r for r in matrix_rows if r["dataset"] == "combined"]

    if frank_runs:
        baseline = next((r for r in frank_runs if "baseline" in r["run_id"]), None)
        tuned = next((r for r in frank_runs if "tuned" in r["run_id"]), None)
        ablation = next((r for r in frank_runs if "ablation" in r["run_id"]), None)

        if baseline and tuned:
            bal_acc_improvement = tuned["balanced_accuracy"] - baseline["balanced_accuracy"]
            f1_improvement = tuned["f1"] - baseline["f1"]
            lines.append("### FRANK (Dev/Calibration)")
            lines.append("")
            lines.append(
                f"- **Baseline Balanced Acc:** {baseline['balanced_accuracy']:.3f} | F1: {baseline['f1']:.3f}"
            )
            lines.append(
                f"- **Tuned Balanced Acc:** {tuned['balanced_accuracy']:.3f} ({bal_acc_improvement:+.3f}) | F1: {tuned['f1']:.3f} ({f1_improvement:+.3f})"
            )
            if ablation:
                ablation_drop_bal = baseline["balanced_accuracy"] - ablation["balanced_accuracy"]
                ablation_drop_f1 = baseline["f1"] - ablation["f1"]
                lines.append(
                    f"- **Ablation Balanced Acc:** {ablation['balanced_accuracy']:.3f} ({ablation_drop_bal:+.3f} vs baseline) | F1: {ablation['f1']:.3f} ({ablation_drop_f1:+.3f})"
                )
                lines.append(
                    f"  → Ablation zeigt {'deutlichen' if ablation_drop_bal > 0.05 else 'geringen'} Effekt der Claim-Extraktion"
                )
            lines.append("")

    if finesumfact_runs:
        final = next((r for r in finesumfact_runs if "final" in r["run_id"]), None)
        ablation = next((r for r in finesumfact_runs if "ablation" in r["run_id"]), None)

        if final:
            lines.append("### FineSumFact (Test)")
            lines.append("")
            lines.append(
                f"- **Final Balanced Acc:** {final['balanced_accuracy']:.3f} | F1: {final['f1']:.3f}"
            )
            if tuned:
                generalization_bal = final["balanced_accuracy"] - tuned["balanced_accuracy"]
                generalization_f1 = final["f1"] - tuned["f1"]
                lines.append(
                    f"- **Generalization:** Balanced Acc {generalization_bal:+.3f} | F1 {generalization_f1:+.3f} vs FRANK Tuned"
                )
                if abs(generalization_bal) < 0.05:
                    lines.append(
                        "  → Gute Generalisierung auf FineSumFact (keine Parameteränderungen nach FRANK)"
                    )
                elif generalization_bal < 0:
                    lines.append(
                        "  → Performance-Drop auf FineSumFact (mögliche Dataset-Differenzen)"
                    )
                else:
                    lines.append("  → Performance-Verbesserung auf FineSumFact")
            if ablation:
                ablation_drop_bal = final["balanced_accuracy"] - ablation["balanced_accuracy"]
                ablation_drop_f1 = final["f1"] - ablation["f1"]
                lines.append(
                    f"- **Ablation Balanced Acc:** {ablation['balanced_accuracy']:.3f} ({ablation_drop_bal:+.3f} vs final) | F1: {ablation['f1']:.3f} ({ablation_drop_f1:+.3f})"
                )
            lines.append("")

    if combined_runs:
        combined = combined_runs[0]
        lines.append("### Combined")
        lines.append("")
        lines.append(
            f"- **Combined Balanced Acc:** {combined['balanced_accuracy']:.3f} | F1: {combined['f1']:.3f}"
        )
        lines.append(f"- **N:** {combined['n']} (FRANK + FineSumFact)")
        lines.append("")

    # Trade-offs
    lines.extend(
        [
            "### Trade-offs",
            "",
        ]
    )

    if tuned:
        lines.append(
            f"- **Recall vs Specificity:** Recall={tuned['recall']:.3f}, Specificity={tuned['specificity']:.3f}"
        )
        if tuned["recall"] > tuned["specificity"]:
            lines.append(
                "  → System priorisiert Recall (findet mehr Fehler, aber auch mehr False Positives)"
            )
        elif tuned["specificity"] > tuned["recall"]:
            lines.append(
                "  → System priorisiert Specificity (weniger False Positives, aber möglicherweise mehr übersehene Fehler)"
            )
        else:
            lines.append("  → Ausgewogenes Verhältnis zwischen Recall und Specificity")
        lines.append("")

    # Ablation effect
    if frank_runs:
        baseline = next((r for r in frank_runs if "baseline" in r["run_id"]), None)
        ablation = next((r for r in frank_runs if "ablation" in r["run_id"]), None)

        if baseline and ablation:
            ablation_effect_bal = baseline["balanced_accuracy"] - ablation["balanced_accuracy"]
            ablation_effect_f1 = baseline["f1"] - ablation["f1"]
            lines.append("### Ablation-Effekt")
            lines.append("")
            lines.append(
                f"- **Balanced Acc-Drop durch Ablation:** {ablation_effect_bal:.3f} | F1-Drop: {ablation_effect_f1:.3f}"
            )
            if ablation_effect_bal > 0.1:
                lines.append(
                    "  → **Starker Effekt:** Claim-Extraktion trägt erheblich zur Performance bei"
                )
            elif ablation_effect_bal > 0.05:
                lines.append("  → **Moderater Effekt:** Claim-Extraktion trägt zur Performance bei")
            else:
                lines.append("  → **Schwacher Effekt:** Claim-Extraktion hat geringen Einfluss")
            lines.append("")

    # Save
    md_path = ROOT / "results" / "evaluation" / "summary.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    with md_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"✅ Saved summary: {md_path}")


if __name__ == "__main__":
    # Load all run results
    results_dir = ROOT / "results" / "evaluation" / "runs" / "results"
    run_results = []

    for json_path in results_dir.glob("*.json"):
        if json_path.name.endswith("_examples.jsonl"):
            continue

        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            run_results.append(data)

    if not run_results:
        print("No run results found. Run evaluation first.")
        exit(1)

    aggregate_results(run_results)
