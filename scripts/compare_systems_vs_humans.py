"""
Vergleicht Agent/Judge vs klassische Baselines (ROUGE, BERTScore, Readability-Formeln).

Input:
- Agent-Run dir (readability/coherence)
- Judge-Run dir (readability/coherence)
- Baseline summary_matrix.csv oder einzelne baseline runs

Output:
- docs/thesis/chapters/classical_metrics_baselines.md
"""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_summary_json(path: Path) -> dict[str, Any] | None:
    """Lädt summary.json."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """Lädt predictions.jsonl."""
    predictions = []
    if not predictions_path.exists():
        return predictions
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return predictions


def compute_metrics_from_predictions(
    predictions: list[dict[str, Any]], score_key: str, gt_key: str = "gt_norm"
) -> dict[str, Any] | None:
    """Berechnet Metriken aus predictions.jsonl."""
    scores = [p.get(score_key) for p in predictions if p.get(score_key) is not None]
    gt_scores = [
        p.get(gt_key)
        for p in predictions
        if p.get(score_key) is not None and p.get(gt_key) is not None
    ]

    if len(scores) != len(gt_scores) or len(scores) == 0:
        return None

    import statistics

    from scipy.stats import pearsonr, spearmanr

    pearson_r, pearson_p = pearsonr(scores, gt_scores)
    spearman_rho, spearman_p = spearmanr(scores, gt_scores)

    mae_val = sum(abs(s - g) for s, g in zip(scores, gt_scores)) / len(scores)
    rmse_val = (sum((s - g) ** 2 for s, g in zip(scores, gt_scores)) / len(scores)) ** 0.5

    mean_gt = statistics.mean(gt_scores)
    ss_res = sum((g - s) ** 2 for s, g in zip(scores, gt_scores))
    ss_tot = sum((g - mean_gt) ** 2 for g in gt_scores)
    r2_val = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Bootstrap CIs (vereinfacht - könnte auch vollständig implementiert werden)
    # Für jetzt ohne CIs, da das aufwändig wäre
    return {
        "pearson": pearson_r,
        "pearson_ci_lower": None,
        "pearson_ci_upper": None,
        "spearman": spearman_rho,
        "spearman_ci_lower": None,
        "spearman_ci_upper": None,
        "mae": mae_val,
        "mae_ci_lower": None,
        "mae_ci_upper": None,
        "rmse": rmse_val,
        "rmse_ci_lower": None,
        "rmse_ci_upper": None,
        "r_squared": r2_val,
        "n": len(scores),
    }


def extract_agent_judge_metrics(
    summary: dict[str, Any], predictions_path: Path | None = None, source: str = "agent"
) -> dict[str, Any] | None:
    """Extrahiert Metriken für Agent oder Judge."""
    if source == "judge" and predictions_path:
        # Versuche Judge-Metriken aus predictions.jsonl zu berechnen
        predictions = load_predictions(predictions_path)
        if predictions:
            judge_metrics = compute_metrics_from_predictions(predictions, "pred_judge")
            if judge_metrics:
                return judge_metrics

    # Fallback: Nutze summary.json (für Agent oder wenn Judge nicht in predictions)
    return {
        "pearson": summary.get("pearson", {}).get("value")
        if isinstance(summary.get("pearson"), dict)
        else summary.get("pearson"),
        "pearson_ci_lower": summary.get("pearson", {}).get("ci_lower")
        if isinstance(summary.get("pearson"), dict)
        else None,
        "pearson_ci_upper": summary.get("pearson", {}).get("ci_upper")
        if isinstance(summary.get("pearson"), dict)
        else None,
        "spearman": summary.get("spearman", {}).get("value")
        if isinstance(summary.get("spearman"), dict)
        else summary.get("spearman"),
        "spearman_ci_lower": summary.get("spearman", {}).get("ci_lower")
        if isinstance(summary.get("spearman"), dict)
        else None,
        "spearman_ci_upper": summary.get("spearman", {}).get("ci_upper")
        if isinstance(summary.get("spearman"), dict)
        else None,
        "mae": summary.get("mae", {}).get("value")
        if isinstance(summary.get("mae"), dict)
        else summary.get("mae"),
        "mae_ci_lower": summary.get("mae", {}).get("ci_lower")
        if isinstance(summary.get("mae"), dict)
        else None,
        "mae_ci_upper": summary.get("mae", {}).get("ci_upper")
        if isinstance(summary.get("mae"), dict)
        else None,
        "rmse": summary.get("rmse", {}).get("value")
        if isinstance(summary.get("rmse"), dict)
        else summary.get("rmse"),
        "rmse_ci_lower": summary.get("rmse", {}).get("ci_lower")
        if isinstance(summary.get("rmse"), dict)
        else None,
        "rmse_ci_upper": summary.get("rmse", {}).get("ci_upper")
        if isinstance(summary.get("rmse"), dict)
        else None,
        "r_squared": summary.get("r_squared"),
        "n": summary.get("n_used"),
    }


def load_baseline_metrics(baseline_matrix_path: Path, target: str) -> dict[str, dict[str, Any]]:
    """Lädt Baseline-Metriken aus summary_matrix.csv."""
    if not baseline_matrix_path.exists():
        return {}

    df = pd.read_csv(baseline_matrix_path)
    target_df = df[df["target"] == target]

    baseline_metrics = {}
    for _, row in target_df.iterrows():
        baseline_name = row["baseline"]
        baseline_metrics[baseline_name] = {
            "pearson": row.get("pearson"),
            "pearson_ci_lower": row.get("pearson_ci_lower"),
            "pearson_ci_upper": row.get("pearson_ci_upper"),
            "spearman": row.get("spearman"),
            "spearman_ci_lower": row.get("spearman_ci_lower"),
            "spearman_ci_upper": row.get("spearman_ci_upper"),
            "mae": row.get("mae"),
            "mae_ci_lower": row.get("mae_ci_lower"),
            "mae_ci_upper": row.get("mae_ci_upper"),
            "rmse": row.get("rmse"),
            "rmse_ci_lower": row.get("rmse_ci_lower"),
            "rmse_ci_upper": row.get("rmse_ci_upper"),
            "r_squared": row.get("r_squared"),
            "n": row.get("n"),
        }

    return baseline_metrics


def format_metric_value(
    value: float | None, ci_lower: float | None = None, ci_upper: float | None = None
) -> str:
    """Formatiert Metrik-Wert mit CI."""
    if value is None:
        return "N/A"
    if ci_lower is not None and ci_upper is not None:
        return f"{value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
    return f"{value:.4f}"


def write_thesis_chapter(
    agent_metrics: dict[str, Any] | None,
    judge_metrics: dict[str, Any] | None,
    baseline_metrics: dict[str, dict[str, Any]],
    target: str,
    has_references: bool,
    out_path: Path,
) -> None:
    """Schreibt Thesis-Kapitel."""
    lines = [
        "# Klassische Metriken als Baselines",
        "",
        f"Dieses Kapitel vergleicht die Agent- und Judge-Methoden mit klassischen Baseline-Metriken für {target.capitalize()}.",
        "",
        "## System vs Humans",
        "",
        "| System | Spearman ρ | Pearson r | MAE | RMSE | R² | n |",
        "|---|---|---|---|---|---|---|",
    ]

    # Agent
    if agent_metrics:
        lines.append(
            f"| **Agent** | {format_metric_value(agent_metrics.get('spearman'), agent_metrics.get('spearman_ci_lower'), agent_metrics.get('spearman_ci_upper'))} | "
            f"{format_metric_value(agent_metrics.get('pearson'), agent_metrics.get('pearson_ci_lower'), agent_metrics.get('pearson_ci_upper'))} | "
            f"{format_metric_value(agent_metrics.get('mae'), agent_metrics.get('mae_ci_lower'), agent_metrics.get('mae_ci_upper'))} | "
            f"{format_metric_value(agent_metrics.get('rmse'), agent_metrics.get('rmse_ci_lower'), agent_metrics.get('rmse_ci_upper'))} | "
            f"{format_metric_value(agent_metrics.get('r_squared'))} | {agent_metrics.get('n', 'N/A')} |"
        )
    else:
        lines.append("| **Agent** | N/A | N/A | N/A | N/A | N/A | N/A |")

    # Judge
    if judge_metrics:
        lines.append(
            f"| **Judge** | {format_metric_value(judge_metrics.get('spearman'), judge_metrics.get('spearman_ci_lower'), judge_metrics.get('spearman_ci_upper'))} | "
            f"{format_metric_value(judge_metrics.get('pearson'), judge_metrics.get('pearson_ci_lower'), judge_metrics.get('pearson_ci_upper'))} | "
            f"{format_metric_value(judge_metrics.get('mae'), judge_metrics.get('mae_ci_lower'), judge_metrics.get('mae_ci_upper'))} | "
            f"{format_metric_value(judge_metrics.get('rmse'), judge_metrics.get('rmse_ci_lower'), judge_metrics.get('rmse_ci_upper'))} | "
            f"{format_metric_value(judge_metrics.get('r_squared'))} | {judge_metrics.get('n', 'N/A')} |"
        )
    else:
        lines.append("| **Judge** | N/A | N/A | N/A | N/A | N/A | N/A |")

    # Baselines
    for baseline_name, metrics in sorted(baseline_metrics.items()):
        baseline_display = baseline_name.upper().replace("_", "-")
        lines.append(
            f"| **{baseline_display}** | {format_metric_value(metrics.get('spearman'), metrics.get('spearman_ci_lower'), metrics.get('spearman_ci_upper'))} | "
            f"{format_metric_value(metrics.get('pearson'), metrics.get('pearson_ci_lower'), metrics.get('pearson_ci_upper'))} | "
            f"{format_metric_value(metrics.get('mae'), metrics.get('mae_ci_lower'), metrics.get('mae_ci_upper'))} | "
            f"{format_metric_value(metrics.get('rmse'), metrics.get('rmse_ci_lower'), metrics.get('rmse_ci_upper'))} | "
            f"{format_metric_value(metrics.get('r_squared'))} | {metrics.get('n', 'N/A')} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
        ]
    )

    if not has_references:
        lines.extend(
            [
                "**ROUGE/BERTScore/BLEU/METEOR nicht berechenbar:** Der SummEval-Datensatz enthält keine Referenz-Zusammenfassungen (Gold-Standard).",
                "Daher können Similarity-Metriken wie ROUGE, BERTScore, BLEU und METEOR nicht berechnet werden, da diese einen Vergleich zwischen System-Output und Referenz erfordern.",
                "",
                "**Readability-Formeln:** Die klassischen Readability-Formeln (Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog Index) wurden berechnet, da diese keine Referenzen benötigen.",
                "Diese Formeln basieren auf statistischen Eigenschaften des Textes (Satzlänge, Silbenanzahl, komplexe Wörter) und messen die Lesbarkeit direkt am Text.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "**Similarity-Metriken:** ROUGE, BERTScore und andere Similarity-Metriken messen die Ähnlichkeit zwischen System-Output und Referenz-Zusammenfassung.",
                "Diese Metriken erfassen Textähnlichkeit, nicht zwingend Lesbarkeit oder Kohärenz im menschlichen Sinne.",
                "",
                "**Readability-Formeln:** Die klassischen Readability-Formeln (Flesch, Flesch-Kincaid, Gunning Fog) messen die Lesbarkeit basierend auf statistischen Textmerkmalen.",
                "",
            ]
        )

    lines.extend(
        [
            "**Korrelationen:** Die Korrelationen (Spearman ρ, Pearson r) zeigen, ob die Baseline-Metriken menschliche Urteile überhaupt treffen können.",
            "Niedrige Korrelationen deuten darauf hin, dass die Baseline-Metriken andere Aspekte messen als menschliche Bewerter.",
            "",
            "**Vergleich mit Agent/Judge:** Die Agent- und Judge-Methoden zeigen [Interpretation basierend auf tatsächlichen Werten einfügen].",
        ]
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Vergleicht Agent/Judge vs Baselines")
    ap.add_argument("--agent_run_dir", type=str, help="Agent-Run-Verzeichnis")
    ap.add_argument("--judge_run_dir", type=str, help="Judge-Run-Verzeichnis")
    ap.add_argument("--baseline_matrix", type=str, help="Baseline summary_matrix.csv")
    ap.add_argument(
        "--target",
        type=str,
        default="readability",
        choices=["readability", "coherence"],
        help="Ziel-Dimension",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="docs/thesis/chapters/classical_metrics_baselines.md",
        help="Output-Pfad",
    )

    args = ap.parse_args()

    agent_metrics = None
    judge_metrics = None
    baseline_metrics = {}
    has_references = False

    # Lade Agent-Metriken
    if args.agent_run_dir:
        agent_dir = Path(args.agent_run_dir)
        agent_summary = load_summary_json(agent_dir / "summary.json")
        if agent_summary:
            agent_metrics = extract_agent_judge_metrics(
                agent_summary, agent_dir / "predictions.jsonl", "agent"
            )

    # Lade Judge-Metriken
    if args.judge_run_dir:
        judge_dir = Path(args.judge_run_dir)
        judge_summary = load_summary_json(judge_dir / "summary.json")
        if judge_summary:
            judge_metrics = extract_agent_judge_metrics(
                judge_summary, judge_dir / "predictions.jsonl", "judge"
            )
    elif args.agent_run_dir:
        # Versuche Judge-Metriken aus dem gleichen Run zu extrahieren
        agent_dir = Path(args.agent_run_dir)
        agent_summary = load_summary_json(agent_dir / "summary.json")
        if agent_summary:
            judge_metrics = extract_agent_judge_metrics(
                agent_summary, agent_dir / "predictions.jsonl", "judge"
            )

    # Lade Baseline-Metriken
    if args.baseline_matrix:
        baseline_matrix_path = Path(args.baseline_matrix)
        baseline_metrics = load_baseline_metrics(baseline_matrix_path, args.target)
        # Prüfe, ob Referenzen vorhanden waren
        if baseline_matrix_path.exists():
            df = pd.read_csv(baseline_matrix_path)
            if not df.empty:
                has_references = df.iloc[0].get("has_references", False)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    write_thesis_chapter(
        agent_metrics,
        judge_metrics,
        baseline_metrics,
        args.target,
        has_references,
        out_path,
    )

    print(f"Thesis-Kapitel geschrieben: {out_path}")


if __name__ == "__main__":
    main()
