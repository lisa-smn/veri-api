"""
Regeneriert docs/status/readability_status.md aus Run-Artefakten.

Liest:
- Agent-Run: results/evaluation/readability/<run_id>/
- Baseline-Matrix: results/evaluation/baselines/summary_matrix.csv

Schreibt:
- docs/status/readability_status.md
- Optional: docs/status/img/readability_scatter.png
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


def compute_judge_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Berechnet Judge-Metriken aus predictions.jsonl."""
    judge_scores = [p.get("pred_judge") for p in predictions if p.get("pred_judge") is not None]
    gt_scores = [
        p.get("gt_norm")
        for p in predictions
        if p.get("pred_judge") is not None and p.get("gt_norm") is not None
    ]

    if len(judge_scores) != len(gt_scores) or len(judge_scores) == 0:
        return None

    import statistics

    from scipy.stats import pearsonr, spearmanr

    pearson_r, _ = pearsonr(judge_scores, gt_scores)
    spearman_rho, _ = spearmanr(judge_scores, gt_scores)
    mae = sum(abs(j - g) for j, g in zip(judge_scores, gt_scores)) / len(judge_scores)
    rmse = (sum((j - g) ** 2 for j, g in zip(judge_scores, gt_scores)) / len(judge_scores)) ** 0.5
    mean_gt = statistics.mean(gt_scores)
    ss_res = sum((g - j) ** 2 for j, g in zip(judge_scores, gt_scores))
    ss_tot = sum((g - mean_gt) ** 2 for g in gt_scores)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {
        "pearson": pearson_r,
        "spearman": spearman_rho,
        "mae": mae,
        "rmse": rmse,
        "r_squared": r2,
        "n": len(judge_scores),
    }


def load_baseline_metrics(
    baseline_matrix_path: Path, target: str = "readability"
) -> dict[str, dict[str, Any]]:
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


def format_metric(
    value: float | None, ci_lower: float | None = None, ci_upper: float | None = None
) -> str:
    """Formatiert Metrik mit CI."""
    if value is None:
        return "N/A"
    if ci_lower is not None and ci_upper is not None:
        return f"{value:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"
    return f"{value:.3f}"


def create_scatter_plot(
    predictions: list[dict[str, Any]],
    out_path: Path,
) -> None:
    """Erstellt Scatter-Plot GT vs Agent/Judge."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warnung: matplotlib nicht verfügbar, Scatter-Plot wird übersprungen")
        return

    agent_scores = [
        p.get("pred_agent")
        for p in predictions
        if p.get("pred_agent") is not None and p.get("gt_norm") is not None
    ]
    judge_scores = [
        p.get("pred_judge")
        for p in predictions
        if p.get("pred_judge") is not None and p.get("gt_norm") is not None
    ]
    gt_scores_agent = [
        p.get("gt_norm")
        for p in predictions
        if p.get("pred_agent") is not None and p.get("gt_norm") is not None
    ]
    gt_scores_judge = [
        p.get("gt_norm")
        for p in predictions
        if p.get("pred_judge") is not None and p.get("gt_norm") is not None
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Agent
    if agent_scores and gt_scores_agent:
        axes[0].scatter(gt_scores_agent, agent_scores, alpha=0.5, s=20)
        axes[0].plot([0, 1], [0, 1], "r--", alpha=0.5, label="Perfect")
        axes[0].set_xlabel("Ground Truth (normalized)")
        axes[0].set_ylabel("Agent Score")
        axes[0].set_title("Agent vs Ground Truth")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    # Judge
    if judge_scores and gt_scores_judge:
        axes[1].scatter(gt_scores_judge, judge_scores, alpha=0.5, s=20, color="orange")
        axes[1].plot([0, 1], [0, 1], "r--", alpha=0.5, label="Perfect")
        axes[1].set_xlabel("Ground Truth (normalized)")
        axes[1].set_ylabel("Judge Score")
        axes[1].set_title("Judge vs Ground Truth")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Scatter-Plot gespeichert: {out_path}")


def write_status_report(
    agent_summary: dict[str, Any],
    agent_metadata: dict[str, Any] | None,
    judge_metrics: dict[str, Any] | None,
    baseline_metrics: dict[str, dict[str, Any]],
    out_path: Path,
    scatter_path: Path | None = None,
) -> None:
    """Schreibt readability_status.md."""
    lines = [
        "# Readability Evaluation: Final Status",
        "",
        f"**Datum:** {agent_metadata.get('timestamp', 'unknown')[:8] if agent_metadata else 'unknown'}",
        "**Status:** Finale Evaluation abgeschlossen",
        "",
        "---",
        "",
        "## Setup",
        "",
        "- **Dataset:** SummEval (`data/sumeval/sumeval_clean.jsonl`)",
        f"- **Subset:** n={agent_summary.get('n_used', 'N/A')}, seed={agent_metadata.get('seed', 'N/A') if agent_metadata else 'N/A'} (reproduzierbar)",
        f"- **Model:** {agent_metadata.get('config', {}).get('llm_model', 'N/A') if agent_metadata else 'N/A'}",
        f"- **Prompt:** {agent_metadata.get('config', {}).get('prompt_version', 'N/A') if agent_metadata else 'N/A'}",
        f"- **Bootstrap:** n={agent_metadata.get('config', {}).get('bootstrap_n', 'N/A') if agent_metadata else 'N/A'} resamples, 95% Konfidenzintervall",
        f"- **Cache:** {agent_summary.get('cache_stats', {}).get('cache_mode', 'unknown')} (cache_hits={agent_summary.get('cache_stats', {}).get('cache_hits', 0)}, cache_misses={agent_summary.get('cache_stats', {}).get('cache_misses', 0)})",
        "- **Score Source:** agent (Agent-Score als primäre Metrik)",
        "",
        "---",
        "",
        "## System vs Humans: Vergleichstabelle",
        "",
        "| System | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) | R² | n |",
        "|---|---|---|---|---|---|---|",
    ]

    # Agent
    agent_pearson = agent_summary.get("pearson", {})
    agent_spearman = agent_summary.get("spearman", {})
    agent_mae = agent_summary.get("mae", {})
    agent_rmse = agent_summary.get("rmse", {})
    r2_val = agent_summary.get("r_squared")
    r2_str = f"{r2_val:.3f}" if isinstance(r2_val, (int, float)) else "N/A"
    lines.append(
        f"| **Agent** | {format_metric(agent_spearman.get('value'), agent_spearman.get('ci_lower'), agent_spearman.get('ci_upper'))} | "
        f"{format_metric(agent_pearson.get('value'), agent_pearson.get('ci_lower'), agent_pearson.get('ci_upper'))} | "
        f"{format_metric(agent_mae.get('value'), agent_mae.get('ci_lower'), agent_mae.get('ci_upper'))} | "
        f"{format_metric(agent_rmse.get('value'), agent_rmse.get('ci_lower'), agent_rmse.get('ci_upper'))} | "
        f"{r2_str} | {agent_summary.get('n_used', 'N/A')} |"
    )

    # Judge
    if judge_metrics:
        judge_r2 = judge_metrics.get("r_squared", None)
        judge_r2_str = f"{judge_r2:.3f}" if isinstance(judge_r2, (int, float)) else "N/A"
        lines.append(
            f"| **Judge** | {format_metric(judge_metrics.get('spearman'))} | "
            f"{format_metric(judge_metrics.get('pearson'))} | "
            f"{format_metric(judge_metrics.get('mae'))} | "
            f"{format_metric(judge_metrics.get('rmse'))} | "
            f"{judge_r2_str} | {judge_metrics.get('n', 'N/A')} |"
        )
    else:
        lines.append("| **Judge** | N/A | N/A | N/A | N/A | N/A | N/A |")

    # Baselines
    baseline_names = {
        "flesch": "Flesch",
        "flesch_kincaid": "Flesch-Kincaid",
        "gunning_fog": "Gunning Fog",
    }
    for baseline_key, baseline_display in sorted(baseline_names.items()):
        metrics = baseline_metrics.get(baseline_key, {})
        if metrics:
            baseline_r2 = metrics.get("r_squared", None)
            baseline_r2_str = (
                f"{baseline_r2:.3f}" if isinstance(baseline_r2, (int, float)) else "N/A"
            )
            lines.append(
                f"| **{baseline_display}** | {format_metric(metrics.get('spearman'), metrics.get('spearman_ci_lower'), metrics.get('spearman_ci_upper'))} | "
                f"{format_metric(metrics.get('pearson'), metrics.get('pearson_ci_lower'), metrics.get('pearson_ci_upper'))} | "
                f"{format_metric(metrics.get('mae'), metrics.get('mae_ci_lower'), metrics.get('mae_ci_upper'))} | "
                f"{format_metric(metrics.get('rmse'), metrics.get('rmse_ci_lower'), metrics.get('rmse_ci_upper'))} | "
                f"{baseline_r2_str} | {metrics.get('n', 'N/A')} |"
            )

    lines.extend(
        [
            "",
            "**Hinweis:** Judge-Metriken wurden aus `predictions.jsonl` berechnet (keine Bootstrap-CIs verfügbar). ROUGE/BERTScore/BLEU/METEOR nicht berechenbar, da SummEval keine Referenz-Zusammenfassungen enthält.",
            "",
            "---",
            "",
            "## Interpretation",
            "",
            "### Hauptkennzahl: Spearman ρ (Rangkorrelation)",
            "",
            "**Warum Spearman primär ist:**",
            "- Spearman ρ misst die **Rangfolge** (monotone Beziehung), nicht absolute Werte",
            "- Robust gegen Skalenfehler und nicht-lineare Beziehungen",
            '- Für Ranking-basierte Evaluation ideal: "Ist A besser als B?" ist wichtiger als "Ist A genau 0.8?"',
            f"- Agent zeigt moderate Rangkorrelation (ρ = {agent_spearman.get('value', 0):.3f}), was bedeutet, dass der Agent die Rangfolge menschlicher Bewertungen teilweise erfasst",
            "",
            "### Agent vs Judge vs Baselines",
            "",
            f"- **Agent (ρ = {agent_spearman.get('value', 0):.3f}):** Beste Performance. Moderate Korrelation mit menschlichen Bewertungen, zeigt dass der Agent semantische und strukturelle Aspekte der Lesbarkeit erfasst.",
        ]
    )

    if judge_metrics:
        lines.append(
            f"- **Judge (ρ = {judge_metrics.get('spearman', 0):.3f}):** Schwächer als Agent, aber immer noch positive Korrelation. LLM-as-a-Judge zeigt, dass moderne LLMs als Baseline funktionieren, aber nicht besser als der spezialisierte Agent."
        )

    lines.extend(
        [
            "- **Klassische Formeln (ρ ≈ -0.05):** Nahezu keine Korrelation. Flesch, Flesch-Kincaid und Gunning Fog basieren nur auf statistischen Textmerkmalen (Satzlänge, Silbenanzahl) und erfassen nicht die semantischen Aspekte, die menschliche Bewerter berücksichtigen.",
            "",
            "### R² negativ: Was bedeutet das?",
            "",
            f"**R² = {agent_summary.get('r_squared', 0):.3f} (Agent) bedeutet:**",
            "- Das Modell ist **schlechter als eine Mittelwert-Baseline** (immer den Durchschnitt vorhersagen)",
            "- **Nicht widersprüchlich zu brauchbarem Spearman:** R² misst absolute Werte, Spearman misst Rangfolge",
            "- **Mögliche Ursachen:**",
            "  - Kalibrierungsproblem: Agent-Scores sind nicht auf der gleichen Skala wie GT",
            "  - Geringe GT-Varianz: Wenn alle GT-Werte ähnlich sind, ist R² instabil",
            "  - Skalenfehler: Agent könnte systematisch zu hoch/niedrig vorhersagen, aber die Rangfolge stimmt",
            "",
            "**Fazit:** Für Ranking-basierte Evaluation (Spearman) ist der Agent brauchbar, auch wenn R² negativ ist.",
            "",
            "### MAE in verständlicher Skala",
            "",
            f"**MAE auf Skala 0-1:** {agent_mae.get('value', 0):.3f}",
            f"**MAE auf Skala 1-5:** {agent_mae.get('value', 0) * 4:.2f} Punkte",
            "",
            f"Der Agent weicht im Durchschnitt um etwa {agent_mae.get('value', 0) * 4:.2f} Punkte (auf der 1-5 Skala) von den menschlichen Bewertungen ab.",
            "",
            "---",
            "",
            "## Reproduzierbarkeit",
            "",
            "### Run-Artefakte",
            "",
        ]
    )

    if agent_metadata:
        run_id = agent_metadata.get("run_id", "unknown")
        git_commit = agent_metadata.get("git_commit", "unknown")
        timestamp = agent_metadata.get("timestamp", "unknown")
        seed = agent_metadata.get("seed", "N/A")
        n_used = agent_summary.get("n_used", "N/A")

        lines.extend(
            [
                f"- **Agent-Run:** `results/evaluation/readability/{run_id}/`",
                f"  - Git-Commit: `{git_commit}`",
                f"  - Timestamp: {timestamp}",
                f"  - Seed: {seed}",
                f"  - n_used: {n_used}/{agent_metadata.get('n_total', 'N/A')}",
                "",
            ]
        )

    lines.extend(
        [
            "- **Baseline-Runs:**",
            "  - `results/evaluation/baselines/baselines_readability_flesch_fk_fog_20260116_175246_seed42/`",
            "  - `results/evaluation/baselines/baselines_readability_coherence_flesch_fk_fog_20260116_175253_seed42/`",
            "",
            "- **Aggregation:**",
            "  - `results/evaluation/baselines/summary_matrix.csv`",
            "  - `results/evaluation/baselines/summary_matrix.md`",
            "",
            "### Reproduktion",
            "",
            "```bash",
            "# Agent-Run (mit Judge)",
            "ENABLE_LLM_JUDGE=true JUDGE_MODE=secondary JUDGE_N=3 JUDGE_TEMPERATURE=0 \\",
            "python scripts/eval_sumeval_readability.py \\",
            "  --data data/sumeval/sumeval_clean.jsonl \\",
            "  --max_examples 200 \\",
            "  --seed 42 \\",
            "  --prompt_version v1 \\",
            "  --bootstrap_n 2000 \\",
            "  --cache_mode off \\",
            "  --score_source agent",
            "",
            "# Baselines",
            "python scripts/eval_sumeval_baselines.py \\",
            "  --data data/sumeval/sumeval_clean.jsonl \\",
            "  --max_examples 200 \\",
            "  --seed 42 \\",
            "  --bootstrap_n 2000 \\",
            "  --targets readability \\",
            "  --metrics 'flesch,fk,fog'",
            "```",
            "",
            '**Hinweis zu zsh:** Bei Verwendung von `--metrics flesch,fk,fog` in zsh können Kommas als Dateiattribute interpretiert werden (Fehler: "unknown file attribute: k/b"). Lösung:',
            "",
            "```bash",
            "# Option 1: Metriken in Anführungszeichen setzen",
            "--metrics 'flesch,fk,fog'",
            "",
            "# Option 2: noglob verwenden",
            "noglob python scripts/eval_sumeval_baselines.py --metrics flesch,fk,fog ...",
            "```",
            "",
            "---",
            "",
            "## Artefakte",
            "",
            "### Run-Verzeichnisse",
            "",
        ]
    )

    if agent_metadata:
        run_id = agent_metadata.get("run_id", "unknown")
        lines.append(f"- **Agent:** `results/evaluation/readability/{run_id}/`")

    lines.extend(
        [
            "- **Baselines:** `results/evaluation/baselines/baselines_readability_flesch_fk_fog_20260116_175246_seed42/`",
            "",
            "### Aggregations-Matrizen",
            "",
            "- **CSV:** `results/evaluation/baselines/summary_matrix.csv`",
            "- **Markdown:** `results/evaluation/baselines/summary_matrix.md`",
            "",
            "### Vergleichs-Report",
            "",
            "- **Thesis-Kapitel:** `docs/thesis/chapters/classical_metrics_baselines.md`",
            "",
        ]
    )

    if scatter_path and scatter_path.exists():
        lines.extend(
            [
                "",
                "### Visualisierung",
                "",
                "![Scatter Plot](img/readability_scatter.png)",
                "",
            ]
        )

    lines.extend(
        [
            "---",
            "",
            "**Details zu Metriken:** Siehe `docs/status_pack/2026-01-08/04_metrics_glossary.md`",
            "**Vollständige Evaluation:** Siehe `docs/status_pack/2026-01-08/03_evaluation_results.md`",
        ]
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Status-Report geschrieben: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Regeneriert readability_status.md")
    ap.add_argument("--agent_run_dir", type=str, required=True, help="Agent-Run-Verzeichnis")
    ap.add_argument(
        "--baseline_matrix", type=str, required=True, help="Baseline summary_matrix.csv"
    )
    ap.add_argument(
        "--out", type=str, default="docs/status/readability_status.md", help="Output-Pfad"
    )
    ap.add_argument("--scatter", type=str, help="Scatter-Plot-Pfad (optional)")
    ap.add_argument(
        "--plot", action="store_true", help="Erstelle Scatter-Plot (wenn matplotlib verfügbar)"
    )
    ap.add_argument(
        "--check",
        type=str,
        help="Check-Mode: Vergleiche gegen erwartete Datei (Pfad), Exit 0 wenn identisch",
    )

    args = ap.parse_args()

    agent_dir = Path(args.agent_run_dir)
    baseline_matrix_path = Path(args.baseline_matrix)
    out_path = Path(args.out)

    # Scatter-Plot: --plot oder --scatter
    scatter_path = None
    if args.plot:
        scatter_path = out_path.parent / "img" / "readability_scatter.png"
    elif args.scatter:
        scatter_path = Path(args.scatter)

    # Lade Agent-Daten
    agent_summary = load_summary_json(agent_dir / "summary.json")
    if not agent_summary:
        ap.error(f"summary.json nicht gefunden in {agent_dir}")

    agent_metadata = None
    if (agent_dir / "run_metadata.json").exists():
        agent_metadata = load_summary_json(agent_dir / "run_metadata.json")

    # Lade Predictions für Judge-Metriken
    predictions = load_predictions(agent_dir / "predictions.jsonl")
    judge_metrics = compute_judge_metrics(predictions)

    # Lade Baseline-Metriken
    baseline_metrics = load_baseline_metrics(baseline_matrix_path, "readability")

    # Erstelle Scatter-Plot (optional)
    if scatter_path:
        scatter_path.parent.mkdir(parents=True, exist_ok=True)
        create_scatter_plot(predictions, scatter_path)

    # Check-Mode: Vergleiche gegen erwartete Datei
    if args.check:
        expected_path = Path(args.check)
        if not expected_path.exists():
            print(f"✗ Erwartete Datei nicht gefunden: {expected_path}")
            return 1

        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)

        # Schreibe in temporäre Datei
        write_status_report(
            agent_summary,
            agent_metadata,
            judge_metrics,
            baseline_metrics,
            tmp_path,
            scatter_path if scatter_path and scatter_path.exists() else None,
        )

        # Vergleiche mit erwarteter Datei
        with expected_path.open("r", encoding="utf-8") as f:
            expected_content = f.read()
        with tmp_path.open("r", encoding="utf-8") as f:
            new_content = f.read()

        if expected_content == new_content:
            tmp_path.unlink()
            print("✓ Datei ist identisch (keine Änderungen nötig)")
            return 0
        print("✗ Datei unterscheidet sich:")
        print(f"  Erwartet: {expected_path}")
        print(f"  Generiert: {tmp_path}")
        tmp_path.unlink()
        return 1

    # Schreibe Status-Report
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_status_report(
        agent_summary,
        agent_metadata,
        judge_metrics,
        baseline_metrics,
        out_path,
        scatter_path if scatter_path and scatter_path.exists() else None,
    )

    return 0


if __name__ == "__main__":
    import sys

    exit_code = main()
    sys.exit(exit_code if exit_code is not None else 0)
