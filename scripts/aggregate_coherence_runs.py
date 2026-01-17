"""
Aggregiert mehrere Coherence-Evaluation-Runs zu einer Vergleichstabelle.

Liest mehrere summary.json (agent + baselines + stress optional) und schreibt:
- summary_matrix.csv (tabellarischer Vergleich)
- summary_matrix.md (human-readable)

Input: Verzeichnisse mit summary.json (z.B. results/evaluation/coherence/*/summary.json)
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
    except Exception as e:
        print(f"Warnung: Konnte {path} nicht laden: {e}")
        return None


def load_run_metadata(path: Path) -> dict[str, Any] | None:
    """Lädt run_metadata.json."""
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    """Extrahiert Metriken aus summary.json."""
    metrics = {}

    # Pearson
    if "pearson" in summary:
        if isinstance(summary["pearson"], dict):
            metrics["pearson"] = summary["pearson"].get("value", 0.0)
            metrics["pearson_ci_lower"] = summary["pearson"].get("ci_lower", 0.0)
            metrics["pearson_ci_upper"] = summary["pearson"].get("ci_upper", 0.0)
        else:
            metrics["pearson"] = summary["pearson"]
            metrics["pearson_ci_lower"] = None
            metrics["pearson_ci_upper"] = None
    else:
        metrics["pearson"] = None
        metrics["pearson_ci_lower"] = None
        metrics["pearson_ci_upper"] = None

    # Spearman
    if "spearman" in summary:
        if isinstance(summary["spearman"], dict):
            metrics["spearman"] = summary["spearman"].get("value", 0.0)
            metrics["spearman_ci_lower"] = summary["spearman"].get("ci_lower", 0.0)
            metrics["spearman_ci_upper"] = summary["spearman"].get("ci_upper", 0.0)
        else:
            metrics["spearman"] = summary["spearman"]
            metrics["spearman_ci_lower"] = None
            metrics["spearman_ci_upper"] = None
    else:
        metrics["spearman"] = None
        metrics["spearman_ci_lower"] = None
        metrics["spearman_ci_upper"] = None

    # MAE
    if "mae" in summary:
        if isinstance(summary["mae"], dict):
            metrics["mae"] = summary["mae"].get("value", 0.0)
            metrics["mae_ci_lower"] = summary["mae"].get("ci_lower", 0.0)
            metrics["mae_ci_upper"] = summary["mae"].get("ci_upper", 0.0)
        else:
            metrics["mae"] = summary["mae"]
            metrics["mae_ci_lower"] = None
            metrics["mae_ci_upper"] = None
    elif "mae_0_1" in summary:
        metrics["mae"] = summary["mae_0_1"]
        metrics["mae_ci_lower"] = None
        metrics["mae_ci_upper"] = None
    else:
        metrics["mae"] = None
        metrics["mae_ci_lower"] = None
        metrics["mae_ci_upper"] = None

    # RMSE
    if "rmse" in summary:
        if isinstance(summary["rmse"], dict):
            metrics["rmse"] = summary["rmse"].get("value", 0.0)
            metrics["rmse_ci_lower"] = summary["rmse"].get("ci_lower", 0.0)
            metrics["rmse_ci_upper"] = summary["rmse"].get("ci_upper", 0.0)
        else:
            metrics["rmse"] = summary["rmse"]
            metrics["rmse_ci_lower"] = None
            metrics["rmse_ci_upper"] = None
    else:
        metrics["rmse"] = None
        metrics["rmse_ci_lower"] = None
        metrics["rmse_ci_upper"] = None

    # R²
    metrics["r_squared"] = summary.get("r_squared")

    # Counts
    metrics["n_used"] = summary.get("n_used")
    metrics["n_failed"] = summary.get("n_failed")

    return metrics


def determine_system_type(summary: dict[str, Any], metadata: dict[str, Any] | None) -> str:
    """Bestimmt System-Typ (agent, rouge_l, bertscore, llm_judge, stress_shuffle, stress_inject)."""
    # Aus summary: method field (für llm_judge)
    method = summary.get("method")
    if method == "llm_judge":
        return "llm_judge"

    # Aus metadata
    if metadata:
        baseline_type = metadata.get("baseline_type")
        if baseline_type:
            return baseline_type
        mode = metadata.get("mode")
        if mode:
            return f"stress_{mode}"

    # Aus summary
    baseline_type = summary.get("baseline_type")
    if baseline_type:
        return baseline_type

    mode = summary.get("mode")
    if mode:
        return f"stress_{mode}"

    # Default: agent
    return "agent"


def extract_model_prompt(metadata: dict[str, Any] | None) -> tuple[str, str]:
    """Extrahiert Model und Prompt-Version aus metadata."""
    if not metadata:
        return "unknown", "unknown"
    config = metadata.get("config", {})
    # Für Judge: judge_model statt llm_model
    model = (
        config.get("judge_model")
        or config.get("llm_model")
        or metadata.get("llm_model")
        or "unknown"
    )
    # Für Judge: rubric_version statt prompt_version
    prompt = config.get("rubric_version") or config.get("prompt_version") or "unknown"
    return model, prompt


def aggregate_runs(run_dirs: list[Path]) -> pd.DataFrame:
    """Aggregiert mehrere Runs zu einer Tabelle."""
    rows = []

    for run_dir in run_dirs:
        summary_path = run_dir / "summary.json"
        metadata_path = run_dir / "run_metadata.json"

        if not summary_path.exists():
            print(f"Warnung: {summary_path} nicht gefunden, überspringe {run_dir}")
            continue

        summary = load_summary_json(summary_path)
        if not summary:
            continue

        metadata = load_run_metadata(metadata_path) if metadata_path.exists() else None

        metrics = extract_metrics(summary)
        system_type = determine_system_type(summary, metadata)
        model, prompt = extract_model_prompt(metadata)

        seed = None
        if metadata:
            seed = metadata.get("seed")

        row = {
            "run_id": run_dir.name,
            "system": system_type,
            "model": model,
            "prompt_version": prompt,
            "seed": seed,
            "pearson": metrics["pearson"],
            "pearson_ci_lower": metrics["pearson_ci_lower"],
            "pearson_ci_upper": metrics["pearson_ci_upper"],
            "spearman": metrics["spearman"],
            "spearman_ci_lower": metrics["spearman_ci_lower"],
            "spearman_ci_upper": metrics["spearman_ci_upper"],
            "mae": metrics["mae"],
            "mae_ci_lower": metrics["mae_ci_lower"],
            "mae_ci_upper": metrics["mae_ci_upper"],
            "rmse": metrics["rmse"],
            "rmse_ci_lower": metrics["rmse_ci_lower"],
            "rmse_ci_upper": metrics["rmse_ci_upper"],
            "r_squared": metrics["r_squared"],
            "n_used": metrics["n_used"],
            "n_failed": metrics["n_failed"],
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def write_summary_md(df: pd.DataFrame, out_path: Path) -> None:
    """Schreibt human-readable Markdown-Tabelle."""
    lines = [
        "# Coherence Evaluation Comparison",
        "",
        "## Summary Matrix",
        "",
    ]

    # Gruppiere nach System-Typ
    for system in df["system"].unique():
        lines.append(f"### {system.upper()}")
        lines.append("")

        # Sortierung: primär nach Spearman, dann Pearson
        system_df = df[df["system"] == system].sort_values(
            ["spearman", "pearson"], ascending=False, na_position="last"
        )

        # Header
        lines.append(
            "| Run ID | Model | Prompt | Seed | Pearson r | Spearman ρ | MAE | RMSE | R² | n_used |"
        )
        lines.append(
            "|--------|-------|--------|------|----------|------------|-----|------|-----|--------|"
        )

        # Rows
        for _, row in system_df.iterrows():
            pearson_str = f"{row['pearson']:.4f}" if pd.notna(row["pearson"]) else "N/A"
            if pd.notna(row["pearson_ci_lower"]) and pd.notna(row["pearson_ci_upper"]):
                pearson_str += f" [{row['pearson_ci_lower']:.4f}, {row['pearson_ci_upper']:.4f}]"

            spearman_str = f"{row['spearman']:.4f}" if pd.notna(row["spearman"]) else "N/A"
            if pd.notna(row["spearman_ci_lower"]) and pd.notna(row["spearman_ci_upper"]):
                spearman_str += f" [{row['spearman_ci_lower']:.4f}, {row['spearman_ci_upper']:.4f}]"

            mae_str = f"{row['mae']:.4f}" if pd.notna(row["mae"]) else "N/A"
            if pd.notna(row["mae_ci_lower"]) and pd.notna(row["mae_ci_upper"]):
                mae_str += f" [{row['mae_ci_lower']:.4f}, {row['mae_ci_upper']:.4f}]"

            rmse_str = f"{row['rmse']:.4f}" if pd.notna(row["rmse"]) else "N/A"
            if pd.notna(row["rmse_ci_lower"]) and pd.notna(row["rmse_ci_upper"]):
                rmse_str += f" [{row['rmse_ci_lower']:.4f}, {row['rmse_ci_upper']:.4f}]"

            r2_str = f"{row['r_squared']:.4f}" if pd.notna(row["r_squared"]) else "N/A"
            n_used_str = str(int(row["n_used"])) if pd.notna(row["n_used"]) else "N/A"
            seed_str = str(int(row["seed"])) if pd.notna(row["seed"]) else "N/A"

            lines.append(
                f"| {row['run_id']} | {row['model']} | {row['prompt_version']} | {seed_str} | "
                f"{pearson_str} | {spearman_str} | {mae_str} | {rmse_str} | {r2_str} | {n_used_str} |"
            )

        lines.append("")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregiert mehrere Coherence-Evaluation-Runs")
    ap.add_argument(
        "run_dirs",
        nargs="+",
        type=str,
        help="Verzeichnisse mit summary.json (z.B. results/evaluation/coherence/*)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="results/evaluation/coherence/summary_matrix",
        help="Output-Pfad (ohne Extension, default: results/evaluation/coherence/summary_matrix)",
    )

    args = ap.parse_args()

    run_dirs = [Path(d) for d in args.run_dirs]

    # Prüfe, ob alle Verzeichnisse existieren
    for d in run_dirs:
        if not d.exists():
            ap.error(f"Verzeichnis nicht gefunden: {d}")
        if not (d / "summary.json").exists():
            ap.error(f"summary.json nicht gefunden in: {d}")

    print(f"Aggregiere {len(run_dirs)} Runs...")
    df = aggregate_runs(run_dirs)

    if df.empty:
        print("Keine gültigen Runs gefunden.")
        return

    # Output
    out_path = Path(args.out)
    csv_path = out_path.with_suffix(".csv")
    md_path = out_path.with_suffix(".md")

    df.to_csv(csv_path, index=False)
    write_summary_md(df, md_path)

    print("\nAggregation abgeschlossen!")
    print(f"  CSV: {csv_path}")
    print(f"  MD:  {md_path}")
    print(f"\n{len(df)} Runs aggregiert:")
    print(df[["run_id", "system", "pearson", "spearman", "mae", "rmse"]].to_string())


if __name__ == "__main__":
    main()
