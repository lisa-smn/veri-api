"""
Aggregiert mehrere Baseline-Evaluation-Runs zu einer Vergleichstabelle.

Liest mehrere summary.json aus results/evaluation/baselines/* und schreibt:
- summary_matrix.csv
- summary_matrix.md
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


def extract_baseline_metrics(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Extrahiert alle Baseline-Metriken aus summary.json."""
    baseline_metrics = summary.get("baseline_metrics", {})
    rows = []

    for metric_key, metric_data in baseline_metrics.items():
        # metric_key ist z.B. "flesch_readability" oder "rougeL_coherence"
        parts = metric_key.rsplit("_", 1)
        if len(parts) == 2:
            baseline_name, target = parts
        else:
            baseline_name = metric_key
            target = "unknown"

        row = {
            "baseline": baseline_name,
            "target": target,
            "pearson": metric_data.get("pearson", {}).get("value"),
            "pearson_ci_lower": metric_data.get("pearson", {}).get("ci_lower"),
            "pearson_ci_upper": metric_data.get("pearson", {}).get("ci_upper"),
            "spearman": metric_data.get("spearman", {}).get("value"),
            "spearman_ci_lower": metric_data.get("spearman", {}).get("ci_lower"),
            "spearman_ci_upper": metric_data.get("spearman", {}).get("ci_upper"),
            "mae": metric_data.get("mae", {}).get("value"),
            "mae_ci_lower": metric_data.get("mae", {}).get("ci_lower"),
            "mae_ci_upper": metric_data.get("mae", {}).get("ci_upper"),
            "rmse": metric_data.get("rmse", {}).get("value"),
            "rmse_ci_lower": metric_data.get("rmse", {}).get("ci_lower"),
            "rmse_ci_upper": metric_data.get("rmse", {}).get("ci_upper"),
            "r_squared": metric_data.get("r_squared"),
            "n": metric_data.get("n"),
        }
        rows.append(row)

    return rows


def aggregate_runs(run_dirs: list[Path]) -> pd.DataFrame:
    """Aggregiert mehrere Runs zu einer Tabelle."""
    all_rows = []

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

        # Extrahiere Baseline-Metriken
        baseline_rows = extract_baseline_metrics(summary)

        for row in baseline_rows:
            row["run_id"] = run_dir.name
            row["seed"] = metadata.get("seed") if metadata else None
            row["has_references"] = summary.get("has_references", False)
            row["n_used"] = summary.get("n_used")
            row["n_no_ref"] = summary.get("n_no_ref", 0)
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df


def write_summary_md(df: pd.DataFrame, out_path: Path) -> None:
    """Schreibt human-readable Markdown-Tabelle."""
    lines = [
        "# Baseline Evaluation Summary Matrix",
        "",
        f"**Total Runs:** {len(df['run_id'].unique())}",
        f"**Total Baseline-Target Combinations:** {len(df)}",
        "",
        "## Summary Table",
        "",
    ]

    # Gruppiere nach Baseline und Target
    for baseline in sorted(df["baseline"].unique()):
        baseline_df = df[df["baseline"] == baseline]
        lines.append(f"### {baseline.upper()}")
        lines.append("")

        for target in sorted(baseline_df["target"].unique()):
            target_df = baseline_df[baseline_df["target"] == target]
            if target_df.empty:
                continue

            # Nimm ersten Eintrag (sollten alle gleich sein für gleiche baseline+target)
            row = target_df.iloc[0]

            lines.extend(
                [
                    f"**Target:** {target}",
                    "",
                    "| Metrik | Wert | 95% CI |",
                    "|---|---|---|",
                    f"| Pearson r | {row['pearson']:.4f} | [{row['pearson_ci_lower']:.4f}, {row['pearson_ci_upper']:.4f}] |",
                    f"| Spearman ρ | {row['spearman']:.4f} | [{row['spearman_ci_lower']:.4f}, {row['spearman_ci_upper']:.4f}] |",
                    f"| MAE | {row['mae']:.4f} | [{row['mae_ci_lower']:.4f}, {row['mae_ci_upper']:.4f}] |",
                    f"| RMSE | {row['rmse']:.4f} | [{row['rmse_ci_lower']:.4f}, {row['rmse_ci_upper']:.4f}] |",
                    f"| R² | {row['r_squared']:.4f} | - |",
                    f"| n | {row['n']} | - |",
                    "",
                ]
            )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregiert mehrere Baseline-Evaluation-Runs")
    ap.add_argument(
        "run_dirs",
        nargs="+",
        type=str,
        help="Verzeichnisse mit summary.json (z.B. results/evaluation/baselines/*)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="results/evaluation/baselines/summary_matrix",
        help="Output-Pfad (ohne Extension, default: results/evaluation/baselines/summary_matrix)",
    )

    args = ap.parse_args()

    # Expandiere Wildcards
    expanded_dirs = []
    for d in args.run_dirs:
        path = Path(d)
        if "*" in str(path) or "?" in str(path):
            parent = path.parent
            pattern = path.name
            if parent.exists():
                expanded_dirs.extend(
                    [parent / p for p in parent.glob(pattern) if (parent / p).is_dir()]
                )
        else:
            expanded_dirs.append(path)

    run_dirs = []
    for d in expanded_dirs:
        if not d.exists():
            print(f"Warnung: Verzeichnis nicht gefunden, überspringe: {d}")
            continue
        if not (d / "summary.json").exists():
            print(f"Warnung: summary.json nicht gefunden, überspringe: {d}")
            continue
        run_dirs.append(d)

    if not run_dirs:
        ap.error("Keine gültigen Runs mit summary.json gefunden!")

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
    print(f"\n{len(df)} Baseline-Target-Kombinationen aggregiert")


if __name__ == "__main__":
    main()
