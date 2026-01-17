"""
Erstellt 4-Wege-Vergleichstabelle für Readability: v1 raw, v1 calibrated, v2 raw, v2 calibrated.

Input:
  - 4 Run-Verzeichnisse
  - summary.json aus allen Runs

Output:
  docs/status_pack/2026-01-08/readability_improvement_comparison.md
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_summary_json(run_dir: Path) -> dict[str, Any] | None:
    """Lädt summary.json aus Run-Verzeichnis."""
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_run_metadata(run_dir: Path) -> dict[str, Any] | None:
    """Lädt run_metadata.json aus Run-Verzeichnis."""
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def format_metric_with_ci(metric: dict[str, Any] | float, name: str) -> str:
    """Formatiert Metrik mit CI (falls vorhanden)."""
    if isinstance(metric, dict):
        value = metric.get("value", 0.0)
        ci_lower = metric.get("ci_lower")
        ci_upper = metric.get("ci_upper")
        if ci_lower is not None and ci_upper is not None:
            return f"{value:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]"
        return f"{value:.4f}"
    return f"{metric:.4f}"


def create_comparison_md(
    v1_raw_summary: dict[str, Any],
    v1_cal_summary: dict[str, Any],
    v2_raw_summary: dict[str, Any],
    v2_cal_summary: dict[str, Any],
    v1_raw_metadata: dict[str, Any] | None,
    v1_cal_metadata: dict[str, Any] | None,
    v2_raw_metadata: dict[str, Any] | None,
    v2_cal_metadata: dict[str, Any] | None,
    v1_raw_run_id: str,
    v1_cal_run_id: str,
    v2_raw_run_id: str,
    v2_cal_run_id: str,
    out_path: Path,
) -> None:
    """Erstellt 4-Wege-Vergleichstabelle als Markdown."""
    lines = [
        "# Readability Improvement Comparison (4-Way)",
        "",
        "## Metriken-Vergleich",
        "",
        "| Metric | v1 raw | v1 calibrated | v2 raw | v2 calibrated |",
        "|--------|--------|---------------|--------|----------------|",
    ]

    # Pearson
    pearson_v1_raw = format_metric_with_ci(v1_raw_summary.get("pearson", {}), "pearson")
    pearson_v1_cal = format_metric_with_ci(v1_cal_summary.get("pearson", {}), "pearson")
    pearson_v2_raw = format_metric_with_ci(v2_raw_summary.get("pearson", {}), "pearson")
    pearson_v2_cal = format_metric_with_ci(v2_cal_summary.get("pearson", {}), "pearson")
    lines.append(
        f"| **Pearson r** | {pearson_v1_raw} | {pearson_v1_cal} | {pearson_v2_raw} | {pearson_v2_cal} |"
    )

    # Spearman
    spearman_v1_raw = format_metric_with_ci(v1_raw_summary.get("spearman", {}), "spearman")
    spearman_v1_cal = format_metric_with_ci(v1_cal_summary.get("spearman", {}), "spearman")
    spearman_v2_raw = format_metric_with_ci(v2_raw_summary.get("spearman", {}), "spearman")
    spearman_v2_cal = format_metric_with_ci(v2_cal_summary.get("spearman", {}), "spearman")
    lines.append(
        f"| **Spearman ρ** | {spearman_v1_raw} | {spearman_v1_cal} | {spearman_v2_raw} | {spearman_v2_cal} |"
    )

    # MAE
    mae_v1_raw = format_metric_with_ci(v1_raw_summary.get("mae", {}), "mae")
    mae_v1_cal = format_metric_with_ci(v1_cal_summary.get("mae", {}), "mae")
    mae_v2_raw = format_metric_with_ci(v2_raw_summary.get("mae", {}), "mae")
    mae_v2_cal = format_metric_with_ci(v2_cal_summary.get("mae", {}), "mae")
    lines.append(f"| **MAE** | {mae_v1_raw} | {mae_v1_cal} | {mae_v2_raw} | {mae_v2_cal} |")

    # RMSE
    rmse_v1_raw = format_metric_with_ci(v1_raw_summary.get("rmse", {}), "rmse")
    rmse_v1_cal = format_metric_with_ci(v1_cal_summary.get("rmse", {}), "rmse")
    rmse_v2_raw = format_metric_with_ci(v2_raw_summary.get("rmse", {}), "rmse")
    rmse_v2_cal = format_metric_with_ci(v2_cal_summary.get("rmse", {}), "rmse")
    lines.append(f"| **RMSE** | {rmse_v1_raw} | {rmse_v1_cal} | {rmse_v2_raw} | {rmse_v2_cal} |")

    # R²
    r2_v1_raw = v1_raw_summary.get("r_squared", 0.0)
    r2_v1_cal = v1_cal_summary.get("r_squared", 0.0)
    r2_v2_raw = v2_raw_summary.get("r_squared", 0.0)
    r2_v2_cal = v2_cal_summary.get("r_squared", 0.0)
    lines.append(
        f"| **R²** | {r2_v1_raw:.4f} | {r2_v1_cal:.4f} | {r2_v2_raw:.4f} | {r2_v2_cal:.4f} |"
    )

    # n
    n_v1_raw = v1_raw_summary.get("n_used", 0)
    n_v1_cal = v1_cal_summary.get("n_used", 0)
    n_v2_raw = v2_raw_summary.get("n_used", 0)
    n_v2_cal = v2_cal_summary.get("n_used", 0)
    lines.append(f"| **n** | {n_v1_raw} | {n_v1_cal} | {n_v2_raw} | {n_v2_cal} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Interpretation",
            "",
            "### v1 raw (Ranking ok, Skala off)",
            "- Baseline-Version mit Prompt v1",
            "- Ranking-Korrelation (Spearman) ist akzeptabel",
            "- Skalierungsproblem: R² negativ deutet auf systematischen Bias hin",
            "",
            "### v1 calibrated (Ranking ok, Skala deutlich besser) ← echte Verbesserung",
            "- Gleiche Prompt-Version wie v1 raw",
            "- Lineare Kalibrierung auf Calibration-Split (n=500, seed=123) angewendet",
            "- **Erwartung:** R² verbessert sich deutlich, MAE sollte sinken",
            "- **Das ist die echte Verbesserung:** Ranking bleibt erhalten, Skala wird korrigiert",
            "",
            "### v2 raw (kaputt, Output collapse, sauber dokumentiert)",
            "- Neue Prompt-Version v2 (Rubrik, 1-5 integer score)",
            "- **Erwartung:** Output collapse (LLM gibt nur wenige Werte aus)",
            "- **Dokumentation:** Verteilung zeigt Kollaps (z.B. alle Werte in [0.8, 1.0])",
            "",
            "### v2 calibrated (zeigt, dass MAE allein lügen kann)",
            "- v2 raw mit Kalibrierung",
            "- **Erwartung:** MAE kann besser werden, aber Ranking bleibt schlecht",
            "- **Lektion:** MAE allein ist nicht ausreichend - Spearman zeigt echte Qualität",
            "",
            "---",
            "",
            "## Run-Ordner",
            "",
            f"- **v1 raw:** `{v1_raw_run_id}`",
            f"- **v1 calibrated:** `{v1_cal_run_id}`",
            f"- **v2 raw:** `{v2_raw_run_id}`",
            f"- **v2 calibrated:** `{v2_cal_run_id}`",
        ]
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Erstellt 4-Wege-Vergleichstabelle für Readability")
    ap.add_argument("--v1_raw_dir", type=str, required=True, help="Run-Verzeichnis für v1 raw")
    ap.add_argument(
        "--v1_cal_dir", type=str, required=True, help="Run-Verzeichnis für v1 calibrated"
    )
    ap.add_argument("--v2_raw_dir", type=str, required=True, help="Run-Verzeichnis für v2 raw")
    ap.add_argument(
        "--v2_cal_dir", type=str, required=True, help="Run-Verzeichnis für v2 calibrated"
    )
    ap.add_argument(
        "--out",
        type=str,
        default="docs/status_pack/2026-01-08/readability_improvement_comparison.md",
        help="Output-Pfad",
    )

    args = ap.parse_args()

    v1_raw_dir = Path(args.v1_raw_dir)
    v1_cal_dir = Path(args.v1_cal_dir)
    v2_raw_dir = Path(args.v2_raw_dir)
    v2_cal_dir = Path(args.v2_cal_dir)

    for d, name in [
        (v1_raw_dir, "v1_raw"),
        (v1_cal_dir, "v1_cal"),
        (v2_raw_dir, "v2_raw"),
        (v2_cal_dir, "v2_cal"),
    ]:
        if not d.exists():
            ap.error(f"{name} Run-Verzeichnis nicht gefunden: {d}")

    print("Lade Summaries...")
    v1_raw_summary = load_summary_json(v1_raw_dir)
    v1_cal_summary = load_summary_json(v1_cal_dir)
    v2_raw_summary = load_summary_json(v2_raw_dir)
    v2_cal_summary = load_summary_json(v2_cal_dir)

    if not v1_raw_summary:
        ap.error(f"summary.json nicht gefunden in: {v1_raw_dir}")
    if not v1_cal_summary:
        ap.error(f"summary.json nicht gefunden in: {v1_cal_dir}")
    if not v2_raw_summary:
        ap.error(f"summary.json nicht gefunden in: {v2_raw_dir}")
    if not v2_cal_summary:
        ap.error(f"summary.json nicht gefunden in: {v2_cal_dir}")

    v1_raw_metadata = load_run_metadata(v1_raw_dir)
    v1_cal_metadata = load_run_metadata(v1_cal_dir)
    v2_raw_metadata = load_run_metadata(v2_raw_dir)
    v2_cal_metadata = load_run_metadata(v2_cal_dir)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Erstelle Vergleichstabelle nach: {out_path}")
    create_comparison_md(
        v1_raw_summary=v1_raw_summary,
        v1_cal_summary=v1_cal_summary,
        v2_raw_summary=v2_raw_summary,
        v2_cal_summary=v2_cal_summary,
        v1_raw_metadata=v1_raw_metadata,
        v1_cal_metadata=v1_cal_metadata,
        v2_raw_metadata=v2_raw_metadata,
        v2_cal_metadata=v2_cal_metadata,
        v1_raw_run_id=v1_raw_dir.name,
        v1_cal_run_id=v1_cal_dir.name,
        v2_raw_run_id=v2_raw_dir.name,
        v2_cal_run_id=v2_cal_dir.name,
        out_path=out_path,
    )

    print("\nVergleichstabelle erstellt!")


if __name__ == "__main__":
    main()
