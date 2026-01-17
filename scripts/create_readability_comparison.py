"""
Erstellt Vergleichstabelle für Readability v1 vs v2.

Input:
  - Run-Verzeichnisse (v1 und v2)
  - summary.json aus beiden Runs

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
    v1_summary: dict[str, Any],
    v2_summary: dict[str, Any],
    v1_metadata: dict[str, Any] | None,
    v2_metadata: dict[str, Any] | None,
    v1_run_id: str,
    v2_run_id: str,
    out_path: Path,
) -> None:
    """Erstellt Vergleichstabelle als Markdown."""
    lines = [
        "# Readability Improvement Comparison (v1 vs v2)",
        "",
        "## Metriken-Vergleich",
        "",
        "| Metric | v1 | v2 |",
        "|--------|----|----|",
    ]

    # Pearson
    pearson_v1 = format_metric_with_ci(v1_summary.get("pearson", {}), "pearson")
    pearson_v2 = format_metric_with_ci(v2_summary.get("pearson", {}), "pearson")
    lines.append(f"| **Pearson r** | {pearson_v1} | {pearson_v2} |")

    # Spearman
    spearman_v1 = format_metric_with_ci(v1_summary.get("spearman", {}), "spearman")
    spearman_v2 = format_metric_with_ci(v2_summary.get("spearman", {}), "spearman")
    lines.append(f"| **Spearman ρ** | {spearman_v1} | {spearman_v2} |")

    # MAE
    mae_v1 = format_metric_with_ci(v1_summary.get("mae", {}), "mae")
    mae_v2 = format_metric_with_ci(v2_summary.get("mae", {}), "mae")
    lines.append(f"| **MAE** | {mae_v1} | {mae_v2} |")

    # RMSE
    rmse_v1 = format_metric_with_ci(v2_summary.get("rmse", {}), "rmse")
    rmse_v2 = format_metric_with_ci(v2_summary.get("rmse", {}), "rmse")
    lines.append(f"| **RMSE** | {rmse_v1} | {rmse_v2} |")

    # R²
    r2_v1 = v1_summary.get("r_squared", 0.0)
    r2_v2 = v2_summary.get("r_squared", 0.0)
    lines.append(f"| **R²** | {r2_v1:.4f} | {r2_v2:.4f} |")

    # n
    n_v1 = v1_summary.get("n_used", 0)
    n_v2 = v2_summary.get("n_used", 0)
    lines.append(f"| **n** | {n_v1} | {n_v2} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Änderungen",
            "",
        ]
    )

    # Änderungen aus Metadaten ableiten
    v1_config = v1_metadata.get("config", {}) if v1_metadata else {}
    v2_config = v2_metadata.get("config", {}) if v2_metadata else {}

    changes = []

    # Prompt Version
    v1_prompt = v1_config.get("prompt_version", "v1")
    v2_prompt = v2_config.get("prompt_version", "v1")
    if v1_prompt != v2_prompt:
        changes.append(
            f"- **Prompt:** {v1_prompt} → {v2_prompt} (Rubrik, 1-5 integer score, volle Skala)"
        )

    # Calibration
    v1_cal = v1_config.get("calibration_path")
    v2_cal = v2_config.get("calibration_path")
    if not v1_cal and v2_cal:
        changes.append(
            "- **Kalibrierung:** Hinzugefügt (lineare Regression auf Calibration-Split, n=500, seed=123)"
        )
    elif v1_cal and v2_cal:
        changes.append("- **Kalibrierung:** Beide Runs verwenden Kalibrierung")
    elif v1_cal and not v2_cal:
        changes.append("- **Kalibrierung:** Entfernt (v2 ohne Kalibrierung)")

    # Threshold (falls geändert)
    # TODO: Falls ISSUE_FALLBACK_THRESHOLD geändert wurde, hier dokumentieren

    if not changes:
        changes.append("- Keine expliziten Änderungen dokumentiert")

    lines.extend(changes)

    lines.extend(
        [
            "",
            "---",
            "",
            "## Ergebnis-Interpretation",
            "",
        ]
    )

    # Interpretation basierend auf Metriken
    spearman_v1_val = (
        v1_summary.get("spearman", {}).get("value", 0.0)
        if isinstance(v1_summary.get("spearman"), dict)
        else v1_summary.get("spearman", 0.0)
    )
    spearman_v2_val = (
        v2_summary.get("spearman", {}).get("value", 0.0)
        if isinstance(v2_summary.get("spearman"), dict)
        else v2_summary.get("spearman", 0.0)
    )

    mae_v1_val = (
        v1_summary.get("mae", {}).get("value", 0.0)
        if isinstance(v1_summary.get("mae"), dict)
        else v1_summary.get("mae", 0.0)
    )
    mae_v2_val = (
        v2_summary.get("mae", {}).get("value", 0.0)
        if isinstance(v2_summary.get("mae"), dict)
        else v2_summary.get("mae", 0.0)
    )

    r2_v1_val = v1_summary.get("r_squared", 0.0)
    r2_v2_val = v2_summary.get("r_squared", 0.0)

    interpretations = []

    if spearman_v2_val > spearman_v1_val:
        interpretations.append(
            f"- **Spearman ρ:** Verbesserung von {spearman_v1_val:.4f} auf {spearman_v2_val:.4f} (höher = bessere Rangfolge-Übereinstimmung)"
        )
    elif spearman_v2_val < spearman_v1_val:
        interpretations.append(
            f"- **Spearman ρ:** Verschlechterung von {spearman_v1_val:.4f} auf {spearman_v2_val:.4f}"
        )
    else:
        interpretations.append(f"- **Spearman ρ:** Unverändert bei {spearman_v1_val:.4f}")

    if mae_v2_val < mae_v1_val:
        interpretations.append(
            f"- **MAE:** Verbesserung von {mae_v1_val:.4f} auf {mae_v2_val:.4f} (niedriger = kleinerer durchschnittlicher Fehler)"
        )
    elif mae_v2_val > mae_v1_val:
        interpretations.append(
            f"- **MAE:** Verschlechterung von {mae_v1_val:.4f} auf {mae_v2_val:.4f}"
        )
    else:
        interpretations.append(f"- **MAE:** Unverändert bei {mae_v1_val:.4f}")

    if r2_v2_val > r2_v1_val:
        interpretations.append(
            f"- **R²:** Verbesserung von {r2_v1_val:.4f} auf {r2_v2_val:.4f} (höher = bessere Kalibrierung, negativ = schlechter als Mittelwert-Baseline)"
        )
    elif r2_v2_val < r2_v1_val:
        interpretations.append(
            f"- **R²:** Verschlechterung von {r2_v1_val:.4f} auf {r2_v2_val:.4f}"
        )
    else:
        interpretations.append(f"- **R²:** Unverändert bei {r2_v1_val:.4f}")

    lines.extend(interpretations)

    lines.extend(
        [
            "",
            "---",
            "",
            "## Run-Ordner",
            "",
            f"- **v1:** `{v1_run_id}`",
            f"- **v2:** `{v2_run_id}`",
        ]
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Erstellt Vergleichstabelle für Readability v1 vs v2")
    ap.add_argument(
        "--v1_run_dir",
        type=str,
        required=True,
        help="Run-Verzeichnis für v1 (z.B. results/evaluation/readability/readability_20260114_184754_gpt-4o-mini_v1_seed42)",
    )
    ap.add_argument("--v2_run_dir", type=str, required=True, help="Run-Verzeichnis für v2")
    ap.add_argument(
        "--out",
        type=str,
        default="docs/status_pack/2026-01-08/readability_improvement_comparison.md",
        help="Output-Pfad",
    )

    args = ap.parse_args()

    v1_run_dir = Path(args.v1_run_dir)
    v2_run_dir = Path(args.v2_run_dir)

    if not v1_run_dir.exists():
        ap.error(f"v1 Run-Verzeichnis nicht gefunden: {v1_run_dir}")
    if not v2_run_dir.exists():
        ap.error(f"v2 Run-Verzeichnis nicht gefunden: {v2_run_dir}")

    print(f"Lade v1 Summary aus: {v1_run_dir}")
    v1_summary = load_summary_json(v1_run_dir)
    if not v1_summary:
        ap.error(f"summary.json nicht gefunden in: {v1_run_dir}")

    print(f"Lade v2 Summary aus: {v2_run_dir}")
    v2_summary = load_summary_json(v2_run_dir)
    if not v2_summary:
        ap.error(f"summary.json nicht gefunden in: {v2_run_dir}")

    v1_metadata = load_run_metadata(v1_run_dir)
    v2_metadata = load_run_metadata(v2_run_dir)

    v1_run_id = v1_run_dir.name
    v2_run_id = v2_run_dir.name

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Erstelle Vergleichstabelle nach: {out_path}")
    create_comparison_md(
        v1_summary=v1_summary,
        v2_summary=v2_summary,
        v1_metadata=v1_metadata,
        v2_metadata=v2_metadata,
        v1_run_id=v1_run_id,
        v2_run_id=v2_run_id,
        out_path=out_path,
    )

    print("\nVergleichstabelle erstellt!")


if __name__ == "__main__":
    main()
