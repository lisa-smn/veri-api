"""
Aggregiert mehrere Factuality-Evaluation-Runs zu einer Vergleichstabelle.

Liest mehrere summary.json (agent + baselines) und schreibt:
- summary_factuality_matrix.csv (tabellarischer Vergleich)
- summary_factuality_matrix.md (human-readable)

Input: Verzeichnisse mit summary.json (z.B. results/evaluation/factuality/*/summary.json)
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
    """
    Extrahiert Metriken aus summary.json.

    Unterstützt zwei Formate:
    1. Binäre Metriken (Agent): TP/FP/TN/FN, Precision, Recall, F1, etc.
    2. Kontinuierliche Metriken (Baselines): Pearson, Spearman, MAE, RMSE
    """
    metrics = {}

    # Dataset-Signature und Counts
    metrics["n_total"] = summary.get("n_total")
    metrics["dataset_signature"] = summary.get("dataset_signature")

    # Binäre Metriken (Agent)
    if "metrics" in summary and isinstance(summary["metrics"], dict):
        agent_metrics = summary["metrics"]

        # Extrahiere Werte aus Dictionary-Format (falls vorhanden)
        def extract_value(metric_dict_or_value):
            if isinstance(metric_dict_or_value, dict):
                return metric_dict_or_value.get("value")
            return metric_dict_or_value

        metrics["precision"] = extract_value(agent_metrics.get("precision"))
        metrics["recall"] = extract_value(agent_metrics.get("recall"))
        metrics["f1"] = extract_value(agent_metrics.get("f1"))
        metrics["balanced_accuracy"] = extract_value(agent_metrics.get("balanced_accuracy"))
        # auroc und specificity können direkt Float sein
        auroc_val = agent_metrics.get("auroc")
        metrics["auroc"] = extract_value(auroc_val) if isinstance(auroc_val, dict) else auroc_val
        metrics["accuracy"] = extract_value(agent_metrics.get("accuracy"))
        specificity_val = agent_metrics.get("specificity")
        metrics["specificity"] = (
            extract_value(specificity_val) if isinstance(specificity_val, dict) else specificity_val
        )

        # Counts
        if "counts" in summary:
            metrics["tp"] = summary["counts"].get("tp")
            metrics["fp"] = summary["counts"].get("fp")
            metrics["tn"] = summary["counts"].get("tn")
            metrics["fn"] = summary["counts"].get("fn")

    # Kontinuierliche Metriken (Baselines)
    if "pearson" in summary:
        if isinstance(summary["pearson"], dict):
            metrics["pearson"] = summary["pearson"].get("value")
            metrics["pearson_ci_lower"] = summary["pearson"].get("ci_lower")
            metrics["pearson_ci_upper"] = summary["pearson"].get("ci_upper")
        else:
            metrics["pearson"] = summary["pearson"]
            metrics["pearson_ci_lower"] = None
            metrics["pearson_ci_upper"] = None

    if "spearman" in summary:
        if isinstance(summary["spearman"], dict):
            metrics["spearman"] = summary["spearman"].get("value")
            metrics["spearman_ci_lower"] = summary["spearman"].get("ci_lower")
            metrics["spearman_ci_upper"] = summary["spearman"].get("ci_upper")
        else:
            metrics["spearman"] = summary["spearman"]
            metrics["spearman_ci_lower"] = None
            metrics["spearman_ci_upper"] = None

    if "mae" in summary:
        if isinstance(summary["mae"], dict):
            metrics["mae"] = summary["mae"].get("value")
            metrics["mae_ci_lower"] = summary["mae"].get("ci_lower")
            metrics["mae_ci_upper"] = summary["mae"].get("ci_upper")
        else:
            metrics["mae"] = summary["mae"]
            metrics["mae_ci_lower"] = None
            metrics["mae_ci_upper"] = None

    if "rmse" in summary:
        if isinstance(summary["rmse"], dict):
            metrics["rmse"] = summary["rmse"].get("value")
            metrics["rmse_ci_lower"] = summary["rmse"].get("ci_lower")
            metrics["rmse_ci_upper"] = summary["rmse"].get("ci_upper")
        else:
            metrics["rmse"] = summary["rmse"]
            metrics["rmse_ci_lower"] = None
            metrics["rmse_ci_upper"] = None

    metrics["r_squared"] = summary.get("r_squared")

    # Counts
    metrics["n_used"] = summary.get("n_used")
    metrics["n_failed"] = summary.get("n_failed")
    if "counts" in summary:
        metrics["n_used"] = (
            summary["counts"].get("tp", 0)
            + summary["counts"].get("fp", 0)
            + summary["counts"].get("tn", 0)
            + summary["counts"].get("fn", 0)
        )

    return metrics


def determine_method_type(summary: dict[str, Any], metadata: dict[str, Any] | None) -> str:
    """Bestimmt Method-Typ (agent, rouge_l, bertscore)."""
    # Aus metadata
    if metadata:
        baseline_type = metadata.get("baseline_type")
        if baseline_type:
            return baseline_type

    # Aus summary
    baseline_type = summary.get("baseline_type")
    if baseline_type:
        return baseline_type

    # Default: agent (hat binäre Metriken)
    if "metrics" in summary and isinstance(summary["metrics"], dict):
        return "agent"

    return "agent"


def extract_model_prompt(metadata: dict[str, Any] | None) -> tuple[str, str]:
    """Extrahiert Model und Prompt-Version aus metadata."""
    if not metadata:
        return "unknown", "unknown"
    config = metadata.get("config", {})
    model = config.get("llm_model") or metadata.get("llm_model") or "unknown"
    prompt = config.get("prompt_version") or "unknown"
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
        method_type = determine_method_type(summary, metadata)
        model, prompt = extract_model_prompt(metadata)

        seed = None
        if metadata:
            seed = metadata.get("seed")

        # Dependency-Status (für Invalid-Run-Erkennung)
        dependencies_ok = True
        allow_dummy = False
        missing_packages = []
        if metadata:
            dependencies_ok = metadata.get("dependencies_ok", True)
            missing_packages = metadata.get("missing_packages", [])
            # allow_dummy kann aus config abgeleitet werden (falls vorhanden)
            config = metadata.get("config", {})
            allow_dummy = config.get("allow_dummy_baseline", False)

        # Markiere als invalid wenn dependencies fehlen oder dummy aktiviert
        is_invalid = not dependencies_ok or allow_dummy

        row = {
            "run_id": run_dir.name,
            "method": method_type,
            "model": model,
            "prompt_version": prompt,
            "seed": seed,
            # Dataset-Signature
            "n_total": metrics.get("n_total"),
            "dataset_signature": metrics.get("dataset_signature"),
            # Dependency-Status
            "dependencies_ok": dependencies_ok,
            "allow_dummy": allow_dummy,
            "missing_packages": ", ".join(missing_packages) if missing_packages else "",
            "is_invalid": is_invalid,
            # Binäre Metriken (Agent)
            "precision": metrics.get("precision"),
            "recall": metrics.get("recall"),
            "f1": metrics.get("f1"),
            "balanced_accuracy": metrics.get("balanced_accuracy"),
            "auroc": metrics.get("auroc"),
            "tp": metrics.get("tp"),
            "fp": metrics.get("fp"),
            "tn": metrics.get("tn"),
            "fn": metrics.get("fn"),
            # Kontinuierliche Metriken (Baselines)
            "pearson": metrics.get("pearson"),
            "pearson_ci_lower": metrics.get("pearson_ci_lower"),
            "pearson_ci_upper": metrics.get("pearson_ci_upper"),
            "spearman": metrics.get("spearman"),
            "spearman_ci_lower": metrics.get("spearman_ci_lower"),
            "spearman_ci_upper": metrics.get("spearman_ci_upper"),
            "mae": metrics.get("mae"),
            "mae_ci_lower": metrics.get("mae_ci_lower"),
            "mae_ci_upper": metrics.get("mae_ci_upper"),
            "rmse": metrics.get("rmse"),
            "rmse_ci_lower": metrics.get("rmse_ci_lower"),
            "rmse_ci_upper": metrics.get("rmse_ci_upper"),
            "r_squared": metrics.get("r_squared"),
            "n": metrics.get("n_used"),
            "n_failed": metrics.get("n_failed"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df


def write_summary_md(df: pd.DataFrame, out_path: Path) -> None:
    """Schreibt human-readable Markdown-Tabelle."""
    lines = [
        "# Factuality Evaluation Comparison",
        "",
        "## Summary Matrix",
        "",
        "### Interpretation",
        "",
        "**Baselines (ROUGE-L, BERTScore) messen Ähnlichkeit zur Referenz, nicht Faktentreue.**",
        "Der Vergleich dient als Proxy-Baseline: Höhere Ähnlichkeit zur Referenz korreliert",
        "oft mit besserer Faktentreue, ist aber kein direkter Faktentreue-Maßstab.",
        "",
        "**Factuality-Agent** misst explizit Faktentreue durch Evidence-Gate und Claim-Verifikation.",
        "",
        "---",
        "",
    ]

    # Invalid-Run-Warnung
    invalid_runs = df[df.get("is_invalid", False) == True]
    if not invalid_runs.empty:
        lines.extend(
            [
                "## ⚠️ WARNUNG: Invalid/Dummy Runs gefunden",
                "",
                "**Die folgenden Runs sind ungültig und wurden aus der Quick Comparison ausgeschlossen:**",
                "",
            ]
        )
        for _, row in invalid_runs.iterrows():
            missing_str = row.get("missing_packages", "") or "unknown"
            allow_dummy_str = "Ja" if row.get("allow_dummy", False) else "Nein"
            lines.append(f"- **{row['run_id']}** ({row['method']}):")
            lines.append(f"  - Fehlende Packages: {missing_str}")
            lines.append(f"  - Allow Dummy: {allow_dummy_str}")
            lines.append("")
        lines.extend(
            [
                "**→ Diese Runs sind nicht für Thesis-Evaluation nutzbar!**",
                "",
                "---",
                "",
            ]
        )

    # Dataset-Konsistenz-Prüfung
    signatures = df["dataset_signature"].dropna().unique()
    n_used_values = df["n"].dropna().unique()

    if len(signatures) > 1:
        lines.append("### ⚠️ WARNUNG: Dataset-Konsistenz")
        lines.append("")
        lines.append("**Unterschiedliche Dataset-Signatures gefunden!**")
        lines.append("")
        for sig in signatures:
            runs_with_sig = df[df["dataset_signature"] == sig]["run_id"].tolist()
            lines.append(f"- Signature `{sig[:16]}...`: {len(runs_with_sig)} Runs")
        lines.append("")
        lines.append("**→ Agent und Baselines nutzen möglicherweise unterschiedliche Datensätze!**")
        lines.append("**→ Vergleich ist nicht fair!**")
        lines.append("")
        lines.append("---")
        lines.append("")

    if len(n_used_values) > 1:
        lines.append("### ⚠️ WARNUNG: Unterschiedliche n_used Werte")
        lines.append("")
        lines.append(f"**Gefundene n_used Werte:** {sorted(n_used_values)}")
        lines.append("")
        lines.append("**→ Agent und Baselines haben unterschiedlich viele Beispiele evaluiert!**")
        lines.append("")
        lines.append("---")
        lines.append("")

    # Kompakte Vergleichstabelle oben (nur vergleichbare Runs)
    lines.append("### Quick Comparison")
    lines.append("")

    # Filter: Nur valide Runs (dependencies_ok=True, allow_dummy=False)
    df_valid = df[df.get("is_invalid", False) == False]

    if df_valid.empty:
        lines.append("**⚠️ Keine validen Runs gefunden! Alle Runs sind invalid/dummy.**")
        lines.append("")
    else:
        # Filter: Nur Runs mit derselben dataset_signature (falls eindeutig)
        if len(signatures) == 1:
            df_comparable = df_valid[df_valid["dataset_signature"] == signatures[0]]
            lines.append(
                f"*Vergleichbare Runs (dataset_signature: {signatures[0][:16]}..., nur valide Runs)*"
            )
        else:
            df_comparable = df_valid
            lines.append(
                "*⚠️ ACHTUNG: Runs mit unterschiedlichen Dataset-Signatures - Vergleich möglicherweise nicht fair!*"
            )

        lines.append("")
        lines.append(
            "| Method | Pearson r | Spearman ρ | MAE | RMSE | F1 | Balanced Acc | N | Dataset Signature |"
        )
        lines.append(
            "|--------|-----------|------------|-----|------|----|--------------|----|-------------------|"
        )

        for _, row in df_comparable.iterrows():
            method = row["method"].upper().replace("_", "-")

        # Kontinuierliche Metriken
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

        # Binäre Metriken (können None sein, wenn nicht Agent-Run)
        f1_val = row["f1"]
        f1_str = f"{f1_val:.4f}" if pd.notna(f1_val) and isinstance(f1_val, (int, float)) else "N/A"
        balanced_acc_val = row["balanced_accuracy"]
        balanced_acc_str = (
            f"{balanced_acc_val:.4f}"
            if pd.notna(balanced_acc_val) and isinstance(balanced_acc_val, (int, float))
            else "N/A"
        )

        n_str = str(int(row["n"])) if pd.notna(row["n"]) else "N/A"
        sig_str = (
            str(row["dataset_signature"])[:16] + "..."
            if pd.notna(row["dataset_signature"])
            else "N/A"
        )

        lines.append(
            f"| {method} | {pearson_str} | {spearman_str} | {mae_str} | {rmse_str} | "
            f"{f1_str} | {balanced_acc_str} | {n_str} | {sig_str} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("")

    # Invalid Runs am Ende dokumentieren (falls vorhanden)
    if not invalid_runs.empty:
        lines.extend(
            [
                "## Invalid/Dummy Runs (Details)",
                "",
                "Diese Runs wurden aus der Quick Comparison ausgeschlossen:",
                "",
                "| Run ID | Method | Missing Packages | Allow Dummy | Status |",
                "|--------|--------|------------------|-------------|--------|",
            ]
        )
        for _, row in invalid_runs.iterrows():
            missing_str = row.get("missing_packages", "") or "N/A"
            allow_dummy_str = "Ja" if row.get("allow_dummy", False) else "Nein"
            status_str = "INVALID" if not row.get("dependencies_ok", True) else "DUMMY"
            lines.append(
                f"| {row['run_id']} | {row['method']} | {missing_str} | {allow_dummy_str} | {status_str} |"
            )
        lines.append("")
        lines.append("---")
        lines.append("")

    # Detaillierte Tabelle nach Method
    for method in df["method"].unique():
        lines.append(f"### {method.upper().replace('_', '-')}")
        lines.append("")

        # Wähle Sortierspalte basierend auf verfügbaren Metriken
        sort_col = "pearson" if pd.notna(df["pearson"]).any() else "f1"
        method_df = df[df["method"] == method].sort_values(
            sort_col, ascending=False, na_position="last"
        )

        # Header (abhängig von Method-Typ)
        if method == "agent":
            lines.append(
                "| Run ID | Model | Prompt | Seed | TP | FP | TN | FN | Precision | Recall | F1 | Balanced Acc | AUROC | N |"
            )
            lines.append(
                "|--------|-------|--------|------|----|----|----|----|-----------|--------|----|--------------|-------|----|"
            )
        else:
            lines.append(
                "| Run ID | Model | Prompt | Seed | Pearson r | Spearman ρ | MAE | RMSE | R² | N |"
            )
            lines.append(
                "|--------|-------|--------|------|----------|------------|-----|------|-----|----|"
            )

        # Rows
        for _, row in method_df.iterrows():
            seed_str = str(int(row["seed"])) if pd.notna(row["seed"]) else "N/A"
            n_str = str(int(row["n"])) if pd.notna(row["n"]) else "N/A"

            if method == "agent":
                tp_str = str(int(row["tp"])) if pd.notna(row["tp"]) else "N/A"
                fp_str = str(int(row["fp"])) if pd.notna(row["fp"]) else "N/A"
                tn_str = str(int(row["tn"])) if pd.notna(row["tn"]) else "N/A"
                fn_str = str(int(row["fn"])) if pd.notna(row["fn"]) else "N/A"
                # Sicherstellen, dass Werte numerisch sind
                precision_val = row["precision"]
                precision_str = (
                    f"{precision_val:.4f}"
                    if pd.notna(precision_val) and isinstance(precision_val, (int, float))
                    else "N/A"
                )
                recall_val = row["recall"]
                recall_str = (
                    f"{recall_val:.4f}"
                    if pd.notna(recall_val) and isinstance(recall_val, (int, float))
                    else "N/A"
                )
                f1_val = row["f1"]
                f1_str = (
                    f"{f1_val:.4f}"
                    if pd.notna(f1_val) and isinstance(f1_val, (int, float))
                    else "N/A"
                )
                balanced_acc_val = row["balanced_accuracy"]
                balanced_acc_str = (
                    f"{balanced_acc_val:.4f}"
                    if pd.notna(balanced_acc_val) and isinstance(balanced_acc_val, (int, float))
                    else "N/A"
                )
                auroc_val = row["auroc"]
                auroc_str = (
                    f"{auroc_val:.4f}"
                    if pd.notna(auroc_val) and isinstance(auroc_val, (int, float))
                    else "N/A"
                )

                lines.append(
                    f"| {row['run_id']} | {row['model']} | {row['prompt_version']} | {seed_str} | "
                    f"{tp_str} | {fp_str} | {tn_str} | {fn_str} | {precision_str} | {recall_str} | "
                    f"{f1_str} | {balanced_acc_str} | {auroc_str} | {n_str} |"
                )
            else:
                pearson_str = f"{row['pearson']:.4f}" if pd.notna(row["pearson"]) else "N/A"
                if pd.notna(row["pearson_ci_lower"]) and pd.notna(row["pearson_ci_upper"]):
                    pearson_str += (
                        f" [{row['pearson_ci_lower']:.4f}, {row['pearson_ci_upper']:.4f}]"
                    )

                spearman_str = f"{row['spearman']:.4f}" if pd.notna(row["spearman"]) else "N/A"
                if pd.notna(row["spearman_ci_lower"]) and pd.notna(row["spearman_ci_upper"]):
                    spearman_str += (
                        f" [{row['spearman_ci_lower']:.4f}, {row['spearman_ci_upper']:.4f}]"
                    )

                mae_str = f"{row['mae']:.4f}" if pd.notna(row["mae"]) else "N/A"
                if pd.notna(row["mae_ci_lower"]) and pd.notna(row["mae_ci_upper"]):
                    mae_str += f" [{row['mae_ci_lower']:.4f}, {row['mae_ci_upper']:.4f}]"

                rmse_str = f"{row['rmse']:.4f}" if pd.notna(row["rmse"]) else "N/A"
                if pd.notna(row["rmse_ci_lower"]) and pd.notna(row["rmse_ci_upper"]):
                    rmse_str += f" [{row['rmse_ci_lower']:.4f}, {row['rmse_ci_upper']:.4f}]"

                r2_str = f"{row['r_squared']:.4f}" if pd.notna(row["r_squared"]) else "N/A"

                lines.append(
                    f"| {row['run_id']} | {row['model']} | {row['prompt_version']} | {seed_str} | "
                    f"{pearson_str} | {spearman_str} | {mae_str} | {rmse_str} | {r2_str} | {n_str} |"
                )

        lines.append("")

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def check_dataset_consistency(df) -> None:
    """Prüft, ob alle Runs dasselbe Dataset verwenden (via dataset_signature) und ob Runs invalid sind."""
    signatures = df["dataset_signature"].dropna().unique()
    n_used_values = df["n"].dropna().unique()
    invalid_runs = df[df.get("is_invalid", False) == True]

    warnings = []

    # Invalid-Run-Warnung
    if not invalid_runs.empty:
        warnings.append(f"⚠️  WARNUNG: {len(invalid_runs)} Invalid/Dummy Runs gefunden!")
        for _, row in invalid_runs.iterrows():
            missing_str = row.get("missing_packages", "") or "unknown"
            warnings.append(
                f"   - {row['run_id']} ({row['method']}): Missing: {missing_str}, Allow Dummy: {row.get('allow_dummy', False)}"
            )
        warnings.append("   → Diese Runs sind nicht für Thesis-Evaluation nutzbar!")
        warnings.append("   → Sie werden aus der Quick Comparison ausgeschlossen.")

    if len(signatures) > 1:
        warnings.append(
            f"⚠️  WARNUNG: Unterschiedliche Dataset-Signatures gefunden: {list(signatures)}"
        )
        warnings.append(
            "   → Agent und Baselines nutzen möglicherweise unterschiedliche Datensätze!"
        )

    if len(n_used_values) > 1:
        warnings.append(f"⚠️  WARNUNG: Unterschiedliche n_used Werte: {list(n_used_values)}")
        warnings.append("   → Agent und Baselines haben unterschiedlich viele Beispiele evaluiert!")

    if warnings:
        print("\n" + "=" * 60)
        print("DATASET-KONSISTENZ-PRÜFUNG")
        print("=" * 60)
        for w in warnings:
            print(w)
        print("=" * 60 + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregiert mehrere Factuality-Evaluation-Runs")
    ap.add_argument(
        "run_dirs",
        nargs="+",
        type=str,
        help="Verzeichnisse mit summary.json (z.B. results/evaluation/factuality/*, results/evaluation/factuality_baselines/*)",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="results/evaluation/summary_factuality_matrix",
        help="Output-Pfad (ohne Extension, default: results/evaluation/summary_factuality_matrix)",
    )
    ap.add_argument(
        "--skip-consistency-check",
        action="store_true",
        help="Überspringe Dataset-Konsistenz-Prüfung",
    )

    args = ap.parse_args()

    # Expandiere Wildcards (falls Shell das nicht gemacht hat)
    expanded_dirs = []
    for d in args.run_dirs:
        path = Path(d)
        if "*" in str(path) or "?" in str(path):
            # Wildcard-Expansion
            parent = path.parent
            pattern = path.name
            if parent.exists():
                expanded_dirs.extend(
                    [parent / p for p in parent.glob(pattern) if (parent / p).is_dir()]
                )
            else:
                print(f"Warnung: Verzeichnis für Wildcard nicht gefunden: {parent}")
        else:
            expanded_dirs.append(path)

    run_dirs = []
    # Prüfe, ob alle Verzeichnisse existieren und summary.json haben
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

    # Dataset-Konsistenz-Prüfung
    if not args.skip_consistency_check:
        check_dataset_consistency(df)

    # Output
    out_path = Path(args.out)
    csv_path = out_path.with_suffix(".csv")
    md_path = out_path.with_suffix(".md")

    df.to_csv(csv_path, index=False)
    write_summary_md(df, md_path)

    # Invalid-Run-Statistik
    invalid_count = len(df[df.get("is_invalid", False) == True])
    valid_count = len(df) - invalid_count

    print("\nAggregation abgeschlossen!")
    print(f"  CSV: {csv_path}")
    print(f"  MD:  {md_path}")
    print(f"\n{len(df)} Runs aggregiert ({valid_count} valide, {invalid_count} invalid/dummy):")
    print(
        df[
            ["run_id", "method", "pearson", "spearman", "f1", "balanced_accuracy", "is_invalid"]
        ].to_string()
    )


if __name__ == "__main__":
    main()
