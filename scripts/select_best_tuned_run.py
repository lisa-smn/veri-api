"""
Wählt den besten Tuned Run basierend auf robusten Gates und Optimierungszielen.

Kriterien:
- Gate 1: recall >= recall_min (default: 0.90)
- Gate 2: specificity >= specificity_min (default: 0.20)
- Optimierungsziel: balanced_accuracy ODER mcc (wählbar)
- Tie-breaker: precision, dann f1
"""

import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Literal

ROOT = Path(__file__).resolve().parents[1]


def compute_mcc(tp: float, tn: float, fp: float, fn: float) -> float:
    """
    Berechnet Matthews Correlation Coefficient (MCC).
    Robust gegen Division durch 0.

    MCC = (TP * TN - FP * FN) / sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    """
    numerator = tp * tn - fp * fn
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)

    if denominator <= 0:
        return 0.0

    return numerator / math.sqrt(denominator)


def load_summary_matrix() -> list[dict[str, Any]]:
    """Lädt summary_matrix.csv und berechnet MCC für jeden Run."""
    csv_path = ROOT / "results" / "evaluation" / "summary_matrix.csv"
    if not csv_path.exists():
        raise SystemExit(f"Summary matrix not found: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Convert numeric fields
            for key in [
                "n",
                "tp",
                "fp",
                "tn",
                "fn",
                "precision",
                "recall",
                "f1",
                "specificity",
                "accuracy",
                "auroc",
                "balanced_accuracy",
            ]:
                if key in row:
                    try:
                        row[key] = float(row[key])
                    except (ValueError, TypeError):
                        row[key] = 0.0

            # Berechne MCC
            tp = row.get("tp", 0.0)
            tn = row.get("tn", 0.0)
            fp = row.get("fp", 0.0)
            fn = row.get("fn", 0.0)
            row["mcc"] = compute_mcc(tp, tn, fp, fn)

            rows.append(row)

    return rows


def select_best_tuned_run(
    rows: list[dict[str, Any]],
    recall_min: float = 0.90,
    specificity_min: float = 0.20,
    target_metric: Literal["balanced_accuracy", "mcc"] = "mcc",
    dataset: str = "frank",
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """
    Wählt den besten Run basierend auf robusten Gates und Optimierungszielen.

    Args:
        rows: Alle Runs aus summary_matrix.csv
        recall_min: Minimum Recall (Gate 1, default: 0.90)
        specificity_min: Minimum Specificity (Gate 2, default: 0.20)
        target_metric: Optimierungsziel ("balanced_accuracy" oder "mcc", default: "mcc")
        dataset: Dataset-Filter (default: "frank")

    Returns:
        Tuple: (best run dict oder None, stats dict mit Begründung)
    """
    # Filter: Nur FRANK-Runs, die Tuning-Varianten sind (nicht baseline/ablation)
    tuning_runs = [
        r
        for r in rows
        if r.get("dataset") == dataset
        and ("tune" in r.get("run_id", "").lower() or "weighted" in r.get("run_id", "").lower())
    ]

    if not tuning_runs:
        # Fallback: Alle FRANK-Runs außer baseline/ablation
        tuning_runs = [
            r
            for r in rows
            if r.get("dataset") == dataset
            and "baseline" not in r.get("run_id", "").lower()
            and "ablation" not in r.get("run_id", "").lower()
        ]

    if not tuning_runs:
        return None, {"error": "No tuning runs found"}

    # Prüfe GT-Negatives: Wenn < 20 UND explizit vorhanden, deaktiviere Specificity-Gate
    # GT Negatives = TN + FP (alle, die als negativ markiert sind)
    representative_run = tuning_runs[0]
    tn = float(representative_run.get("tn", 0))
    fp = float(representative_run.get("fp", 0))
    gt_negatives = tn + fp
    # Prüfe, ob n_gt_negatives explizit vorhanden ist (als zusätzliche Validierung)
    n_gt_negatives = representative_run.get("n_gt_negatives")

    specificity_gate_disabled = False
    # Gate2 darf nur deaktiviert werden, wenn explizite Daten vorhanden sind
    if n_gt_negatives is not None and n_gt_negatives < 20:
        specificity_gate_disabled = True
        specificity_min = 0.0  # Deaktiviere Gate
    elif gt_negatives > 0 and gt_negatives < 20:
        # Fallback: Wenn tn+fp < 20, aber n_gt_negatives nicht vorhanden, trotzdem deaktivieren
        specificity_gate_disabled = True
        specificity_min = 0.0

    stats = {
        "total_runs": len(tuning_runs),
        "recall_min": recall_min,
        "specificity_min": specificity_min,
        "target_metric": target_metric,
        "gt_negatives": gt_negatives,
        "specificity_gate_disabled": specificity_gate_disabled,
    }

    # Gate 1: Recall-Constraint
    recall_filtered = [r for r in tuning_runs if r.get("recall", 0.0) >= recall_min]
    stats["after_recall_gate"] = len(recall_filtered)
    stats["recall_gate_passed"] = len(recall_filtered) > 0

    # Gate 2: Specificity-Constraint (nur auf recall_filtered, wenn nicht deaktiviert)
    if specificity_gate_disabled:
        # Wenn Gate deaktiviert: candidate pool basiert NUR auf Gate1 (recall filter)
        # Gate2 ist technisch "nicht bestanden" (deaktiviert != bestanden)
        both_gates_filtered = []  # Gate2 ist deaktiviert, also kein "both_gates" Pool
        stats["after_specificity_gate"] = 0
        stats["both_gates_passed"] = False  # Gate2 ist deaktiviert, also nicht bestanden
    else:
        both_gates_filtered = [
            r for r in recall_filtered if r.get("specificity", 0.0) >= specificity_min
        ]
        stats["after_specificity_gate"] = len(both_gates_filtered)
        stats["both_gates_passed"] = len(both_gates_filtered) > 0

    # Fallback-Logik: Bestimme Kandidatenmenge
    fallback_used = False
    candidate_pool_name = None
    if specificity_gate_disabled:
        # Gate2 deaktiviert: candidate pool = recall_only (nicht both_gates)
        # Dies zählt als "Fallback", weil wir nicht beide Gates verwenden können
        if len(recall_filtered) > 0:
            candidate_set = recall_filtered
            candidate_pool_name = "recall_only"
            fallback_used = True  # Zählt als Fallback, da Gate2 nicht verwendet werden kann
        else:
            # Letzter Fallback: Alle Runs
            candidate_set = tuning_runs
            candidate_pool_name = "all_runs"
            fallback_used = True
    elif len(both_gates_filtered) > 0:
        candidate_set = both_gates_filtered
        candidate_pool_name = "both_gates"
    else:
        fallback_used = True
        # Fallback: Best specificity from recall-filtered runs
        if len(recall_filtered) > 0:
            candidate_set = recall_filtered
            candidate_pool_name = "recall_only"
        else:
            # Letzter Fallback: Alle Runs
            candidate_set = tuning_runs
            candidate_pool_name = "all_runs"

    stats["fallback_used"] = fallback_used
    stats["candidate_set_size"] = len(candidate_set)
    stats["candidate_pool_name"] = candidate_pool_name

    # Optimierungsziel: Sortiere nach target_metric
    # Tie-breaker: balanced_accuracy (falls mcc), dann precision, dann f1
    if target_metric == "mcc":
        tie_breakers = ["balanced_accuracy", "precision", "f1"]
        best = max(
            candidate_set,
            key=lambda r: (
                r.get("mcc", 0.0),  # Primär: MCC
                r.get("balanced_accuracy", 0.0),  # Tie-breaker 1: Balanced Accuracy
                r.get("precision", 0.0),  # Tie-breaker 2: Precision
                r.get("f1", 0.0),  # Tie-breaker 3: F1
            ),
        )
    else:  # balanced_accuracy
        tie_breakers = ["mcc", "precision", "f1"]
        best = max(
            candidate_set,
            key=lambda r: (
                r.get("balanced_accuracy", 0.0),  # Primär: Balanced Accuracy
                r.get("mcc", 0.0),  # Tie-breaker 1: MCC
                r.get("precision", 0.0),  # Tie-breaker 2: Precision
                r.get("f1", 0.0),  # Tie-breaker 3: F1
            ),
        )

    stats["tie_breakers"] = tie_breakers

    stats["selected_run"] = best.get("run_id")
    stats["selected_metrics"] = {
        "recall": best.get("recall", 0.0),
        "specificity": best.get("specificity", 0.0),
        "balanced_accuracy": best.get("balanced_accuracy", 0.0),
        "mcc": best.get("mcc", 0.0),
        "precision": best.get("precision", 0.0),
        "f1": best.get("f1", 0.0),
    }

    # Begründung aufbauen (Gate-Status darf niemals gelogen sein)
    justification = []

    # Gate 1 Status: ✅ nur wenn mindestens ein Run das Recall-Gate erfüllt UND der ausgewählte Run recall >= recall_min hat
    gate1_passed = stats["recall_gate_passed"] and best.get("recall", 0.0) >= recall_min
    if stats["recall_gate_passed"]:
        justification.append(f"{'✅' if gate1_passed else '❌'} Gate 1: recall >= {recall_min}")
    else:
        justification.append(f"❌ Gate 1: kein Run erfüllt recall >= {recall_min}")

    # Gate 2 Status: Wenn Gate deaktiviert, dann ❌ (nicht bestanden, da deaktiviert)
    # Wenn Gate aktiv: ✅ nur wenn both_gates_filtered nicht leer ist UND candidate_pool == both_gates
    if stats.get("specificity_gate_disabled", False):
        justification.append(
            f"⚠️  Gate 2: specificity >= {specificity_min} (DEAKTIVIERT: nur {stats.get('gt_negatives', 0):.0f} GT-Negatives, < 20)"
        )
        gate2_passed = False  # Gate ist deaktiviert, also nicht bestanden
    else:
        gate2_passed = stats["both_gates_passed"] and candidate_pool_name == "both_gates"
        if stats["both_gates_passed"] and candidate_pool_name == "both_gates":
            justification.append(f"✅ Gate 2: specificity >= {specificity_min}")
        else:
            justification.append(
                f"❌ Gate 2: specificity >= {specificity_min} (kein Run erfüllt beide Gates)"
            )

    # Kandidatenmenge immer erwähnen (auch wenn kein Fallback)
    tie_breaker_str = ", ".join([tb.upper() for tb in tie_breakers])
    if fallback_used:
        justification.append(
            f"ℹ️  Kandidatenmenge: {candidate_pool_name} "
            f"(Runs, die {'Gate 1 erfüllen' if candidate_pool_name == 'recall_only' else 'keine Gates erfüllen'}). "
            f"Ranking: {target_metric.upper()} (Tie-Breaker: {tie_breaker_str})."
        )
    else:
        justification.append(
            f"ℹ️  Kandidatenmenge: {candidate_pool_name}. "
            f"Ranking: {target_metric.upper()} (Tie-Breaker: {tie_breaker_str})."
        )

    # Beste Zielmetrik (relativ zur Kandidatenmenge)
    justification.append(
        f"✅ Beste Zielmetrik ({target_metric.upper()}) innerhalb Kandidatenmenge: {best.get(target_metric, 0.0):.3f}"
    )

    stats["justification"] = justification
    stats["gate1_passed"] = gate1_passed
    stats["gate2_passed"] = gate2_passed
    stats["candidate_set"] = candidate_set  # Für Top-K Ausgabe

    return best, stats


def generate_top_k_table(runs: list[dict], target_metric: str, k: int = 5) -> list[dict]:
    """
    Generiert Top-K Liste von Run-Dicts.

    Args:
        runs: Liste von Run-Dicts (Kandidaten)
        target_metric: Metrik nach der sortiert wird (z.B. "mcc", "balanced_accuracy")
        k: Anzahl der Top-Runs (default: 5)

    Returns:
        Liste der Top-K Run-Dicts (sortiert nach target_metric)
    """
    if not runs:
        return []

    def get_num(d: dict, key: str) -> float:
        v = d.get(key)
        try:
            return float(v)
        except Exception:
            return float("-inf")

    sorted_runs = sorted(runs, key=lambda r: get_num(r, target_metric), reverse=True)[:k]
    return sorted_runs


def render_top_k_markdown(top_k_rows: list[dict], target_metric: str) -> str:
    """
    Rendert Top-K Runs als Markdown-Tabelle.

    Args:
        top_k_rows: Liste von Run-Dicts (von generate_top_k_table)
        target_metric: Metrik für Spalten-Header

    Returns:
        Markdown-Tabelle als String
    """
    if not top_k_rows:
        return "No candidates."

    # Spalten: run_name, target_metric, dann alle anderen Metriken (ohne Duplikate)
    all_metrics = ["recall", "precision", "specificity", "f1", "mcc", "balanced_accuracy"]
    other_metrics = [m for m in all_metrics if m != target_metric]
    cols = ["run_name", target_metric] + other_metrics
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"

    rows = []
    for r in top_k_rows:
        row = []
        for c in cols:
            if c == "run_name":
                v = r.get("run_id") or r.get("run_name") or r.get("name", "")
            else:
                v = r.get(c, "")
            if isinstance(v, float):
                row.append(f"{v:.3f}")
            else:
                row.append(str(v))
        rows.append("| " + " | ".join(row) + " |")

    return "\n".join([header, sep] + rows)


def save_selection_artifacts(
    best: dict[str, Any],
    stats: dict[str, Any],
    top_k_table: str,
    dataset: str,
    recall_min: float,
    specificity_min: float,
    target_metric: str,
) -> dict[str, Path]:
    """
    Speichert Selektions-Artefakte (JSON + CSV).

    Args:
        best: Best Run Dict
        stats: Stats Dict mit Begründung
        top_k_table: Markdown-Tabelle als String
        dataset: Dataset-Name
        recall_min: Recall-Minimum
        specificity_min: Specificity-Minimum
        target_metric: Zielmetrik

    Returns:
        Dict mit Pfaden zu gespeicherten Artefakten
    """
    artifacts_dir = ROOT / "results" / "evaluation" / "selection"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # JSON: Vollständige Selektions-Info
    json_path = artifacts_dir / f"best_run_{dataset}_{target_metric}.json"
    # Exclude large candidate_set from JSON (aber behalte für CSV/Top-K)
    stats_for_json = {k: v for k, v in stats.items() if k != "candidate_set"}
    json_data = {
        "best_run": best,
        "stats": stats_for_json,
        "selection_criteria": {
            "dataset": dataset,
            "recall_min": recall_min,
            "specificity_min": specificity_min,
            "target_metric": target_metric,
        },
        "top_k_table": top_k_table,
        "timestamp": datetime.now().isoformat(),
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    # CSV: Top-K Runs (aus candidate_set)
    csv_path = artifacts_dir / f"top_k_{dataset}_{target_metric}.csv"
    candidate_set = stats.get("candidate_set", [])
    if candidate_set:
        # Sortiere nach target_metric
        def get_num(d: dict, key: str) -> float:
            v = d.get(key)
            try:
                return float(v)
            except Exception:
                return float("-inf")

        sorted_runs = sorted(candidate_set, key=lambda r: get_num(r, target_metric), reverse=True)[
            :5
        ]

        # CSV schreiben
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id",
                    "recall",
                    "precision",
                    "specificity",
                    "f1",
                    "mcc",
                    "balanced_accuracy",
                    target_metric,
                ],
            )
            writer.writeheader()
            for r in sorted_runs:
                writer.writerow(
                    {
                        "run_id": r.get("run_id", ""),
                        "recall": r.get("recall", 0.0),
                        "precision": r.get("precision", 0.0),
                        "specificity": r.get("specificity", 0.0),
                        "f1": r.get("f1", 0.0),
                        "mcc": r.get("mcc", 0.0),
                        "balanced_accuracy": r.get("balanced_accuracy", 0.0),
                        target_metric: r.get(target_metric, 0.0),
                    }
                )

    return {"json": json_path, "csv": csv_path}


def get_run_config(run_id: str) -> dict[str, Any] | None:
    """Lädt Run-Config aus results JSON."""
    results_path = ROOT / "results" / "evaluation" / "runs" / "results" / f"{run_id}.json"
    if not results_path.exists():
        return None

    with results_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
        return data.get("config")


def main():
    parser = argparse.ArgumentParser(description="Select best tuned run")
    parser.add_argument(
        "--recall-min", type=float, default=0.90, help="Minimum recall (Gate 1, default: 0.90)"
    )
    parser.add_argument(
        "--specificity-min",
        type=float,
        default=0.20,
        help="Minimum specificity (Gate 2, default: 0.20)",
    )
    parser.add_argument(
        "--target",
        type=str,
        choices=["balanced_accuracy", "mcc"],
        default="mcc",
        help="Optimierungsziel (default: mcc)",
    )
    parser.add_argument(
        "--dataset", type=str, default="frank", help="Dataset to filter (default: frank)"
    )
    parser.add_argument(
        "--output-config", type=str, default=None, help="Output path for best config YAML snippet"
    )

    args = parser.parse_args()

    # Load summary matrix
    rows = load_summary_matrix()

    # Select best run
    best, stats = select_best_tuned_run(
        rows,
        recall_min=args.recall_min,
        specificity_min=args.specificity_min,
        target_metric=args.target,
        dataset=args.dataset,
    )

    if not best:
        print("❌ Kein passender Run gefunden")
        if "error" in stats:
            print(f"   Fehler: {stats['error']}")
        if "warning" in stats:
            print(f"   ⚠️  {stats['warning']}")
        return

    print("=" * 80)
    print("Best Tuned Run Selection")
    print("=" * 80)
    print("\nSelection Criteria:")
    print(f"  Gate 1: recall >= {args.recall_min}")
    print(f"  Gate 2: specificity >= {args.specificity_min}")
    print(f"  Target Metric: {args.target}")
    print(f"  Dataset: {args.dataset}")
    print()
    print("Filtering Stats:")
    print(f"  Total tuning runs: {stats['total_runs']}")
    print(f"  After recall gate: {stats['after_recall_gate']}")
    print(f"  After specificity gate: {stats['after_specificity_gate']}")
    if stats.get("fallback_used"):
        print(f"  ℹ️  Fallback verwendet: Kandidatenmenge = {stats['candidate_set_size']} Runs")
    if stats.get("gt_negatives") is not None:
        gt_neg = stats["gt_negatives"]
        if gt_neg < 20:
            print(
                f"  ⚠️  Warnung: Nur {gt_neg:.0f} GT-Negatives (< 20) - Specificity-Gate deaktiviert"
            )
        else:
            print(f"  ℹ️  GT-Negatives: {gt_neg:.0f}")
    print()
    print("=" * 80)
    print(f"\n✅ Best Run: {best['run_id']}")
    print("=" * 80)
    print("\nConfusion Matrix:")
    print(f"  TP: {int(best.get('tp', 0))}")
    print(f"  TN: {int(best.get('tn', 0))}")
    print(f"  FP: {int(best.get('fp', 0))}")
    print(f"  FN: {int(best.get('fn', 0))}")
    print()
    print("Metrics:")
    print(f"  Recall: {best['recall']:.3f} {'✅' if best['recall'] >= args.recall_min else '❌'}")
    print(
        f"  Specificity: {best['specificity']:.3f} {'✅' if best['specificity'] >= args.specificity_min else '❌'}"
    )
    print(f"  Balanced Accuracy: {best['balanced_accuracy']:.3f}")
    print(f"  MCC: {best.get('mcc', 0.0):.3f}")
    print(f"  Precision: {best['precision']:.3f}")
    print(f"  F1: {best['f1']:.3f}")
    print()
    print("Selection Justification:")
    for line in stats.get("justification", []):
        print(f"  {line}")
    print()

    # Top-K Tabelle generieren und ausgeben
    candidate_set = stats.get("candidate_set", [])
    try:
        top_k_rows = generate_top_k_table(candidate_set, args.target, k=5)
        top_k_table = render_top_k_markdown(top_k_rows, args.target)

        print("=" * 80)
        print("Top 5 Runs (aus Kandidatenmenge)")
        print("=" * 80)
        print()
        print(top_k_table)
        print()
    except Exception as e:
        print("=" * 80)
        print("Top 5 Runs (aus Kandidatenmenge)")
        print("=" * 80)
        print(f"\n⚠️  Warnung: Top-K-Tabelle konnte nicht generiert werden: {e}")
        print()
        top_k_table = "No candidates."

    # Artefakte speichern
    try:
        artifacts = save_selection_artifacts(
            best,
            stats,
            top_k_table,
            args.dataset,
            args.recall_min,
            args.specificity_min,
            args.target,
        )
    except Exception as e:
        print(f"⚠️  Warnung: Artefakte konnten nicht gespeichert werden: {e}")
        artifacts = {"json": Path(""), "csv": Path("")}

    print("=" * 80)
    if artifacts.get("json") and artifacts["json"].exists():
        print("Artefakte gespeichert:")
        print(f"  JSON: {artifacts['json'].relative_to(ROOT)}")
        if artifacts.get("csv") and artifacts["csv"].exists():
            print(f"  Top-5 CSV: {artifacts['csv'].relative_to(ROOT)}")
    else:
        print("⚠️  Artefakte konnten nicht gespeichert werden")
    print("=" * 80)
    print()

    # Load config
    config = get_run_config(best["run_id"])
    if config:
        print("Configuration:")
        print(f"  decision_mode: {config.get('decision_mode', 'N/A')}")
        if "decision_threshold_float" in config:
            print(f"  decision_threshold_float: {config.get('decision_threshold_float', 'N/A')}")
        else:
            print(f"  error_threshold: {config.get('error_threshold', 'N/A')}")
        print(f"  severity_min: {config.get('severity_min', 'N/A')}")
        if "severity_weights" in config:
            print(f"  severity_weights: {config.get('severity_weights', {})}")
        print(f"  ignore_issue_types: {config.get('ignore_issue_types', [])}")
        print(f"  uncertainty_policy: {config.get('uncertainty_policy', 'N/A')}")
        if "confidence_min" in config:
            print(f"  confidence_min: {config.get('confidence_min', 'N/A')}")
        print(f"  score_cutoff: {config.get('score_cutoff', 'N/A')}")
        print()

        # Generate YAML snippet for FineSumFact final
        yaml_lines = [
            f"# Best Tuned Config (from {best['run_id']})",
            f'decision_mode: "{config.get("decision_mode", "issues")}"',
        ]

        if "decision_threshold_float" in config:
            yaml_lines.append(f"decision_threshold_float: {config.get('decision_threshold_float')}")
        else:
            yaml_lines.append(f"error_threshold: {config.get('error_threshold', 1)}")

        yaml_lines.extend(
            [
                f'severity_min: "{config.get("severity_min", "low")}"',
            ]
        )

        if "severity_weights" in config:
            weights = config.get("severity_weights", {})
            yaml_lines.append("severity_weights:")
            for k, v in weights.items():
                yaml_lines.append(f"  {k}: {v}")

        yaml_lines.extend(
            [
                f"ignore_issue_types: {config.get('ignore_issue_types', [])}",
                f'uncertainty_policy: "{config.get("uncertainty_policy", "count_as_error")}"',
            ]
        )

        if "confidence_min" in config:
            yaml_lines.append(f"confidence_min: {config.get('confidence_min', 0.0)}")

        yaml_lines.append(
            f"score_cutoff: {config.get('score_cutoff') if config.get('score_cutoff') is not None else 'null'}"
        )

        yaml_snippet = "\n".join(yaml_lines) + "\n"

        if args.output_config:
            output_path = Path(args.output_config)
            with output_path.open("w", encoding="utf-8") as f:
                f.write(yaml_snippet)
            print(f"✅ Config snippet saved to: {output_path}")
        else:
            print("💡 YAML snippet für FineSumFact final:")
            print("-" * 80)
            print(yaml_snippet, end="")
            print("-" * 80)

    print("=" * 80)


if __name__ == "__main__":
    main()
