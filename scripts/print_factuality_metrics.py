#!/usr/bin/env python3
"""Druckt Factuality-Metriken als formatierte Textbox für Screenshots"""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def print_metrics_box(run_json_path: Path, dataset_name: str):
    """Druckt Metriken als formatierte Box"""
    with run_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = data["metrics"]
    n = data["n"]
    run_id = data.get("run_id", "unknown")

    print("=" * 60)
    print(f"Factuality Evaluation: {dataset_name.upper()}")
    print("=" * 60)
    print(f"Dataset:     {dataset_name}")
    print(f"Run ID:      {run_id}")
    print(f"n:           {int(n)}")
    print()
    print("Metrics:")
    print(f"  Precision:          {metrics['precision']:.3f}")
    print(f"  Recall:             {metrics['recall']:.3f}")
    print(f"  F1:                 {metrics['f1']:.3f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.3f}")
    print()
    print("Confusion Matrix:")
    print(f"  TP: {int(metrics['tp'])}  FP: {int(metrics['fp'])}")
    print(f"  FN: {int(metrics['fn'])}  TN: {int(metrics['tn'])}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default: Zeige beide Datensätze
        print("\n" + "=" * 60)
        print("FRANK (Dev/Calibration)")
        print("=" * 60)
        frank_path = (
            ROOT / "results" / "evaluation" / "runs" / "results" / "factuality_frank_tuned_v1.json"
        )
        if frank_path.exists():
            print_metrics_box(frank_path, "FRANK")
        else:
            print(f"❌ Datei nicht gefunden: {frank_path}")

        print("\n")

        print("=" * 60)
        print("FineSumFact (Test)")
        print("=" * 60)
        finesumfact_path = (
            ROOT
            / "results"
            / "evaluation"
            / "runs"
            / "results"
            / "factuality_finesumfact_final_v1.json"
        )
        if finesumfact_path.exists():
            print_metrics_box(finesumfact_path, "FineSumFact")
        else:
            print(f"❌ Datei nicht gefunden: {finesumfact_path}")
    else:
        # Einzelner Datensatz
        dataset = sys.argv[1].lower()
        if dataset == "frank":
            path = (
                ROOT
                / "results"
                / "evaluation"
                / "runs"
                / "results"
                / "factuality_frank_tuned_v1.json"
            )
            print_metrics_box(path, "FRANK")
        elif dataset == "finesumfact":
            path = (
                ROOT
                / "results"
                / "evaluation"
                / "runs"
                / "results"
                / "factuality_finesumfact_final_v1.json"
            )
            print_metrics_box(path, "FineSumFact")
        else:
            print(f"❌ Unbekannter Datensatz: {dataset}")
            print("Verwendung: python3 scripts/print_factuality_metrics.py [frank|finesumfact]")
            sys.exit(1)
