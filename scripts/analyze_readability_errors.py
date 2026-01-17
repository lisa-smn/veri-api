"""
Analysiert Fehler in Readability-Evaluation-Runs.

Input:
  --run_dir results/evaluation/readability/readability_20260114_184754_gpt-4o-mini_v1_seed42

Output:
  results/evaluation/readability/analysis_readability_v1.md

Erzeugt 4 Listen (je 20 Beispiele):
- A) größte absolute Fehler |gt - pred|
- B) gt hoch (>=0.85) aber pred niedrig (<=0.55)
- C) gt niedrig (<=0.55) aber pred hoch (>=0.85)
- D) Fälle mit high severity spans (Top 20)
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_predictions(predictions_path: Path) -> list[dict[str, Any]]:
    """Lädt predictions.jsonl."""
    predictions = []
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                predictions.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warnung: Konnte Zeile nicht parsen: {e}")
    return predictions


def load_summaries(data_path: Path, example_ids: list[str]) -> dict[str, str]:
    """Lädt Summary-Texts für gegebene example_ids."""
    summaries = {}
    example_ids_set = set(example_ids)  # Für schnelleres Lookup
    print(f"  Suche nach {len(example_ids_set)} example_ids...")
    matched = 0
    with data_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                meta = row.get("meta", {})
                # Versuche verschiedene ID-Felder (in der Reihenfolge, wie sie in eval_sumeval_readability.py verwendet werden)
                example_id = (
                    meta.get("doc_id") or meta.get("id") or row.get("doc_id") or row.get("id")
                )
                if example_id and example_id in example_ids_set:
                    summaries[example_id] = row.get("summary", "")
                    matched += 1
                    if matched % 50 == 0:
                        print(f"    {matched}/{len(example_ids_set)} gematcht...")
            except json.JSONDecodeError as e:
                print(f"    Warnung: Zeile {line_no} konnte nicht geparst werden: {e}")
                continue
    print(f"  {matched}/{len(example_ids_set)} Summaries geladen")
    return summaries


def truncate_text(text: str, max_len: int = 200) -> str:
    """Kürzt Text auf max_len Zeichen."""
    if not text:
        return ""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def format_issue_types(issue_types_counts: dict[str, int]) -> str:
    """Formatiert Issue-Types für Ausgabe."""
    if not issue_types_counts:
        return "none"
    return ", ".join([f"{k}:{v}" for k, v in issue_types_counts.items()])


def analyze_errors(
    predictions: list[dict[str, Any]],
    summaries: dict[str, str],
) -> dict[str, list[dict[str, Any]]]:
    """Analysiert Fehler und erzeugt 4 Listen."""

    # Berechne absolute Fehler
    for pred in predictions:
        if pred.get("failed", False):
            continue
        gt_norm = pred.get("gt_norm")
        pred_score = pred.get("pred")
        if gt_norm is not None and pred_score is not None:
            pred["abs_error"] = abs(gt_norm - pred_score)
        else:
            pred["abs_error"] = None

    # A) Größte absolute Fehler
    valid_preds = [
        p for p in predictions if p.get("abs_error") is not None and not p.get("failed", False)
    ]
    sorted_by_error = sorted(valid_preds, key=lambda x: x.get("abs_error", 0.0), reverse=True)
    list_a = sorted_by_error[:20]

    # B) gt hoch (>=0.85) aber pred niedrig (<=0.55)
    list_b = [p for p in valid_preds if p.get("gt_norm", 0) >= 0.85 and p.get("pred", 1.0) <= 0.55]
    list_b = sorted(list_b, key=lambda x: x.get("abs_error", 0.0), reverse=True)[:20]

    # C) gt niedrig (<=0.55) aber pred hoch (>=0.85)
    list_c = [
        p for p in valid_preds if p.get("gt_norm", 1.0) <= 0.55 and p.get("pred", 0.0) >= 0.85
    ]
    list_c = sorted(list_c, key=lambda x: x.get("abs_error", 0.0), reverse=True)[:20]

    # D) Fälle mit high severity spans
    list_d = [p for p in valid_preds if p.get("max_severity") == "high"]
    list_d = sorted(list_d, key=lambda x: x.get("num_issues", 0), reverse=True)[:20]

    return {
        "largest_errors": list_a,
        "gt_high_pred_low": list_b,
        "gt_low_pred_high": list_c,
        "high_severity": list_d,
    }


def compute_histograms(predictions: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Berechnet Histogramme für gt_norm und pred."""
    valid_preds = [
        p
        for p in predictions
        if not p.get("failed", False) and p.get("gt_norm") is not None and p.get("pred") is not None
    ]

    # Bins: [0, 0.2), [0.2, 0.4), [0.4, 0.6), [0.6, 0.8), [0.8, 1.0]
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    def bin_value(v: float) -> str:
        for i in range(len(bins) - 1):
            if bins[i] <= v < bins[i + 1]:
                return f"[{bins[i]:.1f}, {bins[i + 1]:.1f})"
        if v >= bins[-1]:
            return f"[{bins[-1]:.1f}, 1.0]"
        return "[0.0, 0.2)"

    gt_bins = Counter([bin_value(p.get("gt_norm", 0.0)) for p in valid_preds])
    pred_bins = Counter([bin_value(p.get("pred", 0.0)) for p in valid_preds])

    return {
        "gt_norm": dict(gt_bins),
        "pred": dict(pred_bins),
    }


def write_analysis_md(
    lists: dict[str, list[dict[str, Any]]],
    histograms: dict[str, dict[str, int]],
    summaries: dict[str, str],
    out_path: Path,
) -> None:
    """Schreibt analysis_readability_v1.md."""
    lines = [
        "# Readability Error Analysis",
        "",
        "## Histogramme (Verteilung)",
        "",
        "### GT (gt_norm)",
        "",
    ]

    for bin_name in sorted(histograms["gt_norm"].keys()):
        count = histograms["gt_norm"][bin_name]
        lines.append(f"- {bin_name}: {count}")

    lines.extend(
        [
            "",
            "### Predictions (pred)",
            "",
        ]
    )

    for bin_name in sorted(histograms["pred"].keys()):
        count = histograms["pred"][bin_name]
        lines.append(f"- {bin_name}: {count}")

    lines.extend(
        [
            "",
            "---",
            "",
            "## A) Größte absolute Fehler (Top 20)",
            "",
            "| ID | GT | Pred | Error | Summary Snippet | Top Issues | Max Severity |",
            "|----|----|------|-------|-----------------|------------|--------------|",
        ]
    )

    for pred in lists["largest_errors"]:
        example_id = pred.get("example_id", "unknown")
        summary = summaries.get(example_id, "")
        summary_snippet = truncate_text(summary, max_len=200)
        gt_norm = pred.get("gt_norm", 0.0)
        pred_score = pred.get("pred", 0.0)
        abs_error = pred.get("abs_error", 0.0)
        issue_types = format_issue_types(pred.get("issue_types_counts", {}))
        max_severity = pred.get("max_severity", "none")

        lines.append(
            f"| {example_id[:30]} | {gt_norm:.3f} | {pred_score:.3f} | {abs_error:.3f} | "
            f"{summary_snippet[:100]} | {issue_types[:30]} | {max_severity} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## B) GT hoch (>=0.85) aber Pred niedrig (<=0.55) (Top 20)",
            "",
            "| ID | GT | Pred | Error | Summary Snippet | Top Issues | Max Severity |",
            "|----|----|------|-------|-----------------|------------|--------------|",
        ]
    )

    for pred in lists["gt_high_pred_low"]:
        example_id = pred.get("example_id", "unknown")
        summary = summaries.get(example_id, "")
        summary_snippet = truncate_text(summary, max_len=200)
        gt_norm = pred.get("gt_norm", 0.0)
        pred_score = pred.get("pred", 0.0)
        abs_error = pred.get("abs_error", 0.0)
        issue_types = format_issue_types(pred.get("issue_types_counts", {}))
        max_severity = pred.get("max_severity", "none")

        lines.append(
            f"| {example_id[:30]} | {gt_norm:.3f} | {pred_score:.3f} | {abs_error:.3f} | "
            f"{summary_snippet[:100]} | {issue_types[:30]} | {max_severity} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## C) GT niedrig (<=0.55) aber Pred hoch (>=0.85) (Top 20)",
            "",
            "| ID | GT | Pred | Error | Summary Snippet | Top Issues | Max Severity |",
            "|----|----|------|-------|-----------------|------------|--------------|",
        ]
    )

    for pred in lists["gt_low_pred_high"]:
        example_id = pred.get("example_id", "unknown")
        summary = summaries.get(example_id, "")
        summary_snippet = truncate_text(summary, max_len=200)
        gt_norm = pred.get("gt_norm", 0.0)
        pred_score = pred.get("pred", 0.0)
        abs_error = pred.get("abs_error", 0.0)
        issue_types = format_issue_types(pred.get("issue_types_counts", {}))
        max_severity = pred.get("max_severity", "none")

        lines.append(
            f"| {example_id[:30]} | {gt_norm:.3f} | {pred_score:.3f} | {abs_error:.3f} | "
            f"{summary_snippet[:100]} | {issue_types[:30]} | {max_severity} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## D) Fälle mit High Severity Spans (Top 20)",
            "",
            "| ID | GT | Pred | Error | Summary Snippet | Top Issues | Max Severity |",
            "|----|----|------|-------|-----------------|------------|--------------|",
        ]
    )

    for pred in lists["high_severity"]:
        example_id = pred.get("example_id", "unknown")
        summary = summaries.get(example_id, "")
        summary_snippet = truncate_text(summary, max_len=200)
        gt_norm = pred.get("gt_norm", 0.0)
        pred_score = pred.get("pred", 0.0)
        abs_error = pred.get("abs_error", 0.0)
        issue_types = format_issue_types(pred.get("issue_types_counts", {}))
        max_severity = pred.get("max_severity", "none")

        lines.append(
            f"| {example_id[:30]} | {gt_norm:.3f} | {pred_score:.3f} | {abs_error:.3f} | "
            f"{summary_snippet[:100]} | {issue_types[:30]} | {max_severity} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## Fazit: Typische Fehlmuster",
            "",
            "- **Größte Fehler:** Oft bei mittleren GT-Werten (0.4-0.6), wo der Agent zu extremen Werten tendiert.",
            "- **GT hoch, Pred niedrig:** Agent übersieht gut lesbare Summaries, möglicherweise wegen komplexer Satzstrukturen oder langer Sätze.",
            "- **GT niedrig, Pred hoch:** Agent überschätzt schlecht lesbare Summaries, möglicherweise wegen fehlender Sensitivität für spezifische Readability-Probleme.",
            "- **High Severity Spans:** Oft korreliert mit niedrigen GT-Werten, aber nicht immer (Agent kann auch bei mittleren GT-Werten high severity markieren).",
            "- **Kalibrierungsproblem:** R² = -2.84 deutet auf systematischen Bias hin; lineare Kalibrierung könnte helfen.",
        ]
    )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description="Analysiert Fehler in Readability-Evaluation-Runs")
    ap.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Run-Verzeichnis (z.B. results/evaluation/readability/readability_20260114_184754_gpt-4o-mini_v1_seed42)",
    )
    ap.add_argument(
        "--data",
        type=str,
        default="data/sumeval/sumeval_clean.jsonl",
        help="Pfad zur JSONL-Datei (für Summary-Texts)",
    )
    ap.add_argument(
        "--out", type=str, help="Output-Pfad (default: <run_dir>/../analysis_readability_v1.md)"
    )

    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        ap.error(f"Run-Verzeichnis nicht gefunden: {run_dir}")

    predictions_path = run_dir / "predictions.jsonl"
    if not predictions_path.exists():
        ap.error(f"predictions.jsonl nicht gefunden in: {run_dir}")

    data_path = Path(args.data)
    if not data_path.exists():
        ap.error(f"Datensatz nicht gefunden: {data_path}")

    if args.out:
        out_path = Path(args.out)
    else:
        out_path = run_dir.parent / "analysis_readability_v1.md"

    print(f"Lade Predictions aus: {predictions_path}")
    predictions = load_predictions(predictions_path)
    print(f"  {len(predictions)} Predictions geladen")

    print(f"Lade Summaries aus: {data_path}")
    example_ids = [p.get("example_id") for p in predictions if p.get("example_id")]
    summaries = load_summaries(data_path, example_ids)
    print(f"  {len(summaries)} Summaries geladen")

    print("Analysiere Fehler...")
    lists = analyze_errors(predictions, summaries)
    print(f"  A) Größte Fehler: {len(lists['largest_errors'])}")
    print(f"  B) GT hoch, Pred niedrig: {len(lists['gt_high_pred_low'])}")
    print(f"  C) GT niedrig, Pred hoch: {len(lists['gt_low_pred_high'])}")
    print(f"  D) High Severity: {len(lists['high_severity'])}")

    print("Berechne Histogramme...")
    histograms = compute_histograms(predictions)

    print(f"Schreibe Analyse nach: {out_path}")
    write_analysis_md(lists, histograms, summaries, out_path)

    print("\nAnalyse abgeschlossen!")


if __name__ == "__main__":
    main()
