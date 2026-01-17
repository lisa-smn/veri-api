#!/usr/bin/env python3
"""
Generiert stichpunktartige Zusammenfassung der Factuality-Agent Evaluation
aus den JSON/JSONL-Artefakten von scripts/test_evidence_gate_eval.py.

Beispiel-Commands:
    # Text-Output (stdout):
    python3 scripts/print_factuality_eval_bullets.py --results results/evaluation/evidence_gate_test/results_non_error.json

    # Markdown + Artefakte + Output-Datei:
    python3 scripts/print_factuality_eval_bullets.py \\
        --results results/evaluation/evidence_gate_test/results_non_error.json \\
        --debug results/evaluation/evidence_gate_test/debug_claims.jsonl \\
        --errors results/evaluation/evidence_gate_test/error_cases.jsonl \\
        --format md \\
        --out results/evaluation/evidence_gate_test/factuality_eval_bullets.md
"""

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Lädt JSON-Datei."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Lädt JSONL-Datei."""
    entries = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def format_number(value: Any, decimals: int = 3) -> str:
    """Formatiert Zahl mit fester Dezimalstellen."""
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return str(value)


def format_percent(value: Any, decimals: int = 1) -> str:
    """Formatiert Prozentwert."""
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return str(value)


def print_section(title: str, format_type: str, output_lines: list[str]):
    """Fügt Sektion-Header hinzu."""
    if format_type == "md":
        output_lines.append(f"\n## {title}\n")
    else:
        output_lines.append(f"\n{title}")
        output_lines.append("-" * len(title))


def print_bullet(text: str, format_type: str, output_lines: list[str], indent: int = 0):
    """Fügt Bulletpoint hinzu."""
    prefix = "  " * indent + ("- " if format_type == "md" else "• ")
    output_lines.append(f"{prefix}{text}")


def generate_bullets(
    results_path: Path,
    debug_path: Path | None = None,
    errors_path: Path | None = None,
    format_type: str = "text",
) -> str:
    """Generiert Stichpunkte aus den Artefakten."""
    output_lines: list[str] = []

    # Lade Results JSON
    results = load_json(results_path)
    metrics = results.get("metrics", {})
    coverage = results.get("coverage", {})
    evidence_stats = results.get("evidence_stats", {})
    claim_level_stats = evidence_stats.get("claim_level", {})
    span_level_stats = evidence_stats.get("span_level", {})

    # 1) Zweck des Faktenagenten
    print_section("1) ZWECK DES FAKTENAGENTEN", format_type, output_lines)
    print_bullet(
        "Prüft automatisch, ob Behauptungen (Claims) in einer Zusammenfassung "
        "durch den Originalartikel belegt sind",
        format_type,
        output_lines,
    )
    print_bullet(
        "Ein 'Claim' ist eine einzelne, überprüfbare Aussage aus der Zusammenfassung "
        "(z.B. 'Die Studie wurde 2023 veröffentlicht' oder 'X hat Y gesagt')",
        format_type,
        output_lines,
    )
    print_bullet(
        "'IssueSpan' markiert problematische Stellen im Text; pro Satz maximal ein Span "
        "(repräsentativ für alle Claims in diesem Satz)",
        format_type,
        output_lines,
    )

    # 2) Evidence-Gate Erklärung
    print_section("2) WAS IST DAS EVIDENCE-GATE (KERNIDEE)", format_type, output_lines)
    print_bullet(
        "Hauptziel: Reduzierung von False Positives (fälschlich als Fehler markierte "
        "korrekte Aussagen)",
        format_type,
        output_lines,
    )
    print_bullet(
        "Harte Regel: 'incorrect' wird nur vergeben, wenn belastbare Evidence "
        "(wörtliches Zitat aus dem Artikel) gefunden wurde",
        format_type,
        output_lines,
    )
    uncertainty_policy = results.get("uncertainty_policy", "unknown")
    if uncertainty_policy != "count_as_error":
        print_bullet(
            f"Uncertainty-Policy: '{uncertainty_policy}' (uncertain Claims zählen nicht "
            "als Fehler oder nur teilweise)",
            format_type,
            output_lines,
        )
    print_bullet(
        "Ohne Evidence: Claim wird als 'uncertain' markiert (kein Fehler, aber "
        "nicht verifizierbar)",
        format_type,
        output_lines,
    )
    print_bullet(
        "Evidence muss wörtlich im Artikel vorkommen (Quote-Matching mit Normalisierung)",
        format_type,
        output_lines,
    )
    print_bullet(
        "Evidence muss den Claim abdecken (Coverage-Check: wichtige Einheiten "
        "wie Zahlen, Namen, Daten müssen im Evidence vorkommen)",
        format_type,
        output_lines,
    )

    # 3) Invarianten
    print_section("3) WICHTIGE INVARIANTEN", format_type, output_lines)
    print_bullet(
        "Wenn kein Evidence-Index gewählt wurde (index = -1): "
        "→ evidence_quote ist None → evidence_found = False",
        format_type,
        output_lines,
    )
    print_bullet(
        "Wenn Evidence-Index gewählt wurde (index >= 0): "
        "→ evidence_quote muss nicht-leer sein UND wörtlich in der Passage vorkommen "
        "→ nur dann evidence_found = True",
        format_type,
        output_lines,
    )
    print_bullet(
        "Label 'correct' oder 'incorrect' wird nur vergeben, wenn evidence_found = True",
        format_type,
        output_lines,
    )
    print_bullet(
        "Coverage-Fail (Evidence deckt Claim nicht vollständig ab) führt NICHT "
        "automatisch zu 'uncertain', sondern reduziert nur die Confidence",
        format_type,
        output_lines,
    )
    print_bullet(
        "Gate-Logik sitzt ausschließlich im ClaimVerifier; der Agent selbst "
        "macht keine zusätzlichen 'Safety-Downgrades'",
        format_type,
        output_lines,
    )
    print_bullet(
        "Verdict (incorrect/uncertain) ist unabhängig von Severity (low/medium/high): "
        "ein 'incorrect' Claim kann severity='low' haben (z.B. bei error_type='OTHER')",
        format_type,
        output_lines,
    )

    # 4) Tests
    print_section("4) WELCHE TESTS WURDEN GEMACHT", format_type, output_lines)
    print_bullet("Unit Tests:", format_type, output_lines, indent=0)
    print_bullet(
        "tests/unit/test_evidence_gate_refactored.py: 10 Tests "
        "(Evidence-Validierung, Gate-Logik, Coverage-Fail)",
        format_type,
        output_lines,
        indent=1,
    )
    print_bullet(
        "tests/unit/test_issue_span_verdict.py: 4 Tests (Verdict vs Severity Trennung)",
        format_type,
        output_lines,
        indent=1,
    )

    # 4.2) Mini-Evaluation Ergebnisse
    print_section("4.2) MINI-EVALUATION ERGEBNISSE", format_type, output_lines)

    # Confusion Matrix
    tp = metrics.get("tp", 0)
    tn = metrics.get("tn", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)

    print_bullet("Konfusionsmatrix:", format_type, output_lines)
    print_bullet(f"TP (True Positives): {tp}", format_type, output_lines, indent=1)
    print_bullet(f"TN (True Negatives): {tn}", format_type, output_lines, indent=1)
    print_bullet(f"FP (False Positives): {fp}", format_type, output_lines, indent=1)
    print_bullet(f"FN (False Negatives): {fn}", format_type, output_lines, indent=1)

    # Ground Truth Distribution
    gt_positives = tp + fn
    gt_negatives = tn + fp
    total_examples = gt_positives + gt_negatives
    positive_rate = (gt_positives / total_examples * 100) if total_examples > 0 else 0.0

    print_bullet("Ground Truth Distribution:", format_type, output_lines)
    print_bullet(f"GT Positives (TP+FN): {gt_positives}", format_type, output_lines, indent=1)
    print_bullet(f"GT Negatives (TN+FP): {gt_negatives}", format_type, output_lines, indent=1)
    print_bullet(f"Positiv-Rate: {positive_rate:.1f}%", format_type, output_lines, indent=1)

    if gt_negatives < 20:
        print_bullet(
            "⚠️  WARN: Specificity ist bei <20 Negativen statistisch instabil; "
            "nicht überinterpretieren.",
            format_type,
            output_lines,
            indent=1,
        )

    # Metriken
    recall = metrics.get("recall", 0.0)
    precision = metrics.get("precision", 0.0)
    f1 = metrics.get("f1", 0.0)
    specificity = metrics.get("specificity", 0.0)
    # Balanced Accuracy kann direkt im metrics sein oder berechnet werden
    balanced_accuracy = metrics.get("balanced_accuracy")
    if balanced_accuracy is None:
        # Fallback: Berechne aus Recall und Specificity
        balanced_accuracy = (recall + specificity) / 2.0 if (recall + specificity) > 0 else 0.0

    print_bullet("Metriken:", format_type, output_lines)
    print_bullet(
        f"Recall: {format_number(recall)} ({format_percent(recall)} der echten Fehler werden erkannt)",
        format_type,
        output_lines,
        indent=1,
    )
    print_bullet(
        f"Precision: {format_number(precision)} ({format_percent(precision)} der als Fehler markierten sind tatsächlich Fehler)",
        format_type,
        output_lines,
        indent=1,
    )
    print_bullet(f"F1-Score: {format_number(f1)}", format_type, output_lines, indent=1)
    print_bullet(f"Specificity: {format_number(specificity)}", format_type, output_lines, indent=1)
    print_bullet(
        f"Balanced Accuracy: {format_number(balanced_accuracy)}",
        format_type,
        output_lines,
        indent=1,
    )

    # Coverage/Abstention
    total_claims = coverage.get("total_claims", 0)
    num_correct = coverage.get("num_correct", 0)
    num_incorrect = coverage.get("num_incorrect", 0)
    num_uncertain = coverage.get("num_uncertain", 0)
    coverage_rate = coverage.get("coverage", 0.0)
    abstention_rate = coverage.get("abstention_rate", 0.0)

    print_bullet("Coverage/Abstention:", format_type, output_lines)
    print_bullet(f"Total Claims: {total_claims}", format_type, output_lines, indent=1)
    print_bullet(f"Correct: {num_correct}", format_type, output_lines, indent=1)
    print_bullet(f"Incorrect: {num_incorrect}", format_type, output_lines, indent=1)
    print_bullet(f"Uncertain: {num_uncertain}", format_type, output_lines, indent=1)
    print_bullet(
        f"Coverage: {format_number(coverage_rate)} ({format_percent(coverage_rate)} der Claims werden als correct/incorrect klassifiziert)",
        format_type,
        output_lines,
        indent=1,
    )
    print_bullet(
        f"Abstention Rate: {format_number(abstention_rate)} ({format_percent(abstention_rate)} der Claims bleiben 'uncertain')",
        format_type,
        output_lines,
        indent=1,
    )

    # Evidence-Statistiken (Claim-Level)
    incorrect_with_evidence = claim_level_stats.get("incorrect_with_evidence", 0)
    incorrect_without_evidence = claim_level_stats.get("incorrect_without_evidence", 0)
    evidence_found_true = claim_level_stats.get("evidence_found_true", 0)
    evidence_found_false = claim_level_stats.get("evidence_found_false", 0)

    print_bullet("Evidence-Statistiken (Claim-Level):", format_type, output_lines)
    print_bullet(
        f"Evidence Found True: {evidence_found_true} Claims", format_type, output_lines, indent=1
    )
    print_bullet(
        f"Evidence Found False: {evidence_found_false} Claims", format_type, output_lines, indent=1
    )
    print_bullet(
        f"Incorrect Claims mit Evidence: {incorrect_with_evidence}",
        format_type,
        output_lines,
        indent=1,
    )
    if incorrect_without_evidence == 0:
        print_bullet(
            f"Incorrect Claims ohne Evidence: {incorrect_without_evidence} (✅ Evidence-Gate funktioniert)",
            format_type,
            output_lines,
            indent=1,
        )
    else:
        print_bullet(
            f"Incorrect Claims ohne Evidence: {incorrect_without_evidence} (⚠️  sollte 0 sein)",
            format_type,
            output_lines,
            indent=1,
        )

    # IssueSpan-Level Stats (falls vorhanden)
    if span_level_stats:
        incorrect_spans_with_evidence = span_level_stats.get("incorrect_with_evidence", 0)
        incorrect_spans_without_evidence = span_level_stats.get("incorrect_without_evidence", 0)
        uncertain_spans = span_level_stats.get("uncertain", 0)

        print_bullet("IssueSpan-Level Evidence Stats:", format_type, output_lines)
        print_bullet(
            f"Incorrect Spans ohne Evidence: {incorrect_spans_without_evidence}",
            format_type,
            output_lines,
            indent=1,
        )
        print_bullet(
            f"Incorrect Spans mit Evidence: {incorrect_spans_with_evidence}",
            format_type,
            output_lines,
            indent=1,
        )
        print_bullet(f"Uncertain Spans: {uncertain_spans}", format_type, output_lines, indent=1)
        print_bullet(
            "(Hinweis: Spans sind repräsentativ pro Satz, daher kann Span-Level < Claim-Level sein)",
            format_type,
            output_lines,
            indent=1,
        )

    # Evidence Selection / Gate Reasons (aus Debug, falls vorhanden)
    if debug_path and debug_path.exists():
        debug_claims = load_jsonl(debug_path)

        # Zähle Gate Reasons
        gate_reasons = {}
        selection_reasons = {}

        for entry in debug_claims:
            for claim in entry.get("claims", []):
                gate_reason = claim.get("gate_reason")
                if gate_reason:
                    gate_reasons[gate_reason] = gate_reasons.get(gate_reason, 0) + 1

                selection_reason = claim.get("evidence_selection_reason")
                if selection_reason:
                    selection_reasons[selection_reason] = (
                        selection_reasons.get(selection_reason, 0) + 1
                    )

        if gate_reasons:
            print_bullet(
                "Gate Reasons (warum wurde ein Claim als correct/incorrect/uncertain markiert?):",
                format_type,
                output_lines,
            )
            for reason, count in sorted(gate_reasons.items(), key=lambda x: -x[1]):
                print_bullet(f"{reason}: {count} Claims", format_type, output_lines, indent=1)

        if selection_reasons:
            print_bullet(
                "Evidence Selection Reasons (warum wurde Evidence gefunden/nicht gefunden?):",
                format_type,
                output_lines,
            )
            for reason, count in sorted(selection_reasons.items(), key=lambda x: -x[1]):
                print_bullet(f"{reason}: {count} Fälle", format_type, output_lines, indent=1)

    # 5) Interpretation
    print_section("5) INTERPRETATION", format_type, output_lines)

    if precision > 0.9:
        print_bullet(
            f"Precision sehr hoch ({format_percent(precision)}): "
            "Wenn der Agent einen Fehler meldet, ist es sehr wahrscheinlich tatsächlich ein Fehler",
            format_type,
            output_lines,
        )
    elif precision > 0.7:
        print_bullet(
            f"Precision moderat ({format_percent(precision)}): "
            "Wenn der Agent einen Fehler meldet, ist es meist wirklich ein Fehler",
            format_type,
            output_lines,
        )

    if recall < 0.7:
        print_bullet(
            f"Recall begrenzt ({format_percent(recall)}): "
            f"{fn} False Negatives - echte Fehler werden übersehen",
            format_type,
            output_lines,
        )
        print_bullet(
            "Hauptursachen: konservatives Gate (nur mit Evidence) + "
            "Retrieval findet nicht immer relevante Passagen",
            format_type,
            output_lines,
            indent=1,
        )

    if fp == 0:
        print_bullet(
            "False Positives: 0 - Evidence-Gate erfüllt Sicherheitsziel perfekt",
            format_type,
            output_lines,
        )
    elif fp <= 2:
        print_bullet(
            f"False Positives sehr niedrig ({fp}): Evidence-Gate erfüllt Sicherheitsziel",
            format_type,
            output_lines,
        )

    if incorrect_without_evidence == 0:
        print_bullet(
            "Keine 'incorrect' Claims ohne Evidence (✅ Evidence-Gate funktioniert)",
            format_type,
            output_lines,
        )

    if coverage_rate > 0.7:
        print_bullet(
            f"Coverage akzeptabel ({format_percent(coverage_rate)}): "
            "Mehrheit der Claims wird als correct/incorrect klassifiziert",
            format_type,
            output_lines,
        )

    if balanced_accuracy > 0.6:
        print_bullet(
            f"Balanced Accuracy ({format_number(balanced_accuracy)}) zeigt: "
            "System ist besser als Zufall, aber Trade-off zwischen Recall und Specificity",
            format_type,
            output_lines,
        )

    # 6) Artefakte
    print_section("6) ROHDATEN/ARTEFAKTE FÜR NACHVOLLZIEHBARKEIT", format_type, output_lines)

    print_bullet(
        f"{results_path.name}: Metriken, Coverage/Abstention, Evidence-Stats, "
        "Ground Truth Distribution",
        format_type,
        output_lines,
    )

    if debug_path and debug_path.exists():
        print_bullet(
            f"{debug_path.name}: Pro Claim: verdict, evidence_found, confidence, "
            "Gate-Entscheidungsweg, Evidence-Selection-Details, Coverage-Status, Raw LLM-Output",
            format_type,
            output_lines,
        )

    if errors_path and errors_path.exists():
        error_cases = load_jsonl(errors_path)
        print_bullet(
            f"{errors_path.name}: {len(error_cases)} False Positives und False Negatives "
            "mit Top Issue Span, Top Claims, Claim-Label-Counts (für gezielte Fehleranalyse)",
            format_type,
            output_lines,
        )

    # 7) Hinweis zu Variabilität
    print_section("7) HINWEISE ZU VARIABILITÄT", format_type, output_lines)
    print_bullet(
        "Zahlen können zwischen Runs leicht schwanken (LLM-Sampling/Context-Auswahl)",
        format_type,
        output_lines,
    )
    print_bullet("Fokus auf Trends, nicht einzelne Dezimalstellen", format_type, output_lines)

    return "\n".join(output_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generiert stichpunktartige Zusammenfassung der Factuality-Agent Evaluation"
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Pfad zu results/evaluation/evidence_gate_test/results_*.json",
    )
    parser.add_argument(
        "--debug",
        type=Path,
        default=None,
        help="Optional: Pfad zu debug_claims.jsonl",
    )
    parser.add_argument(
        "--errors",
        type=Path,
        default=None,
        help="Optional: Pfad zu error_cases.jsonl",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "md"],
        default="text",
        help="Output-Format: text oder md (default: text)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional: Pfad für Output-Datei (sonst stdout)",
    )

    args = parser.parse_args()

    # Prüfe ob Results-Datei existiert
    if not args.results.exists():
        print(f"❌ Fehler: Results-Datei nicht gefunden: {args.results}")
        return 1

    # Generiere Stichpunkte
    try:
        output = generate_bullets(
            results_path=args.results,
            debug_path=args.debug,
            errors_path=args.errors,
            format_type=args.format,
        )

        # Output schreiben
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with args.out.open("w", encoding="utf-8") as f:
                f.write(output)
            print(f"✅ Stichpunkte gespeichert: {args.out}")
        else:
            print(output)

        return 0
    except Exception as e:
        print(f"❌ Fehler beim Generieren der Stichpunkte: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
