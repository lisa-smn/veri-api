"""
Kleiner Test-Run für Evidence-Gate Evaluation.

Führt 50 FRANK Examples aus, um zu prüfen:
- Ob Specificity steigt (weniger False Positives)
- Ob Recall nicht komplett kollabiert
- Ob incorrect ohne evidence zu uncertain wird
- Uncertainty-Policy Effekte (non_error, weight_0.5, count_as_error)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.openai_client import OpenAIClient
from app.services.agents.factuality.factuality_agent import FactualityAgent
from app.services.analysis.metrics import BinaryMetrics


def load_frank_examples(dataset_path: Path, max_examples: int = 50) -> list[dict[str, Any]]:
    """Lädt FRANK Examples."""
    examples = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            if line.strip():
                ex = json.loads(line)
                examples.append(ex)
    return examples


def parse_has_error(raw_label: Any) -> bool | None:
    """Robustes Parsing von has_error."""
    if isinstance(raw_label, bool):
        return raw_label
    if isinstance(raw_label, str):
        s = raw_label.lower().strip()
        if s in ("true", "1", "yes", "error", "fehler"):
            return True
        if s in ("false", "0", "no", "correct", "korrekt"):
            return False
    if isinstance(raw_label, int):
        return bool(raw_label)
    return None


def count_effective_issues(issue_spans, uncertainty_policy: str) -> float:
    """
    Zählt effektive Issues basierend auf uncertainty_policy.

    Args:
        issue_spans: Liste von IssueSpans
        uncertainty_policy: "non_error", "weight_0.5", oder "count_as_error"

    Returns:
        Effektive Anzahl von Issues (float für weight_0.5)
    """
    if not issue_spans:
        return 0.0

    count = 0.0
    for span in issue_spans:
        # NEU: Primär verdict Feld verwenden (explizites Uncertainty-Signal)
        is_uncertain = span.verdict == "uncertain"

        # Fallback: Nur wenn verdict fehlt (alte Runs), nutze Heuristik
        if span.verdict is None:
            # Heuristik für Backward Compatibility (alte Daten ohne verdict)
            is_uncertain = (
                span.severity == "low"
                or "uncertain" in (span.message or "").lower()
                or "nicht sicher verifizierbar" in (span.message or "").lower()
            )

        if is_uncertain:
            if uncertainty_policy == "non_error":
                # Uncertain zählt nicht als Fehler
                continue
            if uncertainty_policy in ("weight_0.5", "weight_0_5"):
                # Uncertain zählt als 0.5
                count += 0.5
            else:  # count_as_error
                # Uncertain zählt wie incorrect
                count += 1.0
        else:
            # Incorrect zählt immer als 1.0
            count += 1.0

    return count


def main():
    parser = argparse.ArgumentParser(description="Evidence-Gate Test Evaluation")
    parser.add_argument(
        "--uncertainty-policy",
        type=str,
        default="count_as_error",
        choices=["non_error", "weight_0.5", "weight_0_5", "count_as_error"],
        help="Uncertainty-Policy: non_error, weight_0.5 (oder weight_0_5), oder count_as_error",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Maximale Anzahl von Examples (default: 50)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug-Modus: Schreibe debug_claims.jsonl mit detaillierten Claim-Informationen",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict-Mode: Keine silent fallbacks, Exception bei Schema-Verletzungen",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Cache deaktivieren: Alle LLM-Calls werden neu ausgeführt",
    )
    parser.add_argument(
        "--decision-threshold",
        type=float,
        default=1.0,
        help="Schwelle für pred_has_error: effective_issues >= threshold (default: 1.0)",
    )
    args = parser.parse_args()

    dataset_path = ROOT / "data" / "frank" / "frank_clean.jsonl"
    max_examples = args.max_examples
    uncertainty_policy = args.uncertainty_policy
    debug_mode = args.debug
    strict_mode = args.strict
    no_cache = args.no_cache
    decision_threshold = args.decision_threshold

    # Normalisiere Alias: weight_0_5 -> weight_0.5
    if uncertainty_policy == "weight_0_5":
        uncertainty_policy = "weight_0.5"

    print("=" * 80)
    print("Evidence-Gate Test Evaluation")
    print("=" * 80)
    print(f"\nDataset: {dataset_path}")
    print(f"Max Examples: {max_examples}")
    print(f"Uncertainty Policy: {uncertainty_policy}")
    print(f"Decision Threshold: {decision_threshold}")
    print(f"Debug Mode: {'✅' if debug_mode else '❌'}")
    print(f"Strict Mode: {'✅' if strict_mode else '❌'}")
    print(f"Cache: {'❌ (deaktiviert)' if no_cache else '✅ (aktiviert)'}")
    print()

    # Load examples
    examples = load_frank_examples(dataset_path, max_examples)
    print(f"✅ {len(examples)} Examples geladen")
    print()

    # Setup agent
    llm_client = OpenAIClient(model_name="gpt-4o-mini")

    # Strict-Mode: Erstelle ClaimVerifier mit strict_mode
    if strict_mode:
        from app.services.agents.factuality.claim_verifier import LLMClaimVerifier

        claim_verifier = LLMClaimVerifier(
            llm_client,
            use_evidence_retriever=True,
            evidence_retriever_top_k=5,
            strict_mode=strict_mode,  # WICHTIG: Verwende die Variable aus CLI-Args
        )
        # RUNTIME CHECK: Beweis, dass strict_mode korrekt gesetzt ist
        print(
            f"RUNTIME CHECK claim_verifier.strict_mode: {claim_verifier.strict_mode}, expected: {strict_mode}"
        )
        assert claim_verifier.strict_mode == strict_mode, (
            f"verifier.strict_mode ist {claim_verifier.strict_mode}, erwartet {strict_mode}"
        )

        agent = FactualityAgent(llm_client, claim_verifier=claim_verifier)

        # RUNTIME CHECK: Beweis, dass strict_mode im Agent-Verifier gesetzt ist
        print(
            f"RUNTIME CHECK agent.claim_verifier.strict_mode: {agent.claim_verifier.strict_mode}, expected: {strict_mode}"
        )
        assert agent.claim_verifier.strict_mode == strict_mode, (
            f"agent.claim_verifier.strict_mode ist {agent.claim_verifier.strict_mode}, erwartet {strict_mode}"
        )

        print("⚠️  Strict-Mode aktiviert (--strict)")
    else:
        agent = FactualityAgent(llm_client)

    print("Starting evaluation...")
    print()

    # Evaluate
    metrics = BinaryMetrics()
    predictions = []
    ground_truths = []
    results = []

    # Coverage/Abstention Tracking
    num_correct = 0
    num_incorrect = 0
    num_uncertain = 0

    # IssueSpan-Level Statistics (repräsentativ pro Satz, max 1 pro Satz)
    incorrect_spans_without_evidence = 0
    incorrect_spans_with_evidence = 0
    uncertain_spans_count = 0

    # Claim-Level Statistics (alle Claims)
    claim_evidence_found_true = 0
    claim_evidence_found_false = 0
    incorrect_claims_with_evidence = 0
    incorrect_claims_without_evidence = 0

    # Debug: Claim-Details für debug_claims.jsonl
    debug_claims = []

    for i, ex in enumerate(examples):
        if (i + 1) % 10 == 0:
            print(f"  Processing {i + 1}/{len(examples)}...")

        has_error = parse_has_error(ex.get("has_error"))
        if has_error is None:
            continue

        try:
            agent_result = agent.run(
                ex["article"],
                ex["summary"],
                meta={"source": "evidence_gate_test", "example_id": ex.get("example_id", i)},
            )
        except (ValueError, RuntimeError) as e:
            # Strict Mode Exceptions: Nicht schlucken, sondern weiterwerfen
            if strict_mode and (
                "Schema-Verletzung" in str(e) or "TRIPWIRE" in str(e) or "Strict-Mode" in str(e)
            ):
                print("\n❌ STRICT MODE EXCEPTION (nicht geschluckt):")
                print(f"   {e}")
                print(f"   Example ID: {ex.get('example_id', i)}")
                raise  # Weiterwerfen, nicht schlucken
            # Andere Exceptions: wie bisher behandeln
            raise

        try:
            # Binary prediction: has_error basierend auf uncertainty_policy und decision_threshold
            effective_issues = count_effective_issues(agent_result.issue_spans, uncertainty_policy)
            pred_has_error = effective_issues >= decision_threshold

            # Coverage/Abstention Tracking: Zähle Claims nach Label
            # (Wir müssen die Claims aus details extrahieren, da AgentResult keine direkte Claim-Liste hat)
            details = agent_result.details or {}
            claims_data = details.get("claims", [])

            # Debug: Sammle Claim-Details
            example_debug_claims = []

            for claim_data in claims_data:
                if not isinstance(claim_data, dict):
                    continue

                claim_label = claim_data.get("label_final") or claim_data.get("label")
                if claim_label == "correct":
                    num_correct += 1
                elif claim_label == "incorrect":
                    num_incorrect += 1
                elif claim_label == "uncertain":
                    num_uncertain += 1

                # Claim-Level Evidence Statistics
                evidence_found = claim_data.get("evidence_found")
                # Fallback: Wenn evidence_found None, prüfe evidence_quote und selected_evidence_index
                if evidence_found is None:
                    evidence_quote = claim_data.get("evidence_quote")
                    selected_idx = claim_data.get("selected_evidence_index", -1)
                    evidence_found = bool(evidence_quote) and selected_idx >= 0

                if evidence_found is True:
                    claim_evidence_found_true += 1
                elif evidence_found is False:
                    claim_evidence_found_false += 1

                # Claim-Level: Incorrect Claims mit/ohne Evidence
                if claim_label == "incorrect":
                    if evidence_found is True:
                        incorrect_claims_with_evidence += 1
                    elif evidence_found is False:
                        incorrect_claims_without_evidence += 1

                # Debug: Speichere Claim-Details
                if debug_mode:
                    # FIX: Verwende direkt Claim-Felder, nicht JSON-Parse von raw_verifier_output_preview
                    # (raw_verifier_output_preview ist auf 800 chars gekürzt und oft nicht parsbares JSON)
                    # Die Felder label_raw, label_final, etc. wurden bereits im ClaimVerifier gesetzt

                    example_debug_claims.append(
                        {
                            "verdict": claim_label,
                            "evidence_found": claim_data.get("evidence_found"),
                            "selected_evidence_index": claim_data.get("selected_evidence_index"),
                            "evidence_quote": claim_data.get("evidence_quote"),
                            "confidence": claim_data.get("confidence"),
                            "claim_text": claim_data.get("text", "")[:200],  # Kürze auf 200 Zeichen
                            "llm_called": True,  # Immer True, da wir LLM verwenden
                            "parse_ok": claim_data.get("parse_ok", True),
                            "parse_error": claim_data.get("parse_error"),
                            "schema_violation_reason": claim_data.get("schema_violation_reason"),
                            # Repair-Status (aus schema_violation_reason extrahiert)
                            "repair_attempted": "repair"
                            in (claim_data.get("schema_violation_reason") or "").lower(),
                            "repair_success": claim_data.get("schema_violation_reason")
                            == "repair_success",
                            # Gate-Entscheidungsweg (direkt aus Claim-Feldern)
                            "label_raw": claim_data.get("label_raw"),
                            "label_final": claim_data.get("label_final") or claim_data.get("label"),
                            "selected_evidence_index_raw": claim_data.get(
                                "selected_evidence_index_raw"
                            ),
                            "evidence_quote_raw": claim_data.get("evidence_quote_raw"),
                            "evidence_selection_reason": claim_data.get(
                                "evidence_selection_reason"
                            ),  # ok | no_passage_selected | index_out_of_range | empty_quote | quote_not_in_passage
                            "evidence_found": claim_data.get("evidence_found"),
                            "selected_evidence_index": claim_data.get("selected_evidence_index"),
                            "evidence_quote": claim_data.get("evidence_quote"),
                            "selected_passage_preview": claim_data.get("selected_passage_preview"),
                            "quote_is_substring_of_passage": claim_data.get(
                                "quote_is_substring_of_passage"
                            ),
                            "coverage_ok": claim_data.get("coverage_ok"),
                            "coverage_note": claim_data.get("coverage_note"),
                            "gate_reason": claim_data.get("gate_reason"),
                            # raw_verifier_output_preview: Immer loggen (800 chars)
                            "raw_verifier_output_preview": claim_data.get("raw_verifier_output")
                            or None,
                            # Final-Felder (nach Parsing/Normalisierung)
                            "selected_evidence_index": claim_data.get("selected_evidence_index"),
                            "evidence_quote": claim_data.get("evidence_quote"),
                            # Raw-Felder für Debug (was kam wirklich vom LLM? - unverändert)
                            "selected_evidence_index_raw": claim_data.get(
                                "selected_evidence_index_raw"
                            ),
                            "evidence_quote_raw": claim_data.get("evidence_quote_raw"),
                            "retrieved_passages_count": len(
                                claim_data.get("retrieved_passages", [])
                            ),
                            "retrieval_scores": claim_data.get("retrieval_scores", [])[
                                :5
                            ],  # Top-5 Scores
                            "retrieved_passages_preview": [
                                p[:120] + "..." if len(p) > 120 else p
                                for p in claim_data.get("retrieved_passages", [])[
                                    :5
                                ]  # Top-5 Passagen
                            ],
                            "evidence_context_preview": (
                                claim_data.get("retrieved_passages", [])[0][:200] + "..."
                                if claim_data.get("retrieved_passages")
                                else None
                            ),
                        }
                    )

            if debug_mode:
                debug_claims.append(
                    {
                        "example_id": ex.get("example_id", i),
                        "gt_has_error": has_error,
                        "num_claims": len(claims_data),
                        "claims": example_debug_claims,
                    }
                )

            # IssueSpan-Level Evidence Statistics (repräsentativ pro Satz)
            # NEU: Nutze verdict primär (trennt incorrect von uncertain unabhängig von severity)
            for span in agent_result.issue_spans:
                v = getattr(span, "verdict", None)  # Primär: verdict Feld

                if v == "incorrect":
                    # Explizit incorrect (auch wenn severity="low")
                    if span.evidence_found is False:
                        incorrect_spans_without_evidence += 1
                    elif span.evidence_found is True:
                        incorrect_spans_with_evidence += 1
                elif v == "uncertain":
                    # Explizit uncertain
                    uncertain_spans_count += 1
                else:
                    # Backward fallback (alte Daten ohne verdict): nutze alte Heuristik
                    is_uncertain = (
                        (span.severity == "low")
                        or ("uncertain" in (span.message or "").lower())
                        or ("nicht sicher" in (span.message or "").lower())
                    )
                    if is_uncertain:
                        uncertain_spans_count += 1
                    else:
                        # Als incorrect behandeln (alte Heuristik)
                        if span.evidence_found is False:
                            incorrect_spans_without_evidence += 1
                        elif span.evidence_found is True:
                            incorrect_spans_with_evidence += 1

            predictions.append(pred_has_error)
            ground_truths.append(has_error)

            # Sammle Claim-Label-Counts für error_cases.jsonl
            claim_label_counts = {"correct": 0, "incorrect": 0, "uncertain": 0}
            top_claims_list = []

            for claim_data in claims_data:
                if not isinstance(claim_data, dict):
                    continue
                claim_label = claim_data.get("label_final") or claim_data.get("label")
                if claim_label in claim_label_counts:
                    claim_label_counts[claim_label] += 1

                # Sammle Top Claims (max 3, incorrect first, dann confidence desc)
                if len(top_claims_list) < 3 or claim_label == "incorrect":
                    top_claims_list.append(
                        {
                            "text": (claim_data.get("text") or "")[:200],
                            "label_final": claim_label,
                            "confidence": claim_data.get("confidence"),
                            "gate_reason": claim_data.get("gate_reason"),
                            "selection_reason": claim_data.get("evidence_selection_reason"),
                            "evidence_found": claim_data.get("evidence_found"),
                            "coverage_ok": claim_data.get("coverage_ok"),
                            "coverage_note": claim_data.get("coverage_note"),
                            "evidence_quote_preview": (claim_data.get("evidence_quote") or "")[
                                :200
                            ],
                        }
                    )

            # Sortiere Top Claims: incorrect first, dann confidence desc
            top_claims_list.sort(
                key=lambda x: (
                    0 if x["label_final"] == "incorrect" else 1,
                    -(x.get("confidence") or 0.0),
                )
            )
            top_claims_list = top_claims_list[:3]

            # Top Issue Span
            top_issue_span = None
            if agent_result.issue_spans:
                top_span = agent_result.issue_spans[0]  # Erster Span (repräsentativ)
                top_issue_span = {
                    "message": top_span.message,
                    "severity": top_span.severity,
                    "evidence_found": top_span.evidence_found,
                }

            results.append(
                {
                    "example_id": ex.get("example_id", i),
                    "gt_has_error": has_error,
                    "pred_has_error": pred_has_error,
                    "num_issues": len(agent_result.issue_spans),
                    "effective_issues": effective_issues,
                    "score": agent_result.score,
                    # Zusätzliche Felder für error_cases.jsonl
                    "decision_threshold": decision_threshold,
                    "uncertainty_policy": uncertainty_policy,
                    "num_issue_spans": len(agent_result.issue_spans),
                    "top_issue_span": top_issue_span,
                    "num_claims": len(claims_data),
                    "claim_label_counts": claim_label_counts,
                    "top_claims": top_claims_list,
                }
            )

            # Update metrics
            if pred_has_error and has_error:
                metrics.tp += 1
            elif pred_has_error and not has_error:
                metrics.fp += 1
            elif not pred_has_error and not has_error:
                metrics.tn += 1
            else:
                metrics.fn += 1

        except (ValueError, RuntimeError) as e:
            # Strict Mode Exceptions: Nicht schlucken, sondern weiterwerfen
            if strict_mode and (
                "Schema-Verletzung" in str(e) or "TRIPWIRE" in str(e) or "Strict-Mode" in str(e)
            ):
                print("\n❌ STRICT MODE EXCEPTION (nicht geschluckt):")
                print(f"   {e}")
                print(f"   Example ID: {ex.get('example_id', i)}")
                raise  # Weiterwerfen, nicht schlucken
            # Andere Exceptions: wie bisher behandeln
            print(f"  ⚠️  Error processing example {i}: {e}")
            continue
        except Exception as e:
            # Catch-All für andere Exceptions (nicht Strict Mode)
            print(f"  ⚠️  Error processing example {i}: {e}")
            continue

    # Compute final metrics (BinaryMetrics hat Properties, keine compute() Methode)
    balanced_accuracy = (metrics.recall + metrics.specificity) / 2.0

    # Coverage/Abstention Metriken
    total_claims = num_correct + num_incorrect + num_uncertain
    coverage = (num_correct + num_incorrect) / total_claims if total_claims > 0 else 0.0
    abstention_rate = num_uncertain / total_claims if total_claims > 0 else 0.0

    print()
    print("=" * 80)
    print("Results")
    print("=" * 80)

    # Ground Truth Distribution
    gt_positives = metrics.tp + metrics.fn
    gt_negatives = metrics.tn + metrics.fp
    total_examples = gt_positives + gt_negatives
    positive_rate = (gt_positives / total_examples * 100) if total_examples > 0 else 0.0

    print("\nGround Truth Distribution:")
    print(f"  GT Positives (TP+FN): {gt_positives}")
    print(f"  GT Negatives (TN+FP): {gt_negatives}")
    print(f"  Positiv-Rate: {positive_rate:.1f}%")

    # Stabilitätswarnung
    if gt_negatives < 20:
        print(
            "\n⚠️  WARN: Specificity ist bei <20 Negativen statistisch instabil; nicht überinterpretieren."
        )

    print()
    print("Confusion Matrix:")
    print(f"  TP: {metrics.tp}")
    print(f"  TN: {metrics.tn}")
    print(f"  FP: {metrics.fp}")
    print(f"  FN: {metrics.fn}")
    print()
    print("Metrics:")
    print(f"  Recall: {metrics.recall:.3f}")
    print(f"  Specificity: {metrics.specificity:.3f}")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  F1: {metrics.f1:.3f}")
    print(f"  Balanced Accuracy: {balanced_accuracy:.3f}")
    print()
    print("Coverage/Abstention:")
    print(f"  Total Claims: {total_claims}")
    print(f"  Correct: {num_correct}")
    print(f"  Incorrect: {num_incorrect}")
    print(f"  Uncertain: {num_uncertain}")
    print(f"  Coverage: {coverage:.3f} ({(num_correct + num_incorrect)}/{total_claims})")
    print(f"  Abstention Rate: {abstention_rate:.3f} ({num_uncertain}/{total_claims})")
    print()

    # Claim-Level Evidence Statistics (alle Claims)
    print("Claim-Level Evidence Stats:")
    print(f"  Incorrect Claims mit Evidence: {incorrect_claims_with_evidence}")
    print(
        f"  Incorrect Claims ohne Evidence: {incorrect_claims_without_evidence}  (Expectation: 0)"
    )
    print(f"  Claims mit Evidence (gesamt): {claim_evidence_found_true}")
    print(f"  Claims ohne Evidence (gesamt): {claim_evidence_found_false}")
    print()

    # IssueSpan-Level Evidence Statistics (repräsentativ pro Satz)
    print("IssueSpan-Level Evidence Stats:")
    print(f"  Incorrect Spans ohne Evidence: {incorrect_spans_without_evidence}")
    print(f"  Incorrect Spans mit Evidence: {incorrect_spans_with_evidence}")
    print(f"  Uncertain Spans: {uncertain_spans_count}")
    print(
        "  (Hinweis: Spans sind repräsentativ pro Satz, daher kann Span-Level < Claim-Level sein)"
    )
    print()

    # Counts für selection_reason und gate_reason (aus debug_claims.jsonl)
    if debug_mode:
        from collections import Counter

        selection_reasons = []
        gate_reasons = []
        evidence_found_counts = {"true": 0, "false": 0}

        for debug_claim in debug_claims:
            for claim in debug_claim.get("claims", []):
                if claim.get("evidence_selection_reason"):
                    selection_reasons.append(claim["evidence_selection_reason"])
                if claim.get("gate_reason"):
                    gate_reasons.append(claim["gate_reason"])
                if claim.get("evidence_found") is not None:
                    evidence_found_counts["true" if claim["evidence_found"] else "false"] += 1

        print("Evidence Selection Reasons:")
        for reason, count in Counter(selection_reasons).most_common():
            print(f"  {reason}: {count}")
        print()

        print("Gate Reasons:")
        for reason, count in Counter(gate_reasons).most_common():
            print(f"  {reason}: {count}")
        print()

        print("Evidence Found Counts:")
        print(f"  True: {evidence_found_counts['true']}")
        print(f"  False: {evidence_found_counts['false']}")
        print()

    # Claim-Level Check (primär, da alle Claims erfasst werden)
    if incorrect_claims_without_evidence > 0:
        print(
            f"⚠️  WARNUNG: {incorrect_claims_without_evidence} 'incorrect' Claims ohne Evidence gefunden!"
        )
        print("   → Sollten durch Evidence-Gate zu 'uncertain' downgraded werden")
    else:
        print("✅ Keine 'incorrect' Claims ohne Evidence - Evidence-Gate funktioniert!")

    # Span-Level Check (sekundär, da repräsentativ)
    if incorrect_spans_without_evidence > 0:
        print(
            f"⚠️  INFO: {incorrect_spans_without_evidence} 'incorrect' Spans ohne Evidence (Span-Level, repräsentativ)"
        )

    print()
    print("=" * 80)

    # Save results
    output_dir = ROOT / "results" / "evaluation" / "evidence_gate_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / f"results_{uncertainty_policy}.json"
    with results_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "uncertainty_policy": uncertainty_policy,
                "metrics": {
                    "tp": metrics.tp,
                    "tn": metrics.tn,
                    "fp": metrics.fp,
                    "fn": metrics.fn,
                    "recall": metrics.recall,
                    "specificity": metrics.specificity,
                    "precision": metrics.precision,
                    "f1": metrics.f1,
                    "balanced_accuracy": balanced_accuracy,
                },
                "coverage": {
                    "total_claims": total_claims,
                    "num_correct": num_correct,
                    "num_incorrect": num_incorrect,
                    "num_uncertain": num_uncertain,
                    "coverage": coverage,
                    "abstention_rate": abstention_rate,
                },
                "evidence_stats": {
                    "claim_level": {
                        "incorrect_with_evidence": incorrect_claims_with_evidence,
                        "incorrect_without_evidence": incorrect_claims_without_evidence,
                        "evidence_found_true": claim_evidence_found_true,
                        "evidence_found_false": claim_evidence_found_false,
                    },
                    "span_level": {
                        "incorrect_with_evidence": incorrect_spans_with_evidence,
                        "incorrect_without_evidence": incorrect_spans_without_evidence,
                        "uncertain": uncertain_spans_count,
                    },
                },
                "n": len(results),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"✅ Results saved to: {results_file.relative_to(ROOT)}")

    # Debug: Schreibe debug_claims.jsonl
    if debug_mode:
        debug_file = output_dir / "debug_claims.jsonl"
        with debug_file.open("w", encoding="utf-8") as f:
            for entry in debug_claims:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"✅ Debug claims saved to: {debug_file.relative_to(ROOT)}")

    # FP/FN Debug-Export: error_cases.jsonl
    error_cases = []
    for result in results:
        gt_has_error = result["gt_has_error"]
        pred_has_error = result["pred_has_error"]

        # Nur FP und FN Fälle
        is_fp = (not gt_has_error) and pred_has_error
        is_fn = gt_has_error and (not pred_has_error)

        if is_fp or is_fn:
            error_cases.append(
                {
                    "example_id": result["example_id"],
                    "gt_has_error": gt_has_error,
                    "pred_has_error": pred_has_error,
                    "decision_threshold": result["decision_threshold"],
                    "uncertainty_policy": result["uncertainty_policy"],
                    "effective_issues": result["effective_issues"],
                    "num_issue_spans": result["num_issue_spans"],
                    "top_issue_span": result["top_issue_span"],
                    "num_claims": result["num_claims"],
                    "claim_label_counts": result["claim_label_counts"],
                    "top_claims": result["top_claims"],
                }
            )

    if error_cases:
        error_file = output_dir / "error_cases.jsonl"
        with error_file.open("w", encoding="utf-8") as f:
            for entry in error_cases:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(
            f"✅ Error cases saved to: {error_file.relative_to(ROOT)} ({len(error_cases)} FP/FN cases)"
        )
    else:
        print("ℹ️  No error cases (FP/FN) to export")

    print()


if __name__ == "__main__":
    main()
