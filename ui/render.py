"""Rendering-Funktionen für Explainability und Highlights."""

from typing import Any

import streamlit as st


def render_explainability(
    result: dict[str, Any], summary_text: str, explainability_enabled: bool = True
) -> None:
    """
    Rendert Explainability-Ergebnisse als Tabelle + Summary-Highlights.

    Args:
        result: API Response mit Explainability-Daten
        summary_text: Summary-Text für Highlights
        explainability_enabled: Ob Explainability aktiviert ist (default: True)
    """
    # Guardrail: Wenn Explainability deaktiviert ist
    if not explainability_enabled:
        st.info("Explainability disabled. Enable it to see findings and highlights.")
        return

    # Defensive Checks: Wenn keine Daten vorhanden
    if not result:
        st.info("No issues detected above threshold.")
        st.caption("💡 Tip: Load an example with has_error=True or lower Min Severity.")
        return

    if "explainability" not in result:
        st.info("No issues detected above threshold.")
        st.caption("💡 Tip: Load an example with has_error=True or lower Min Severity.")
        return

    exp = result["explainability"]

    # Findings-Tabelle
    findings = exp.get("findings", [])
    if findings:
        st.subheader("Findings")

        # Filter
        col1, col2 = st.columns(2)
        with col1:
            filter_dimension = st.selectbox(
                "Dimension filtern",
                ["Alle"] + list(set(f.get("dimension", "") for f in findings)),
                key="filter_dim",
            )
        with col2:
            filter_severity = st.selectbox(
                "Min. Severity",
                ["Alle", "high", "medium", "low"],
                key="filter_sev",
            )

        # Filter anwenden
        filtered = findings
        if filter_dimension != "Alle":
            filtered = [f for f in filtered if f.get("dimension") == filter_dimension]
        if filter_severity != "Alle":
            severity_order = {"high": 3, "medium": 2, "low": 1}
            min_sev = severity_order[filter_severity]
            filtered = [
                f for f in filtered if severity_order.get(f.get("severity", "low"), 0) >= min_sev
            ]

        # Tabelle
        if filtered:
            import pandas as pd

            table_data = []
            for f in filtered:
                span = f.get("span", {})
                table_data.append(
                    {
                        "Dimension": f.get("dimension", ""),
                        "Severity": f.get("severity", ""),
                        "Message": f.get("message", "")[:100] + "..."
                        if len(f.get("message", "")) > 100
                        else f.get("message", ""),
                        "Start": span.get("start_char") if span else None,
                        "End": span.get("end_char") if span else None,
                        "Confidence": f.get("confidence"),
                    }
                )
            df = pd.DataFrame(table_data)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No issues detected above threshold.")
            st.caption("💡 Tip: Load an example with has_error=True or lower Min Severity.")
    else:
        # Keine Findings vorhanden, aber Explainability-Daten existieren
        st.info("No issues detected above threshold.")
        st.caption("💡 Tip: Load an example with has_error=True or lower Min Severity.")

    # Summary-Highlights (mit Filter)
    if summary_text:
        st.subheader("Summary mit Highlights")
        # Verwende gefilterte Findings falls vorhanden, sonst alle
        filtered_for_highlights = st.session_state.get("filtered_findings", findings)
        render_summary_highlights(summary_text, filtered_for_highlights)


def render_summary_highlights(summary_text: str, findings: list[dict[str, Any]]) -> None:
    """Rendert Summary mit farbigen Markierungen für Spans."""
    if not findings:
        st.text(summary_text)
        return

    # Sammle alle Spans mit Severity und Dimension
    spans_with_severity = []
    for f in findings:
        span = f.get("span")
        if span and span.get("start_char") is not None and span.get("end_char") is not None:
            spans_with_severity.append(
                {
                    "start": span["start_char"],
                    "end": span["end_char"],
                    "severity": f.get("severity", "low"),
                    "dimension": f.get("dimension", ""),
                    "message": f.get("message", ""),
                }
            )

    if not spans_with_severity:
        st.text(summary_text)
        return

    # Preprocessing: Deduplicate und Merge Overlaps
    spans_processed = _deduplicate_and_merge_spans(spans_with_severity)

    # Sortiere: start desc (für reverse insertion), dann severity desc
    severity_order = {"high": 3, "medium": 2, "low": 1}
    spans_processed.sort(key=lambda x: (-x["start"], -severity_order.get(x["severity"], 0)))

    # Rendere mit HTML-Markierungen (von hinten nach vorne, damit Indizes stabil bleiben)
    highlighted_text = summary_text
    for span in spans_processed:
        start = max(0, span["start"])
        end = min(len(highlighted_text), span["end"])

        if start >= end:
            continue

        # HTML-Marker basierend auf Severity
        severity = span["severity"]
        if severity == "high":
            marker_start = '<mark style="background-color: #ffcccc;">'
            marker_end = "</mark>"
        elif severity == "medium":
            marker_start = '<mark style="background-color: #fff4cc;">'
            marker_end = "</mark>"
        else:
            marker_start = '<mark style="background-color: #e6f3ff;">'
            marker_end = "</mark>"

        # Insert von hinten nach vorne (reverse order)
        highlighted_text = (
            highlighted_text[:start]
            + marker_start
            + highlighted_text[start:end]
            + marker_end
            + highlighted_text[end:]
        )

    st.markdown(highlighted_text, unsafe_allow_html=True)

    # Legende
    st.caption("Legende: 🔴 high | 🟡 medium | 🔵 low")


def _deduplicate_and_merge_spans(spans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Dedupliziert identische Spans und merged überlappende Spans.

    Regel:
    - Identische Ranges: behalte einen mit höchster Severity
    - Überlappende Spans: merge zu einem Span mit max Severity
    """
    if not spans:
        return []

    # Sortiere: start asc, end asc, severity desc
    severity_order = {"high": 3, "medium": 2, "low": 1}
    spans_sorted = sorted(
        spans, key=lambda x: (x["start"], x["end"], -severity_order.get(x["severity"], 0))
    )

    merged = []
    for span in spans_sorted:
        start = span["start"]
        end = span["end"]
        severity = span["severity"]
        dimensions = {span["dimension"]}

        # Prüfe ob mit vorherigen Spans überlappt
        overlap_found = False
        for i, existing in enumerate(merged):
            ex_start = existing["start"]
            ex_end = existing["end"]

            # Überlappung: start < ex_end und end > ex_start
            if start < ex_end and end > ex_start:
                # Merge: erweitere Range und setze max Severity
                merged[i] = {
                    "start": min(start, ex_start),
                    "end": max(end, ex_end),
                    "severity": max(
                        severity, existing["severity"], key=lambda s: severity_order.get(s, 0)
                    ),
                    "dimension": existing.get("dimension", ""),  # Behalte erste Dimension
                    "message": existing.get("message", ""),
                }
                overlap_found = True
                break

        if not overlap_found:
            # Keine Überlappung: füge hinzu
            merged.append(
                {
                    "start": start,
                    "end": end,
                    "severity": severity,
                    "dimension": span["dimension"],
                    "message": span["message"],
                }
            )

    return merged


def render_scores(result: dict[str, Any]) -> None:
    """Rendert Scores pro Dimension."""
    st.subheader("Scores")

    overall = result.get("overall_score", 0.0)
    st.metric("Overall Score", f"{overall:.3f}")

    col1, col2, col3 = st.columns(3)

    with col1:
        fact = result.get("factuality", {})
        st.metric("Factuality", f"{fact.get('score', 0.0):.3f}", help=fact.get("explanation", ""))

    with col2:
        coh = result.get("coherence", {})
        st.metric("Coherence", f"{coh.get('score', 0.0):.3f}", help=coh.get("explanation", ""))

    with col3:
        read = result.get("readability", {})
        st.metric("Readability", f"{read.get('score', 0.0):.3f}", help=read.get("explanation", ""))


def render_comparison(result: dict[str, Any], summary_text: str) -> None:
    """
    Rendert Vergleich: Agent vs LLM-as-a-Judge vs Classical Baseline.

    Args:
        result: API Response mit Scores
        summary_text: Summary-Text für Baseline-Berechnung
    """
    import baselines

    st.caption("Vergleich mit alternativen Methoden (nur für Demo, keine Bootstrap-CIs)")

    col1, col2, col3 = st.columns(3)

    # Spalte 1: Agent (System Output)
    with col1:
        st.markdown("**Agent**")
        st.caption("System output", help="Agent-basierte Bewertung (semantisch)")

        fact_agent = result.get("factuality", {}).get("score")
        coh_agent = result.get("coherence", {}).get("score")
        read_agent = result.get("readability", {}).get("score")

        if fact_agent is not None:
            st.metric("Factuality", f"{fact_agent:.3f}")
        else:
            st.metric("Factuality", "N/A")

        if coh_agent is not None:
            st.metric("Coherence", f"{coh_agent:.3f}")
        else:
            st.metric("Coherence", "N/A")

        if read_agent is not None:
            st.metric("Readability", f"{read_agent:.3f}")
        else:
            st.metric("Readability", "N/A")

    # Spalte 2: LLM-as-a-Judge
    with col2:
        st.markdown("**LLM-as-a-Judge**")
        st.caption("LLM judge (optional)", help="LLM-basierte Bewertung als Baseline")

        # Prüfe ob Judge-Daten im Response vorhanden sind
        judge_data = result.get("judge", {})
        judge_error = result.get("judge_error")
        judge_available = result.get("judge_available", True)

        if judge_error:
            st.warning(f"⚠️ Judge error: {judge_error}")
            st.caption("Agent scores are still available")

        if judge_data and judge_available:
            # Factuality Judge (binary verdict + confidence)
            fact_judge = judge_data.get("factuality", {})
            if fact_judge:
                error_present = fact_judge.get("error_present")
                confidence = fact_judge.get("confidence")
                if error_present is not None:
                    verdict_text = "Error detected" if error_present else "No error"
                    st.metric(
                        "Factuality",
                        verdict_text,
                        help=f"Confidence: {confidence:.3f}" if confidence is not None else None,
                    )
                elif confidence is not None:
                    st.metric("Factuality", f"{confidence:.3f}", help="Confidence score")
                else:
                    st.metric("Factuality", "N/A")
            else:
                st.metric("Factuality", "N/A")

            # Coherence Judge (optional, falls vorhanden)
            coh_judge = judge_data.get("coherence", {}).get("score")
            if coh_judge is not None:
                st.metric("Coherence", f"{coh_judge:.3f}")
            else:
                st.metric("Coherence", "N/A")

            # Readability Judge (optional, falls vorhanden)
            read_judge = judge_data.get("readability", {}).get("score")
            if read_judge is not None:
                st.metric("Readability", f"{read_judge:.3f}")
            else:
                st.metric("Readability", "N/A")
        else:
            # Keine Judge-Daten verfügbar
            st.info("Not available in /verify response")
            st.caption("💡 Enable 'Run LLM Judge' in Advanced Options to see judge verdicts")
            st.metric("Factuality", "N/A")
            st.metric("Coherence", "N/A")
            st.metric("Readability", "N/A")

    # Spalte 3: Classical Baseline
    with col3:
        st.markdown("**Classical Baseline**")
        st.caption("Formula / heuristic", help="Statistische Formeln (keine LLM)")

        # Readability: Flesch
        read_baseline = baselines.compute_readability_baseline(summary_text)
        if read_baseline.get("flesch") is not None:
            flesch_norm = read_baseline["flesch"]
            flesch_raw = read_baseline["raw_flesch"]
            st.metric(
                "Readability (Flesch)", f"{flesch_norm:.3f}", help=f"Raw: {flesch_raw:.1f}/100"
            )
        else:
            st.metric("Readability (Flesch)", "N/A")

        # Coherence: N/A (keine zuverlässige klassische Baseline)
        st.metric("Coherence", "N/A", help="No reliable classical baseline")
        st.caption("(no formula available)")

        # Factuality: N/A (keine klassische Baseline)
        st.metric("Factuality", "N/A", help="No classical baseline")
        st.caption("(requires semantic understanding)")
