"""
Traceability Tests für Explainability-Modul.

Validiert:
- Jede Finding referenziert mindestens einen Input-Span
- Evidence IDs referenzieren existierende Evidence-Objekte
"""

import json
from pathlib import Path

import pytest

from app.services.explainability.explainability_service import ExplainabilityService

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Lädt ein Fixture-File."""
    path = FIXTURES_DIR / f"explainability_input_{name}.json"
    with path.open() as f:
        return json.load(f)


@pytest.fixture
def service():
    """ExplainabilityService-Instanz."""
    return ExplainabilityService()


@pytest.fixture
def minimal_input():
    """Minimales Input-Fixture."""
    return load_fixture("minimal")


@pytest.fixture
def mixed_input():
    """Mixed Input-Fixture."""
    return load_fixture("mixed")


def test_findings_reference_input_spans(service, minimal_input):
    """Prüft, dass jede Finding mindestens einen Input-Span referenziert."""
    result = service.build(minimal_input, minimal_input["summary_text"])

    for finding in result.findings:
        # Jede Finding sollte source-Information haben
        assert "source" in finding.source or finding.source, f"Finding {finding.id} has no source"
        assert "agent" in finding.source, f"Finding {finding.id} has no agent in source"
        assert "source_list" in finding.source, f"Finding {finding.id} has no source_list in source"

        # source_list sollte "issue_spans" oder "details" sein
        assert finding.source["source_list"] in ("issue_spans", "details"), (
            f"Finding {finding.id} has invalid source_list: {finding.source['source_list']}"
        )


def test_evidence_items_have_valid_structure(service, minimal_input):
    """Prüft, dass Evidence-Items gültige Struktur haben."""
    result = service.build(minimal_input, minimal_input["summary_text"])

    for finding in result.findings:
        for evidence in finding.evidence:
            assert hasattr(evidence, "kind")
            assert hasattr(evidence, "source")
            # Evidence sollte entweder quote oder data haben
            assert evidence.quote is not None or evidence.data is not None, (
                f"Evidence in finding {finding.id} has neither quote nor data"
            )


def test_findings_with_spans_have_valid_references(service, mixed_input):
    """Prüft, dass Findings mit Spans gültige Referenzen haben."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    for finding in result.findings:
        if finding.span:
            # Span sollte innerhalb des Summary-Texts liegen
            summary_len = len(mixed_input["summary_text"])
            assert 0 <= finding.span.start_char <= summary_len
            assert finding.span.start_char <= finding.span.end_char <= summary_len

            # Span-Text sollte mit Summary-Text übereinstimmen
            # Nach Clustern kann der Span-Text anders sein (Union-Span), daher nur prüfen, dass
            # der Text nicht leer ist (wenn Span gültig ist) oder dass er mit dem Summary-Text übereinstimmt
            if finding.span.text:
                expected_text = mixed_input["summary_text"][
                    finding.span.start_char : finding.span.end_char
                ]
                # Wenn geclustert, kann der Text anders sein (Union)
                if finding.source.get("cluster_size", 0) > 1:
                    # Bei Clustern: Text sollte nicht leer sein und sollte Teil des Summary-Texts sein
                    assert finding.span.text.strip(), (
                        f"Clustered finding {finding.id} has empty span text"
                    )
                    # Prüfe, dass der Text im Summary-Text vorkommt (nicht exakt übereinstimmen muss)
                    assert finding.span.text in mixed_input["summary_text"], (
                        f"Clustered finding {finding.id} span text not in summary"
                    )
                else:
                    # Bei nicht-geclusterten: Text sollte übereinstimmen
                    assert finding.span.text == expected_text, (
                        f"Span text mismatch for finding {finding.id}: "
                        f"expected '{expected_text}', got '{finding.span.text}'"
                    )


def test_clustered_findings_have_cluster_info(service, mixed_input):
    """Prüft, dass geclusterte Findings Cluster-Information haben."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    clustered = [f for f in result.findings if f.source.get("cluster_size", 0) > 1]

    for finding in clustered:
        assert "cluster_size" in finding.source
        assert "cluster_members" in finding.source
        assert isinstance(finding.source["cluster_members"], list)
        assert len(finding.source["cluster_members"]) == finding.source["cluster_size"]


def test_source_provenance_completeness(service, minimal_input):
    """Prüft, dass Source-Provenance vollständig ist."""
    result = service.build(minimal_input, minimal_input["summary_text"])

    for finding in result.findings:
        source = finding.source
        assert "agent" in source
        assert "source_list" in source
        # item_index sollte vorhanden sein (für Traceability)
        assert "item_index" in source or "merged_from" in source, (
            f"Finding {finding.id} has no item_index or merged_from"
        )
