"""
Contract Tests für Explainability-Modul.

Validiert:
- Output hat alle required fields
- Typen stimmen
- Ranges stimmen (score in [0,1] etc.)
- Jede Dimension ist vorhanden (auch wenn leer)
"""

import json
from pathlib import Path

import pytest

from app.services.explainability.explainability_models import (
    Dimension,
    ExplainabilityResult,
    Finding,
)
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


@pytest.fixture
def edgecases_input():
    """Edgecases Input-Fixture."""
    return load_fixture("edgecases")


def test_output_has_all_required_fields(service, minimal_input):
    """Prüft, dass Output alle required fields hat."""
    result = service.build(minimal_input, minimal_input["summary_text"])

    assert isinstance(result, ExplainabilityResult)
    assert hasattr(result, "summary")
    assert hasattr(result, "findings")
    assert hasattr(result, "by_dimension")
    assert hasattr(result, "top_spans")
    assert hasattr(result, "stats")
    assert hasattr(result, "version")


def test_output_types(service, minimal_input):
    """Prüft, dass Typen stimmen."""
    result = service.build(minimal_input, minimal_input["summary_text"])

    assert isinstance(result.summary, list)
    assert all(isinstance(s, str) for s in result.summary)
    assert isinstance(result.findings, list)
    assert isinstance(result.by_dimension, dict)
    assert isinstance(result.top_spans, list)
    assert isinstance(result.stats.num_findings, int)
    assert isinstance(result.version, str)


def test_every_dimension_present(service, minimal_input):
    """Prüft, dass jede Dimension vorhanden ist (auch wenn leer)."""
    result = service.build(minimal_input, minimal_input["summary_text"])

    assert Dimension.factuality in result.by_dimension
    assert Dimension.coherence in result.by_dimension
    assert Dimension.readability in result.by_dimension

    # Auch wenn leer, muss es eine Liste sein
    assert isinstance(result.by_dimension[Dimension.factuality], list)
    assert isinstance(result.by_dimension[Dimension.coherence], list)
    assert isinstance(result.by_dimension[Dimension.readability], list)


def test_finding_required_fields(service, minimal_input):
    """Prüft, dass jedes Finding alle required fields hat."""
    result = service.build(minimal_input, minimal_input["summary_text"])

    for finding in result.findings:
        assert isinstance(finding, Finding)
        assert isinstance(finding.id, str)
        assert finding.id  # nicht leer
        assert isinstance(finding.dimension, Dimension)
        # Severity ist Literal, kann nicht mit isinstance geprüft werden
        assert finding.severity in ("low", "medium", "high")
        assert isinstance(finding.message, str)
        assert finding.message  # nicht leer
        assert isinstance(finding.evidence, list)
        assert isinstance(finding.source, dict)


def test_stats_ranges(service, minimal_input):
    """Prüft, dass Stats in gültigen Ranges sind."""
    result = service.build(minimal_input, minimal_input["summary_text"])

    stats = result.stats
    assert stats.num_findings >= 0
    assert stats.num_high_severity >= 0
    assert stats.num_medium_severity >= 0
    assert stats.num_low_severity >= 0
    assert stats.coverage_chars >= 0
    assert 0.0 <= stats.coverage_ratio <= 1.0

    # Summe der Severities sollte gleich num_findings sein
    assert (
        stats.num_high_severity + stats.num_medium_severity + stats.num_low_severity
        == stats.num_findings
    )


def test_empty_input_produces_valid_output(service, edgecases_input):
    """Prüft, dass leere Inputs gültigen Output produzieren."""
    result = service.build(edgecases_input, edgecases_input["summary_text"])

    assert isinstance(result, ExplainabilityResult)
    assert isinstance(result.summary, list)
    assert len(result.summary) > 0  # Executive Summary sollte existieren
    assert isinstance(result.findings, list)
    assert isinstance(result.stats.num_findings, int)
    assert result.stats.num_findings == 0  # Keine Findings bei leerem Input


def test_span_validity(service, minimal_input):
    """Prüft, dass Spans gültig sind (wenn vorhanden)."""
    result = service.build(minimal_input, minimal_input["summary_text"])

    for finding in result.findings:
        if finding.span:
            assert finding.span.start_char >= 0
            assert finding.span.end_char >= finding.span.start_char
            assert finding.span.end_char <= len(minimal_input["summary_text"])


def test_top_spans_structure(service, minimal_input):
    """Prüft, dass Top-Spans korrekt strukturiert sind."""
    result = service.build(minimal_input, minimal_input["summary_text"])

    # Nur prüfen, wenn Top-Spans vorhanden sind
    if not result.top_spans:
        pytest.skip("No top spans in result")

    for top_span in result.top_spans:
        assert top_span.span is not None
        assert isinstance(top_span.dimension, Dimension)
        assert isinstance(
            top_span.severity, str
        )  # Severity ist Literal, aber als str repräsentiert
        assert top_span.severity in ("low", "medium", "high")
        assert isinstance(top_span.finding_id, str)
        assert isinstance(top_span.rank_score, float)
        assert top_span.rank_score >= 0
