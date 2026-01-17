"""
Determinism Tests für Explainability-Modul.

Validiert:
- Gleicher Input → exakt gleicher Output (deep equality)
- Listen sind stabil sortiert
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


def test_deterministic_output_same_input(service, minimal_input):
    """Prüft, dass identischer Input exakt identischen Output produziert."""
    result1 = service.build(minimal_input, minimal_input["summary_text"])
    result2 = service.build(minimal_input, minimal_input["summary_text"])

    # Deep equality check
    assert result1.model_dump() == result2.model_dump(), "Output not deterministic"


def test_deterministic_output_multiple_runs(service, minimal_input):
    """Prüft, dass 10x identischer Input exakt identischen Output produziert."""
    results = []
    for _ in range(10):
        result = service.build(minimal_input, minimal_input["summary_text"])
        results.append(result.model_dump())

    # Alle sollten identisch sein
    first = results[0]
    for i, result in enumerate(results[1:], 1):
        assert result == first, f"Run {i + 1} differs from first run"


def test_findings_stable_sorting(service, mixed_input):
    """Prüft, dass Findings stabil sortiert sind."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    # Findings sollten nach Ranking sortiert sein (höchste Priorität zuerst)
    if len(result.findings) > 1:
        finding_ids = [f.id for f in result.findings]
        # IDs sollten in stabiler Reihenfolge sein
        result2 = service.build(mixed_input, mixed_input["summary_text"])
        finding_ids2 = [f.id for f in result2.findings]
        assert finding_ids == finding_ids2, "Finding order not stable"


def test_top_spans_stable_sorting(service, mixed_input):
    """Prüft, dass Top-Spans stabil sortiert sind."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    if len(result.top_spans) > 1:
        top_span_ids = [
            (ts.span.start_char, ts.span.end_char, ts.dimension.value) for ts in result.top_spans
        ]
        result2 = service.build(mixed_input, mixed_input["summary_text"])
        top_span_ids2 = [
            (ts.span.start_char, ts.span.end_char, ts.dimension.value) for ts in result2.top_spans
        ]
        assert top_span_ids == top_span_ids2, "Top spans order not stable"


def test_finding_ids_deterministic(service, minimal_input):
    """Prüft, dass Finding-IDs deterministisch sind."""
    result1 = service.build(minimal_input, minimal_input["summary_text"])
    result2 = service.build(minimal_input, minimal_input["summary_text"])

    ids1 = [f.id for f in result1.findings]
    ids2 = [f.id for f in result2.findings]

    assert ids1 == ids2, "Finding IDs not deterministic"


def test_different_inputs_produce_different_outputs(service):
    """Prüft, dass unterschiedliche Inputs unterschiedliche Outputs produzieren."""
    minimal = load_fixture("minimal")
    mixed = load_fixture("mixed")

    result1 = service.build(minimal, minimal["summary_text"])
    result2 = service.build(mixed, mixed["summary_text"])

    # Sollten unterschiedlich sein
    assert result1.model_dump() != result2.model_dump(), "Different inputs produced same output"


def test_by_dimension_stable_sorting(service, mixed_input):
    """Prüft, dass by_dimension Findings stabil sortiert sind."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    for dimension in result.by_dimension:
        finding_ids = [f.id for f in result.by_dimension[dimension]]
        result2 = service.build(mixed_input, mixed_input["summary_text"])
        finding_ids2 = [f.id for f in result2.by_dimension[dimension]]
        assert finding_ids == finding_ids2, f"by_dimension[{dimension}] order not stable"
