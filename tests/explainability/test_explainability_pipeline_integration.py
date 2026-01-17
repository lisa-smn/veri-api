"""
Mini-Integration Test für Explainability in der Pipeline.

Simuliert Pipeline-Ausgabe mit gemockten Agent-Outputs (keine LLMs).
"""

import json
from pathlib import Path

import pytest

from app.models.pydantic import AgentResult, PipelineResult
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


def test_pipeline_result_integration(service, minimal_input):
    """Prüft, dass ExplainabilityService mit PipelineResult funktioniert."""
    # Erstelle PipelineResult-ähnliches Objekt
    factuality = AgentResult(
        name="factuality",
        score=minimal_input["factuality"]["score"],
        explanation=minimal_input["factuality"]["explanation"],
        issue_spans=minimal_input["factuality"]["issue_spans"],
    )
    coherence = AgentResult(
        name="coherence",
        score=minimal_input["coherence"]["score"],
        explanation=minimal_input["coherence"]["explanation"],
        issue_spans=minimal_input["coherence"]["issue_spans"],
    )
    readability = AgentResult(
        name="readability",
        score=minimal_input["readability"]["score"],
        explanation=minimal_input["readability"]["explanation"],
        issue_spans=minimal_input["readability"]["issue_spans"],
    )

    pipeline_result = PipelineResult(
        factuality=factuality,
        coherence=coherence,
        readability=readability,
        overall_score=0.7,
    )

    # Build Explainability
    result = service.build(pipeline_result, minimal_input["summary_text"])

    # Prüfe, dass Result gültig ist
    assert result is not None
    assert result.version == "m9_v1"
    assert len(result.findings) > 0  # Sollte Findings haben


def test_pipeline_result_dict_compatibility(service, minimal_input):
    """Prüft, dass ExplainabilityService auch mit dict-Input funktioniert."""
    # Build mit dict (wie in Pipeline)
    result = service.build(minimal_input, minimal_input["summary_text"])

    assert result is not None
    assert result.version == "m9_v1"


def test_explainability_in_pipeline_result(service, minimal_input):
    """Prüft, dass Explainability in PipelineResult eingebettet werden kann."""
    factuality = AgentResult(
        name="factuality",
        score=minimal_input["factuality"]["score"],
        explanation=minimal_input["factuality"]["explanation"],
        issue_spans=minimal_input["factuality"]["issue_spans"],
    )
    coherence = AgentResult(
        name="coherence",
        score=minimal_input["coherence"]["score"],
        explanation=minimal_input["coherence"]["explanation"],
        issue_spans=minimal_input["coherence"]["issue_spans"],
    )
    readability = AgentResult(
        name="readability",
        score=minimal_input["readability"]["score"],
        explanation=minimal_input["readability"]["explanation"],
        issue_spans=minimal_input["readability"]["issue_spans"],
    )

    pipeline_result = PipelineResult(
        factuality=factuality,
        coherence=coherence,
        readability=readability,
        overall_score=0.7,
        explainability=None,
    )

    # Build Explainability
    explainability = service.build(pipeline_result, minimal_input["summary_text"])

    # Setze in PipelineResult
    pipeline_result.explainability = explainability

    # Prüfe, dass es funktioniert
    assert pipeline_result.explainability is not None
    assert pipeline_result.explainability.version == "m9_v1"
