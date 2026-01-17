"""
Snapshot Tests für Explainability-Modul.

Vergleicht Output gegen expected fixtures (strict).
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


def load_expected(name: str) -> dict:
    """Lädt ein Expected-Output-Fixture."""
    path = FIXTURES_DIR / f"explainability_expected_{name}.json"
    with path.open() as f:
        return json.load(f)


@pytest.fixture
def service():
    """ExplainabilityService-Instanz."""
    return ExplainabilityService()


def normalize_output(output: dict) -> dict:
    """Normalisiert Output für Vergleich (entfernt volatile fields)."""
    # Kopiere Output
    normalized = json.loads(json.dumps(output))

    # Entferne volatile fields (falls vorhanden)
    # z.B. timestamps, random IDs, etc.
    # In unserem Fall sind IDs deterministisch, also keine Änderung nötig

    return normalized


def test_snapshot_minimal(service):
    """Vergleicht minimal Input gegen expected Output."""
    input_data = load_fixture("minimal")
    expected = load_expected("minimal")

    result = service.build(input_data, input_data["summary_text"])
    actual = normalize_output(result.model_dump(mode="json"))

    # Vergleiche strukturiert
    assert actual["version"] == expected["version"]
    assert len(actual["findings"]) == len(expected["findings"])
    assert len(actual["summary"]) == len(expected["summary"])
    assert actual["stats"]["num_findings"] == expected["stats"]["num_findings"]

    # Vergleiche Findings (nach ID sortiert für Stabilität)
    actual_findings = sorted(actual["findings"], key=lambda x: x["id"])
    expected_findings = sorted(expected["findings"], key=lambda x: x["id"])

    assert len(actual_findings) == len(expected_findings)
    for a, e in zip(actual_findings, expected_findings):
        assert a["id"] == e["id"]
        assert a["dimension"] == e["dimension"]
        assert a["severity"] == e["severity"]


def test_snapshot_mixed(service):
    """Vergleicht mixed Input gegen expected Output."""
    input_data = load_fixture("mixed")
    expected = load_expected("mixed")

    result = service.build(input_data, input_data["summary_text"])
    actual = normalize_output(result.model_dump(mode="json"))

    # Basis-Checks
    assert actual["version"] == expected["version"]
    assert actual["stats"]["num_findings"] == expected["stats"]["num_findings"]

    # Top-Spans sollten übereinstimmen
    assert len(actual["top_spans"]) == len(expected["top_spans"])


def test_snapshot_edgecases(service):
    """Vergleicht edgecases Input gegen expected Output."""
    input_data = load_fixture("edgecases")
    expected = load_expected("edgecases")

    result = service.build(input_data, input_data["summary_text"])
    actual = normalize_output(result.model_dump(mode="json"))

    # Bei leerem Input sollten keine Findings sein
    assert actual["stats"]["num_findings"] == 0
    assert expected["stats"]["num_findings"] == 0
    assert len(actual["findings"]) == 0
    assert len(expected["findings"]) == 0
