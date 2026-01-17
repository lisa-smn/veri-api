"""
Aggregation Tests für Explainability-Modul.

Validiert:
- Counts/Stats stimmen
- Top-Spans Auswahl korrekt (höchste rank_score)
- Dedupe/Clustern funktioniert
"""

import json
from pathlib import Path

import pytest

from app.services.explainability.explainability_models import Dimension
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
def mixed_input():
    """Mixed Input-Fixture (hat überlappende Spans)."""
    return load_fixture("mixed")


def test_stats_counts_match_findings(service, mixed_input):
    """Prüft, dass Stats-Counts mit Findings übereinstimmen."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    # Counts pro Severity
    high_count = sum(1 for f in result.findings if f.severity == "high")
    medium_count = sum(1 for f in result.findings if f.severity == "medium")
    low_count = sum(1 for f in result.findings if f.severity == "low")

    assert result.stats.num_high_severity == high_count
    assert result.stats.num_medium_severity == medium_count
    assert result.stats.num_low_severity == low_count
    assert result.stats.num_findings == len(result.findings)


def test_by_dimension_counts_match(service, mixed_input):
    """Prüft, dass by_dimension Counts mit Findings übereinstimmen."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    for dimension in Dimension:
        dim_findings = [f for f in result.findings if f.dimension == dimension]
        assert len(result.by_dimension[dimension]) == len(dim_findings)


def test_top_spans_sorted_by_rank_score(service, mixed_input):
    """Prüft, dass Top-Spans nach rank_score sortiert sind (absteigend)."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    if len(result.top_spans) > 1:
        scores = [ts.rank_score for ts in result.top_spans]
        assert scores == sorted(scores, reverse=True)


def test_top_spans_no_duplicates(service, mixed_input):
    """Prüft, dass Top-Spans keine Duplikate haben (gleiche start_char, end_char, dimension)."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    seen = set()
    for top_span in result.top_spans:
        key = (
            top_span.span.start_char,
            top_span.span.end_char,
            top_span.dimension.value,
        )
        assert key not in seen, f"Duplicate top_span: {key}"
        seen.add(key)


def test_top_spans_limit(service, mixed_input):
    """Prüft, dass Top-Spans das Limit (default: 5) nicht überschreiten."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    assert len(result.top_spans) <= service.top_k


def test_dedupe_removes_identical_ids(service, mixed_input):
    """Prüft, dass Deduplizierung identische IDs entfernt."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    finding_ids = [f.id for f in result.findings]
    assert len(finding_ids) == len(set(finding_ids)), "Duplicate finding IDs found"


def test_clustering_merges_overlapping_spans(service, mixed_input):
    """Prüft, dass überlappende Spans (innerhalb derselben Dimension) geclustert werden."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    # Prüfe, ob es Findings mit cluster_size gibt (indiziert Clustering)
    clustered = [f for f in result.findings if f.source.get("cluster_size", 0) > 1]

    # Bei mixed_input sollten überlappende Spans in coherence/readability geclustert sein
    # (coherence: 75-95 und 75-100 überlappen, readability: 120-160 und 120-165 überlappen)
    assert len(clustered) > 0, "Expected clustered findings for overlapping spans"


def test_findings_sorted_by_rank(service, mixed_input):
    """Prüft, dass Findings nach Ranking sortiert sind (höchste Priorität zuerst)."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    if len(result.findings) > 1:
        # Berechne Scores für alle Findings
        def score(f):
            sev_w = service.weights.severity[f.severity]
            dim_w = service.weights.dimension[f.dimension]
            span_len = 1
            if f.span:
                span_len = max(1, f.span.end_char - f.span.start_char)
            span_w = 1.0 + __import__("math").log(1.0 + span_len)
            return sev_w * dim_w * span_w

        scores = [score(f) for f in result.findings]
        # Sollte absteigend sortiert sein
        assert scores == sorted(scores, reverse=True), "Findings not sorted by rank"


def test_coverage_calculation(service, mixed_input):
    """Prüft, dass Coverage korrekt berechnet wird."""
    result = service.build(mixed_input, mixed_input["summary_text"])

    summary_len = len(mixed_input["summary_text"])
    assert result.stats.coverage_chars <= summary_len
    assert result.stats.coverage_ratio == result.stats.coverage_chars / summary_len
