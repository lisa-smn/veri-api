"""
Dieser Test überprüft den /verify-Endpoint auf korrekte Integration der
Coherence- und Readability-Agenten.

Der Test patcht die Methode `VerificationService.verify`, um eine kontrollierte
PipelineResult-Instanz zurückzugeben und externe Abhängigkeiten wie Datenbank
oder LLM auszuschließen. Dadurch wird ausschließlich das Routing, die
Serialisierung sowie die Response-Struktur des Endpoints getestet.

Geprüft wird insbesondere:
- dass der Endpoint erfolgreich antwortet (HTTP 200)
- dass Coherence- und Readability-Ergebnisse im Response enthalten sind
- dass Scores und IssueSpans korrekt serialisiert werden

Der Test stellt sicher, dass der Readability-Agent vollständig in die API
integriert ist und konsistent mit bestehenden Agenten (z.B. Coherence)
zurückgegeben wird.

"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from app.api.routes import router
from app.models.pydantic import AgentResult, IssueSpan, PipelineResult


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_verify_returns_coherence_and_readability(monkeypatch, client):
    """
    API-Test für /verify:
    - patcht VerificationService.verify
    - prüft, dass coherence + readability korrekt zurückgegeben werden
    """

    # --- Dummy-AgentResults ---
    coherence_result = AgentResult(
        name="coherence",
        score=0.3,
        explanation="Coherence problem.",
        issue_spans=[
            IssueSpan(
                start_char=0,
                end_char=10,
                message="Widerspruch",
                severity="high",
            )
        ],
        details={"num_issues": 1},
    )

    readability_result = AgentResult(
        name="readability",
        score=0.2,
        explanation="Readability problem.",
        issue_spans=[
            IssueSpan(
                start_char=5,
                end_char=20,
                message="Sehr langer Satz",
                severity="medium",
            )
        ],
        details={"num_issues": 1},
    )

    factuality_result = AgentResult(
        name="factuality",
        score=0.9,
        explanation="Mostly correct.",
        issue_spans=[],
        details={},
    )

    pipeline_result = PipelineResult(
        factuality=factuality_result,
        coherence=coherence_result,
        readability=readability_result,
        overall_score=(0.9 + 0.3 + 0.2) / 3.0,
    )

    # --- Patch VerificationService.verify ---
    def fake_verify(self, req, db):
        return 123, pipeline_result  # run_id, PipelineResult

    from app.services.verification_service import VerificationService

    monkeypatch.setattr(VerificationService, "verify", fake_verify)

    # --- Call API ---
    resp = client.post(
        "/verify",
        json={
            "article_text": "Some article text.",
            "summary_text": "A very bad, unreadable summary.",
            "meta": {},
        },
    )

    assert resp.status_code == 200
    data = resp.json()

    # --- Assertions ---
    assert data["run_id"] == 123
    assert "overall_score" in data

    assert "coherence" in data
    assert data["coherence"]["score"] == 0.3
    assert len(data["coherence"]["issue_spans"]) == 1

    assert "readability" in data
    assert data["readability"]["score"] == 0.2
    assert len(data["readability"]["issue_spans"]) == 1
