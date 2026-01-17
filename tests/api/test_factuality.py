"""
Dieser Test schickt eine Zusammenfassung an die API, die bewusst falsch ist
(„Paris“ statt „Berlin“), und prüft: erkennt das System den Fehler?

Er zeigt, dass /verify nicht nur „irgendwas“ antwortet,
sondern den Fehler klar markiert (Score runter, falscher Claim in den Details
und eine markierte Stelle im Text als IssueSpan).

"""

from fastapi.testclient import TestClient

from app.server import app


def test_factuality_endpoint_detects_issue_span():
    client = TestClient(app)

    payload = {
        "dataset": "custom",
        "article_text": "Lisa wohnt in Berlin und studiert Softwareentwicklung.",
        "summary_text": "Lisa wohnt in Paris und studiert Medizin.",
        "llm_model": "gpt-4o-mini",
    }

    response = client.post("/verify", json=payload)

    # Debug-Ausgaben (nur falls du es wirklich brauchst; sonst raus damit)
    print("STATUS:", response.status_code)
    print("BODY:", response.text)

    assert response.status_code == 200

    data = response.json()
    f = data["factuality"]

    assert f["score"] < 1.0

    # neuer Key aus deinem Agent: num_incorrect statt num_errors
    assert f["details"]["num_incorrect"] >= 1

    # optional, aber sinnvoll: API liefert issue_spans
    assert "issue_spans" in f
    assert len(f["issue_spans"]) >= 1

    # Neue Struktur: Labels hängen an details["claims"]; sentences enthalten nur Rohtext.
    claims = f["details"].get("claims", [])
    incorrect = [c for c in claims if c.get("label") == "incorrect"]
    assert len(incorrect) >= 1
    assert any("Paris" in (c.get("sentence") or c.get("text") or "") for c in incorrect)

    # Optional: issue_spans enthalten ebenfalls den Claim-Text
    assert any("Paris" in (sp.get("message") or "") for sp in f.get("issue_spans", []))
