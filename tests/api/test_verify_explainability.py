from fastapi.testclient import TestClient
from app.server import app


def test_verify_returns_explainability():
    client = TestClient(app)

    payload = {
        "dataset": "custom",
        "article_text": "Berlin hat 3.7 Millionen Einwohner.",
        "summary_text": "Berlin hat 7.3 Millionen Einwohner.",
        "llm_model": "gpt-4o-mini",
    }

    r = client.post("/verify", json=payload)
    assert r.status_code == 200, r.text

    data = r.json()
    assert "explainability" in data
    assert data["explainability"] is not None

    exp = data["explainability"]
    assert exp["version"] == "m9_v1"
    assert "findings" in exp
    assert "top_spans" in exp
