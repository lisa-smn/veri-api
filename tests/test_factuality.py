from fastapi.testclient import TestClient
from app.server import app


def test_factuality_endpoint_detects_error():
    client = TestClient(app)

    payload = {
        "dataset": "custom",
        "article_text": "Lisa wohnt in Berlin und studiert Softwareentwicklung.",
        "summary_text": "Lisa wohnt in Paris und studiert Medizin.",
        "llm_model": "gpt-4o-mini"
    }

    response = client.post("/verify", json=payload)

    # Debug-Ausgaben:
    print("STATUS:", response.status_code)
    print("BODY:", response.text)

    assert response.status_code == 200

    data = response.json()
    f = data["factuality"]

    assert f["score"] < 1.0
    assert f["details"]["num_errors"] >= 1

    sentences = f["details"]["sentences"]
    error_sents = [s for s in sentences if s["label"] == "incorrect"]
    assert any("Paris" in s["sentence"] for s in error_sents)
