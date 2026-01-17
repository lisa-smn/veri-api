import pytest

from app.llm.fake_client import FakeLLMClient
from app.services.agents.factuality.factuality_agent import FactualityAgent


def test_factuality_agent_does_not_penalize_meta_readability_sentence():
    llm = FakeLLMClient()
    agent = FactualityAgent(llm)

    article = "Kurzer Artikeltext. Er ist nur dazu da, dass die Pipeline etwas Kontext hat."
    summary = (
        "Das ist ein unfassbar langer Satz, der immer weitergeht, mit vielen Kommas und Einschüben."
    )

    res = agent.run(
        article_text=article, summary_text=summary, meta={"test_case": "meta_readability"}
    )

    # Neue Semantik: Wenn keine überprüfbaren Faktenbehauptungen extrahiert werden können,
    # liefert der Agent einen "unknown/empty check" Score (EMPTY_CHECK_SCORE = 0.5).
    assert res.score == pytest.approx(0.5, abs=1e-6)
    assert res.details is not None
    assert res.details.get("claims") == []
    assert res.details.get("num_incorrect") == 0

    assert res.name == "factuality"
    assert res.issue_spans == []  # keine falschen Fakten, also keine Spans

    # Details sollten transparent zeigen, dass Sätze geskippt wurden
    assert res.details is not None
    assert "skipped_sentences" in res.details
    assert isinstance(res.details["skipped_sentences"], list)
    assert len(res.details["skipped_sentences"]) >= 1
