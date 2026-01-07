import pytest

from app.models.pydantic import AgentResult
from app.services.agents.coherence.coherence_agent import CoherenceAgent
from app.services.agents.coherence.coherence_models import CoherenceIssue


class DummyCoherenceEvaluator:
    def evaluate(self, article_text: str, summary_text: str):
        issues = [
            CoherenceIssue(
                type="CONTRADICTION",
                severity="high",
                summary_span="This is bad",
                comment="Widerspruch im Satz.",
            )
        ]
        return 0.2, issues, "Kurz erklärt."


def test_coherence_agent_run_returns_agentresult():
    agent = CoherenceAgent(llm_client=None, evaluator=DummyCoherenceEvaluator())  # llm_client wird nicht genutzt
    res = agent.run("article", "This is bad. Another sentence.", meta=None)

    assert isinstance(res, AgentResult)
    assert res.name == "coherence"
    assert 0.0 <= res.score <= 1.0
    assert res.explanation

    # wichtig: Pydantic serialisiert als issue_spans
    assert isinstance(res.issue_spans, list)
    assert len(res.issue_spans) == 1
    span = res.issue_spans[0]
    assert span.message
    assert span.severity == "high"


def test_coherence_agent_span_mapping_sets_offsets_when_found():
    agent = CoherenceAgent(llm_client=None, evaluator=DummyCoherenceEvaluator())
    summary = "This is bad. Another sentence."
    res = agent.run("article", summary, meta=None)

    span = res.issue_spans[0]
    assert span.start_char is not None
    assert span.end_char is not None
    assert summary[span.start_char:span.end_char] == "This is bad"

