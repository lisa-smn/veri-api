from app.models.pydantic import AgentResult
from app.services.agents.readability.readability_agent import ReadabilityAgent
from app.services.agents.readability.readability_models import ReadabilityIssue


class DummyReadabilityEvaluator:
    def evaluate(self, article_text: str, summary_text: str):
        issues = [
            ReadabilityIssue(
                type="LONG_SENTENCE",
                severity="medium",
                summary_span="This sentence is extremely long and painful to read",
                comment="Sehr langer Satz, erschwert Lesefluss.",
            )
        ]
        return 0.4, issues, ""  # explanation leer -> Agent-Fallback wird getestet


def test_readability_agent_run_returns_agentresult_and_fallback_explanation():
    agent = ReadabilityAgent(llm_client=None, evaluator=DummyReadabilityEvaluator())
    summary = "This sentence is extremely long and painful to read. Short."
    res = agent.run("article", summary, meta=None)

    assert isinstance(res, AgentResult)
    assert res.name == "readability"
    assert 0.0 <= res.score <= 1.0

    # explanation sollte durch Fallback gebaut werden
    assert res.explanation
    assert "Score" in res.explanation

    assert isinstance(res.issue_spans, list)
    assert len(res.issue_spans) == 1
    assert res.issue_spans[0].severity == "medium"


def test_readability_agent_span_mapping_handles_missing_span():
    class MissingSpanEval:
        def evaluate(self, article_text: str, summary_text: str):
            issues = [
                ReadabilityIssue(
                    type="HARD_TO_PARSE",
                    severity="high",
                    summary_span="DOES NOT EXIST",
                    comment="Kann nicht gemappt werden.",
                )
            ]
            return 0.1, issues, "Explanation."

    agent = ReadabilityAgent(llm_client=None, evaluator=MissingSpanEval())
    res = agent.run("article", "Real summary text.", meta=None)

    span = res.issue_spans[0]
    assert span.start_char is None
    assert span.end_char is None
    assert span.severity == "high"
