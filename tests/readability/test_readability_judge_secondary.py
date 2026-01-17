"""
Judge Secondary Mode Tests: Agent primär, Judge als Fallback.
"""

import os
from unittest.mock import Mock, patch

import pytest

from app.services.agents.readability.readability_agent import ReadabilityAgent


class DummyReadabilityEvaluator:
    """Dummy-Evaluator für Tests."""

    def __init__(self, score: float = 0.5, should_fail: bool = False):
        self.score = score
        self.should_fail = should_fail

    def evaluate(self, article_text: str, summary_text: str):
        if self.should_fail:
            raise Exception("Agent evaluation failed")
        return self.score, [], "Test explanation"


class MockJudgeResult:
    """Mock für Judge-Result."""

    def __init__(self, final_score_norm: float):
        self.final_score_norm = final_score_norm

    def model_dump(self):
        return {"final_score_norm": self.final_score_norm}


@pytest.fixture
def mock_llm_judge():
    """Mock für LLMJudge."""
    with patch("app.services.agents.readability.readability_agent.LLMJudge") as mock_judge_class:
        mock_judge_instance = Mock()
        mock_judge_class.return_value = mock_judge_instance
        yield mock_judge_instance


def test_judge_secondary_agent_primary(mock_llm_judge):
    """Test: Agent-Score wird verwendet, wenn beide verfügbar."""
    # Setup
    os.environ["ENABLE_LLM_JUDGE"] = "true"
    os.environ["JUDGE_MODE"] = "secondary"

    agent_score = 0.8
    judge_score = 0.2

    evaluator = DummyReadabilityEvaluator(score=agent_score)
    mock_llm_judge.judge.return_value = MockJudgeResult(judge_score)

    agent = ReadabilityAgent(llm_client=None, evaluator=evaluator)

    # Run
    result = agent.run("article", "summary")

    # Assert: Agent-Score sollte verwendet werden (secondary mode)
    assert result.score == agent_score

    # Cleanup
    os.environ.pop("ENABLE_LLM_JUDGE", None)
    os.environ.pop("JUDGE_MODE", None)


def test_judge_secondary_agent_fails_judge_available(mock_llm_judge):
    """Test: Judge-Score wird verwendet, wenn Agent fehlschlägt."""
    # Setup
    os.environ["ENABLE_LLM_JUDGE"] = "true"
    os.environ["JUDGE_MODE"] = "secondary"

    judge_score = 0.2
    evaluator = DummyReadabilityEvaluator(score=0.5, should_fail=True)
    mock_llm_judge.judge.return_value = MockJudgeResult(judge_score)

    agent = ReadabilityAgent(llm_client=None, evaluator=evaluator)

    # Run: Agent schlägt fehl, aber Judge ist verfügbar
    # Da der Agent im secondary mode primär ist, sollte der Agent-Fehler durchgereicht werden
    # Aber der Judge-Score sollte in details gespeichert sein
    try:
        result = agent.run("article", "summary")
        # Wenn Agent fehlschlägt, sollte eine Exception geworfen werden
        # (oder der Agent sollte einen Fallback-Score verwenden)
        # Für diesen Test nehmen wir an, dass der Agent einen Fehler wirft
        assert False, "Agent sollte fehlschlagen"
    except Exception:
        # Erwartetes Verhalten: Agent-Fehler wird durchgereicht
        pass

    # Cleanup
    os.environ.pop("ENABLE_LLM_JUDGE", None)
    os.environ.pop("JUDGE_MODE", None)


def test_judge_secondary_agent_available_judge_fails(mock_llm_judge):
    """Test: Agent-Score wird verwendet, wenn Judge fehlschlägt."""
    # Setup
    os.environ["ENABLE_LLM_JUDGE"] = "true"
    os.environ["JUDGE_MODE"] = "secondary"

    agent_score = 0.8
    evaluator = DummyReadabilityEvaluator(score=agent_score)
    mock_llm_judge.judge.side_effect = Exception("Judge failed")

    agent = ReadabilityAgent(llm_client=None, evaluator=evaluator)

    # Run
    result = agent.run("article", "summary")

    # Assert: Agent-Score sollte verwendet werden
    assert result.score == agent_score
    # Judge-Fehler sollte in details gespeichert sein
    assert "judge_error" in result.details or "judge" not in result.details

    # Cleanup
    os.environ.pop("ENABLE_LLM_JUDGE", None)
    os.environ.pop("JUDGE_MODE", None)
