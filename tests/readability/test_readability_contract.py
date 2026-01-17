"""
Contract-Tests für ReadabilityAgent: Score-Range & Robustheit.
"""

from app.services.agents.readability.readability_agent import ReadabilityAgent


class DummyReadabilityEvaluator:
    """Dummy-Evaluator für Tests ohne LLM-Calls."""

    def __init__(self, score: float = 0.5, issues: list = None):
        self.score = score
        self.issues = issues or []

    def evaluate(self, article_text: str, summary_text: str):
        return self.score, self.issues, "Test explanation"


def test_readability_agent_score_is_float():
    """Prüft, dass der Score ein float ist."""
    agent = ReadabilityAgent(llm_client=None, evaluator=DummyReadabilityEvaluator(score=0.7))
    result = agent.run("article", "summary")

    assert isinstance(result.score, float)


def test_readability_agent_score_in_range_0_1():
    """Prüft, dass der Score im Bereich [0, 1] liegt."""
    # Test mit verschiedenen Scores
    for score in [0.0, 0.3, 0.5, 0.7, 1.0, -0.5, 1.5]:
        evaluator = DummyReadabilityEvaluator(score=score)
        agent = ReadabilityAgent(llm_client=None, evaluator=evaluator)
        result = agent.run("article", "summary")

        assert 0.0 <= result.score <= 1.0, f"Score {result.score} außerhalb [0, 1]"


def test_readability_agent_handles_empty_string():
    """Prüft, dass leerer String nicht crasht."""
    agent = ReadabilityAgent(llm_client=None, evaluator=DummyReadabilityEvaluator(score=0.5))
    result = agent.run("", "")

    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


def test_readability_agent_handles_whitespace_only():
    """Prüft, dass nur Whitespace nicht crasht."""
    agent = ReadabilityAgent(llm_client=None, evaluator=DummyReadabilityEvaluator(score=0.5))
    result = agent.run("   ", "   ")

    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


def test_readability_agent_handles_unicode():
    """Prüft, dass Unicode-Zeichen nicht crasht."""
    agent = ReadabilityAgent(llm_client=None, evaluator=DummyReadabilityEvaluator(score=0.5))
    result = agent.run("ÄÖÜ ß 漢字", "Test")

    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


def test_readability_agent_handles_very_long_text():
    """Prüft, dass sehr langer Text nicht crasht."""
    long_text = "Sentence. " * 5000
    agent = ReadabilityAgent(llm_client=None, evaluator=DummyReadabilityEvaluator(score=0.5))
    result = agent.run(long_text, long_text)

    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


def test_readability_agent_handles_simple_text():
    """Prüft, dass einfacher Text funktioniert."""
    agent = ReadabilityAgent(llm_client=None, evaluator=DummyReadabilityEvaluator(score=0.8))
    result = agent.run("Hi.", "Hi.")

    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0
    assert result.score == 0.8  # Sollte nicht geclamped werden, wenn bereits in Range
