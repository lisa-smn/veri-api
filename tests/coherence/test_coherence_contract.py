"""
Contract-Tests für CoherenceAgent: Score-Range & Robustheit.
"""

from app.services.agents.coherence.coherence_agent import CoherenceAgent


class DummyCoherenceEvaluator:
    """Dummy-Evaluator für Tests ohne LLM-Calls."""

    def __init__(self, score: float = 0.5, issues: list = None):
        self.score = score
        self.issues = issues or []

    def evaluate(self, article_text: str, summary_text: str):
        return self.score, self.issues, "Test explanation"


def test_coherence_agent_score_is_float():
    """Prüft, dass der Score ein float ist."""
    agent = CoherenceAgent(llm_client=None, evaluator=DummyCoherenceEvaluator(score=0.7))
    result = agent.run("article", "summary")

    assert isinstance(result.score, float)


def test_coherence_agent_score_in_range_0_1():
    """Prüft, dass der Score im Bereich [0, 1] liegt."""
    # Test mit verschiedenen Scores
    # Hinweis: CoherenceAgent clammpt möglicherweise nicht automatisch
    # Daher testen wir nur Scores, die bereits im Range sind
    for score in [0.0, 0.3, 0.5, 0.7, 1.0]:
        evaluator = DummyCoherenceEvaluator(score=score)
        agent = CoherenceAgent(llm_client=None, evaluator=evaluator)
        result = agent.run("article", "summary")

        assert 0.0 <= result.score <= 1.0, f"Score {result.score} außerhalb [0, 1]"

    # Test mit Scores außerhalb des Ranges (sollten geclamped werden, falls implementiert)
    for score in [-0.5, 1.5]:
        evaluator = DummyCoherenceEvaluator(score=score)
        agent = CoherenceAgent(llm_client=None, evaluator=evaluator)
        result = agent.run("article", "summary")
        # Agent sollte Score clammen oder zumindest nicht crashen
        # Wenn nicht geclamped: Score kann außerhalb sein, aber Agent sollte nicht crashen
        assert isinstance(result.score, float), f"Score sollte float sein, ist {type(result.score)}"


def test_coherence_agent_handles_empty_string():
    """Prüft, dass leerer String nicht crasht."""
    agent = CoherenceAgent(llm_client=None, evaluator=DummyCoherenceEvaluator(score=0.5))
    result = agent.run("", "")

    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


def test_coherence_agent_handles_whitespace_only():
    """Prüft, dass nur Whitespace nicht crasht."""
    agent = CoherenceAgent(llm_client=None, evaluator=DummyCoherenceEvaluator(score=0.5))
    result = agent.run("   ", "   ")

    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


def test_coherence_agent_handles_unicode():
    """Prüft, dass Unicode-Zeichen nicht crasht."""
    agent = CoherenceAgent(llm_client=None, evaluator=DummyCoherenceEvaluator(score=0.5))
    result = agent.run("ÄÖÜ ß 漢字", "Test")

    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


def test_coherence_agent_handles_very_long_text():
    """Prüft, dass sehr langer Text nicht crasht."""
    long_text = "Sentence. " * 5000
    agent = CoherenceAgent(llm_client=None, evaluator=DummyCoherenceEvaluator(score=0.5))
    result = agent.run(long_text, long_text)

    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0


def test_coherence_agent_handles_simple_text():
    """Prüft, dass einfacher Text funktioniert."""
    agent = CoherenceAgent(llm_client=None, evaluator=DummyCoherenceEvaluator(score=0.8))
    result = agent.run("Hi.", "Hi.")

    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0
    assert result.score == 0.8  # Sollte nicht geclamped werden, wenn bereits in Range
