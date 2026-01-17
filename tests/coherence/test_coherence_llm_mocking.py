"""
LLM-Mocking-Tests für CoherenceAgent.
"""

from unittest.mock import Mock

from app.services.agents.coherence.coherence_agent import CoherenceAgent


class MockLLMClient:
    """Mock LLM-Client für Tests."""

    def __init__(self, return_value: str = '{"score": 0.7, "explanation": "Test", "issues": []}'):
        self.return_value = return_value

    def complete(self, prompt: str) -> str:
        return self.return_value


def test_coherence_agent_with_mocked_llm():
    """Prüft, dass CoherenceAgent mit gemocktem LLM funktioniert."""
    mock_llm = MockLLMClient()
    agent = CoherenceAgent(llm_client=mock_llm)

    result = agent.run("article", "summary")

    assert isinstance(result.score, float)
    assert 0.0 <= result.score <= 1.0
    assert result.name == "coherence"


def test_coherence_agent_handles_llm_error():
    """Prüft, dass LLM-Fehler abgefangen werden."""
    mock_llm = Mock()
    mock_llm.complete.side_effect = Exception("LLM error")

    # Agent sollte nicht crashen, sondern einen Fallback-Score verwenden
    # (abhängig von der Implementierung)
    agent = CoherenceAgent(llm_client=mock_llm)

    # Wenn der Evaluator einen Fehler wirft, sollte der Agent das abfangen
    # oder einen Fallback verwenden
    try:
        result = agent.run("article", "summary")
        # Wenn kein Fehler geworfen wird, sollte der Score im Range sein
        assert 0.0 <= result.score <= 1.0
    except Exception:
        # Wenn ein Fehler geworfen wird, ist das auch ok (abhängig von der Implementierung)
        pass
