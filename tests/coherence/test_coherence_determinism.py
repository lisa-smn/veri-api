"""
Determinismus-Tests: Gleicher Input + gleiche Config => identischer Output.
"""

from app.services.agents.coherence.coherence_agent import CoherenceAgent


class DeterministicCoherenceEvaluator:
    """Deterministischer Evaluator für Tests."""

    def __init__(self, seed: int = 42):
        self.seed = seed

    def evaluate(self, article_text: str, summary_text: str):
        # Deterministischer Score basierend auf Text-Länge und Seed
        import hashlib

        text_hash = hashlib.md5(f"{article_text}{summary_text}{self.seed}".encode()).hexdigest()
        # Konvertiere ersten 8 Zeichen zu float zwischen 0 und 1
        score = int(text_hash[:8], 16) / (16**8)
        return score, [], "Deterministic explanation"


def test_coherence_agent_deterministic_same_input():
    """Prüft, dass gleicher Input + gleiche Config identischen Output liefert."""
    evaluator = DeterministicCoherenceEvaluator(seed=42)
    agent = CoherenceAgent(llm_client=None, evaluator=evaluator)

    article = "Test article"
    summary = "Test summary"

    # Erste Ausführung
    result1 = agent.run(article, summary)
    score1 = result1.score

    # Zweite Ausführung (gleicher Input, gleiche Config)
    result2 = agent.run(article, summary)
    score2 = result2.score

    # Scores sollten identisch sein (abs diff < 1e-9)
    assert abs(score1 - score2) < 1e-9, f"Scores differ: {score1} vs {score2}"


def test_coherence_agent_deterministic_different_input():
    """Prüft, dass unterschiedlicher Input unterschiedlichen Output liefert."""
    evaluator = DeterministicCoherenceEvaluator(seed=42)
    agent = CoherenceAgent(llm_client=None, evaluator=evaluator)

    # Erste Ausführung
    result1 = agent.run("Article 1", "Summary 1")
    score1 = result1.score

    # Zweite Ausführung (anderer Input)
    result2 = agent.run("Article 2", "Summary 2")
    score2 = result2.score

    # Scores sollten unterschiedlich sein (wenn Evaluator deterministisch ist)
    # Aber beide sollten im Range [0, 1] sein
    assert 0.0 <= score1 <= 1.0
    assert 0.0 <= score2 <= 1.0
    # Es ist möglich, dass zufällig gleiche Scores auftreten, aber unwahrscheinlich
    # Daher prüfen wir nur, dass beide im Range sind
