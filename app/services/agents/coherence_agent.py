from app.models.schemas import AgentResult


class CoherenceAgent:
    def run(self, article: str, summary: str, meta: dict | None = None) -> AgentResult:
        # DUMMY-LOGIK: längere Summary = höherer Score
        length = len(summary.split())
        score = 1.0 if length > 50 else 0.6

        return AgentResult(
            score=score,
            errors=[],
            explanation=f"Dummy-CoherenceAgent: Score basierend auf Länge ({length} Tokens)."
        )
