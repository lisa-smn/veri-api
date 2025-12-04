from app.models.schemas import AgentResult


class ReadabilityAgent:
    def run(self, article: str, summary: str, meta: dict | None = None) -> AgentResult:
        # DUMMY-LOGIK: immer perfekter Score
        return AgentResult(
            score=1.0,
            errors=[],
            explanation="Dummy-ReadabilityAgent: immer 1.0."
        )
