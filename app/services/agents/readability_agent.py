from typing import Any, Dict

from app.models.pydantic import AgentResult


class ReadabilityAgent:
    def __init__(self) -> None:
        ...

    def run(
        self,
        article_text: str,
        summary_text: str,
        meta: Dict[str, Any] | None = None,
    ) -> AgentResult:
        # Vollkommen dummer Platzhalter: immer perfekt lesbar
        explanation = "Dummy-ReadabilityAgent: immer 1.0."

        return AgentResult(
            name="readability",
            score=1.0,
            explanation=explanation,
            errors=[],
            details=None,
        )
