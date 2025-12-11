from typing import Any, Dict

from app.models.pydantic import AgentResult


class CoherenceAgent:
    def __init__(self) -> None:
        ...

    def run(
        self,
        article_text: str,
        summary_text: str,
        meta: Dict[str, Any] | None = None,
    ) -> AgentResult:
        # Dumme Heuristik: Coherence-Score basiert auf Länge der Summary
        num_tokens = len(summary_text.split())
        score = min(1.0, max(0.0, num_tokens / 10.0))

        explanation = f"Dummy-CoherenceAgent: Score basierend auf Länge ({num_tokens} Tokens)."

        return AgentResult(
            name="coherence",
            score=score,
            explanation=explanation,
            errors=[],      # aktuell keine Fehlerliste genutzt
            details=None,   # kannst du später mit Satzinfos etc. füllen
        )
