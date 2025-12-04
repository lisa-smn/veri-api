from app.models.schemas import AgentResult, ErrorSpan


class FactualityAgent:
    def run(self, article: str, summary: str, meta: dict | None = None) -> AgentResult:
        # DUMMY-LOGIK: immer Score 0.8, mit Beispiel-Fehler, falls "XYZ" in der Summary steht
        errors: list[ErrorSpan] = []

        idx = summary.find("XYZ")
        if idx != -1:
            errors.append(ErrorSpan(
                start_char=idx,
                end_char=idx + 3,
                message="Beispielhafte faktische Unstimmigkeit (Dummy).",
                severity="major",
            ))

        return AgentResult(
            score=0.8,
            errors=errors,
            explanation="Dummy-FactualityAgent: fester Score 0.8."
        )
