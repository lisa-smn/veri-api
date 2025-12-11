from dataclasses import asdict
from typing import Dict, Any, List

from app.llm.llm_client import LLMClient
from app.models.pydantic import AgentResult, ErrorSpan
from app.services.agents.coherence.coherence_models import CoherenceIssue
from app.services.agents.coherence.coherence_evaluator import (
    CoherenceEvaluator,
    LLMCoherenceEvaluator,
)


class CoherenceAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        evaluator: CoherenceEvaluator | None = None,
    ):
        """
        Bewertet die Kohärenz einer Summary relativ zum Artikel.

        Nutzt intern einen CoherenceEvaluator (i.d.R. LLMCoherenceEvaluator),
        um Score, Issues und Erklärung zu bestimmen.
        """
        self.llm = llm_client
        self.evaluator = evaluator or LLMCoherenceEvaluator(llm_client)

    def run(
        self,
        article_text: str,
        summary_text: str,
        meta: Dict[str, Any] | None = None,
    ) -> AgentResult:
        """
        Führt die Kohärenzbewertung durch und liefert ein AgentResult,
        kompatibel mit der bestehenden Verifikationspipeline.
        """

        score, issues, explanation = self.evaluator.evaluate(article_text, summary_text)

        # Fallback, falls das LLM keine sinnvolle Erklärung liefert
        if not explanation:
            explanation = self._build_global_explanation(score, issues)

        # Fehlerliste analog zum FactualityAgent: lesbare Strings
        errors = self._build_error_spans(summary_text, issues)

        details = {
            "issues": [asdict(i) for i in issues],
            "num_issues": len(issues),
        }

        return AgentResult(
            name="coherence",
            score=score,
            explanation=explanation,
            errors=errors,
            details=details,
        )

    # ---------- Hilfsfunktionen ---------- #

    def _build_global_explanation(
        self,
        score: float,
        issues: List[CoherenceIssue],
    ) -> str:
        if issues:
            first = issues[0]
            return (
                f"Score {score:.2f}. Es wurden {len(issues)} Kohärenzprobleme erkannt. "
                f"Beispiel: [{first.severity}] {first.type} in '{first.summary_span}' – {first.comment}"
            )
        return f"Score {score:.2f}. Die Summary wirkt insgesamt kohärent, es wurden keine Probleme erkannt."

    def _build_error_spans(
            self,
            summary_text: str,
            issues: List[CoherenceIssue],
    ) -> List[ErrorSpan]:
        error_spans: List[ErrorSpan] = []

        for i, issue in enumerate(issues):
            # sehr grober Versuch: wenn summary_span direkt im Summary vorkommt → Span setzen
            start = summary_text.find(issue.summary_span) if issue.summary_span else -1
            if start != -1:
                end = start + len(issue.summary_span)
            else:
                start, end = None, None

            message = (
                f"[{issue.severity}] {issue.type} in '{issue.summary_span}' – {issue.comment}"
            )

            error_spans.append(
                ErrorSpan(
                    start_char=start,
                    end_char=end,
                    message=message,
                    severity=issue.severity,
                )
            )

        return error_spans

