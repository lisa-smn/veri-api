from dataclasses import asdict
import os
from typing import Any, Literal

from app.llm.llm_client import LLMClient
from app.models.pydantic import AgentResult, IssueSpan
from app.services.agents.coherence.coherence_models import CoherenceIssue
from app.services.agents.coherence.coherence_verifier import (
    CoherenceEvaluator,
    LLMCoherenceEvaluator,
)
from app.services.judges.llm_judge import LLMJudge


class CoherenceAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        evaluator: CoherenceEvaluator | None = None,
        enable_judge: bool | None = None,
        judge_mode: Literal["primary", "secondary", "diagnostic"] = "secondary",
        judge_model: str | None = None,
        judge_prompt_version: str | None = None,
        judge_n: int | None = None,
        judge_aggregation: Literal["mean", "median", "majority"] = "mean",
        judge_temperature: float | None = None,
    ):
        """
        Bewertet die Kohärenz einer Summary relativ zum Artikel.

        Nutzt intern einen CoherenceEvaluator (i.d.R. LLMCoherenceEvaluator),
        um Score, Issues und Erklärung zu bestimmen.
        """
        self.llm = llm_client
        self.evaluator = evaluator or LLMCoherenceEvaluator(llm_client)

        # Judge-Integration (via ENV oder Parameter)
        enable_judge_env = os.getenv("ENABLE_LLM_JUDGE", "false").lower() == "true"
        self.enable_judge = enable_judge if enable_judge is not None else enable_judge_env
        self.judge_mode = os.getenv("JUDGE_MODE", judge_mode).lower()
        if self.judge_mode not in ("primary", "secondary", "diagnostic"):
            self.judge_mode = "secondary"

        self.llm_judge = None
        if self.enable_judge:
            judge_model_val = judge_model or os.getenv("JUDGE_MODEL", "gpt-4o-mini")
            judge_prompt_version_val = judge_prompt_version or os.getenv(
                "JUDGE_PROMPT_VERSION", "v1"
            )
            judge_n_val = judge_n if judge_n is not None else int(os.getenv("JUDGE_N", "1"))
            judge_aggregation_val = os.getenv("JUDGE_AGGREGATION", judge_aggregation).lower()
            if judge_aggregation_val not in ("mean", "median", "majority"):
                judge_aggregation_val = "mean"
            judge_temperature_val = (
                judge_temperature
                if judge_temperature is not None
                else float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
            )

            self.llm_judge = LLMJudge(
                llm_client=llm_client,
                default_model=judge_model_val,
                default_prompt_version=judge_prompt_version_val,
                default_n=judge_n_val,
                default_temperature=judge_temperature_val,
                default_aggregation=judge_aggregation_val,
            )

    def run(
        self,
        article_text: str,
        summary_text: str,
        meta: dict[str, Any] | None = None,
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

        # Judge-Integration
        judge_result = None
        judge_score = None
        if self.enable_judge and self.llm_judge:
            try:
                judge_result = self.llm_judge.judge(
                    dimension="coherence",
                    article_text=article_text,
                    summary_text=summary_text,
                )
                judge_score = judge_result.final_score_norm
                details["judge"] = judge_result.model_dump()
            except Exception as e:
                # Bei Fehler: Judge-Ergebnis optional, Agent-Score bleibt primär
                details["judge_error"] = str(e)

        # Score-Auswahl basierend auf JUDGE_MODE
        final_score = score
        if self.judge_mode == "primary" and judge_score is not None:
            final_score = judge_score
        elif self.judge_mode == "secondary" and judge_score is not None:
            # Agent-Score bleibt primär, Judge-Score wird in details gespeichert
            details["judge_score"] = judge_score
            details["agent_score"] = score

        return AgentResult(
            name="coherence",
            score=final_score,
            explanation=explanation,
            issue_spans=errors,
            details=details,
        )

    # ---------- Hilfsfunktionen ---------- #

    def _build_global_explanation(
        self,
        score: float,
        issues: list[CoherenceIssue],
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
        issues: list[CoherenceIssue],
    ) -> list[IssueSpan]:
        error_spans: list[IssueSpan] = []

        for i, issue in enumerate(issues):
            # sehr grober Versuch: wenn summary_span direkt im Summary vorkommt → Span setzen
            start = summary_text.find(issue.summary_span) if issue.summary_span else -1
            if start != -1:
                end = start + len(issue.summary_span)
            else:
                start, end = None, None

            message = f"[{issue.severity}] {issue.type} in '{issue.summary_span}' – {issue.comment}"

            error_spans.append(
                IssueSpan(
                    start_char=start,
                    end_char=end,
                    message=message,
                    severity=issue.severity,
                    issue_type=issue.type,  # Explizit setzen für spätere Analyse
                )
            )

        return error_spans
