"""
ReadabilityAgent: Wrapper um die Readability-Bewertung einer Summary.

- Ruft einen ReadabilityEvaluator auf (standardmäßig LLMReadabilityEvaluator).
- Normalisiert und validiert Score/Issues.
- Erzeugt (best-effort) issue_spans im Summary-Text.
- Liefert ein AgentResult kompatibel zur bestehenden Pipeline.

Scope: Lesbarkeit (Satzlänge, Komplexität, Überladung, Interpunktion).
Nicht Scope: Fakten, Kohärenz, Stil/Tonalität.
"""

from dataclasses import asdict
import os
from typing import Any, Literal

from app.llm.llm_client import LLMClient
from app.models.pydantic import AgentResult, IssueSpan
from app.services.agents.readability.readability_models import ReadabilityIssue
from app.services.agents.readability.readability_verifier import (
    LLMReadabilityEvaluator,
    ReadabilityEvaluator,
)
from app.services.judges.llm_judge import LLMJudge


class ReadabilityAgent:
    # Wenn Score deutlich schlecht ist, erwarten wir i.d.R. auch Issues.
    # (Schwelle kannst du später in Config ziehen.)
    ISSUE_FALLBACK_THRESHOLD = 0.7

    def __init__(
        self,
        llm_client: LLMClient,
        evaluator: ReadabilityEvaluator | None = None,
        prompt_version: str = "v1",
        enable_judge: bool | None = None,
        judge_mode: Literal["primary", "secondary", "diagnostic"] = "secondary",
        judge_model: str | None = None,
        judge_prompt_version: str | None = None,
        judge_n: int | None = None,
        judge_aggregation: Literal["mean", "median", "majority"] = "mean",
        judge_temperature: float | None = None,
    ):
        self.llm = llm_client
        self.evaluator = evaluator or LLMReadabilityEvaluator(
            llm_client, prompt_version=prompt_version
        )

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
        score, issues, explanation = self.evaluator.evaluate(article_text, summary_text)

        score = self._clamp_0_1(score)
        issues = self._sanitize_issues(issues)

        # Falls der Evaluator einen schlechten Score liefert, aber keine Issues:
        # minimale, deterministische Fallback-Issues aus Heuristiken erzeugen.
        if not issues and score < self.ISSUE_FALLBACK_THRESHOLD:
            issues = self._fallback_issues(summary_text)

        if not explanation:
            explanation = self._build_global_explanation(score, issues)

        issue_spans = self._build_issue_spans(summary_text, issues)

        details = {
            "issues": [asdict(i) for i in issues],
            "num_issues": len(issues),
            # optional für Debugging:
            # "meta": meta,
        }

        # Judge-Integration
        judge_result = None
        judge_score = None
        if self.enable_judge and self.llm_judge:
            try:
                judge_result = self.llm_judge.judge(
                    dimension="readability",
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
            name="readability",
            score=final_score,
            explanation=explanation,
            issue_spans=issue_spans,
            details=details,
        )

    # ---------- intern ---------- #

    @staticmethod
    def _clamp_0_1(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        if v < 0.0:
            return 0.0
        if v > 1.0:
            return 1.0
        return v

    @staticmethod
    def _sanitize_issues(issues: Any) -> list[ReadabilityIssue]:
        if not isinstance(issues, list):
            return []
        out: list[ReadabilityIssue] = []
        for it in issues:
            if isinstance(it, ReadabilityIssue):
                # severity defensiv normalisieren
                if it.severity not in ("low", "medium", "high"):
                    it.severity = "medium"
                out.append(it)
        return out

    def _build_global_explanation(self, score: float, issues: list[ReadabilityIssue]) -> str:
        if issues:
            first = issues[0]
            return (
                f"Score {score:.2f}. Es wurden {len(issues)} Lesbarkeitsprobleme erkannt. "
                f"Beispiel: [{first.severity}] {first.type} in '{first.summary_span}' – {first.comment}"
            )
        return f"Score {score:.2f}. Die Summary wirkt insgesamt gut lesbar, es wurden keine Probleme erkannt."

    def _build_issue_spans(
        self, summary_text: str, issues: list[ReadabilityIssue]
    ) -> list[IssueSpan]:
        spans: list[IssueSpan] = []

        for issue in issues:
            start, end = None, None
            snippet = (issue.summary_span or "").strip()

            if snippet:
                pos = summary_text.find(snippet)
                if pos != -1:
                    start = pos
                    end = pos + len(snippet)

            msg = f"[{issue.severity}] {issue.type} in '{snippet}' – {issue.comment}"

            spans.append(
                IssueSpan(
                    start_char=start,
                    end_char=end,
                    message=msg,
                    severity=issue.severity,
                )
            )
        return spans

    # ---------- Fallback-Heuristiken ---------- #

    def _fallback_issues(self, summary_text: str) -> list[ReadabilityIssue]:
        """
        Minimaler deterministischer Fallback, falls LLM keine Issues liefert,
        aber der Score schlecht ist. Ziel: issue_spans sind nicht leer.
        """
        text = (summary_text or "").strip()
        if not text:
            return [
                ReadabilityIssue(
                    type="OTHER",
                    severity="high",
                    summary_span="",
                    comment="Die Summary ist leer oder enthält keinen verwertbaren Text.",
                )
            ]

        # einfache Kennzahlen
        words = text.split()
        word_count = len(words)
        comma_count = text.count(",")
        has_parentheses = "(" in text and ")" in text

        issues: list[ReadabilityIssue] = []

        # LONG_SENTENCE: sehr grob, aber zuverlässig
        if word_count >= 30:
            span = " ".join(words[: min(20, word_count)])
            issues.append(
                ReadabilityIssue(
                    type="LONG_SENTENCE",
                    severity="high" if word_count >= 45 else "medium",
                    summary_span=span,
                    comment=f"Sehr lange Satzstruktur ({word_count} Wörter) erschwert das Lesen.",
                )
            )

        if comma_count >= 4:
            span = text[: min(120, len(text))]
            issues.append(
                ReadabilityIssue(
                    type="PUNCTUATION_OVERLOAD",
                    severity="medium",
                    summary_span=span,
                    comment=f"Viele Kommata ({comma_count}) deuten auf starke Verschachtelung hin.",
                )
            )

        if has_parentheses:
            # best-effort: markiere den Teil um die Klammern
            start = text.find("(")
            end = text.find(")", start + 1)
            snippet = (
                text[start : min(end + 1, len(text))]
                if start != -1 and end != -1
                else text[: min(120, len(text))]
            )
            issues.append(
                ReadabilityIssue(
                    type="NESTED_CLAUSE",
                    severity="low",
                    summary_span=snippet,
                    comment="Einschübe/Klammern erhöhen syntaktische Komplexität.",
                )
            )

        if not issues:
            issues.append(
                ReadabilityIssue(
                    type="OTHER",
                    severity="medium",
                    summary_span=text[: min(120, len(text))],
                    comment="Die Summary wirkt schwer lesbar, konnte aber nicht weiter aufgeschlüsselt werden.",
                )
            )

        return issues
