from __future__ import annotations

import json
from typing import Any, Protocol

from app.llm.llm_client import LLMClient
from app.services.agents.coherence.coherence_models import CoherenceIssue


class CoherenceEvaluator(Protocol):
    def evaluate(
        self, article_text: str, summary_text: str
    ) -> tuple[float, list[CoherenceIssue], str]:
        """
        :return: (score, issues, explanation)
        """
        ...


class LLMCoherenceEvaluator:
    """
    Bewertet NUR Kohärenz der SUMMARY.

    Scope (Coherence):
    - interne logische Konsistenz (keine Selbstwidersprüche)
    - nachvollziehbarer Informationsfluss / Reihenfolge
    - keine unnötigen Wiederholungen (Redundanz)
    - klare Referenzen (Pronomen/Bezüge verständlich)

    Nicht Scope:
    - Lesbarkeit (Satzlänge, Kommas, Stil, Grammatik)
    - Tonalität
    - Faktentreue gegenüber dem Artikel (das ist Factuality)
    """

    ISSUE_REQUIRED_BELOW = 0.7
    MAX_ISSUES = 8

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def evaluate(
        self, article_text: str, summary_text: str
    ) -> tuple[float, list[CoherenceIssue], str]:
        prompt = self._build_prompt(article_text, summary_text)
        raw = self.llm.complete(prompt)
        data = self._parse_output(raw)

        score = self._clamp01(data.get("score", 0.0))
        explanation = (data.get("explanation") or "").strip()

        issues_raw = data.get("issues") or []
        if not isinstance(issues_raw, list):
            issues_raw = []

        issues: list[CoherenceIssue] = []
        for issue_dict in issues_raw[: self.MAX_ISSUES]:
            if not isinstance(issue_dict, dict):
                continue

            t = issue_dict.get("type", "OTHER")
            if t not in (
                "LOGICAL_INCONSISTENCY",
                "CONTRADICTION",
                "REDUNDANCY",
                "ORDERING",
                "OTHER",
            ):
                t = "OTHER"

            sev = issue_dict.get("severity", "medium")
            if sev not in ("low", "medium", "high"):
                sev = "medium"

            span = (issue_dict.get("summary_span") or "").strip()
            comment = (issue_dict.get("comment") or "").strip()
            hint = issue_dict.get("hint")
            if hint is not None and not isinstance(hint, str):
                hint = str(hint)

            issues.append(
                CoherenceIssue(
                    type=t,
                    severity=sev,
                    summary_span=span,
                    comment=comment,
                    hint=hint,
                )
            )

        # summary_span muss substring sein, sonst Mapping kaputt
        issues = self._ensure_spans_are_substrings(summary_text, issues)

        # Safety-net: score niedrig, aber keine Issues geliefert -> erzwingen
        if score < self.ISSUE_REQUIRED_BELOW and not issues:
            safe = (summary_text or "").strip()[:120]
            issues = [
                CoherenceIssue(
                    type="OTHER",
                    severity="medium",
                    summary_span=safe,
                    comment="Score ist niedrig, aber es wurden keine konkreten Kohärenz-Issues geliefert (Fallback).",
                    hint=None,
                )
            ]

        # Fallback explanation (nur Kohärenz-Aspekte)
        if not explanation:
            explanation = (
                "Die Summary wirkt insgesamt kohärent."
                if not issues
                else "Die Summary enthält Kohärenzprobleme (z.B. Sprünge, Widersprüche, Redundanz oder unklare Bezüge)."
            )

        return score, issues, explanation

    def _build_prompt(self, article: str, summary: str) -> str:
        return f"""
Du bewertest NUR die KOHÄRENZ (Coherence) der SUMMARY.

Kohärenz bedeutet hier:
- interne logische Konsistenz (keine Selbstwidersprüche)
- nachvollziehbarer Informationsfluss / Reihenfolge (keine abrupten Sprünge ohne Übergang)
- keine unnötigen Wiederholungen (Redundanz)
- klare Referenzen (Pronomen/Bezüge müssen verständlich sein)

Bewerte NICHT:
- Lesbarkeit (Satzlänge, Kommas, Stil, Grammatik)
- Tonalität
- faktische Korrektheit gegenüber dem Artikel (das ist ein anderer Agent)

Gib NUR JSON zurück, ohne Text außerhalb des JSON.

Schema:
{{
  "score": 0.0,  # float in [0,1] (1 = sehr kohärent, 0 = sehr inkohärent)
  "explanation": "1–2 Sätze globale Begründung (nur Kohärenz-Aspekte)",
  "issues": [
    {{
      "type": "LOGICAL_INCONSISTENCY" | "CONTRADICTION" | "REDUNDANCY" | "ORDERING" | "OTHER",
      "severity": "low" | "medium" | "high",
      "summary_span": "wörtlicher Auszug aus der SUMMARY (kurz, exakt kopiert)",
      "comment": "kurze Erklärung, warum das ein Kohärenzproblem ist",
      "hint": "optional: konkrete Reparaturidee"
    }}
  ]
}}

WICHTIG:
- summary_span MUSS direkt aus der SUMMARY kopiert sein (Substring), damit wir die Stelle mappen können.
- Maximal {self.MAX_ISSUES} Issues.
- Wenn score < {self.ISSUE_REQUIRED_BELOW}, gib MINDESTENS 1 Issue zurück.
- Keine Hinweise zu Lesbarkeit/Stil/Grammatik. Nur Kohärenz.

ARTIKEL (nur Kontext, NICHT faktisch prüfen):
{article}

SUMMARY (zu bewerten):
{summary}
""".strip()

    def _parse_output(self, raw: str) -> dict[str, Any]:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            json_str = raw[start:end]
            data = json.loads(json_str)
        except Exception:
            return {
                "score": 0.0,
                "issues": [],
                "explanation": f"Antwort nicht parsbar: {raw[:200]}",
            }

        if not isinstance(data, dict):
            return {"score": 0.0, "issues": [], "explanation": ""}

        if not isinstance(data.get("issues", []), list):
            data["issues"] = []

        if not isinstance(data.get("explanation", ""), str):
            data["explanation"] = ""

        return data

    @staticmethod
    def _ensure_spans_are_substrings(
        summary_text: str, issues: list[CoherenceIssue]
    ) -> list[CoherenceIssue]:
        safe = (summary_text or "").strip()[:120]
        out: list[CoherenceIssue] = []
        for iss in issues:
            span = (iss.summary_span or "").strip()
            if (span and summary_text and span not in summary_text) or ((not span) and safe):
                iss.summary_span = safe
            out.append(iss)
        return out

    @staticmethod
    def _clamp01(score: Any) -> float:
        try:
            s = float(score)
        except Exception:
            s = 0.0
        if s < 0.0:
            s = 0.0
        if s > 1.0:
            s = 1.0
        return s
