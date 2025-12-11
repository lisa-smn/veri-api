from typing import Protocol, List, Tuple, Any
import json

from app.llm.llm_client import LLMClient
from app.services.agents.coherence.coherence_models import CoherenceIssue


class CoherenceEvaluator(Protocol):
    def evaluate(self, article_text: str, summary_text: str) -> Tuple[float, List[CoherenceIssue], str]:
        """
        :return: (score, issues, explanation)
        """
        ...


class LLMCoherenceEvaluator:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def evaluate(self, article_text: str, summary_text: str) -> Tuple[float, List[CoherenceIssue], str]:
        prompt = self._build_prompt(article_text, summary_text)
        raw = self.llm.complete(prompt)
        data = self._parse_output(raw)

        score = self._normalize_score(data.get("score"))
        issues_raw = data.get("issues") or []
        issues: List[CoherenceIssue] = []

        for issue_dict in issues_raw:
            if not isinstance(issue_dict, dict):
                continue
            try:
                issues.append(CoherenceIssue(**issue_dict))
            except TypeError:
                # LLM hat Felder vermurkst → Issue ignorieren, aber Rest behalten
                continue

        explanation = data.get("explanation", "")

        return score, issues, explanation

    def _build_prompt(self, article: str, summary: str) -> str:
        return f"""
You are an expert in textual coherence.

Given the following ARTICLE and SUMMARY, evaluate how coherent the SUMMARY is.

COHERENCE means:
- logical flow between sentences
- no abrupt topic jumps
- no unnecessary repetition
- no internal contradictions
- clear ordering of information

Return ONLY a valid JSON object with the following fields:

{{
  "score": 0.0,
  "issues": [
    {{
      "type": "LOGICAL_INCONSISTENCY" | "CONTRADICTION" | "REDUNDANCY" | "ORDERING" | "OTHER",
      "severity": "low" | "medium" | "high",
      "summary_span": "short description which part of the summary is affected",
      "comment": "short explanation of the problem"
    }}
  ],
  "explanation": "2-3 sentence explanation of overall coherence"
}}

Rules:
- Score must be between 0 and 1 (1 = perfectly coherent).
- If there are no issues, use an empty list for "issues".
- Do not add any text outside the JSON. No comments, no markdown, no prose.

ARTICLE:
{article}

SUMMARY:
{summary}
""".strip()

    def _parse_output(self, raw: str) -> dict:
        """
        Robust JSON-Parsing wie beim ClaimVerifier:
        - versucht erst, das JSON-Innere zu extrahieren
        - liefert im Fehlerfall ein sinnvolles Fallback-Dict
        """
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            json_str = raw[start:end]
            data = json.loads(json_str)
        except Exception:
            return {
                "score": 0.5,
                "issues": [],
                "explanation": f"Antwort nicht parsbar: {raw[:200]}",
            }

        # Safety-Net für 'issues' und 'explanation'
        if not isinstance(data.get("issues", []), list):
            data["issues"] = []

        if not isinstance(data.get("explanation", ""), str):
            data["explanation"] = ""

        return data

    def _normalize_score(self, score: Any) -> float:
        try:
            s = float(score)
        except Exception:
            s = 0.5
        if s < 0.0:
            s = 0.0
        if s > 1.0:
            s = 1.0
        return s
