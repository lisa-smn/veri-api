"""
LLM-as-a-Judge Modul für generische Bewertung von Textqualität.

Unterstützt:
- Readability, Coherence, Factuality
- Strict JSON Output
- Committee (mehrere Judgements)
- Robustes Parsing mit Retry
- Degeneration Guardrails
"""

from app.services.judges.llm_judge import LLMJudge
from app.services.judges.parsing import normalize_rating, parse_judge_json

__all__ = ["LLMJudge", "normalize_rating", "parse_judge_json"]
