from pydantic import BaseModel
from typing import List, Optional, Dict

# Eingabemodell für die Verifikation
class VerifyRequest(BaseModel):
    text: str       # Originaltext
    summary: str    # Zusammenfassung

# Ausgabemodell für die Scores
class VerifyResponse(BaseModel):
    factual_score: float
    coherence_score: float
    fluency_score: float
    overall_score: float


class ErrorSpan(BaseModel):
    start_char: int
    end_char: int
    message: str
    severity: Optional[str] = None  # z.B. "minor", "major"


class AgentResult(BaseModel):
    score: float              # 0..1
    errors: List[ErrorSpan]
    explanation: Optional[str] = None


class PipelineResult(BaseModel):
    factuality: AgentResult
    coherence: AgentResult
    readability: AgentResult
    overall_score: float


class VerifyRequest(BaseModel):
    dataset: Optional[str] = None
    article_text: str
    summary_text: str
    llm_model: Optional[str] = None
    meta: Optional[Dict[str, str]] = None


class VerifyResponse(BaseModel):
    run_id: int
    overall_score: float
    factuality: AgentResult
    coherence: AgentResult
    readability: AgentResult
