from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ErrorSpan(BaseModel):
    start_char: int
    end_char: int
    message: str
    severity: Optional[str] = None  # z.B. "minor", "major"


class AgentResult(BaseModel):
    name: str
    score: float
    explanation: str
    errors: List[str] = []
    details: Optional[Dict[str, Any]] = None


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
