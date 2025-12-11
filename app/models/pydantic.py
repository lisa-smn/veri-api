from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field


class ErrorSpan(BaseModel):
    """
    Markiert einen fehlerhaften oder relevanten Ausschnitt im Text.
    Kann von Agenten genutzt werden, um Stellen im Summary / Artikel zu markieren.
    """
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    message: str
    severity: Optional[Literal["low", "medium", "high"]] = None


class AgentResult(BaseModel):
    """
    Standardisiertes Ergebnis eines Agenten (Factuality, Coherence, Readability, ...).
    """
    name: str
    score: float
    explanation: str
    # Liste strukturierter Fehler/Spans; kann leer sein.
    errors: List[ErrorSpan] = Field(default_factory=list)
    # Agent-spezifische Zusatzinfos (Claims, Issues, Roh-LLM-Output, etc.).
    details: Optional[Dict[str, Any]] = None


class PipelineResult(BaseModel):
    """
    Internes Ergebnis der gesamten Verifikationspipeline.
    Wird typischerweise im Service und in der Persistenz verwendet.
    """
    factuality: AgentResult
    coherence: AgentResult
    readability: AgentResult
    overall_score: float


class VerifyRequest(BaseModel):
    """
    Request-Body für den /verify-Endpoint.
    """
    dataset: Optional[str] = None
    article_text: str
    summary_text: str
    llm_model: Optional[str] = None
    meta: Optional[Dict[str, str]] = None


class VerifyResponse(BaseModel):
    """
    Response-Body für den /verify-Endpoint.
    """
    run_id: int
    overall_score: float
    factuality: AgentResult
    coherence: AgentResult
    readability: AgentResult
