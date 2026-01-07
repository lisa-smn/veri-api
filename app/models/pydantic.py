from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field, AliasChoices
from app.services.explainability.explainability_models import ExplainabilityResult

class IssueSpan(BaseModel):
    """
    Markiert einen problematischen oder relevanten Ausschnitt im Text.
    Wird von Agenten genutzt, um Stellen im Summary / Artikel zu markieren.
    
    Repräsentiert ein "Issue" (kann "incorrect" oder "uncertain" sein).
    Der Name "IssueSpan" ist konsistenter als "ErrorSpan", da nicht alle Issues
    Fehler sind (uncertain Issues sind keine klaren Fehler).
    """
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    message: str
    severity: Optional[Literal["low", "medium", "high"]] = None
    issue_type: Optional[str] = None
    # Für gewichtete Decision Logic
    confidence: Optional[float] = None  # Confidence des Claims (0.0..1.0)
    mapping_confidence: Optional[float] = None  # Wie sicher ist das Span-Mapping (0.0..1.0)
    evidence_found: Optional[bool] = None  # Wurde belastbare Evidence gefunden?
    # Explizites Uncertainty-Signal (trennt verdict von severity)
    verdict: Optional[Literal["incorrect", "uncertain"]] = None  # Sicherheit/Status (incorrect vs uncertain), unabhängig von severity

class AgentResult(BaseModel):
    """
    Standardisiertes Ergebnis eines Agenten
    (Factuality, Coherence, Readability, ...).
    """
    name: str
    score: float
    explanation: str

    # Kanonischer Name: issue_spans
    # Akzeptiert beim Einlesen auch "errors" (Legacy-Alias für Backward Compatibility),
    # serialisiert aber immer als "issue_spans".
    issue_spans: List[IssueSpan] = Field(
        default_factory=list,
        validation_alias=AliasChoices("issue_spans", "errors"),
        serialization_alias="issue_spans",
    )

    # Agent-spezifische Zusatzinfos
    # (Claims, CoherenceIssues, Readability-Metriken, Roh-LLM-Output, etc.)
    details: Optional[Dict[str, Any]] = None


class VerifyRequest(BaseModel):
    """
    Request-Body für den /verify-Endpoint.
    """
    dataset: Optional[str] = None
    article_text: str
    summary_text: str
    llm_model: Optional[str] = None
    meta: Optional[Dict[str, str]] = None


class PipelineResult(BaseModel):
    """
    Internes Ergebnis der gesamten Verifikationspipeline.
    Wird im Service und in der Persistenz verwendet.
    """
    factuality: AgentResult
    coherence: AgentResult
    readability: AgentResult
    overall_score: float

    explainability: Optional[ExplainabilityResult] = None


class VerifyResponse(BaseModel):
    """
    Response-Body für den /verify-Endpoint.
    """
    run_id: int
    overall_score: float
    factuality: AgentResult
    coherence: AgentResult
    readability: AgentResult

    explainability: Optional[ExplainabilityResult] = None