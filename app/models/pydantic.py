from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field

from app.services.explainability.explainability_models import ExplainabilityResult


class IssueSpan(BaseModel):
    """
    Markiert einen problematischen oder relevanten Ausschnitt im Text.
    Wird von Agenten genutzt, um Stellen im Summary / Artikel zu markieren.

    Repräsentiert ein "Issue" (kann "incorrect" oder "uncertain" sein).
    Der Name "IssueSpan" ist konsistenter als "ErrorSpan", da nicht alle Issues
    Fehler sind (uncertain Issues sind keine klaren Fehler).
    """

    start_char: int | None = None
    end_char: int | None = None
    message: str
    severity: Literal["low", "medium", "high"] | None = None
    issue_type: str | None = None
    # Für gewichtete Decision Logic
    confidence: float | None = None  # Confidence des Claims (0.0..1.0)
    mapping_confidence: float | None = None  # Wie sicher ist das Span-Mapping (0.0..1.0)
    evidence_found: bool | None = None  # Wurde belastbare Evidence gefunden?
    # Explizites Uncertainty-Signal (trennt verdict von severity)
    verdict: Literal["incorrect", "uncertain"] | None = (
        None  # Sicherheit/Status (incorrect vs uncertain), unabhängig von severity
    )


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
    issue_spans: list[IssueSpan] = Field(
        default_factory=list,
        validation_alias=AliasChoices("issue_spans", "errors"),
        serialization_alias="issue_spans",
    )

    # Agent-spezifische Zusatzinfos
    # (Claims, CoherenceIssues, Readability-Metriken, Roh-LLM-Output, etc.)
    details: dict[str, Any] | None = None


class VerifyRequest(BaseModel):
    """
    Request-Body für den /verify-Endpoint.
    """

    dataset: str | None = None
    article_text: str
    summary_text: str
    llm_model: str | None = None
    meta: dict[str, str] | None = None
    # LLM-as-a-Judge (optional, default OFF)
    run_llm_judge: bool = False
    judge_mode: str | None = None  # "primary" or "secondary"
    judge_n: int | None = None  # Number of judges (default: 3)
    judge_temperature: float | None = None  # Default: 0.0
    judge_aggregation: str | None = None  # "majority" or "mean"


class PipelineResult(BaseModel):
    """
    Internes Ergebnis der gesamten Verifikationspipeline.
    Wird im Service und in der Persistenz verwendet.
    """

    factuality: AgentResult
    coherence: AgentResult
    readability: AgentResult
    overall_score: float

    explainability: ExplainabilityResult | None = None
    # LLM-as-a-Judge results (optional)
    judge: dict[str, Any] | None = None
    judge_error: str | None = None
    judge_available: bool = True


class VerifyResponse(BaseModel):
    """
    Response-Body für den /verify-Endpoint.
    """

    run_id: int
    overall_score: float
    factuality: AgentResult
    coherence: AgentResult
    readability: AgentResult

    explainability: ExplainabilityResult | None = None
    # LLM-as-a-Judge results (optional)
    judge: dict[str, Any] | None = None
    judge_error: str | None = None
    judge_available: bool = True


# ---------------------------
# LLM-as-a-Judge Models
# ---------------------------


class JudgeOutput(BaseModel):
    """
    Einzelnes Judge-Urteil (ein LLM-Call).
    """

    dimension: Literal["readability", "coherence", "factuality"]
    rating_raw: float | int  # z.B. 1-5 oder 0-100
    score_norm: float = Field(ge=0.0, le=1.0)  # 0-1 normalisiert
    rationale: str  # kurze Begründung
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)  # 0-1, optional
    flags: list[str] = Field(default_factory=list)  # z.B. ["collapse_warning", "parse_retry"]
    model: str
    prompt_version: str
    raw_json: dict[str, Any]  # geparster JSON-Inhalt
    raw_text: str | None = None  # fallback logging


class CommitteeStats(BaseModel):
    """
    Statistik über ein Committee von Judgements.
    """

    n: int
    mean: float
    std: float | None = None
    min: float
    max: float


class JudgeResult(BaseModel):
    """
    Ergebnis eines LLM-as-a-Judge Calls (kann mehrere Judgements enthalten).
    """

    outputs: list[JudgeOutput]
    committee: CommitteeStats | None = None
    final_score_norm: float = Field(ge=0.0, le=1.0)
    aggregation: Literal["mean", "median", "majority"] = "mean"
