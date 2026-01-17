from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class EvidenceSpan:
    """Strukturierte Evidence-Span mit Position und Text."""

    start_char: int | None = None  # Start-Position im article_text (optional)
    end_char: int | None = None  # End-Position im article_text (optional)
    text: str = ""  # Wörtlicher Textauszug (evidence_quote)
    source: str = "article"  # Quelle: "article" | "context"


@dataclass
class Claim:
    id: str
    sentence_index: int
    sentence: str  # ganzer Satz, aus dem der Claim stammt
    text: str  # eigentlicher Claim

    # Diese Felder werden durch den ClaimVerifier gefüllt:
    label: Literal["correct", "incorrect", "uncertain"] | None = None
    confidence: float | None = None
    error_type: Literal["ENTITY", "NUMBER", "DATE", "OTHER"] | None = None
    explanation: str | None = None
    evidence: list[str] | None = None  # Legacy: Liste von Evidence-Strings
    # Für besseres Uncertainty-Handling und Evidence-Gate
    evidence_found: bool | None = None  # Wurde belastbare Evidence gefunden?
    evidence_spans: list[str] | None = None  # Legacy: Liste von Evidence-Strings
    # Neue strukturierte Evidence-Felder
    evidence_spans_structured: list[EvidenceSpan] = field(
        default_factory=list
    )  # Strukturierte Evidence-Spans
    evidence_quote: str | None = None  # Kurzer Textauszug (max 1-2 Sätze, zusammengefasst)
    rationale: str | None = None  # Kurzer Grund für verdict
    # Evidence Retrieval
    retrieved_passages: list[str] = field(default_factory=list)  # Top-k Passagen vom Retriever
    retrieval_scores: list[float] = field(default_factory=list)  # Scores für jede Passage
    selected_evidence_index: int = -1  # Index der ausgewählten Passage (0..k-1) oder -1 wenn keine
    # Debug-Felder
    parse_ok: bool = True  # Wurde Verifier-Output korrekt geparst?
    parse_error: str | None = None  # Parse-Fehler (falls vorhanden)
    schema_violation_reason: str | None = None  # Grund für Schema-Verletzung (falls vorhanden)
    raw_verifier_output: str | None = (
        None  # Roher LLM-Output (gekürzt auf 800 Zeichen, für Debugging)
    )
    # Raw-Felder (was kam wirklich vom LLM?)
    selected_evidence_index_raw: Any | None = None  # Raw-Wert aus JSON (vor Parsing)
    evidence_quote_raw: Any | None = None  # Raw-Wert aus JSON (vor Parsing)
