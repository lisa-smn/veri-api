from dataclasses import dataclass, field
from typing import List, Optional, Literal, Any


@dataclass
class EvidenceSpan:
    """Strukturierte Evidence-Span mit Position und Text."""
    start_char: Optional[int] = None  # Start-Position im article_text (optional)
    end_char: Optional[int] = None     # End-Position im article_text (optional)
    text: str = ""                     # Wörtlicher Textauszug (evidence_quote)
    source: str = "article"            # Quelle: "article" | "context"


@dataclass
class Claim:
    id: str
    sentence_index: int
    sentence: str          # ganzer Satz, aus dem der Claim stammt
    text: str              # eigentlicher Claim

    # Diese Felder werden durch den ClaimVerifier gefüllt:
    label: Optional[Literal["correct", "incorrect", "uncertain"]] = None
    confidence: Optional[float] = None
    error_type: Optional[Literal["ENTITY", "NUMBER", "DATE", "OTHER"]] = None
    explanation: Optional[str] = None
    evidence: Optional[List[str]] = None  # Legacy: Liste von Evidence-Strings
    # Für besseres Uncertainty-Handling und Evidence-Gate
    evidence_found: Optional[bool] = None  # Wurde belastbare Evidence gefunden?
    evidence_spans: Optional[List[str]] = None  # Legacy: Liste von Evidence-Strings
    # Neue strukturierte Evidence-Felder
    evidence_spans_structured: List[EvidenceSpan] = field(default_factory=list)  # Strukturierte Evidence-Spans
    evidence_quote: Optional[str] = None  # Kurzer Textauszug (max 1-2 Sätze, zusammengefasst)
    rationale: Optional[str] = None  # Kurzer Grund für verdict
    # Evidence Retrieval
    retrieved_passages: List[str] = field(default_factory=list)  # Top-k Passagen vom Retriever
    retrieval_scores: List[float] = field(default_factory=list)  # Scores für jede Passage
    selected_evidence_index: int = -1  # Index der ausgewählten Passage (0..k-1) oder -1 wenn keine
    # Debug-Felder
    parse_ok: bool = True  # Wurde Verifier-Output korrekt geparst?
    parse_error: Optional[str] = None  # Parse-Fehler (falls vorhanden)
    schema_violation_reason: Optional[str] = None  # Grund für Schema-Verletzung (falls vorhanden)
    raw_verifier_output: Optional[str] = None  # Roher LLM-Output (gekürzt auf 800 Zeichen, für Debugging)
    # Raw-Felder (was kam wirklich vom LLM?)
    selected_evidence_index_raw: Optional[Any] = None  # Raw-Wert aus JSON (vor Parsing)
    evidence_quote_raw: Optional[Any] = None  # Raw-Wert aus JSON (vor Parsing)
