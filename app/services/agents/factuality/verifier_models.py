"""
Clean-Code Models für Claim Verifier.
Definiert strukturierte Datenmodelle für LLM Output, Evidence Selection und Gate Decisions.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


class VerifierLLMOutput(BaseModel):
    """
    Strukturiertes Modell für LLM Output des Claim Verifiers.
    Wird aus JSON geparst und validiert.
    """

    label: Literal["correct", "incorrect", "uncertain"] = "uncertain"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    error_type: Literal["ENTITY", "NUMBER", "DATE", "OTHER"] | None = None
    explanation: str = ""
    selected_evidence_index: int = Field(default=-1, ge=-1)
    evidence_quote: str | None = None


@dataclass(frozen=True)
class EvidenceSelection:
    """
    Repräsentiert die Evidence-Auswahl mit Validierung.
    Frozen dataclass für Immutability.
    """

    index: int  # -1 wenn keine Passage ausgewählt, sonst 0..k-1
    quote: str | None  # None wenn keine Evidence, sonst wörtlicher Quote
    passage: str | None  # None wenn keine Passage, sonst die ausgewählte Passage
    quote_in_passage: bool  # True wenn quote ein Substring von passage ist
    evidence_found: bool  # True wenn belastbare Evidence vorhanden ist
    reason: (
        str  # ok | no_passage_selected | index_out_of_range | empty_quote | quote_not_in_passage
    )


@dataclass(frozen=True)
class GateDecision:
    """
    Repräsentiert die Gate-Entscheidung mit finalem Label und Confidence.
    Frozen dataclass für Immutability.
    """

    label_raw: str  # Original-Label vom LLM
    label_final: str  # Finales Label nach Gate (correct|incorrect|uncertain)
    confidence: float  # Finale Confidence (geclampft)
    gate_reason: str  # ok | no_evidence | parse_error | schema_violation | coverage_fail
    coverage_ok: bool  # True wenn Coverage-Check bestanden
    coverage_note: str  # Kurze Notiz zum Coverage-Status
