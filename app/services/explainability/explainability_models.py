"""
Enthält die Datenmodelle für das Explainability-Modul.

-“Erklärung” der Verifikation als klar strukturierte, reproduzierbare Daten.

- Span: eine Textstelle in der Summary (Start/Ende in Zeichen + optional Text).
- Finding: ein einzelnes Problem (Dimension, Schweregrad, kurze Erklärung,
  optional Span, Evidence/Details und eine Empfehlung).
- ExplainabilityResult: der komplette Explainability-Report:
  - Executive Summary (kurze Zusammenfassung in wenigen Sätzen)
  - Liste aller Findings (normalisiert, dedupliziert, sortiert)
  - Findings gruppiert nach Dimension (factuality/coherence/readability)
  - Top-Spans (wichtigste Stellen, gerankt)
  - Stats (z.B. Anzahl Findings, Coverage)
  - version (z.B. "m9_v1" für Vergleichbarkeit)

Diese Modelle sorgen dafür, dass die Ausgabe stabil bleibt, leicht testbar ist
und später gespeichert (Postgres/Neo4j) und in Swagger gut angezeigt werden kann.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class Dimension(str, Enum):
    factuality = "factuality"
    coherence = "coherence"
    readability = "readability"


Severity = Literal["low", "medium", "high"]


class Span(BaseModel):
    start_char: int = Field(ge=0)
    end_char: int = Field(ge=0)
    text: Optional[str] = None


class EvidenceItem(BaseModel):
    kind: str = "generic"              # e.g. "quote", "claim", "metric"
    source: Optional[str] = None       # e.g. "agent:factuality"
    quote: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class Finding(BaseModel):
    id: str
    dimension: Dimension
    severity: Severity
    message: str
    span: Optional[Span] = None
    evidence: List[EvidenceItem] = Field(default_factory=list)
    recommendation: Optional[str] = None
    source: Dict[str, Any] = Field(default_factory=dict)  # agent + issue_type + claim_id + merged_from ...


class ExplainabilityStats(BaseModel):
    num_findings: int
    num_high_severity: int
    num_medium_severity: int
    num_low_severity: int
    coverage_chars: int
    coverage_ratio: float


class TopSpan(BaseModel):
    span: Span
    dimension: Dimension
    severity: Severity
    finding_id: str
    rank_score: float


class ExplainabilityResult(BaseModel):
    summary: List[str] = Field(default_factory=list)  # 3–6 Sätze
    findings: List[Finding] = Field(default_factory=list)
    by_dimension: Dict[Dimension, List[Finding]] = Field(default_factory=dict)
    top_spans: List[TopSpan] = Field(default_factory=list)
    stats: ExplainabilityStats
    version: str = "m9_v1"
