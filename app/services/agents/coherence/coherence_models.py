from dataclasses import dataclass
from typing import Literal


@dataclass
class CoherenceIssue:
    type: Literal[
        "LOGICAL_INCONSISTENCY",
        "CONTRADICTION",
        "REDUNDANCY",
        "ORDERING",
        "OTHER",
    ]
    severity: Literal["low", "medium", "high"]
    summary_span: str  # z.B. "Sentence 2–3" oder ein kurzer Textauszug
    comment: str  # kurze Beschreibung des Problems

    # Optional für spätere Erweiterung:
    hint: str | None = None  # z.B. "Merge sentences 2 and 3."
