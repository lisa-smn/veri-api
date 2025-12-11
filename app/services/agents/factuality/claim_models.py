from dataclasses import dataclass
from typing import List, Optional, Literal


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
    evidence: Optional[List[str]] = None
