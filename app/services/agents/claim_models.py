from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Claim:
    id: str
    sentence_index: int
    sentence: str          # ganzer Satz, aus dem der Claim stammt
    text: str              # eigentlicher Claim

    # Diese Felder werden durch den ClaimVerifier gefüllt:
    label: Optional[str] = None        # "correct" | "incorrect" | "uncertain"
    confidence: Optional[float] = None
    error_type: Optional[str] = None   # "ENTITY" | "NUMBER" | "DATE" | ...
    explanation: Optional[str] = None
    evidence: Optional[List[str]] = None
