"""
Ablation-Varianten für Claim-Extraktion.
"""

from app.services.agents.factuality.claim_models import Claim


class NoOpClaimExtractor:
    """
    Ablation: Keine Claim-Extraktion.
    Gibt leere Liste zurück, damit Fallback (ganzer Satz) verwendet wird.
    """

    def extract_claims(self, sentence: str, sentence_index: int) -> list[Claim]:
        return []


class SentenceOnlyExtractor:
    """
    Ablation: Nur Satz-basierte Claims (keine LLM-Extraktion).
    Gibt direkt den ganzen Satz als Claim zurück.
    """

    def extract_claims(self, sentence: str, sentence_index: int) -> list[Claim]:
        s = (sentence or "").strip()
        if not s:
            return []
        return [
            Claim(
                id=f"s{sentence_index}_sentence_only",
                sentence_index=sentence_index,
                sentence=s,
                text=s,
            )
        ]
