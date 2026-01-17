"""
Ablation-Varianten für Claim-Verifikation.
"""

from app.services.agents.factuality.claim_models import Claim


class NoOpClaimVerifier:
    """
    Ablation: Keine Claim-Verifikation.
    Markiert alle Claims als "uncertain" (neutral).
    """

    def verify(self, article_text: str, claim: Claim) -> Claim:
        claim.label = "uncertain"
        claim.confidence = 0.5
        claim.explanation = "Ablation: Claim-Verifikation deaktiviert"
        claim.error_type = None
        claim.evidence = []
        return claim


class AlwaysCorrectVerifier:
    """
    Ablation: Alle Claims als "correct" markieren.
    """

    def verify(self, article_text: str, claim: Claim) -> Claim:
        claim.label = "correct"
        claim.confidence = 1.0
        claim.explanation = "Ablation: Alle Claims als correct markiert"
        claim.error_type = None
        claim.evidence = []
        return claim
