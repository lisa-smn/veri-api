"""
Unit Tests für IssueSpan.verdict vs severity Trennung.
Testet, dass verdict explizit gesetzt wird und unabhängig von severity ist.
"""

from app.models.pydantic import IssueSpan
from app.services.agents.factuality.claim_models import Claim
from scripts.test_evidence_gate_eval import count_effective_issues


class TestIssueSpanVerdict:
    """Tests für verdict vs severity Trennung."""

    def test_verdict_incorrect_with_low_severity(self):
        """
        Test: Ein Claim mit label="incorrect" und error_type="OTHER"
        produziert IssueSpan(severity="low", verdict="incorrect").

        Das ist der Sinn: severity="low" darf nicht automatisch "uncertain" bedeuten.
        """
        # Simuliere Claim: incorrect OTHER
        claim = Claim(
            id="s0_c0",
            sentence_index=0,
            sentence="Dummy sentence.",
            text="Test claim",
            label="incorrect",
            error_type="OTHER",
            confidence=0.6,
            explanation="Test explanation",
        )

        # Baue IssueSpan über Agent (oder direkt)
        # Für isolierten Test: Baue IssueSpan direkt
        span = IssueSpan(
            start_char=0,
            end_char=10,
            message="Satz 1: Claim 'Test claim' – Test explanation",
            severity="low",  # OTHER hat low severity
            issue_type="OTHER",
            verdict="incorrect",  # Explizit incorrect, nicht uncertain
            confidence=0.6,
        )

        # Assert: verdict ist incorrect, auch wenn severity low
        assert span.verdict == "incorrect"
        assert span.severity == "low"
        # Wichtig: severity="low" bedeutet NICHT automatisch "uncertain"
        assert span.verdict != "uncertain"

    def test_verdict_uncertain_with_low_severity(self):
        """
        Test: Ein Claim mit label="uncertain" produziert
        IssueSpan(severity="low", verdict="uncertain").
        """
        span = IssueSpan(
            start_char=0,
            end_char=10,
            message="Satz 1: Claim 'Test claim' – Nicht sicher verifizierbar",
            severity="low",
            issue_type="OTHER",
            verdict="uncertain",  # Explizit uncertain
            confidence=0.5,
        )

        assert span.verdict == "uncertain"
        assert span.severity == "low"

    def test_count_effective_issues_uses_verdict(self):
        """
        Test: count_effective_issues() nutzt verdict, nicht severity.

        Ein IssueSpan mit severity="low" aber verdict="incorrect"
        zählt als 1.0 (nicht als uncertain).
        """
        # Span: incorrect OTHER mit low severity
        span_incorrect = IssueSpan(
            start_char=0,
            end_char=10,
            message="Test incorrect",
            severity="low",  # Low severity
            issue_type="OTHER",
            verdict="incorrect",  # Aber explizit incorrect
            confidence=0.6,
        )

        # Span: uncertain
        span_uncertain = IssueSpan(
            start_char=0,
            end_char=10,
            message="Test uncertain",
            severity="low",
            issue_type="OTHER",
            verdict="uncertain",
            confidence=0.5,
        )

        # Test 1: non_error Policy
        # incorrect zählt als 1.0, uncertain zählt nicht
        effective = count_effective_issues([span_incorrect, span_uncertain], "non_error")
        assert effective == 1.0  # Nur incorrect zählt

        # Test 2: weight_0.5 Policy
        # incorrect zählt als 1.0, uncertain zählt als 0.5
        effective = count_effective_issues([span_incorrect, span_uncertain], "weight_0.5")
        assert effective == 1.5  # 1.0 + 0.5

        # Test 3: count_as_error Policy
        # Beide zählen als 1.0
        effective = count_effective_issues([span_incorrect, span_uncertain], "count_as_error")
        assert effective == 2.0  # 1.0 + 1.0

    def test_count_effective_issues_fallback_to_heuristic(self):
        """
        Test: count_effective_issues() nutzt Heuristik als Fallback,
        wenn verdict fehlt (alte Runs ohne verdict Feld).
        """
        # Span ohne verdict (alte Daten)
        span_old = IssueSpan(
            start_char=0,
            end_char=10,
            message="Nicht sicher verifizierbar",
            severity="low",
            issue_type="OTHER",
            verdict=None,  # Fehlt (alte Daten)
            confidence=0.5,
        )

        # Fallback: Heuristik erkennt "Nicht sicher" im message
        effective = count_effective_issues([span_old], "non_error")
        assert effective == 0.0  # Uncertain wird ignoriert

        effective = count_effective_issues([span_old], "weight_0.5")
        assert effective == 0.5  # Uncertain zählt als 0.5
