"""
Unit Tests für Evidence-Gate Logik.
Testet _validate_evidence() und _apply_gate() als reine Funktionen.
"""

from app.llm.fake_client import FakeLLMClient
from app.services.agents.factuality.claim_verifier import LLMClaimVerifier
from app.services.agents.factuality.verifier_models import EvidenceSelection, VerifierLLMOutput


class TestValidateEvidence:
    """Tests für _validate_evidence() Methode."""

    def setup_method(self):
        """Setup: Erstelle LLMClaimVerifier Instanz."""
        llm_client = FakeLLMClient()
        self.verifier = LLMClaimVerifier(llm_client, strict_mode=False)

    def test_idx_minus_one_no_evidence(self):
        """Test 1: idx=-1, quote=None => evidence_found False, reason no_passage_selected"""
        out = VerifierLLMOutput(
            label="uncertain",
            selected_evidence_index=-1,
            evidence_quote=None,
        )
        passages = ["Passage 1", "Passage 2"]

        selection = self.verifier._validate_evidence(out, passages)

        assert selection.index == -1
        assert selection.quote is None
        assert selection.passage is None
        assert selection.quote_in_passage is False
        assert selection.evidence_found is False
        assert selection.reason == "no_passage_selected"

    def test_idx_zero_empty_quote(self):
        """Test 2: idx=0, quote="" => evidence_found False, reason empty_quote"""
        out = VerifierLLMOutput(
            label="correct",
            selected_evidence_index=0,
            evidence_quote="",
        )
        passages = ["Passage 1", "Passage 2"]

        selection = self.verifier._validate_evidence(out, passages)

        assert selection.index == -1
        assert selection.quote is None
        assert selection.evidence_found is False
        assert selection.reason == "empty_quote"

    def test_idx_zero_quote_not_in_passage(self):
        """Test 3: idx=0, quote not in passage => evidence_found False, reason quote_not_in_passage"""
        out = VerifierLLMOutput(
            label="incorrect",
            selected_evidence_index=0,
            evidence_quote="This quote is not in the passage",
        )
        passages = ["Passage 1 contains some text", "Passage 2"]

        selection = self.verifier._validate_evidence(out, passages)

        assert selection.index == -1
        assert selection.quote is None
        assert selection.evidence_found is False
        assert selection.reason == "quote_not_in_passage"

    def test_idx_out_of_range(self):
        """Test: idx out of range => evidence_found False, reason index_out_of_range"""
        out = VerifierLLMOutput(
            label="correct",
            selected_evidence_index=5,  # Out of range (nur 2 Passagen)
            evidence_quote="Some quote",
        )
        passages = ["Passage 1", "Passage 2"]

        selection = self.verifier._validate_evidence(out, passages)

        assert selection.index == -1
        assert selection.quote is None
        assert selection.evidence_found is False
        assert selection.reason == "index_out_of_range"

    def test_valid_evidence(self):
        """Test: idx=0, quote in passage => evidence_found True, reason ok"""
        out = VerifierLLMOutput(
            label="correct",
            selected_evidence_index=0,
            evidence_quote="contains some text",
        )
        passages = ["Passage 1 contains some text", "Passage 2"]

        selection = self.verifier._validate_evidence(out, passages)

        assert selection.index == 0
        assert selection.quote == "contains some text"
        assert selection.passage == "Passage 1 contains some text"
        assert selection.quote_in_passage is True
        assert selection.evidence_found is True
        assert selection.reason == "ok"


class TestApplyGate:
    """Tests für _apply_gate() Methode."""

    def setup_method(self):
        """Setup: Erstelle LLMClaimVerifier Instanz."""
        llm_client = FakeLLMClient()
        self.verifier = LLMClaimVerifier(
            llm_client,
            require_evidence_for_correct=True,
            strict_mode=False,
        )

    def test_incorrect_no_evidence(self):
        """Test 4: label incorrect + no evidence => label_final uncertain, gate_reason no_evidence"""
        out = VerifierLLMOutput(label="incorrect", confidence=0.9)
        selection = EvidenceSelection(
            index=-1,
            quote=None,
            passage=None,
            quote_in_passage=False,
            evidence_found=False,
            reason="no_passage_selected",
        )
        coverage_ok = True
        coverage_note = ""

        decision = self.verifier._apply_gate(out, selection, coverage_ok, coverage_note)

        assert decision.label_raw == "incorrect"
        assert decision.label_final == "uncertain"
        assert decision.confidence == 0.5  # Clamped
        assert decision.gate_reason == "no_evidence"

    def test_correct_require_evidence_no_evidence(self):
        """Test 5: label correct + require_evidence_for_correct + no evidence => label_final uncertain"""
        out = VerifierLLMOutput(label="correct", confidence=0.9)
        selection = EvidenceSelection(
            index=-1,
            quote=None,
            passage=None,
            quote_in_passage=False,
            evidence_found=False,
            reason="no_passage_selected",
        )
        coverage_ok = True
        coverage_note = ""

        decision = self.verifier._apply_gate(out, selection, coverage_ok, coverage_note)

        assert decision.label_raw == "correct"
        assert decision.label_final == "uncertain"
        assert decision.confidence == 0.55  # Clamped
        assert decision.gate_reason == "no_evidence"

    def test_incorrect_evidence_coverage_fail(self):
        """Test 6: label incorrect + evidence_found True + coverage_fail => label_final incorrect, confidence clamped, gate_reason coverage_fail"""
        out = VerifierLLMOutput(label="incorrect", confidence=0.9)
        selection = EvidenceSelection(
            index=0,
            quote="Some quote",
            passage="Passage with quote",
            quote_in_passage=True,
            evidence_found=True,
            reason="ok",
        )
        coverage_ok = False
        coverage_note = "soft coverage too low"

        decision = self.verifier._apply_gate(out, selection, coverage_ok, coverage_note)

        assert decision.label_raw == "incorrect"
        assert decision.label_final == "incorrect"  # Label bleibt incorrect
        assert decision.confidence == 0.5  # Clamped wegen Coverage-Fail
        assert decision.gate_reason == "coverage_fail"
        assert decision.coverage_ok is False

    def test_correct_with_evidence(self):
        """Test: label correct + evidence_found True => label_final correct, gate_reason ok"""
        out = VerifierLLMOutput(label="correct", confidence=0.9)
        selection = EvidenceSelection(
            index=0,
            quote="Some quote",
            passage="Passage with quote",
            quote_in_passage=True,
            evidence_found=True,
            reason="ok",
        )
        coverage_ok = True
        coverage_note = ""

        decision = self.verifier._apply_gate(out, selection, coverage_ok, coverage_note)

        assert decision.label_raw == "correct"
        assert decision.label_final == "correct"
        assert decision.confidence == 0.9
        assert decision.gate_reason == "ok"

    def test_uncertain_no_evidence(self):
        """Test: label uncertain + no evidence => label_final uncertain, gate_reason ok"""
        out = VerifierLLMOutput(label="uncertain", confidence=0.5)
        selection = EvidenceSelection(
            index=-1,
            quote=None,
            passage=None,
            quote_in_passage=False,
            evidence_found=False,
            reason="no_passage_selected",
        )
        coverage_ok = False
        coverage_note = "no evidence"

        decision = self.verifier._apply_gate(out, selection, coverage_ok, coverage_note)

        assert decision.label_raw == "uncertain"
        assert decision.label_final == "uncertain"
        assert decision.confidence == 0.5
        assert decision.gate_reason == "ok"  # Uncertain ist immer ok
