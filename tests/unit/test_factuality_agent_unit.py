import pytest

from app.llm.llm_client import LLMClient
from app.services.agents.factuality.factuality_agent import FactualityAgent
from app.services.agents.factuality.claim_models import Claim


class FakeLLMAlwaysIncorrectSentence(LLMClient):
    """
    Fake für den Satz-Fallback-Pfad:
    _check_sentence() bekommt immer 'incorrect' zurück.
    """
    def complete(self, prompt: str, **kwargs) -> str:
        return """
        {
          "label": "incorrect",
          "confidence": 0.95,
          "explanation": "Im Quelltext steht Berlin, nicht Paris."
        }
        """


class FakeClaimExtractorEmpty:
    """Extractor liefert absichtlich keine Claims -> Fallback-Pfad."""
    def extract_claims(self, sentence: str, sentence_index: int):
        return []


class FakeClaimExtractorOneIncorrect:
    """Extractor liefert genau einen Claim, der als 'incorrect' verifiziert werden kann."""
    def extract_claims(self, sentence: str, sentence_index: int):
        return [
            Claim(
                id=f"s{sentence_index}_c0",
                sentence_index=sentence_index,
                sentence=sentence,
                text="Lisa wohnt in Paris.",
            )
        ]


class FakeClaimVerifierMarksIncorrect:
    """Verifier markiert jeden Claim als incorrect."""
    def verify(self, article_text: str, claim: Claim) -> Claim:
        claim.label = "incorrect"
        claim.confidence = 0.9
        claim.error_type = "ENTITY"
        claim.explanation = "Im Artikel steht Berlin."
        claim.evidence = ["Lisa wohnt in Berlin."]
        return claim


class FakeLLMForExtractorAndVerifier(LLMClient):
    """
    FakeLLM für den echten LLMClaimExtractor + LLMClaimVerifier Pfad.
    Unterscheidet deterministisch anhand deiner echten Prompt-Texte.
    """
    def complete(self, prompt: str, **kwargs) -> str:
        # ClaimExtractor-Prompt erkennen
        if (
            "Extrahiere alle FAKTISCHEN Behauptungen" in prompt
            and "Gib NUR JSON im folgenden Format zurück" in prompt
            and '"claims"' in prompt
        ):
            return '{"claims": [{"text": "Lisa wohnt in Paris."}]}'

        # ClaimVerifier-Prompt erkennen
        if (
            "Du bekommst einen QUELLTEXT (Artikel) und eine einzelne Behauptung (Claim)." in prompt
            and "Gib NUR JSON zurück" in prompt
            and '"error_type"' in prompt
        ):
            return """
            {
              "label": "incorrect",
              "confidence": 0.9,
              "error_type": "ENTITY",
              "explanation": "Im Artikel steht Berlin.",
              "evidence": ["Lisa wohnt in Berlin."]
            }
            """

        raise AssertionError(f"Unerwarteter Prompt in FakeLLM:\n{prompt[:400]}")


def test_factuality_agent_fallback_sentence_path_sets_issue_spans_and_num_incorrect():
    llm = FakeLLMAlwaysIncorrectSentence()

    # Erzwinge Fallback-Pfad: ClaimExtractor liefert keine Claims
    agent = FactualityAgent(
        llm_client=llm,
        claim_extractor=FakeClaimExtractorEmpty(),
        claim_verifier=None,  # wird nicht genutzt im Fallback
    )

    article = "Lisa wohnt in Berlin."
    summary = "Lisa wohnt in Paris."

    result = agent.run(article, summary, meta={})

    assert result.name == "factuality"
    assert result.score < 1.0
    assert result.details is not None
    assert result.details["num_incorrect"] >= 1
    assert "sentences" in result.details
    assert len(result.issue_spans) >= 1


def test_factuality_agent_claim_path_uses_extractor_and_verifier_fakes():
    llm = FakeLLMAlwaysIncorrectSentence()  # wird hier nicht genutzt

    agent = FactualityAgent(
        llm_client=llm,
        claim_extractor=FakeClaimExtractorOneIncorrect(),
        claim_verifier=FakeClaimVerifierMarksIncorrect(),
    )

    article = "Lisa wohnt in Berlin."
    summary = "Lisa wohnt in Paris."

    result = agent.run(article, summary, meta={})

    assert result.name == "factuality"
    assert result.score < 1.0
    assert result.details is not None
    assert result.details["num_incorrect"] == 1
    assert "claims" in result.details
    assert len(result.details["claims"]) >= 1
    assert len(result.issue_spans) == 1

    span = result.issue_spans[0]
    assert span.message
    # start/end können None sein, wenn find() fehlschlägt, aber bei diesem Text sollte es klappen
    assert span.start_char is None or span.start_char >= 0


def test_factuality_agent_claim_path_with_real_llm_classes_fake_llm():
    llm = FakeLLMForExtractorAndVerifier()
    agent = FactualityAgent(llm)  # nutzt LLMClaimExtractor & LLMClaimVerifier

    article = "Lisa wohnt in Berlin."
    summary = "Lisa wohnt in Paris."

    result = agent.run(article, summary, meta={})

    assert result.name == "factuality"
    assert result.score < 1.0
    assert result.details is not None

    # Wir erwarten Claim-Pfad (nicht Satz-Fallback)
    assert "claims" in result.details
    assert len(result.details["claims"]) == 1
    assert result.details["num_incorrect"] == 1

    claim = result.details["claims"][0]
    assert claim["label"] == "incorrect"
    assert claim["error_type"] == "ENTITY"

    assert len(result.issue_spans) >= 1
