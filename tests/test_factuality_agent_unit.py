from services.agents.factuality.factuality_agent import FactualityAgent
from app.llm.llm_client import LLMClient
from services.agents.factuality.claim_models import Claim


# --- Alter Test: Satz-basierter FakeLLM für Regression --- #

class FakeLLMClient(LLMClient):
    def complete(self, prompt: str, **kwargs) -> str:
        # Tut so, als hätte das LLM geprüft und klar falschen Satz erkannt
        return """
        {
          "label": "incorrect",
          "confidence": 0.95,
          "explanation": "Im Quelltext steht Berlin, nicht Paris."
        }
        """


def test_factuality_agent_detects_obvious_error():
    llm = FakeLLMClient()
    agent = FactualityAgent(llm)

    article = "Lisa wohnt in Berlin."
    summary = "Lisa wohnt in Paris."

    result = agent.run(article, summary, meta={})

    assert result.score < 1.0
    assert result.details["num_errors"] >= 1


# --- Neuer Teil: Claim-basierte Tests --- #


class FakeLLMClientForExtractor(LLMClient):
    """
    Dieser Fake-LLM wird NUR für den ClaimExtractor benutzt.
    Er gibt deterministisch genau einen Claim zurück.
    """
    def complete(self, prompt: str, **kwargs) -> str:
        return '{"claims": [{"text": "Das ist ein falscher Claim."}]}'


class FakeClaimExtractor:
    """
    Einfacher Fake-Extractor:
    - tut so, als hätte das LLM einen Claim extrahiert
    """
    def extract_claims(self, sentence: str, sentence_index: int):
        # Wir ignorieren hier die LLM-Antwort und geben direkt einen Claim zurück.
        return [
            Claim(
                id="c1",
                sentence_index=sentence_index,
                sentence=sentence,
                text=sentence,
            )
        ]


class FakeClaimVerifier:
    """
    Fake-ClaimVerifier:
    - alle Claims, die das Wort "falsch" enthalten -> incorrect
    - alle anderen -> correct
    """
    def verify(self, article_text: str, claim: Claim) -> Claim:
        if "falsch" in claim.text.lower():
            claim.label = "incorrect"
            claim.confidence = 0.9
            claim.explanation = "Test: enthält 'falsch'."
            claim.error_type = "OTHER"
        else:
            claim.label = "correct"
            claim.confidence = 0.9
            claim.explanation = "Test: kein Fehler gefunden."
            claim.error_type = None
        claim.evidence = []
        return claim


def test_factuality_agent_uses_claim_verifier():
    llm = FakeLLMClientForExtractor()  # wird hier praktisch nicht gebraucht
    agent = FactualityAgent(
        llm_client=llm,
        claim_extractor=FakeClaimExtractor(),
        claim_verifier=FakeClaimVerifier(),
    )

    result = agent.run(
        article_text="egal",
        summary_text="Das ist ein falscher Claim.",
        meta={},
    )

    assert result.details["num_errors"] == 1
    assert result.score < 1.0


def test_factuality_agent_claim_path_used():
    class FakeLLMForClaims(LLMClient):
        def complete(self, prompt: str, **kwargs) -> str:
            # Unterscheide ClaimExtractor- und ClaimVerifier-Calls anhand des Prompts
            if '"claims"' in prompt:
                # Antwort für LLMClaimExtractor: liefert einen Claim
                return '{"claims": [{"text": "Lisa wohnt in Paris."}]}'
            else:
                # Antwort für LLMClaimVerifier: bewertet diesen Claim als falsch
                return """
                {
                  "label": "incorrect",
                  "confidence": 0.9,
                  "error_type": "ENTITY",
                  "explanation": "Im Artikel steht Berlin.",
                  "evidence": ["Lisa wohnt in Berlin."]
                }
                """

    llm = FakeLLMForClaims()
    agent = FactualityAgent(llm)  # nutzt LLMClaimExtractor & LLMClaimVerifier

    article = "Lisa wohnt in Berlin."
    summary = "Lisa wohnt in Paris."

    result = agent.run(article, summary, meta={})

    # Wir erwarten: Claim-Pfad wurde genutzt, kein Fallback
    assert result.details["num_errors"] == 1
    assert result.score < 1.0
    assert len(result.details["claims"]) == 1

    claim = result.details["claims"][0]
    assert claim["label"] == "incorrect"
    assert claim["error_type"] == "ENTITY"