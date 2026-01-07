"""
Dieser Unit-Test zeigt, dass der Factuality-Agent wirklich das tut, was er soll:
Er nimmt Artikel + Summary, extrahiert daraus überprüfbare Fakten-Behauptungen (Claims),
prüft diese gegen den Artikel und markiert falsche Aussagen als "incorrect".
Wichtig ist dabei nicht nur der Score, sondern auch die nachvollziehbaren Details:
- in `details` steht, wie viele Claims geprüft wurden und wie viele davon falsch sind
- in `issue_spans` wird die betroffene Textstelle in der Summary verortet (Evidence)
Der Test benutzt bewusst einfache, klare Beispielsätze, die durch die eingebauten Filter kommen,
damit wirklich der Prüfprozess getestet wird (und nicht nur, ob ein Satz vorher weggefiltert wurde).
So kann man sicher sein, dass die Factuality-Pipeline reproduzierbar Fehler erkennt und sauber strukturiert ausgibt.
"""

import json
from typing import List

from app.services.agents.factuality.factuality_agent import FactualityAgent
from app.services.agents.factuality.claim_models import Claim


# -----------------------------
# Fakes (deterministisch)
# -----------------------------

class FakeClaimExtractorEmpty:
    """Erzwingt: Extractor liefert keine Claims -> Agent nutzt Fallback-Claim pro Satz."""
    def extract_claims(self, sentence: str, sentence_index: int) -> List[Claim]:
        return []


class FakeClaimExtractorOneClaim:
    """Liefert genau einen Claim, der ein Substring des Satzes ist."""
    def extract_claims(self, sentence: str, sentence_index: int) -> List[Claim]:
        # Substring muss im Satz vorkommen, sonst wird später Mapping/Spans schwierig.
        text = "Lisa wohnt in Paris"
        assert text in sentence  # Test-Fail früh & eindeutig, falls jemand Summary ändert
        return [
            Claim(
                id=f"s{sentence_index}_c0",
                sentence_index=sentence_index,
                sentence=sentence,
                text=text,
            )
        ]


class FakeClaimVerifierMarksIncorrect:
    """Markiert jeden Claim als incorrect (deterministisch)."""
    def verify(self, article_text: str, claim: Claim) -> Claim:
        claim.label = "incorrect"
        claim.confidence = 0.95
        claim.error_type = "ENTITY"
        claim.explanation = "Im Quelltext steht Berlin, nicht Paris."
        claim.evidence = ["Lisa wohnt in Berlin"]
        return claim


class FakeLLMForExtractorAndVerifier:
    """
    Fake-LLM, der EXAKT die JSON-Formate liefert, die
    - LLMClaimExtractor._parse_response
    - LLMClaimVerifier._parse_output
    erwarten.
    """
    def complete(self, prompt: str) -> str:
        # Extractor-Prompt erkennen
        if "Extrahiere NUR überprüfbare, FAKTISCHE Behauptungen" in prompt and '"claims"' in prompt:
            # Muss Substring des SATZES sein, sonst droppt der Extractor den Claim.
            return json.dumps({"claims": [{"text": "Lisa wohnt in Paris"}]})

        # Verifier-Prompt erkennen
        if "Du bekommst einen QUELLTEXT (Artikel) und eine einzelne Behauptung (Claim)" in prompt and '"label"' in prompt:
            return json.dumps(
                {
                    "label": "incorrect",
                    "confidence": 0.95,
                    "error_type": "ENTITY",
                    "explanation": "Im Quelltext steht Berlin, nicht Paris.",
                    "evidence": ["Lisa wohnt in Berlin"],
                }
            )

        # Fallback: parsbar, neutral
        return json.dumps(
            {
                "label": "uncertain",
                "confidence": 0.0,
                "error_type": None,
                "explanation": "Fallback",
                "evidence": [],
            }
        )


# -----------------------------
# Tests
# -----------------------------

def test_factuality_agent_fallback_claim_when_extractor_empty_sets_incorrect_and_issue_span():
    """
    Extractor liefert [] -> Agent erstellt Fallback-Claim pro Satz -> Verifier markiert incorrect.
    Erwartung: num_incorrect>=1, score<1.0, issue_spans vorhanden.
    """
    llm = FakeLLMForExtractorAndVerifier()  # wird hier nicht genutzt, aber ok
    agent = FactualityAgent(
        llm_client=llm,
        claim_extractor=FakeClaimExtractorEmpty(),
        claim_verifier=FakeClaimVerifierMarksIncorrect(),
    )

    article = "Lisa wohnt in Berlin und studiert Softwareentwicklung."
    summary = "Lisa wohnt in Paris und studiert Medizin"

    result = agent.run(article_text=article, summary_text=summary, meta={})

    assert result.name == "factuality"
    assert result.score < 1.0
    assert result.details is not None

    # neue Contract-Felder
    assert "sentences" in result.details
    assert "claims" in result.details
    assert result.details.get("num_incorrect", 0) >= 1
    assert result.details.get("num_claims", 0) >= 1

    # Labels kommen über claims
    claims = result.details["claims"]
    assert any(c.get("label") == "incorrect" for c in claims)

    # Issue spans müssen existieren und den Claim-Text referenzieren
    assert result.issue_spans is not None
    assert len(result.issue_spans) >= 1
    assert any("Paris" in (sp.message or "") for sp in result.issue_spans)


def test_factuality_agent_claim_path_uses_extractor_and_verifier_fakes():
    """
    Claim-Pfad mit FakeExtractor + FakeVerifier: deterministisch und unabhängig von LLM-Prompts.
    """
    llm = FakeLLMForExtractorAndVerifier()  # nicht genutzt, aber ok
    agent = FactualityAgent(
        llm_client=llm,
        claim_extractor=FakeClaimExtractorOneClaim(),
        claim_verifier=FakeClaimVerifierMarksIncorrect(),
    )

    article = "Lisa wohnt in Berlin und studiert Softwareentwicklung."
    summary = "Lisa wohnt in Paris und studiert Medizin"

    result = agent.run(article_text=article, summary_text=summary, meta={})

    assert result.name == "factuality"
    assert result.score < 1.0
    assert result.details is not None

    # Claim ist da und incorrect
    claims = result.details.get("claims", [])
    assert len(claims) >= 1
    assert any(c.get("label") == "incorrect" for c in claims)

    # Counts sind konsistent zum Claim-Label
    assert result.details.get("num_incorrect") == 1
    assert result.details.get("num_claims") >= 1

    # Spans existieren
    assert result.issue_spans is not None
    assert len(result.issue_spans) >= 1


def test_factuality_agent_with_real_llm_extractor_and_verifier_using_fake_llm():
    """
    Nutzt die REALEN LLMClaimExtractor/LLMClaimVerifier Klassen,
    aber mit FakeLLM, das die erwarteten JSON-Schemata liefert.
    """
    llm = FakeLLMForExtractorAndVerifier()
    agent = FactualityAgent(llm_client=llm)  # default: LLMClaimExtractor + LLMClaimVerifier

    article = "Lisa wohnt in Berlin und studiert Softwareentwicklung."
    summary = "Lisa wohnt in Paris und studiert Medizin"

    result = agent.run(article_text=article, summary_text=summary, meta={})

    assert result.name == "factuality"
    assert result.score < 1.0
    assert result.details is not None

    claims = result.details.get("claims", [])
    assert len(claims) >= 1

    # mindestens ein incorrect Claim aus dem Verifier
    incorrect = [c for c in claims if c.get("label") == "incorrect"]
    assert len(incorrect) >= 1

    # Verifier-Felder müssen da sein (Claim dataclass Contract)
    c0 = incorrect[0]
    assert c0.get("confidence") is not None
    assert c0.get("explanation")
    assert "evidence" in c0
    assert isinstance(c0["evidence"], list)

    # Spans müssen existieren
    assert result.issue_spans is not None
    assert len(result.issue_spans) >= 1
