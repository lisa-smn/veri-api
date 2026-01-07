from app.services.agents.factuality.claim_extractor import LLMClaimExtractor
from app.llm.fake_client import FakeLLMClient


def test_claim_extractor_skips_readability_meta_sentence():
    extractor = LLMClaimExtractor(FakeLLMClient())

    s = "Das ist ein unfassbar langer Satz, mit vielen Kommas und Einschüben, der schwer verständlich ist."
    assert extractor._is_obvious_non_claim_sentence(s) is True

    # Wenn es ein Meta-Satz ist, soll auch extract_claims() leer zurückgeben
    claims = extractor.extract_claims(s, sentence_index=0)
    assert claims == []
