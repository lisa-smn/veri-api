from dataclasses import dataclass
from typing import List, Dict, Any
import json
import re

from app.llm.llm_client import LLMClient
from app.models.pydantic import AgentResult
from app.services.agents.claim_models import Claim
from app.services.agents.claim_extractor import ClaimExtractor, LLMClaimExtractor
from app.services.agents.claim_verifier import ClaimVerifier, LLMClaimVerifier


@dataclass
class SentenceResult:
    index: int
    sentence: str
    label: str
    confidence: float
    explanation: str


class FactualityAgent:
    def __init__(
        self,
        llm_client: LLMClient,
        claim_extractor: ClaimExtractor | None = None,
        claim_verifier: ClaimVerifier | None = None,
    ):
        self.llm = llm_client
        self.claim_extractor = claim_extractor or LLMClaimExtractor(llm_client)
        self.claim_verifier = claim_verifier or LLMClaimVerifier(llm_client)

    def run(
        self,
        article_text: str,
        summary_text: str,
        meta: Dict[str, Any] | None = None,
    ) -> AgentResult:
        sentences = self._split(summary_text)

        # 1) Claims aus allen Sätzen extrahieren
        raw_claims: List[Claim] = []
        for i, s in enumerate(sentences):
            claims = self.claim_extractor.extract_claims(s, i)
            raw_claims.extend(claims)

        # Falls keine Claims gefunden wurden, fallback auf Satzlogik
        if not raw_claims:
            results = [self._check_sentence(article_text, s, i) for i, s in enumerate(sentences)]
            score = self._compute_score_from_sentences(results)
            explanation = self._build_global_explanation_from_sentences(score, results)
            return AgentResult(
                name="factuality",
                score=score,
                explanation=explanation,
                details={
                    "sentences": [r.__dict__ for r in results],
                    "claims": [],
                    "num_errors": sum(1 for r in results if r.label == "incorrect"),
                },
                errors=[
                    f"Satz {r.index}: '{r.sentence}' – {r.explanation}"
                    for r in results
                    if r.label == "incorrect"
                ],
            )

        # 2) Jede Behauptung faktisch prüfen
        verified_claims: List[Claim] = [
            self.claim_verifier.verify(article_text, c) for c in raw_claims
        ]

        # 3) Score aus Claims berechnen
        score = self._compute_score_from_claims(verified_claims)

        # 4) Erklärung aus Claims bauen
        explanation = self._build_global_explanation_from_claims(score, verified_claims)

        # 5) Fehler-Liste aus Claims ableiten (Strings, wie bisher)
        error_claims = [c for c in verified_claims if c.label == "incorrect"]
        errors = [
            f"Satz {c.sentence_index}: Claim '{c.text}' – {c.explanation}"
            for c in error_claims
        ]

        # 6) Details zusammenbauen
        return AgentResult(
            name="factuality",
            score=score,
            explanation=explanation,
            details={
                "sentences": [
                    {"index": i, "text": s}
                    for i, s in enumerate(sentences)
                ],
                "claims": [c.__dict__ for c in verified_claims],
                "num_errors": len(error_claims),
            },
            errors=errors,
        )

    # --------- Hilfsfunktionen für Text & LLM ---------- #

    def _split(self, text: str) -> List[str]:
        parts = re.split(r"[.!?]\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _build_prompt(self, article: str, sentence: str) -> str:
        return f"""
Du bekommst einen QUELLTEXT und einen zu prüfenden SATZ.

Entscheide:
- correct → Satz wird durch Quelle gestützt
- incorrect → Satz widerspricht Quelle
- uncertain → Quelle enthält nicht genug Informationen

Gib das Ergebnis als JSON zurück:

{{
  "label": "correct" | "incorrect" | "uncertain",
  "confidence": 0.0,
  "explanation": "kurze Begründung"
}}

QUELLTEXT:
{article}

SATZ:
{sentence}
""".strip()

    def _check_sentence(self, article_text: str, sentence: str, index: int) -> SentenceResult:
        prompt = self._build_prompt(article_text, sentence)
        raw = self.llm.complete(prompt)
        data = self._parse_output(raw)
        return SentenceResult(
            index=index,
            sentence=sentence,
            label=data["label"],
            confidence=data["confidence"],
            explanation=data["explanation"],
        )

    def _parse_output(self, raw: str) -> Dict[str, Any]:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            json_str = raw[start:end]
            data = json.loads(json_str)
        except Exception:
            return {
                "label": "uncertain",
                "confidence": 0.0,
                "explanation": f"Antwort nicht parsbar: {raw[:200]}",
            }

        label = data.get("label", "uncertain")
        if label not in ("correct", "incorrect", "uncertain"):
            label = "uncertain"

        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        return {
            "label": label,
            "confidence": confidence,
            "explanation": data.get("explanation", ""),
        }

    # --------- M5-Kompatibilität (Satz-basiert, Fallback) --------- #

    def _compute_score(self, results: List[SentenceResult]) -> float:
        # Wrapper für alte Aufrufer, falls irgendwo noch genutzt
        return self._compute_score_from_sentences(results)

    def _compute_score_from_sentences(self, results: List[SentenceResult]) -> float:
        if not results:
            return 1.0
        correct = sum(1 for r in results if r.label == "correct")
        return correct / len(results)

    def _build_global_explanation_from_sentences(self, score: float, results: List[SentenceResult]) -> str:
        errors = [r for r in results if r.label == "incorrect"]
        if errors:
            first = errors[0]
            return (
                f"Score {score:.2f}. Es wurden {len(errors)} Fehler erkannt. "
                f"Beispiel Fehler: Satz {first.index}: '{first.sentence}' – {first.explanation}"
            )
        return f"Score {score:.2f}. Alle Sätze sind durch den Artikel gedeckt."

    # --------- Claim-basierte Hilfsfunktionen (M6) --------- #

    def _compute_score_from_claims(self, claims: List[Claim]) -> float:
        if not claims:
            return 1.0
        correct = sum(1 for c in claims if c.label == "correct")
        return correct / len(claims)

    def _build_global_explanation_from_claims(self, score: float, claims: List[Claim]) -> str:
        errors = [c for c in claims if c.label == "incorrect"]
        if errors:
            first = errors[0]
            return (
                f"Score {score:.2f}. Es wurden {len(errors)} falsche Claims erkannt. "
                f"Beispiel Fehler: Satz {first.sentence_index}: Claim '{first.text}' – {first.explanation}"
            )
        return f"Score {score:.2f}. Alle geprüften Claims sind durch den Artikel gedeckt oder unklar."
