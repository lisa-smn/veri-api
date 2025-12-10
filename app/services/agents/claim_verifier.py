# app/services/agents/claim_verifier.py
from typing import Protocol
from app.services.agents.claim_models import Claim
from app.llm.llm_client import LLMClient
import json


class ClaimVerifier(Protocol):
    def verify(self, article_text: str, claim: Claim) -> Claim:
        ...

class LLMClaimVerifier:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def verify(self, article_text: str, claim: Claim) -> Claim:
        prompt = self._build_prompt(article_text, claim.text)
        raw = self.llm.complete(prompt)
        data = self._parse_output(raw)

        claim.label = data["label"]
        claim.confidence = data["confidence"]
        claim.explanation = data["explanation"]
        claim.error_type = data["error_type"]
        claim.evidence = data["evidence"]
        return claim

    def _build_prompt(self, article: str, claim_text: str) -> str:
        return f"""
Du bekommst einen QUELLTEXT (Artikel) und eine einzelne Behauptung (Claim).

Entscheide:
- "correct"   → Claim wird durch die Quelle gestützt
- "incorrect" → Claim widerspricht der Quelle
- "uncertain" → Quelle enthält nicht genug Informationen

Zusätzlich:
- bestimme einen groben Fehlertyp, falls "incorrect":
  - "ENTITY"  → falscher Name, Person, Ort etc.
  - "NUMBER"  → falsche Zahl, Menge, Prozent etc.
  - "DATE"    → falsches Datum, Jahr, Reihenfolge
  - "OTHER"   → sonstige Widersprüche

Gib NUR JSON zurück:

{{
  "label": "correct" | "incorrect" | "uncertain",
  "confidence": 0.0,
  "error_type": "ENTITY" | "NUMBER" | "DATE" | "OTHER" | null,
  "explanation": "kurze Begründung",
  "evidence": ["optional ein oder zwei relevante Zitate aus der Quelle"]
}}

QUELLTEXT:
{article}

CLAIM:
{claim_text}
""".strip()

    def _parse_output(self, raw: str) -> dict:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            json_str = raw[start:end]
            data = json.loads(json_str)
        except Exception:
            return {
                "label": "uncertain",
                "confidence": 0.0,
                "error_type": None,
                "explanation": f"Antwort nicht parsbar: {raw[:200]}",
                "evidence": [],
            }

        label = data.get("label", "uncertain")
        if label not in ("correct", "incorrect", "uncertain"):
            label = "uncertain"

        try:
            confidence = float(data.get("confidence", 0.0))
        except Exception:
            confidence = 0.0

        error_type = data.get("error_type")
        if error_type not in ("ENTITY", "NUMBER", "DATE", "OTHER", None):
            error_type = None

        evidence = data.get("evidence") or []
        if not isinstance(evidence, list):
            evidence = [str(evidence)]

        return {
            "label": label,
            "confidence": confidence,
            "error_type": error_type,
            "explanation": data.get("explanation", ""),
            "evidence": evidence,
        }


