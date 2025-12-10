from typing import Protocol, List
import json
from app.llm.llm_client import LLMClient
from app.services.agents.claim_models import Claim


class ClaimExtractor(Protocol):
    def extract_claims(self, sentence: str, sentence_index: int) -> List[Claim]:
        ...


class LLMClaimExtractor:
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract_claims(self, sentence: str, sentence_index: int) -> List[Claim]:
        prompt = self._build_prompt(sentence)
        raw = self.llm.complete(prompt)
        return self._parse_response(raw, sentence, sentence_index)

    def _build_prompt(self, sentence: str) -> str:
        return f"""
Du bekommst GENAU EINEN Satz aus einer Zusammenfassung.

Deine Aufgabe:
- Extrahiere alle FAKTISCHEN Behauptungen (Claims) aus diesem Satz.
- Ein Claim soll möglichst atomar sein (eine konkrete Aussage).
- Ignoriere reine Meinungen, Bewertungen oder Stil (z.B. "interessant", "schön", "wichtig").

Gib NUR JSON im folgenden Format zurück:

{{
  "claims": [
    {{"text": "Claim 1"}},
    {{"text": "Claim 2"}}
  ]
}}

Regeln:
- Wenn der Satz mehrere Fakten enthält, zerlege sie in mehrere Claims.
- Verwende den Originalinhalt, aber kürze so, dass der Claim als eigenständiger Satz verständlich ist.
- Wenn es KEINE klaren Fakten gibt, gib {{ "claims": [] }} zurück.
- KEIN zusätzlicher Text, KEINE Erklärungen, NUR JSON.

Beispiele:

Satz:
"Lisa wohnt in Berlin und studiert Softwareentwicklung."
Antwort:
{{
  "claims": [
    {{"text": "Lisa wohnt in Berlin."}},
    {{"text": "Lisa studiert Softwareentwicklung."}}
  ]
}}

Satz:
"Die EU wurde 1993 gegründet und hat heute 27 Mitgliedstaaten."
Antwort:
{{
  "claims": [
    {{"text": "Die EU wurde 1993 gegründet."}},
    {{"text": "Die EU hat 27 Mitgliedstaaten."}}
  ]
}}

Satz:
"Lisa findet Berlin schön und liebt Kaffee."
Antwort:
{{
  "claims": [
    {{"text": "Lisa liebt Kaffee."}}
  ]
}}

Jetzt der zu verarbeitende Satz:

SATZ:
"{sentence}"
""".strip()

    def _parse_response(self, raw: str, sentence: str, sentence_index: int) -> List[Claim]:

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            json_str = raw[start:end]
            data = json.loads(json_str)
        except Exception:
            return []

        claims_data = data.get("claims", [])
        claims: List[Claim] = []

        for i, c in enumerate(claims_data):
            text = (c.get("text") or "").strip()
            if not text:
                continue
            claim_id = f"s{sentence_index}_c{i}"
            claims.append(
                Claim(
                    id=claim_id,
                    sentence_index=sentence_index,
                    sentence=sentence,
                    text=text,
                )
            )

        return claims
