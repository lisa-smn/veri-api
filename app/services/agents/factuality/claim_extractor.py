"""
Claim Extractor: Extrahiert atomare, faktische Claims aus Sätzen.

LLM-basierte Extraktion mit Substring-Constraint:
- Claims müssen rückverfolgbar im Satz stehen
- Filtert Meta/Readability/Opinion
- Robuster Substring-Check (Whitespace/Quotes normalisiert)
"""

from __future__ import annotations

import json
import re
from typing import Protocol

from app.llm.llm_client import LLMClient
from app.services.agents.factuality.claim_models import Claim


class ClaimExtractor(Protocol):
    def extract_claims(self, sentence: str, sentence_index: int) -> list[Claim]: ...


class LLMClaimExtractor:
    """
    Extrahiert atomare, faktische Claims aus einem Satz.

    Ziel: hoher Recall ohne Müll (Meta/Readability/Opinion), und Claims müssen
    rückverfolgbar im Satz stehen (Substring-Constraint).

    Änderungen ggü. strenger Vorversion:
    - KEIN Gate "digit oder Großbuchstaben" mehr (killt sonst echten Recall).
    - Opinion/Modalität wird nicht pauschal gefiltert; nur "reine" Meinung ohne Anker.
    - Substring-Check ist robuster (Whitespace/Quotes) und normalisiert auf echten Satz-Substring.
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def extract_claims(self, sentence: str, sentence_index: int) -> list[Claim]:
        s = (sentence or "").strip()
        if not s:
            return []

        # Heuristik: offensichtliche Meta-/Bewertungs-/Fragesätze früh rausfiltern
        if self._is_obvious_non_claim_sentence(s):
            return []

        prompt = self._build_prompt(s)
        raw = self.llm.complete(prompt)
        return self._parse_response(raw, s, sentence_index)

    def _is_obvious_non_claim_sentence(self, sentence: str) -> bool:
        """
        Sehr leichte Filter, um LLM-Calls auf "offensichtlich keine Fakten" zu sparen.
        Wichtig: NICHT zu aggressiv, sonst verlierst du Recall.
        """
        s = sentence.strip()
        low = s.lower()

        # Fragen bringen selten sauber verifizierbare Claims (und werden oft halluziniert)
        if s.endswith("?"):
            return True

        # Meta/Readability/Stil/Struktur
        meta_markers = [
            "dieser satz",
            "die summary",
            "die zusammenfassung",
            "der text",
            "im text",
            "lesbar",
            "lesbarkeit",
            "verständlichkeit",
            "schwer verständlich",
            "langer satz",
            "viele kommas",
            "verschachtel",
            "stil",
            "ton",
            "grammatik",
            "orthografie",
            "rechtschreibung",
            "formulierung",
            "this sentence",
            "this summary",
            "the summary",
            "the text",
            "readable",
            "readability",
            "style",
            "tone",
            "grammar",
        ]
        if any(m in low for m in meta_markers):
            return True

        # Reine Bewertung ohne überprüfbaren Kern
        pure_eval = [
            "wichtig",
            "interessant",
            "schön",
            "schlecht",
            "gut",
            "gelungen",
            "unpassend",
            "important",
            "interesting",
            "nice",
            "bad",
            "good",
            "well written",
        ]
        # Wenn Satz extrem kurz UND nur Bewertung: raus
        if len(self._tokens(low)) <= 5 and any(w in low for w in pure_eval):
            return True

        # Modals/Meinung nicht pauschal rauswerfen, nur wenn keine Anker vorhanden sind
        opinion_markers = [
            "ich finde",
            "meiner meinung",
            "aus meiner sicht",
            "wirkt",
            "scheint",
            "könnte",
            "wahrscheinlich",
            "vermutlich",
            "eventuell",
            "möglicherweise",
            "i think",
            "in my opinion",
            "seems",
            "appears",
            "might",
            "probably",
            "perhaps",
            "maybe",
        ]
        if any(m in low for m in opinion_markers):
            # Wenn es trotz Modalität konkrete Anker gibt: NICHT filtern
            has_digit = any(ch.isdigit() for ch in s)
            has_namedish = bool(re.search(r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]+\b", s))
            has_quote = '"' in s or "“" in s or "”" in s or "'" in s
            if not (has_digit or has_namedish or has_quote):
                return True

        # Sehr kurze Füllsätze
        filler = {
            "however",
            "therefore",
            "overall",
            "in summary",
            "zusammenfassend",
            "insgesamt",
            "jedoch",
            "daher",
        }
        if len(self._tokens(low)) <= 3 and any(w in low for w in filler):
            return True

        return False

    def _build_prompt(self, sentence: str) -> str:
        return f"""
Du bekommst GENAU EINEN Satz aus einer Zusammenfassung.

Deine Aufgabe:
Extrahiere NUR überprüfbare, FAKTISCHE Behauptungen (Claims).

Ein Claim ist nur dann gültig, wenn:
- er konkret und überprüfbar ist (gegen einen Artikeltext)
- er KEINE reine Meinung/Bewertung ist
- er NICHT über den Text selbst spricht (keine Meta-Aussagen über Lesbarkeit/Stil/Struktur)

Ignoriere explizit:
- Aussagen über Lesbarkeit/Stil/Ton ("schwer verständlich", "langer Satz", "viele Kommas", "gut geschrieben")
- reine Bewertungen/Meinungen ohne überprüfbaren Kern ("wichtig", "interessant", "schön", "schlecht")
- reine rhetorische Aussagen ohne Faktengehalt
- Fragen (wenn der Satz eine Frage ist, gib claims=[])

WICHTIG:
- Jeder Claim MUSS als wörtlicher Auszug direkt aus dem Satz stammen (exakt kopiert).
- Formuliere nicht um, erfinde nichts.
- Wenn Modalwörter vorkommen ("scheint", "könnte", "probably"), extrahiere wenn möglich den konkreten faktischen Kern
  als Substring (z.B. aus "Es scheint, dass X Y ist" -> "X Y ist"), aber nur wenn dieser Kern wörtlich im Satz steht.
- Wenn der Satz keine klaren Fakten enthält, gib claims=[] zurück.

Gib NUR JSON im folgenden Format zurück:

{{
  "claims": [
    {{"text": "Claim 1 (exakt aus dem Satz kopiert)"}},
    {{"text": "Claim 2 (exakt aus dem Satz kopiert)"}}
  ]
}}

Regeln:
- Maximal 5 Claims.
- Wenn mehrere Fakten enthalten sind: in mehrere atomare Claims zerlegen.
- Keine Duplikate, keine überlappenden Wiederholungen (nicht denselben Claim in längerer Form nochmal ausgeben).
- KEIN zusätzlicher Text, NUR JSON.

SATZ:
"{sentence}"
""".strip()

    def _parse_response(self, raw: str, sentence: str, sentence_index: int) -> list[Claim]:
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            data = json.loads(raw[start:end])
        except Exception:
            return []

        claims_data = data.get("claims", [])
        if not isinstance(claims_data, list):
            return []

        claims: list[Claim] = []
        seen: set[str] = set()

        for i, c in enumerate(claims_data[:5]):
            if not isinstance(c, dict):
                continue

            text = (c.get("text") or "").strip()
            if not text:
                continue

            # Kürzen/Entschärfen von Quote-Rändern (LLMs lieben Anführungszeichen)
            text = self._strip_outer_quotes(text)

            # Mindestlänge
            if len(text) < 4:
                continue

            # Substring-Constraint robust prüfen und auf echten Substring normalisieren
            normalized = self._normalize_to_sentence_substring(sentence, text)
            if not normalized:
                # lieber droppen als falsche Fakten weiterreichen
                continue

            normalized = normalized.strip()
            if normalized in seen:
                continue
            seen.add(normalized)

            claim_id = f"s{sentence_index}_c{i}"
            claims.append(
                Claim(
                    id=claim_id,
                    sentence_index=sentence_index,
                    sentence=sentence,
                    text=normalized,
                )
            )

        return claims

    # ------------------------ Helpers ------------------------ #

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", text or "")

    @staticmethod
    def _strip_outer_quotes(text: str) -> str:
        t = text.strip()
        if (t.startswith('"') and t.endswith('"')) or (t.startswith("“") and t.endswith("”")):
            return t[1:-1].strip()
        if t.startswith("'") and t.endswith("'"):
            return t[1:-1].strip()
        return t

    @staticmethod
    def _normalize_quotes(text: str) -> str:
        # Curly quotes -> straight quotes
        return (text or "").replace("“", '"').replace("”", '"').replace("’", "'").replace("‘", "'")

    def _normalize_to_sentence_substring(self, sentence: str, candidate: str) -> str | None:
        """
        Gibt einen Substring aus 'sentence' zurück, der semantisch dem 'candidate' entspricht,
        aber exakt im Satz vorkommt. Akzeptiert kleine Abweichungen bei Whitespace/Quotes.
        """
        if not sentence or not candidate:
            return None

        # 1) Direkter Treffer
        if candidate in sentence:
            return candidate

        s0 = self._normalize_quotes(sentence)
        c0 = self._normalize_quotes(candidate)

        if c0 in s0:
            # Finde die Position im normalisierten Satz und mappe auf Original
            # (Weiter mit Whitespace-flexiblem Matching unten)
            pass

        # 2) Whitespace-flexibles Matching (Tokens exakt, Whitespace variabel)
        # Beispiel: "A  B" vs "A B" oder Zeilenumbrüche.
        toks = re.findall(r"\S+", c0.strip())
        if not toks:
            return None

        pattern = r"\s+".join(re.escape(t) for t in toks)
        m = re.search(pattern, s0)
        if not m:
            return None

        # Match aus normalisiertem Satz zurückgeben, aber best-effort auf Original-Substring:
        # Suche denselben Match (als Text) im Originalsatz.
        matched = m.group(0)
        matched = matched.strip()

        # Versuche direkt im Originalsatz zu finden (nach Quote-Normalisierung schwierig)
        if matched in sentence:
            return matched

        # Falls nicht: nutze denselben Regex im Originalsatz
        m2 = re.search(pattern, sentence)
        if m2:
            return m2.group(0).strip()

        # Letzter Fallback: gib das normalisierte Match zurück (sollte selten sein)
        # (Wenn du 100% "muss exakt Substring sein" willst: return None statt matched)
        return matched
