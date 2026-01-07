"""
Claim Verifier: Verifiziert Claims gegen Artikel mit Evidence-Gate.

Pipeline-basierte Architektur mit klaren Invarianten:
- Evidence-Retrieval (sliding windows)
- LLM-basierte Verifikation
- Evidence-Validierung (Quote-Matching, Coverage-Check)
- Gate-Logik (correct/incorrect nur mit evidence_found=True)

Wichtig: Gate-Logik sitzt ausschließlich hier, keine zusätzlichen
Safety-Downgrades im FactualityAgent.
"""

from __future__ import annotations

from typing import Protocol, List, Tuple, Optional
import json
import re

from app.services.agents.factuality.claim_models import Claim, EvidenceSpan
from app.services.agents.factuality.evidence_retriever import EvidenceRetriever
from app.services.agents.factuality.verifier_models import (
    VerifierLLMOutput,
    EvidenceSelection,
    GateDecision,
)
from app.llm.llm_client import LLMClient


class ClaimVerifier(Protocol):
    def verify(self, article_text: str, claim: Claim) -> Claim:
        ...


class LLMClaimVerifier:
    """
    Verifiziert einen einzelnen Claim gegen den Artikel.
    
    Clean-Code Architektur:
    - Pipeline-basierte verify() Methode
    - Klare Invarianten für Evidence und Gate
    - Keine doppelten Gates, keine __dict__ Hacks
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        top_k_sentences: int = 8,
        neighbor_window: int = 1,
        max_context_chars: int = 6000,
        require_evidence_for_correct: bool = True,
        min_soft_token_coverage: float = 0.25,
        max_evidence: int = 2,
        use_evidence_retriever: bool = True,
        evidence_retriever_top_k: int = 5,
        strict_mode: bool = False,
    ):
        self.llm = llm_client
        self.top_k_sentences = top_k_sentences
        self.neighbor_window = neighbor_window
        self.max_context_chars = max_context_chars
        self.require_evidence_for_correct = require_evidence_for_correct
        self.min_soft_token_coverage = min_soft_token_coverage
        self.max_evidence = max_evidence
        self.strict_mode = strict_mode
        
        # Evidence Retriever
        self.use_evidence_retriever = use_evidence_retriever
        if use_evidence_retriever:
            self.evidence_retriever = EvidenceRetriever(top_k=evidence_retriever_top_k)
        else:
            self.evidence_retriever = None

    def verify(self, article_text: str, claim: Claim) -> Claim:
        """
        Pipeline-basierte Verifikation mit klaren Schritten.
        """
        # 1) Retrieve Passages
        passages, scores = self._retrieve_passages(article_text, claim.text)
        
        # 2) Call LLM
        raw = self._call_llm(passages, claim.text)
        
        # 3) Parse LLM Output
        parsed, parse_error = self._parse_llm_output(raw)
        
        # 4) Validate Evidence (harte Invarianten)
        selection = self._validate_evidence(parsed, passages)
        
        # 5) Coverage Check
        coverage_ok, coverage_note = self._coverage_check(claim.text, selection)
        
        # 6) Apply Gate
        decision = self._apply_gate(parsed, selection, coverage_ok, coverage_note)
        
        # 7) Populate Claim
        return self._populate_claim(
            claim, raw, parsed, passages, scores, selection, decision, parse_error
        )

    def _retrieve_passages(
        self, article_text: str, claim_text: str
    ) -> Tuple[List[str], List[float]]:
        """
        Retrieviert relevante Passagen für den Claim.
        Returns: (passages, scores)
        """
        passages = []
        scores = []
        
        if self.use_evidence_retriever and self.evidence_retriever:
            passages, scores = self.evidence_retriever.retrieve(article_text, claim_text)
        
        # Fallback: Wenn Retriever keine Passagen findet, verwende alten Context-Ansatz
        if not passages:
            context = self._select_context(
                article_text,
                claim_text,
                top_k=self.top_k_sentences,
                neighbor_window=self.neighbor_window,
                max_chars=self.max_context_chars,
            )
            # Split context in Passagen (Sätze)
            passages = self._split_sentences(context)
            scores = [0.5] * len(passages)  # Dummy-Scores
        
        # Stelle sicher, dass mindestens eine Passage vorhanden ist
        if not passages:
            passages = [article_text[:self.max_context_chars]]
            scores = [0.0]
        
        return passages, scores

    def _call_llm(self, passages: List[str], claim_text: str) -> str:
        """
        Ruft LLM mit Prompt auf.
        """
        if passages:
            evidence_context_list = passages
            evidence_context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages))
        else:
            evidence_context_list = []
            evidence_context = ""
        
        prompt = self._build_prompt(evidence_context, evidence_context_list, claim_text)
        return self.llm.complete(prompt)

    def _parse_llm_output(
        self, raw: str
    ) -> Tuple[VerifierLLMOutput, Optional[str]]:
        """
        Parst LLM Output zu VerifierLLMOutput.
        Returns: (parsed_output, parse_error)
        """
        parse_error = None
        
        try:
            # Extrahiere JSON aus raw
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start < 0 or end <= start:
                raise ValueError("Kein JSON-Objekt gefunden")
            
            data = json.loads(raw[start:end])
            
            # Validiere mit Pydantic
            parsed = VerifierLLMOutput.model_validate(data)
            return parsed, None
            
        except Exception as e:
            parse_error = str(e)
            
            if self.strict_mode:
                raise ValueError(
                    f"JSON Parsing Fehler in Strict-Mode: {parse_error}. "
                    f"Raw Output Preview (800 chars):\n{raw[:800]}"
                ) from e
            
            # Non-Strict: Default Output
            return VerifierLLMOutput(
                label="uncertain",
                confidence=0.0,
                error_type=None,
                explanation=f"Antwort nicht parsbar: {parse_error}",
                selected_evidence_index=-1,
                evidence_quote=None,
            ), parse_error

    def _normalize_for_match(self, s: str) -> str:
        """
        Normalisiert String für robustes Quote-Matching.
        - strip()
        - Whitespace-Sequenzen zu single space
        - Typografische Quotes zu normalen Quotes
        - Optional: Unicode-Normalisierung (NFKC)
        """
        if not s:
            return ""
        
        # Strip
        s = s.strip()
        
        # Whitespace-Sequenzen zu single space
        s = re.sub(r'\s+', ' ', s)
        
        # Typografische Quotes zu normalen Quotes
        s = s.replace('"', '"').replace('"', '"')  # Typografische doppelte Quotes
        s = s.replace(''', "'").replace(''', "'")  # Typografische einfache Quotes
        
        # Unicode-Normalisierung (NFKC: kompatibel + komponiert)
        try:
            import unicodedata
            s = unicodedata.normalize('NFKC', s)
        except ImportError:
            # Fallback wenn unicodedata nicht verfügbar
            pass
        
        return s

    def _validate_evidence(
        self, out: VerifierLLMOutput, passages: List[str]
    ) -> EvidenceSelection:
        """
        Validiert Evidence-Auswahl mit harten Invarianten.
        
        Invarianten:
        - selected_evidence_index == -1 => evidence_quote is None => evidence_found = False
        - selected_evidence_index >= 0 => evidence_quote ist nicht leer UND ist Substring der Passage => evidence_found = True
        """
        idx = out.selected_evidence_index
        quote = out.evidence_quote.strip() if out.evidence_quote else None
        
        # Invariante 1: idx == -1 => keine Evidence
        if idx == -1:
            return EvidenceSelection(
                index=-1,
                quote=None,
                passage=None,
                quote_in_passage=False,
                evidence_found=False,
                reason="no_passage_selected",
            )
        
        # Invariante 2: idx out of range
        if idx < 0 or idx >= len(passages):
            return EvidenceSelection(
                index=-1,
                quote=None,
                passage=None,
                quote_in_passage=False,
                evidence_found=False,
                reason="index_out_of_range",
            )
        
        # Invariante 3: quote empty
        if not quote:
            return EvidenceSelection(
                index=-1,
                quote=None,
                passage=None,
                quote_in_passage=False,
                evidence_found=False,
                reason="empty_quote",
            )
        
        # Invariante 4: quote muss Substring der Passage sein (mit normalisiertem Matching)
        selected_passage = passages[idx]
        # Normalisiere für Matching, aber behalte Originale für Logging
        normalized_quote = self._normalize_for_match(quote)
        normalized_passage = self._normalize_for_match(selected_passage)
        quote_in_passage = normalized_quote in normalized_passage
        
        if not quote_in_passage:
            return EvidenceSelection(
                index=-1,
                quote=None,  # Original quote wird nicht gespeichert wenn nicht in Passage
                passage=None,
                quote_in_passage=False,
                evidence_found=False,
                reason="quote_not_in_passage",
            )
        
        # Alles OK: Evidence gefunden
        # Speichere ORIGINAL quote und passage (nicht normalisiert) für Logging/Debug
        return EvidenceSelection(
            index=idx,
            quote=quote,  # Original quote
            passage=selected_passage,  # Original passage
            quote_in_passage=True,
            evidence_found=True,
            reason="ok",
        )

    def _coverage_check(
        self, claim_text: str, selection: EvidenceSelection
    ) -> Tuple[bool, str]:
        """
        Prüft Coverage: Evidence muss Claim abdecken.
        Verwendet die GESAMTE Passage für Coverage-Check.
        """
        if not selection.evidence_found:
            return False, "no evidence"
        
        if not selection.passage:
            return False, "no passage"
        
        # Verwende die gesamte Passage für Coverage
        evidence = [selection.passage]
        return self._evidence_covers_claim(claim_text, evidence)

    def _apply_gate(
        self,
        out: VerifierLLMOutput,
        selection: EvidenceSelection,
        coverage_ok: bool,
        coverage_note: str,
    ) -> GateDecision:
        """
        Wendet Gate-Logik an.
        
        Gate Invarianten:
        - label in {"correct","incorrect"} ist nur erlaubt, wenn evidence_found == True
        - label == "correct" ohne Evidence => "uncertain" (wenn require_evidence_for_correct=True)
        - label == "incorrect" ohne Evidence => "uncertain"
        - Coverage-Fail ist KEIN harter Downgrade-Grund, sondern clamp der Confidence
        """
        label_raw = out.label
        label_final = label_raw
        conf = self._clamp01(out.confidence)
        gate_reason = "ok"
        
        # Gate 1: correct ohne Evidence (wenn require_evidence_for_correct=True)
        # Muss VOR Gate 2 kommen, da es spezifischer ist
        if (
            label_raw == "correct"
            and self.require_evidence_for_correct
            and not selection.evidence_found
        ):
            label_final = "uncertain"
            conf = min(conf, 0.55)
            gate_reason = "no_evidence"
        
        # Gate 2: incorrect ohne Evidence => uncertain
        elif label_raw == "incorrect" and not selection.evidence_found:
            label_final = "uncertain"
            conf = min(conf, 0.5)
            gate_reason = "no_evidence"
        
        # Gate 3: incorrect mit Evidence aber Coverage-Fail => Confidence clamp (label bleibt incorrect)
        elif label_raw == "incorrect" and selection.evidence_found and not coverage_ok:
            conf = min(conf, 0.5)
            gate_reason = "coverage_fail"
        
        return GateDecision(
            label_raw=label_raw,
            label_final=label_final,
            confidence=conf,
            gate_reason=gate_reason,
            coverage_ok=coverage_ok,
            coverage_note=coverage_note,
        )

    def _populate_claim(
        self,
        claim: Claim,
        raw: str,
        parsed: VerifierLLMOutput,
        passages: List[str],
        scores: List[float],
        selection: EvidenceSelection,
        decision: GateDecision,
        parse_error: Optional[str],
    ) -> Claim:
        """
        Populiert Claim mit Ergebnissen.
        """
        # Basis-Felder
        claim.label = decision.label_final
        claim.confidence = decision.confidence
        claim.error_type = parsed.error_type if decision.label_final == "incorrect" else None
        claim.explanation = parsed.explanation
        
        # Evidence-Felder
        claim.selected_evidence_index = selection.index
        claim.evidence_quote = selection.quote
        claim.evidence_found = selection.evidence_found
        claim.retrieved_passages = passages
        claim.retrieval_scores = scores
        
        # Legacy evidence
        claim.evidence = [selection.quote] if selection.evidence_found and selection.quote else []
        claim.evidence_spans = claim.evidence
        
        # Evidence Spans Structured
        claim.evidence_spans_structured = []
        if selection.evidence_found and selection.quote:
            claim.evidence_spans_structured.append(
                EvidenceSpan(
                    text=selection.quote,
                    source="article",
                )
            )
        
        # Rationale
        claim.rationale = (
            parsed.explanation[:150].rstrip() + ("…" if len(parsed.explanation) > 150 else "")
            if parsed.explanation
            else None
        )
        
        # Debug-Felder
        claim.parse_ok = parse_error is None
        claim.parse_error = parse_error
        claim.raw_verifier_output = raw[:800] + ("..." if len(raw) > 800 else "")
        claim.selected_evidence_index_raw = parsed.selected_evidence_index
        claim.evidence_quote_raw = parsed.evidence_quote
        
        # Schema violation reason (wenn selection.reason != "ok" und label_raw in ("correct","incorrect"))
        if selection.reason != "ok" and decision.label_raw in ("correct", "incorrect"):
            claim.schema_violation_reason = selection.reason
        else:
            claim.schema_violation_reason = None
        
        # Debug-Felder für Gate-Entscheidungsweg (in __dict__ für Serialisierung)
        claim.__dict__["label_raw"] = decision.label_raw
        claim.__dict__["label_final"] = decision.label_final
        claim.__dict__["gate_reason"] = decision.gate_reason
        claim.__dict__["coverage_ok"] = decision.coverage_ok
        claim.__dict__["coverage_note"] = decision.coverage_note
        claim.__dict__["quote_is_substring_of_passage"] = selection.quote_in_passage
        claim.__dict__["selected_passage_preview"] = (
            selection.passage[:200] if selection.passage else None
        )
        claim.__dict__["evidence_selection_reason"] = selection.reason
        
        return claim

    # ------------------------ Helper Methods ------------------------ #

    def _build_prompt(
        self, context: str, evidence_context_list: List[str], claim_text: str
    ) -> str:
        """
        Baut Prompt mit evidence_context_list (wenn vorhanden) oder altem context.
        """
        if evidence_context_list:
            passages_text = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(evidence_context_list))
            return f"""
Du bekommst mehrere EVIDENCE-PASSAGEN aus einem Artikel (nummeriert [0] bis [{len(evidence_context_list)-1}]) und eine einzelne Behauptung (Claim).

Deine Aufgabe:
Entscheide, ob der Claim durch eine der EVIDENCE-PASSAGEN gestützt wird.

KRITISCH:
- Verwende ausschließlich die EVIDENCE-PASSAGEN. Keine Weltkenntnis, keine Vermutungen.
- Du MUSST eine Passage auswählen (selected_evidence_index: 0..{len(evidence_context_list)-1}) ODER explizit -1 (keine Evidence).
- "correct" NUR, wenn du eine Passage findest, die den Claim klar stützt.
- "incorrect" NUR, wenn du eine Passage findest, die dem Claim klar widerspricht.
- Wenn keine Passage relevant ist: selected_evidence_index=-1, evidence_quote=null, label="uncertain".

Labels:
- "correct"   → Claim wird durch eine Passage gestützt
- "incorrect" → Claim widerspricht einer Passage
- "uncertain" → Keine Passage ist relevant / zu vage

Fehlertyp (nur wenn "incorrect"):
- "ENTITY"  → falscher Name/Person/Ort/Organisation
- "NUMBER"  → falsche Zahl/Menge/Prozent
- "DATE"    → falsches Datum/Jahr/Reihenfolge
- "OTHER"   → sonstiger Widerspruch

EVIDENCE-AUSWAHL (VERPFLICHTEND - SCHEMA):
- Wenn du eine relevante Passage findest: selected_evidence_index = 0..{len(evidence_context_list)-1}
- Wenn keine Passage relevant ist: selected_evidence_index = -1 (NICHT null!)
- evidence_quote MUSS gesetzt sein (nicht null, nicht leer), wenn selected_evidence_index >= 0:
  * Kopiere einen wörtlichen Auszug (max 1-2 Sätze) aus der ausgewählten Passage
  * evidence_quote MUSS ein exakter Substring der ausgewählten Passage [selected_evidence_index] sein
- Wenn selected_evidence_index = -1: evidence_quote MUSS null sein (nicht leerer String)

Gib NUR JSON zurück:

{{
  "label": "correct" | "incorrect" | "uncertain",
  "confidence": 0.0,
  "error_type": "ENTITY" | "NUMBER" | "DATE" | "OTHER" | null,
  "explanation": "kurze Begründung (1-2 Sätze)",
  "selected_evidence_index": 0 | 1 | 2 | ... | {len(evidence_context_list)-1} | -1,
  "evidence_quote": "Wörtlicher Auszug aus Passage [selected_evidence_index] (MUSS gesetzt sein wenn index >= 0, MUSS null sein wenn index = -1)"
}}

EVIDENCE-PASSAGEN:
{passages_text}

CLAIM:
{claim_text}
""".strip()
        else:
            return f"""
Du bekommst einen KONTEXT-AUSZUG aus einem Artikel und eine einzelne Behauptung (Claim).

Deine Aufgabe:
Entscheide, ob der Claim durch den KONTEXT gestützt wird.

KRITISCH:
- Verwende ausschließlich den KONTEXT. Keine Weltkenntnis, keine Vermutungen.
- Wenn der Claim nicht EXPLIZIT im Kontext gestützt wird: label="uncertain".
- "correct" NUR, wenn du 1–2 kurze wörtliche Zitate aus dem Kontext findest, die den Claim direkt stützen.
- "incorrect" NUR, wenn du 1–2 kurze wörtliche Zitate findest, die dem Claim widersprechen.

Labels:
- "correct"   → Claim wird durch den Kontext gestützt
- "incorrect" → Claim widerspricht dem Kontext
- "uncertain" → Kontext enthält nicht genug Informationen / ist zu vage / Claim nicht explizit gestützt

Fehlertyp (nur wenn "incorrect"):
- "ENTITY"  → falscher Name/Person/Ort/Organisation
- "NUMBER"  → falsche Zahl/Menge/Prozent
- "DATE"    → falsches Datum/Jahr/Reihenfolge
- "OTHER"   → sonstiger Widerspruch

Gib NUR JSON zurück:

{{
  "label": "correct" | "incorrect" | "uncertain",
  "confidence": 0.0,
  "error_type": "ENTITY" | "NUMBER" | "DATE" | "OTHER" | null,
  "explanation": "kurze Begründung (1-2 Sätze)",
  "selected_evidence_index": -1,
  "evidence_quote": null
}}

KONTEXT:
{context}

CLAIM:
{claim_text}
""".strip()

    def _evidence_covers_claim(
        self, claim_text: str, evidence: List[str]
    ) -> Tuple[bool, str]:
        """
        Heuristik gegen "plausibility bias":
        - harte Einheiten (Zahlen/Daten/Entities) müssen im Evidence vorkommen, sonst -> nicht ok
        - weiche Tokens (Inhaltswörter) sollen zu einem Mindestanteil vorkommen
        """
        if not claim_text.strip():
            return False, "leerer Claim"

        if not evidence:
            return False, "keine Evidence"

        ev_join = " ".join(evidence)

        hard_numbers = set(re.findall(r"\b\d{1,4}\b", claim_text))
        hard_dates = set(
            re.findall(
                r"\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)[a-z]*\b",
                claim_text.lower(),
            )
        )
        hard_entities = set(self._cap_words(claim_text))

        missing_hard: List[str] = []

        # Zahlen müssen (wenn im Claim vorhanden) im Evidence auftauchen
        for n in hard_numbers:
            if n not in ev_join:
                missing_hard.append(f"NUMBER:{n}")

        # "Entities" (kapitalisierte Wörter) müssen (wenn vorhanden) im Evidence auftauchen
        # Erweiterte Liste generischer Tokens (case-insensitive)
        generic_tokens = {
            "The", "A", "An", "BBC", "Mr", "Mrs", "Dr", "Prof", "Inc", "Ltd",
            "USA", "UK", "EU", "UN", "NATO", "WHO", "UNESCO",
            "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        }
        generic_tokens_lower = {t.lower() for t in generic_tokens}
        
        for ent in hard_entities:
            if ent and ent.lower() not in generic_tokens_lower and ent not in generic_tokens:
                # Case-insensitive Vergleich
                if ent.lower() not in ev_join.lower() and ent not in ev_join:
                    missing_hard.append(f"ENTITY:{ent}")

        # Datumstokens (Monatsnamen etc.)
        for d in hard_dates:
            if d and d not in ev_join.lower():
                missing_hard.append(f"DATE:{d}")

        if missing_hard:
            return False, "fehlende Einheiten im Evidence: " + ", ".join(missing_hard[:6])

        # Soft coverage: Inhaltstokens
        soft = self._tokenize_soft(claim_text)
        if not soft:
            return True, ""

        hit = sum(1 for t in soft if t in ev_join.lower())
        cov = hit / max(1, len(soft))

        if cov < self.min_soft_token_coverage:
            return False, f"soft coverage {cov:.2f} < {self.min_soft_token_coverage:.2f}"

        return True, ""

    def _select_context(
        self,
        article_text: str,
        claim_text: str,
        *,
        top_k: int,
        neighbor_window: int,
        max_chars: int,
    ) -> str:
        """Fallback: Selektiert Kontext aus Artikel (alte Methode)."""
        sentences = self._split_sentences(article_text)
        if not sentences:
            return article_text[:max_chars]

        claim_tokens = self._tokenize(claim_text)
        claim_nums = set(re.findall(r"\d{1,4}", claim_text))
        claim_ents = self._cap_words(claim_text)

        scored: List[Tuple[int, float]] = []
        for i, s in enumerate(sentences):
            stoks = self._tokenize(s)
            if not stoks:
                continue

            overlap = len(claim_tokens & stoks) / (len(claim_tokens) + 1.0)

            nums = set(re.findall(r"\d{1,4}", s))
            num_boost = 0.0
            if claim_nums and nums:
                shared = claim_nums & nums
                num_boost = 0.8 if shared else 0.2

            ent_boost = 0.0
            if claim_ents:
                hit = sum(1 for w in claim_ents if w and w in s)
                if hit:
                    ent_boost = min(0.6, 0.2 * hit)

            score = overlap + num_boost + ent_boost
            scored.append((i, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        top = [i for i, _ in scored[: max(1, top_k)]]

        picked = set()
        for idx in top:
            for j in range(idx - neighbor_window, idx + neighbor_window + 1):
                if 0 <= j < len(sentences):
                    picked.add(j)

        ordered = [sentences[i] for i in sorted(picked)]
        context = "\n".join(s.strip() for s in ordered if s.strip())

        if not context:
            context = article_text[:max_chars]

        if len(context) > max_chars:
            context = context[:max_chars].rstrip() + "…"

        return context

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        if not text:
            return []
        lines = [ln.strip() for ln in re.split(r"\n+", text) if ln.strip()]
        out: List[str] = []
        for ln in lines:
            parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9\"“(])", ln)
            for p in parts:
                p = p.strip()
                if p:
                    out.append(p)
        return [s for s in out if len(s) >= 2]

    @staticmethod
    def _tokenize(text: str) -> set:
        toks = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", (text or "").lower())
        stop = {
            "der", "die", "das", "und", "oder", "ein", "eine", "einer", "einem",
            "to", "the", "a", "an", "of", "in", "on", "for", "with", "at", "by",
            "ist", "sind", "war", "were", "is", "are", "was",
        }
        return {t for t in toks if t not in stop and len(t) > 1}

    @staticmethod
    def _tokenize_soft(text: str) -> List[str]:
        toks = re.findall(r"[A-Za-zÄÖÜäöüß]+", (text or "").lower())
        stop = {
            "der", "die", "das", "und", "oder", "ein", "eine", "einer", "einem", "den", "dem", "des",
            "to", "the", "a", "an", "of", "in", "on", "for", "with", "at", "by", "from", "as",
            "ist", "sind", "war", "were", "is", "are", "was", "be", "been", "being",
            "this", "that", "these", "those",
        }
        toks = [t for t in toks if t not in stop and len(t) > 3]
        seen = set()
        out = []
        for t in toks:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    @staticmethod
    def _cap_words(text: str) -> List[str]:
        return re.findall(r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]{2,}\b", text or "")

    @staticmethod
    def _clamp01(x) -> float:
        try:
            v = float(x)
        except Exception:
            return 0.0
        return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v

