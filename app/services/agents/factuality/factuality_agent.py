from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re
from typing import Any

from app.llm.llm_client import LLMClient
from app.models.pydantic import AgentResult, IssueSpan
from app.services.agents.factuality.claim_extractor import ClaimExtractor, LLMClaimExtractor
from app.services.agents.factuality.claim_models import Claim
from app.services.agents.factuality.claim_verifier import ClaimVerifier, LLMClaimVerifier


@dataclass
class SentenceResult:
    index: int
    sentence: str
    label: str  # correct|incorrect|uncertain|skipped
    confidence: float
    explanation: str


class FactualityAgent:
    """
    Factuality-Agent (satzbasiert, benchmark-nah):
    - Zerlegt Summary in Sätze (mit stabilen Char-Spans)
    - Extrahiert Claims pro Satz (Fallback: ganzer Satz als Claim)
    - Verifiziert Claims gegen den Artikel
    - Aggregiert zu Satz-Labels + Gesamtscore

    Wichtig:
    - "uncertain" beeinflusst den Score (neutral gewichtet)
    - "uncertain" wird auch als IssueSpan ausgegeben (low severity),
      damit score<1 nicht mehr mit num_issues=0 endet.
    """

    # Wie "uncertain" in den Score eingeht: 1.0=correct, 0.0=incorrect, 0.5=neutral
    UNCERTAIN_WEIGHT = 0.5

    # Wenn wir trotz nicht-leerer Summary nichts prüfen konnten: "unknown" statt "perfekt"
    EMPTY_CHECK_SCORE = 0.5

    def __init__(
        self,
        llm_client: LLMClient,
        claim_extractor: ClaimExtractor | None = None,
        claim_verifier: ClaimVerifier | None = None,
        *,
        use_claim_extraction: bool = True,
        use_claim_verification: bool = True,
        use_spans: bool = True,
        strict_mode: bool = False,
    ) -> None:
        """
        Args:
            llm_client: LLM-Client für Extraktion/Verifikation
            claim_extractor: Optional custom extractor
            claim_verifier: Optional custom verifier
            use_claim_extraction: Wenn False, wird keine LLM-basierte Claim-Extraktion durchgeführt (Ablation)
            use_claim_verification: Wenn False, werden Claims nicht verifiziert (Ablation)
            use_spans: Wenn False, werden keine Issue-Spans generiert (Ablation)
            strict_mode: Wenn True, wird LLMClaimVerifier im Strict-Mode betrieben (fail-fast bei Schema-Verletzungen)
        """
        self.llm = llm_client
        self.use_claim_extraction = use_claim_extraction
        self.use_claim_verification = use_claim_verification
        self.use_spans = use_spans

        # Setup extractor
        if claim_extractor is not None:
            self.claim_extractor = claim_extractor
        elif not use_claim_extraction:
            # Ablation: Keine Claim-Extraktion
            from app.services.agents.factuality.ablation_extractor import NoOpClaimExtractor

            self.claim_extractor = NoOpClaimExtractor()
        else:
            self.claim_extractor = LLMClaimExtractor(llm_client)

        # Setup verifier
        if claim_verifier is not None:
            self.claim_verifier = claim_verifier
        elif not use_claim_verification:
            # Ablation: Keine Claim-Verifikation
            from app.services.agents.factuality.ablation_verifier import NoOpClaimVerifier

            self.claim_verifier = NoOpClaimVerifier()
        else:
            # Aktiviere Evidence Retriever standardmäßig
            # Strict-Mode kann über FactualityAgent-Parameter gesetzt werden
            self.claim_verifier = LLMClaimVerifier(
                llm_client,
                use_evidence_retriever=True,
                evidence_retriever_top_k=5,
                strict_mode=strict_mode,  # Wird von FactualityAgent-Parameter übernommen
            )

    def run(
        self,
        article_text: str,
        summary_text: str,
        meta: dict[str, Any] | None = None,
    ) -> AgentResult:
        summary_text = (summary_text or "").strip()
        article_text = (article_text or "").strip()
        meta = meta or {}

        # Normalisiere Tausenderpunkte VOR der Verarbeitung (damit Zahlen korrekt extrahiert werden)
        from app.services.agents.factuality.number_normalization import (
            normalize_text_for_number_extraction,
        )

        summary_text = normalize_text_for_number_extraction(summary_text)
        article_text = normalize_text_for_number_extraction(article_text)

        # Sätze + Char-Spans im Originaltext (wichtig fürs Span-Mapping)
        sent_spans = self._split_into_sentences_with_spans(summary_text)
        sentences = [s for (s, _, _) in sent_spans]

        # ---------- 1) Claim-Extraktion (mit Fallback-Claim pro Satz) ---------- #
        raw_claims: list[Claim] = []
        skipped_sentences: list[dict[str, Any]] = []

        for i, s in enumerate(sentences):
            s = (s or "").strip()
            if not s:
                continue

            # Nur klare Meta-/Text-über-Text Sätze rausfiltern.
            # Der Extractor macht den Rest.
            if self._is_meta_sentence(s):
                skipped_sentences.append({"index": i, "sentence": s, "reason": "meta"})
                continue

            if self.use_claim_extraction:
                claims = self.claim_extractor.extract_claims(s, i)
            else:
                # Ablation: Keine Claim-Extraktion, direkt Fallback
                claims = []

            # Wenn Extractor claims=[] liefert, prüfen wir notfalls den ganzen Satz als Claim.
            # (sonst geht dir Recall verloren, gerade bei FRANK-ähnlichen Outputs)
            if not claims:
                claims = [
                    Claim(
                        id=f"s{i}_c0_fallback",
                        sentence_index=i,
                        sentence=s,
                        text=s.strip(),
                    )
                ]

            raw_claims.extend(claims)

        # ---------- 2) Wenn wirklich keine Claims da sind ---------- #
        if not raw_claims:
            if not sentences:
                # Leere Summary → korrekt im Sinne "nichts behauptet"
                return AgentResult(
                    name="factuality",
                    score=1.0,
                    explanation="Score 1.00. Leere Summary.",
                    details={
                        "score_basis": "sentence",
                        "sentences": [],
                        "sentence_results": [],
                        "claims": [],
                        "num_claims": 0,
                        "num_incorrect": 0,
                        "num_uncertain": 0,
                        "num_checked_sentences": 0,
                        "num_incorrect_sentences": 0,
                        "num_uncertain_sentences": 0,
                        "skipped_sentences": skipped_sentences,
                        "meta": meta,
                        "num_issues": 0,
                    },
                    issue_spans=[],
                )

            # Non-empty Summary, aber nichts Prüfbahres → "unknown"
            score = self.EMPTY_CHECK_SCORE
            explanation = (
                f"Score {score:.2f}. Es konnten keine überprüfbaren Faktenbehauptungen extrahiert werden "
                f"(Summary enthält überwiegend Meta-/nicht-faktische Aussagen)."
            )

            # Damit score<1 nicht mit num_issues=0 endet: ein low IssueSpan über die Summary
            issue_spans = [
                IssueSpan(
                    start_char=0,
                    end_char=len(summary_text) if summary_text else None,
                    message="Keine überprüfbaren Faktenbehauptungen extrahiert (unverifizierbar / Meta-dominiert).",
                    severity="low",
                    issue_type="OTHER",
                )
            ]

            return AgentResult(
                name="factuality",
                score=score,
                explanation=explanation,
                details={
                    "score_basis": "sentence",
                    "sentences": [{"index": i, "text": s} for i, s in enumerate(sentences)],
                    "sentence_results": [],
                    "claims": [],
                    "num_claims": 0,
                    "num_incorrect": 0,
                    "num_uncertain": 0,
                    "num_checked_sentences": 0,
                    "num_incorrect_sentences": 0,
                    "num_uncertain_sentences": 0,
                    "skipped_sentences": skipped_sentences,
                    "meta": meta,
                    "num_issues": len(issue_spans),
                },
                issue_spans=issue_spans,
            )

        # ---------- 3) Claim-Verifikation ---------- #
        verified_claims: list[Claim] = []
        for c in raw_claims:
            try:
                if self.use_claim_verification:
                    verified_claim = self.claim_verifier.verify(article_text, c)
                    # Gate-Logik existiert GENAU EINMAL: im ClaimVerifier.
                    # Kein Safety-Downgrade mehr hier (wurde entfernt für Clean-Code).
                    verified_claims.append(verified_claim)
                else:
                    # Ablation: Keine Verifikation, direkt als uncertain markieren
                    c.label = "uncertain"
                    c.confidence = 0.5
                    c.explanation = "Ablation: Claim-Verifikation deaktiviert"
                    c.error_type = None
                    c.evidence = []
                    c.evidence_found = False
                    verified_claims.append(c)
            except Exception as e:
                # Fail-safe: lieber "uncertain" als Run-Abbruch
                c.label = "uncertain"
                c.confidence = 0.0
                c.error_type = None
                c.explanation = f"Verifier-Fehler: {type(e).__name__}: {e}"
                c.evidence = []
                c.evidence_found = False
                verified_claims.append(c)

        # ---------- 4) Satz-Ergebnisse + Score (satzbasiert) ---------- #
        sentence_results = self._aggregate_sentence_results(sentences, verified_claims)
        score = self._compute_score_from_sentences(sentence_results)
        explanation = self._build_global_explanation(score, sentence_results)

        # ---------- 5) Issue Spans (incorrect + uncertain sichtbar machen) ---------- #
        if self.use_spans:
            rep_claims = self._representative_claims_for_spans(verified_claims)
            issue_spans = self._build_issue_spans_from_claims(
                summary_text=summary_text,
                sent_spans=sent_spans,
                claims=rep_claims,
            )
        else:
            # Ablation: Keine Issue-Spans
            issue_spans = []

        incorrect_claims = [c for c in verified_claims if c.label == "incorrect"]

        # Saubere Serialisierung: Nur bekannte Felder + Debug-Felder aus __dict__
        claims_dicts = []
        for c in verified_claims:
            claim_dict = {}
            # Basis-Felder
            claim_dict["id"] = c.id
            claim_dict["sentence_index"] = c.sentence_index
            claim_dict["sentence"] = c.sentence
            claim_dict["text"] = c.text
            # Verifier-Felder
            claim_dict["label"] = c.label
            claim_dict["confidence"] = c.confidence
            claim_dict["error_type"] = c.error_type
            claim_dict["explanation"] = c.explanation
            # Evidence-Felder (immer vorhanden, auch wenn None/-1)
            claim_dict["selected_evidence_index"] = c.selected_evidence_index
            claim_dict["evidence_quote"] = c.evidence_quote
            claim_dict["evidence_found"] = c.evidence_found
            claim_dict["retrieved_passages"] = c.retrieved_passages
            claim_dict["retrieval_scores"] = c.retrieval_scores
            # Legacy
            claim_dict["evidence"] = c.evidence or []
            claim_dict["evidence_spans"] = c.evidence_spans or []
            # Debug-Felder (aus __dict__)
            claim_dict["label_raw"] = c.__dict__.get("label_raw")
            claim_dict["label_final"] = c.__dict__.get("label_final")
            claim_dict["gate_reason"] = c.__dict__.get("gate_reason")
            claim_dict["coverage_ok"] = c.__dict__.get("coverage_ok")
            claim_dict["coverage_note"] = c.__dict__.get("coverage_note")
            claim_dict["evidence_selection_reason"] = c.__dict__.get("evidence_selection_reason")
            claim_dict["parse_ok"] = c.parse_ok
            claim_dict["parse_error"] = c.parse_error
            claim_dict["raw_verifier_output"] = c.raw_verifier_output
            claim_dict["selected_evidence_index_raw"] = c.selected_evidence_index_raw
            claim_dict["evidence_quote_raw"] = c.evidence_quote_raw
            claims_dicts.append(claim_dict)

        return AgentResult(
            name="factuality",
            score=score,
            explanation=explanation,
            details={
                "score_basis": "sentence",
                "sentences": [{"index": i, "text": s} for i, s in enumerate(sentences)],
                "sentence_results": [sr.__dict__ for sr in sentence_results],
                "claims": claims_dicts,
                # Claim-level counts
                "num_incorrect": len(incorrect_claims),
                "num_uncertain": sum(1 for c in verified_claims if c.label == "uncertain"),
                "num_claims": len(verified_claims),
                # Sentence-level counts (benchmark-näher)
                "num_checked_sentences": sum(1 for sr in sentence_results if sr.label != "skipped"),
                "num_incorrect_sentences": sum(
                    1 for sr in sentence_results if sr.label == "incorrect"
                ),
                "num_uncertain_sentences": sum(
                    1 for sr in sentence_results if sr.label == "uncertain"
                ),
                "skipped_sentences": skipped_sentences,
                "meta": meta,
                "num_issues": len(issue_spans),
            },
            issue_spans=issue_spans,
        )

    # ----------------- sentence aggregation ----------------- #

    def _aggregate_sentence_results(
        self, sentences: list[str], claims: list[Claim]
    ) -> list[SentenceResult]:
        by_sentence: defaultdict[int, list[Claim]] = defaultdict(list)
        for c in claims:
            by_sentence[int(c.sentence_index)].append(c)

        results: list[SentenceResult] = []
        for i, s in enumerate(sentences):
            s = (s or "").strip()
            if not s:
                continue

            if self._is_meta_sentence(s) and i not in by_sentence:
                results.append(SentenceResult(i, s, "skipped", 0.0, "Meta-/nicht-faktischer Satz."))
                continue

            cs = by_sentence.get(i, [])
            if not cs:
                # sollte selten sein (Fallback-Claims), aber sicher ist sicher
                results.append(
                    SentenceResult(
                        i, s, "uncertain", 0.0, "Keine verifizierbaren Claims extrahiert."
                    )
                )
                continue

            incorrect = [c for c in cs if c.label == "incorrect"]
            uncertain = [c for c in cs if c.label == "uncertain"]
            correct = [c for c in cs if c.label == "correct"]

            if incorrect:
                c0 = max(incorrect, key=lambda x: float(x.confidence or 0.0))
                conf = float(c0.confidence or 0.0)
                expl = c0.explanation or "Widerspruch zum Artikelkontext."
                results.append(SentenceResult(i, s, "incorrect", conf, expl))
                continue

            if uncertain:
                c0 = max(uncertain, key=lambda x: float(x.confidence or 0.0))
                conf = float(c0.confidence or 0.0)
                expl = c0.explanation or "Anhand des Artikels nicht sicher verifizierbar."
                results.append(SentenceResult(i, s, "uncertain", conf, expl))
                continue

            # alles korrekt
            conf = float(min((c.confidence or 1.0) for c in correct)) if correct else 1.0
            results.append(SentenceResult(i, s, "correct", conf, "Keine Inkonsistenzen gefunden."))

        return results

    # ----------------- scoring ----------------- #

    def _compute_score_from_sentences(self, sentence_results: list[SentenceResult]) -> float:
        checked = [sr for sr in sentence_results if sr.label != "skipped"]
        if not checked:
            return self.EMPTY_CHECK_SCORE

        total = len(checked)
        correct = sum(1 for sr in checked if sr.label == "correct")
        uncertain = sum(1 for sr in checked if sr.label == "uncertain")
        return (correct + self.UNCERTAIN_WEIGHT * uncertain) / total

    # ----------------- explanations ----------------- #

    def _build_global_explanation(
        self, score: float, sentence_results: list[SentenceResult]
    ) -> str:
        incorrect = [sr for sr in sentence_results if sr.label == "incorrect"]
        uncertain = [sr for sr in sentence_results if sr.label == "uncertain"]

        if incorrect:
            sr = incorrect[0]
            return (
                f"Score {score:.2f}. Es wurden faktische Inkonsistenzen erkannt. "
                f"Beispiel: Satz {sr.index + 1}: {sr.explanation}"
            )

        if uncertain:
            sr = uncertain[0]
            return (
                f"Score {score:.2f}. Keine eindeutigen Widersprüche, aber einige Aussagen sind nicht sicher verifizierbar. "
                f"Beispiel: Satz {sr.index + 1}: {sr.explanation}"
            )

        return f"Score {score:.2f}. Es wurden keine faktischen Probleme erkannt."

    # ----------------- issue spans ----------------- #

    def _representative_claims_for_spans(self, claims: list[Claim]) -> list[Claim]:
        """
        Pro Satz maximal 1 Claim als 'repräsentativ' für IssueSpans:
        - wenn incorrect vorhanden: stärkster incorrect
        - sonst wenn uncertain vorhanden: stärkster uncertain
        - sonst: kein Span
        """
        by_sentence: defaultdict[int, list[Claim]] = defaultdict(list)
        for c in claims:
            by_sentence[int(c.sentence_index)].append(c)

        reps: list[Claim] = []
        for idx, cs in by_sentence.items():
            incorrect = [c for c in cs if c.label == "incorrect"]
            if incorrect:
                reps.append(max(incorrect, key=lambda x: float(x.confidence or 0.0)))
                continue
            uncertain = [c for c in cs if c.label == "uncertain"]
            if uncertain:
                reps.append(max(uncertain, key=lambda x: float(x.confidence or 0.0)))
        return reps

    def _build_issue_spans_from_claims(
        self,
        summary_text: str,
        sent_spans: list[tuple[str, int, int]],
        claims: list[Claim],
    ) -> list[IssueSpan]:
        spans: list[IssueSpan] = []
        sent_map: dict[int, tuple[str, int, int]] = {
            i: sent_spans[i] for i in range(len(sent_spans))
        }

        for c in claims:
            if c.label not in ("incorrect", "uncertain"):
                continue

            # Satz-Span als sichere Fallback-Region
            sent = sent_map.get(int(c.sentence_index))
            if sent:
                sent_text, sent_start, sent_end = sent
            else:
                sent_text, sent_start, sent_end = ("", None, None)

            start, end = self._locate_claim_span(
                summary_text=summary_text,
                sentence_text=sent_text,
                sentence_start=sent_start,
                sentence_end=sent_end,
                claim_text=c.text,
            )

            if c.label == "incorrect":
                issue_type = (c.error_type or "OTHER").upper()
                message = (
                    f"Satz {c.sentence_index + 1}: Claim '{c.text}' – {c.explanation}"
                    if c.explanation
                    else f"Satz {c.sentence_index + 1}: Claim '{c.text}' ist faktisch inkonsistent."
                )
                severity = self._map_issue_type_to_severity(issue_type)
                verdict = "incorrect"  # Explizit: incorrect Claim
            else:
                # uncertain sichtbar machen, aber low severity
                issue_type = "OTHER"
                message = (
                    f"Satz {c.sentence_index + 1}: Claim '{c.text}' – Nicht sicher verifizierbar (Quelle zu vage/fehlend). "
                    f"{c.explanation or ''}"
                ).strip()
                severity = "low"
                verdict = "uncertain"  # Explizit: uncertain Claim

            # Für gewichtete Decision Logic: confidence und evidence_found speichern
            confidence = float(c.confidence or 0.5)
            evidence_found = c.evidence_found if c.evidence_found is not None else bool(c.evidence)

            spans.append(
                IssueSpan(
                    start_char=start,
                    end_char=end,
                    message=message,
                    severity=severity,
                    verdict=verdict,  # Explizites verdict (incorrect vs uncertain)
                    issue_type=issue_type,
                    confidence=confidence,
                    mapping_confidence=1.0,  # Wird später beim Mapping gesetzt
                    evidence_found=evidence_found,
                )
            )

        return spans

    def _locate_claim_span(
        self,
        summary_text: str,
        sentence_text: str,
        sentence_start: int | None,
        sentence_end: int | None,
        claim_text: str,
    ) -> tuple[int | None, int | None]:
        """
        Robust:
        - versuche Claim innerhalb des Satzes zu finden (Offset korrekt)
        - sonst: markiere ganzen Satz
        - sonst: None/None
        """
        claim_text = (claim_text or "").strip()
        if sentence_start is None or sentence_end is None:
            # Notfall: globaler Find (kann falsches Vorkommen treffen, aber besser als nichts)
            if claim_text:
                pos = summary_text.find(claim_text)
                if pos != -1:
                    return pos, pos + len(claim_text)
            return None, None

        if claim_text and sentence_text:
            local = sentence_text.find(claim_text)
            if local != -1:
                return sentence_start + local, sentence_start + local + len(claim_text)

        # Fallback: Satz markieren
        return sentence_start, sentence_end

    def _map_issue_type_to_severity(self, issue_type: str | None) -> str:
        t = (issue_type or "").upper()
        if t in {"NUMBER", "DATE"}:
            return "high"
        if t in {"ENTITY", "NAME", "LOCATION"}:
            return "medium"
        return "low"

    # ----------------- text helpers ----------------- #

    def _split_into_sentences_with_spans(self, text: str) -> list[tuple[str, int, int]]:
        """
        Liefert [(sentence_text, start_char, end_char)] im Originaltext.
        Stabil und span-freundlich: splittet an Satzendzeichen und Newlines.
        """
        if not text:
            return []

        spans: list[tuple[str, int, int]] = []
        for m in re.finditer(r"[^.!?\n]+[.!?]?", text):
            raw = m.group(0)
            if not raw or not raw.strip():
                continue

            # Trim ohne Index zu verlieren
            lead = len(raw) - len(raw.lstrip())
            trail = len(raw) - len(raw.rstrip())

            start = m.start() + lead
            end = m.end() - trail

            s = text[start:end].strip()
            if s:
                spans.append((s, start, end))

        return spans

    def _is_meta_sentence(self, sentence: str) -> bool:
        """
        Filtert nur klare Meta-Aussagen (Text über Text / Stil / Lesbarkeit).
        Nicht aggressiv, sonst verlierst du Recall.
        """
        s = (sentence or "").strip()
        if not s:
            return True

        low = s.lower()

        # Fragen sind i.d.R. keine faktischen Behauptungen
        if s.endswith("?"):
            return True

        meta_markers = [
            # DE
            "dieser satz",
            "die summary",
            "der text",
            "die zusammenfassung",
            "lesbarkeit",
            "kohärenz",
            "verständlichkeit",
            "schwer verständlich",
            "gut lesbar",
            "langer satz",
            "viele kommas",
            "verschachtel",
            "stil",
            "ton",
            "grammatik",
            "orthografie",
            "rechtschreibung",
            # EN
            "this sentence",
            "this summary",
            "the summary",
            "the text",
            "readability",
            "coherence",
            "fluency",
            "hard to read",
            "easy to read",
            "long sentence",
            "many commas",
            "writing style",
            "grammar",
        ]
        if any(m in low for m in meta_markers):
            return True

        # sehr kurze Fragmente sind selten sinnvolle Faktenbehauptungen
        tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", low)
        if len(tokens) < 3:
            return True

        return False
