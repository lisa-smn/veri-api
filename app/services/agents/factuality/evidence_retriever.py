"""
Evidence Retriever: Findet relevante Passagen im Artikel für einen Claim.

Deterministischer Retriever:
- Splittet Artikel in sliding-window Passagen (2-3 Sätze)
- Scored Passagen via Jaccard-Similarity (Token-Overlap)
- Boost für Zahlen/Entities im Claim
- Liefert top_k Passagen für Claim-Verifikation
"""

from __future__ import annotations

import re

from app.services.agents.factuality.number_normalization import normalize_text_for_number_extraction


class EvidenceRetriever:
    """
    Deterministischer Retriever für Evidence-Passagen.

    Splittet article_text in Sätze/Absätze und scored sie gegen den Claim
    basierend auf Token-Overlap (Jaccard-Similarity).
    """

    def __init__(
        self,
        top_k: int = 5,
        min_passage_length: int = 10,
        window_size: int = 3,  # Anzahl Sätze pro Passage
        window_stride: int = 1,  # Schrittweite für sliding window
    ):
        """
        Args:
            top_k: Anzahl der Top-Passagen, die zurückgegeben werden
            min_passage_length: Minimale Länge einer Passage (in Zeichen)
            window_size: Anzahl Sätze pro Passage (sliding window)
            window_stride: Schrittweite für sliding window
        """
        self.top_k = top_k
        self.min_passage_length = min_passage_length
        self.window_size = window_size
        self.window_stride = window_stride

    def retrieve(
        self,
        article_text: str,
        claim_text: str,
    ) -> tuple[list[str], list[float]]:
        """
        Retrieviert die top_k relevantesten Passagen für einen Claim.

        Args:
            article_text: Vollständiger Artikel-Text
            claim_text: Claim-Text

        Returns:
            Tuple von (passages, scores):
            - passages: Liste der top_k Passagen (Strings)
            - scores: Liste der Scores für jede Passage (float, 0.0-1.0)
        """
        if not article_text or not claim_text:
            return [], []

        # Split in Passagen (Sätze)
        passages = self._split_into_passages(article_text)

        if not passages:
            return [], []

        # Tokenize claim
        claim_tokens = self._tokenize(claim_text)
        if not claim_tokens:
            return [], []

        # Score jede Passage (auch niedrige Scores behalten)
        scored: list[tuple[str, float]] = []
        for passage in passages:
            if len(passage.strip()) < self.min_passage_length:
                continue

            score = self._score_passage(passage, claim_tokens, claim_text)
            scored.append((passage, score))

        # Sortiere nach Score (absteigend)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Nimm top_k (immer, auch wenn Scores niedrig sind)
        # Falls weniger Passagen vorhanden sind, nimm alle
        top_passages = [p for p, _ in scored[: self.top_k]]
        top_scores = [s for _, s in scored[: self.top_k]]

        # Stelle sicher, dass wir mindestens eine Passage haben (falls vorhanden)
        if not top_passages and scored:
            top_passages = [scored[0][0]]
            top_scores = [scored[0][1]]

        return top_passages, top_scores

    def _split_into_passages(self, text: str) -> list[str]:
        """
        Splittet Text in Passagen (sliding window über Sätze).
        """
        if not text:
            return []

        # Split nach Satzzeichen
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9"("])', text)

        # Filtere leere Sätze
        sentences = [
            s.strip() for s in sentences if s.strip() and len(s.strip()) >= self.min_passage_length
        ]

        if not sentences:
            return []

        # Sliding Window: Baue Passagen aus window_size Sätzen
        passages = []

        # Wenn zu wenige Sätze für window_size, nutze alle verfügbaren
        if len(sentences) < self.window_size:
            # Fallback: Eine Passage aus allen Sätzen
            combined = " ".join(sentences).strip()
            if len(combined) >= self.min_passage_length:
                passages.append(combined)
        else:
            # Sliding window
            for i in range(0, len(sentences) - self.window_size + 1, self.window_stride):
                window_sentences = sentences[i : i + self.window_size]
                passage = " ".join(window_sentences).strip()
                if len(passage) >= self.min_passage_length:
                    passages.append(passage)

        # Fallback: Wenn keine Passagen gebaut wurden, nutze einzelne Sätze
        if not passages:
            passages = [s for s in sentences if len(s) >= self.min_passage_length]

        return passages

    def _tokenize(self, text: str) -> set:
        """
        Tokenisiert Text in ein Set von Tokens (lowercase, ohne Stopwords).
        """
        if not text:
            return set()

        # Extrahiere Wörter (alphanumerisch + Umlaute)
        tokens = re.findall(r"[A-Za-zÄÖÜäöüß0-9]+", text.lower())

        # Stopwords entfernen
        stopwords = {
            "der",
            "die",
            "das",
            "und",
            "oder",
            "ein",
            "eine",
            "einer",
            "einem",
            "to",
            "the",
            "a",
            "an",
            "of",
            "in",
            "on",
            "for",
            "with",
            "at",
            "by",
            "ist",
            "sind",
            "war",
            "were",
            "is",
            "are",
            "was",
        }

        return {t for t in tokens if t not in stopwords and len(t) > 1}

    def _score_passage(self, passage: str, claim_tokens: set, claim_text: str) -> float:
        """
        Scored eine Passage gegen Claim-Tokens (Jaccard-Similarity).

        Args:
            passage: Passage-Text
            claim_tokens: Set von Claim-Tokens
            claim_text: Original Claim-Text (für Zahl/Entity-Extraktion)

        Returns:
            Score zwischen 0.0 und 1.0
        """
        passage_tokens = self._tokenize(passage)

        if not passage_tokens or not claim_tokens:
            return 0.0

        # Jaccard-Similarity: |A ∩ B| / |A ∪ B|
        intersection = len(claim_tokens & passage_tokens)
        union = len(claim_tokens | passage_tokens)

        if union == 0:
            return 0.0

        jaccard = intersection / union

        # Boost für Zahlen: Extrahiere direkt aus claim_text (nicht aus claim_tokens)
        # Normalisiere Tausenderpunkte VOR der Extraktion
        claim_text_norm = normalize_text_for_number_extraction(claim_text)
        passage_norm = normalize_text_for_number_extraction(passage)
        numbers_claim = set(re.findall(r"\d+", claim_text_norm))
        numbers_passage = set(re.findall(r"\d+", passage_norm))
        if numbers_claim and numbers_passage:
            shared_numbers = len(numbers_claim & numbers_passage)
            if shared_numbers > 0:
                jaccard += 0.2  # Boost für Zahlen-Overlap

        # Boost für Entities: Capitalized tokens aus claim_text
        entities_claim = set(re.findall(r"\b[A-ZÄÖÜ][A-Za-zÄÖÜäöüß]{2,}\b", claim_text))
        generic_tokens = {
            "The",
            "A",
            "An",
            "BBC",
            "Mr",
            "Mrs",
            "Dr",
            "Prof",
            "Inc",
            "Ltd",
            "USA",
            "UK",
        }
        entities_claim = {e for e in entities_claim if e not in generic_tokens}

        if entities_claim:
            # Prüfe ob Entities in Passage vorkommen (case-insensitive)
            passage_lower = passage.lower()
            hit_count = sum(
                1 for ent in entities_claim if ent.lower() in passage_lower or ent in passage
            )
            if hit_count > 0:
                entity_boost = min(0.15, 0.05 * hit_count)  # Max 0.15 Boost
                jaccard += entity_boost

        # Cap bei 1.0
        return min(1.0, jaccard)
