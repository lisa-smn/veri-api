"""
Dieses Modul stellt einfache Hilfsfunktionen zur Vorverarbeitung von Summary-Texten
für den Readability-Agent bereit.

Analog zum coherence_extractor zerlegt der Readability-Extractor die Summary in
einzelne Sätze und berechnet pro Satz grundlegende, heuristische Metriken zur
Abschätzung der Lesbarkeit. Dazu gehören u.a. Wortanzahl, Interpunktionsdichte
und durchschnittliche Wortlänge.

Der Extractor selbst trifft keine Bewertungs- oder Scoring-Entscheidungen.
Er dient ausschließlich dazu, strukturierte Rohinformationen bereitzustellen,
die im Readability-Verifier zur Identifikation konkreter Lesbarkeitsprobleme
(z.B. überlange oder stark verschachtelte Sätze) verwendet werden.

Die Implementierung ist bewusst einfach gehalten (String-basierter Satzsplit),
um:
- deterministisches Verhalten sicherzustellen
- externe NLP-Abhängigkeiten zu vermeiden
- die Architektur konsistent zu M7 (Coherence Agent) zu halten

Eine spätere Ersetzung durch robustere Satzsegmentierung (z.B. spaCy) ist möglich,
ohne die Agent- oder Verifier-Schnittstellen zu verändern.

"""

from dataclasses import dataclass
import re


@dataclass
class ReadabilitySentenceInfo:
    index: int
    text: str
    word_count: int
    comma_count: int
    avg_word_length: float


def split_summary_into_sentences(summary_text: str) -> list[ReadabilitySentenceInfo]:
    """
    Sehr einfacher, naiver Satzsplitter für den Readability-Agent.

    Analog zum Coherence-Extractor:
    - kein NLP
    - keine Abhängigkeiten
    - ausreichend für M8

    Liefert pro Satz einfache Lesbarkeits-Metriken.
    """

    raw_sentences = [s.strip() for s in summary_text.split(".") if s.strip()]
    sentences: list[ReadabilitySentenceInfo] = []

    for i, sentence in enumerate(raw_sentences):
        words = re.findall(r"\b\w+\b", sentence)
        word_count = len(words)

        comma_count = sentence.count(",")

        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0.0

        sentences.append(
            ReadabilitySentenceInfo(
                index=i,
                text=sentence,
                word_count=word_count,
                comma_count=comma_count,
                avg_word_length=avg_word_length,
            )
        )

    return sentences
