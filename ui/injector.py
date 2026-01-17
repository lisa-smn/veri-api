"""
Error-Injection für Demo-Zwecke.

Erlaubt gezielte Modifikationen der Summary nach Dimension (Factuality/Coherence/Readability)
und Severity (low/medium/high) für Demo-Tests.
"""

import re


def inject(summary: str, mode: str, severity: str) -> tuple[str, str]:
    """
    Injiziert einen Fehler in die Summary basierend auf Dimension und Severity.

    Args:
        summary: Original Summary-Text
        mode: "factuality", "coherence", "readability", oder "none"
        severity: "low", "medium", "high"

    Returns:
        (modified_text, change_log): Modifizierter Text und Beschreibung der Änderung
    """
    if not summary or len(summary.strip()) < 10:
        return summary, "No-op (summary too short)"

    if mode.lower() == "none" or not mode:
        return summary, "No injection (mode: none)"

    mode = mode.lower()
    severity = severity.lower()

    if mode == "factuality":
        return _inject_factuality(summary, severity)
    if mode == "coherence":
        return _inject_coherence(summary, severity)
    if mode == "readability":
        return _inject_readability(summary, severity)
    return summary, f"No-op (unknown mode: {mode})"


def _inject_factuality(summary: str, severity: str) -> tuple[str, str]:
    """Injiziert Factuality-Fehler (falsche Zahlen/Daten)."""
    # Strategie 1: Ändere erste Zahl
    number_match = re.search(r"\b\d+(?:\.\d+)?\b", summary)
    if number_match:
        num_str = number_match.group()
        num = float(num_str)

        if severity == "high":
            new_num = int(num * 1.5) if num > 0 else int(num + 1000)
        elif severity == "medium":
            new_num = int(num + 100) if num > 0 else int(num + 100)
        else:  # low
            new_num = int(num + 10) if num > 0 else int(num + 10)

        new_text = summary[: number_match.start()] + str(new_num) + summary[number_match.end() :]
        return new_text, f"Factuality: changed number {num_str} -> {new_num}"

    # Strategie 2: Ändere Jahreszahl
    year_match = re.search(r"\b(19|20)\d{2}\b", summary)
    if year_match:
        year = int(year_match.group())
        new_year = year + 1
        new_text = summary[: year_match.start()] + str(new_year) + summary[year_match.end() :]
        return new_text, f"Factuality: changed year {year} -> {new_year}"

    # Strategie 3: Füge falschen Fakt hinzu
    if severity == "high":
        addition = " [INCORRECT: The event occurred in 2025, not as stated.]"
    elif severity == "medium":
        addition = " [INCORRECT: The number of participants was 500, not as mentioned.]"
    else:  # low
        addition = " [Note: Some details may require verification.]"

    return summary + addition, f"Factuality: added incorrect fact marker ({severity})"


def _inject_coherence(summary: str, severity: str) -> tuple[str, str]:
    """Injiziert Coherence-Fehler (Satz-Reihenfolge, Widersprüche)."""
    # Split in Sätze (einfacher Regex)
    sentences = re.split(r"([.!?]+)", summary)
    # Paare wieder zusammenführen
    sentence_pairs = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence_pairs.append(sentences[i] + sentences[i + 1])
        else:
            sentence_pairs.append(sentences[i])

    # Filter leere Sätze
    sentence_pairs = [s.strip() for s in sentence_pairs if s.strip()]

    if len(sentence_pairs) < 2:
        return summary, "No-op (summary too short for coherence injection)"

    if severity == "high":
        # Reverse alle Sätze + füge Widerspruch hinzu
        reversed_sentences = list(reversed(sentence_pairs))
        new_text = " ".join(reversed_sentences)
        contradiction = " However, this contradicts the previous statement."
        new_text += contradiction
        return new_text, "Coherence: reversed sentence order + added contradiction"
    if severity == "medium":
        # Reverse erste 3 Sätze
        if len(sentence_pairs) >= 3:
            reversed_part = list(reversed(sentence_pairs[:3]))
            new_text = " ".join(reversed_part + sentence_pairs[3:])
            return new_text, "Coherence: reversed first 3 sentences"
        # Nur 2 Sätze: swap
        new_text = " ".join([sentence_pairs[1], sentence_pairs[0]])
        return new_text, "Coherence: swapped sentence order (1<->2)"
    # low
    # Swap erste zwei Sätze
    new_text = " ".join([sentence_pairs[1], sentence_pairs[0]] + sentence_pairs[2:])
    return new_text, "Coherence: swapped sentence order (1<->2)"


def _inject_readability(summary: str, severity: str) -> tuple[str, str]:
    """Injiziert Readability-Fehler (Satzlänge, Struktur)."""
    # Split in Sätze
    sentences = re.split(r"([.!?]+)", summary)
    sentence_pairs = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            sentence_pairs.append(sentences[i] + sentences[i + 1])
        else:
            sentence_pairs.append(sentences[i])

    sentence_pairs = [s.strip() for s in sentence_pairs if s.strip()]

    if len(sentence_pairs) < 2:
        return summary, "No-op (summary too short for readability injection)"

    if severity == "high":
        # Erstelle einen sehr langen Run-on-Satz
        # Entferne Interpunktion zwischen Sätzen und verbinde mit "and"
        combined = " and ".join([s.rstrip(".,!?;") for s in sentence_pairs])
        new_text = combined + "."
        return new_text, "Readability: created very long run-on sentence"
    if severity == "medium":
        # Entferne Interpunktion + füge Filler-Wörter hinzu
        # Merge erste zwei Sätze
        if len(sentence_pairs) >= 2:
            merged = (
                sentence_pairs[0].rstrip(".,!?;")
                + " and furthermore "
                + sentence_pairs[1].rstrip(".,!?;")
                + "."
            )
            new_text = merged + " " + " ".join(sentence_pairs[2:])
            return new_text, "Readability: merged sentences + added filler words"
        # Nur ein Satz: füge Filler hinzu
        new_text = summary.replace(".", ", and furthermore,").replace("!", ", and furthermore!")
        return new_text, "Readability: added filler words"
    # low
    # Merge zwei Sätze zu einem längeren
    if len(sentence_pairs) >= 2:
        merged = sentence_pairs[0].rstrip(".,!?;") + ", " + sentence_pairs[1].lstrip()
        new_text = merged + " " + " ".join(sentence_pairs[2:])
        return new_text, "Readability: merged first two sentences"
    return summary, "No-op (summary too short for readability injection)"
