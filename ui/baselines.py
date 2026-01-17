"""Klassische Baseline-Metriken für UI-Vergleich (lokal, schnell)."""

import re


def count_syllables(word: str) -> int:
    """
    Zählt Silben in einem Wort (vereinfachte Heuristik).
    """
    word = word.lower().strip()
    if not word:
        return 1

    # Entferne Endungen, die keine Silbe sind
    word = re.sub(r"e$", "", word)

    vowels = "aeiouy"
    syllable_count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            syllable_count += 1
        prev_was_vowel = is_vowel

    return max(1, syllable_count)


def flesch_reading_ease(text: str) -> float | None:
    """
    Flesch Reading Ease Score (0-100, höher = leichter).
    Normalisiert auf [0,1]: score / 100

    Returns:
        Score in [0,1] oder None bei Fehler
    """
    if not text or not text.strip():
        return None

    try:
        sentences = re.split(r"[.!?]+", text)
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return 0.5  # Neutral

        words = re.findall(r"\b\w+\b", text.lower())
        if not words:
            return 0.5

        total_syllables = sum(count_syllables(word) for word in words)

        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        avg_syllables_per_word = total_syllables / len(words) if words else 0

        # Flesch Reading Ease Formula
        score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)

        # Clamp auf [0, 100] und normalisiere auf [0, 1]
        score = max(0, min(100, score))
        return score / 100.0
    except Exception:
        return None


def compute_readability_baseline(summary_text: str) -> dict:
    """
    Berechnet Readability Baseline (Flesch).

    Returns:
        dict mit {"flesch": float, "raw_flesch": float} oder None-Werte
    """
    flesch_norm = flesch_reading_ease(summary_text)
    if flesch_norm is not None:
        raw_flesch = flesch_norm * 100  # Zurück zu 0-100 Skala für Anzeige
    else:
        raw_flesch = None

    return {
        "flesch": flesch_norm,
        "raw_flesch": raw_flesch,
    }
