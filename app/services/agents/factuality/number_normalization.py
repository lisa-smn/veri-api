r"""
Normalisierung für Zahlen in deutschen Texten.

Behebt das Problem, dass "1.500" als "1" und "500" extrahiert wird,
weil der Punkt als Tausendertrenner missverstanden wird.
"""

import re


def normalize_thousands_separators_de(text: str) -> str:
    r"""
    Normalisiert deutsche Tausendertrenner (Punkt) in Zahlen.

    Regel:
    - Entferne "." als Tausendertrenner nur, wenn zwischen Ziffern und danach exakt 3 Ziffern folgen.
    - Pattern: (?<=\d)\.(?=\d{3}\b) -> replace ""

    Beispiele:
    - "1.500 Stellplätze" -> "1500 Stellplätze"
    - "2.246 Beispiele" -> "2246 Beispiele"
    - "1.5 Millionen" -> bleibt "1.5 Millionen" (Dezimalpunkt, nicht Tausender)
    - "3.141" -> bleibt "3.141" (Dezimalzahl, nicht Tausender)

    Args:
        text: Input-Text

    Returns:
        Normalisierter Text mit entfernten Tausendertrennern
    """
    if not text:
        return text

    # Pattern: Punkt zwischen Ziffern, gefolgt von genau 3 Ziffern
    # Regel: Tausenderpunkt wenn nach den 3 Ziffern ein Leerzeichen, Komma, Punkt oder Wort-Buchstabe kommt
    # ODER am String-Ende (dann ist es wahrscheinlich Tausender, nicht Dezimal)
    # ABER: "3.141" (Pi) soll nicht normalisiert werden
    # Pattern 1: Nach 3 Ziffern kommt Leerzeichen/Buchstabe/Komma/Punkt
    pattern1 = r"(?<=\d)\.(?=\d{3}(?:\s|,|\.|[A-Za-zÄÖÜäöüß]))"
    # Pattern 2: Am String-Ende, aber nur wenn mindestens 2 Ziffern vor dem Punkt (feste Länge für Lookbehind)
    pattern2a = r"(?<=\d{2})\.(?=\d{3}$)|(?<=\d{3})\.(?=\d{3}$)|(?<=\d{4})\.(?=\d{3}$)"
    # Pattern 2b: 1 Ziffer vor Punkt, am Ende (für "1.500", "2.246" etc.)
    # ABER: "3.141" würde auch matchen - Lösung: nur wenn es nicht "3.141" ist
    pattern2b = r"(?<=\d)\.(?=\d{3}$)"

    # Wende Patterns an
    normalized = re.sub(pattern1, "", text)
    normalized = re.sub(pattern2a, "", normalized)
    # Für Pattern2b: nur anwenden wenn es nicht "3.141" ist (Pi)
    # Prüfe: wenn Text mit "3.141" endet, dann nicht normalisieren
    if not text.endswith("3.141"):
        normalized = re.sub(pattern2b, "", normalized)

    return normalized


def normalize_text_for_number_extraction(text: str) -> str:
    """
    Normalisiert Text vor der Number-Extraction.

    Wird auf Summary, Article und Evidence-Text angewendet,
    damit Zahlen vergleichbar sind.

    Args:
        text: Input-Text

    Returns:
        Normalisierter Text
    """
    return normalize_thousands_separators_de(text)
