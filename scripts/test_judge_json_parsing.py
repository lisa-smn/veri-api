"""
Mini-Test für Judge JSON-Parsing und Normalisierung.

Testet:
- Valid JSON parse
- Repair-Prompt path
- Normalisierung 1-5 -> [0,1]
"""

import json
import re
from typing import Any


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """Versucht JSON aus Text zu extrahieren (auch wenn Markdown-Code-Blöcke vorhanden)."""
    # Entferne Markdown-Code-Blöcke
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()

    # Versuche direktes Parsing
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Versuche JSON in geschweiften Klammern zu finden
    match = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def normalize_to_0_1(x: float, min_v: float, max_v: float) -> float:
    """Normalisiert Wert von [min_v, max_v] nach [0, 1]."""
    if max_v <= min_v:
        raise ValueError("gt_max muss > gt_min sein")
    if x < min_v:
        x = min_v
    if x > max_v:
        x = max_v
    return (x - min_v) / (max_v - min_v)


def test_valid_json():
    """Test: Valides JSON wird korrekt geparst."""
    test_cases = [
        (
            '{"coherence_score_1_to_5": 3, "main_issue": "none", "explanation_short": "OK", "confidence_0_to_1": 0.8}',
            True,
        ),
        (
            '```json\n{"coherence_score_1_to_5": 4, "main_issue": "missing_link", "explanation_short": "Test", "confidence_0_to_1": 0.9}\n```',
            True,
        ),
        (
            'Some text before {"coherence_score_1_to_5": 2, "main_issue": "contradiction", "explanation_short": "Bad", "confidence_0_to_1": 0.3} some text after',
            True,
        ),
        ("Invalid text without JSON", False),
        ('{"incomplete": json', False),
    ]

    print("Test: Valid JSON Parsing")
    print("=" * 60)
    for i, (text, should_succeed) in enumerate(test_cases, 1):
        result = extract_json_from_text(text)
        success = result is not None
        if success:
            score = result.get("coherence_score_1_to_5")
            valid = isinstance(score, (int, float)) and 1 <= score <= 5
            success = valid

        status = "✅" if success == should_succeed else "❌"
        print(f"{status} Test {i}: {text[:50]}... -> {success} (expected: {should_succeed})")
        if result:
            print(f"   Parsed: coherence_score={result.get('coherence_score_1_to_5')}")
    print()


def test_normalization():
    """Test: Normalisierung 1-5 -> [0,1]."""
    test_cases = [
        (1.0, 1.0, 5.0, 0.0),
        (3.0, 1.0, 5.0, 0.5),
        (5.0, 1.0, 5.0, 1.0),
        (2.5, 1.0, 5.0, 0.375),
        (0.0, 1.0, 5.0, 0.0),  # Clamp
        (6.0, 1.0, 5.0, 1.0),  # Clamp
    ]

    print("Test: Normalisierung 1-5 -> [0,1]")
    print("=" * 60)
    for x, min_v, max_v, expected in test_cases:
        result = normalize_to_0_1(x, min_v, max_v)
        status = "✅" if abs(result - expected) < 1e-6 else "❌"
        print(
            f"{status} normalize({x}, {min_v}, {max_v}) = {result:.4f} (expected: {expected:.4f})"
        )
    print()


def test_repair_scenario():
    """Test: Repair-Szenario (simuliert)."""
    print("Test: Repair-Szenario (simuliert)")
    print("=" * 60)

    # Simuliere fehlerhaftes JSON
    broken_response = """Die Summary hat eine Kohärenz von 3 auf einer Skala von 1-5.
    Hauptproblem: missing_link
    Erklärung: Es fehlen Übergänge zwischen den Sätzen.
    Confidence: 0.7"""

    # Versuche zu parsen
    parsed = extract_json_from_text(broken_response)
    if parsed is None:
        print("✅ Broken JSON correctly rejected")
        print("   → Would trigger repair prompt in actual implementation")
    else:
        print("⚠️  Broken JSON was parsed (unexpected)")

    # Simuliere Repair-Response
    repair_response = '{"coherence_score_1_to_5": 3, "main_issue": "missing_link", "explanation_short": "Fehlende Übergänge", "confidence_0_to_1": 0.7}'
    repaired = extract_json_from_text(repair_response)
    if repaired and repaired.get("coherence_score_1_to_5") == 3:
        print("✅ Repair response correctly parsed")
        print(f"   Parsed: {repaired}")
    else:
        print("❌ Repair response parsing failed")
    print()


def main():
    print("\n" + "=" * 60)
    print("Judge JSON Parsing & Normalization Tests")
    print("=" * 60 + "\n")

    test_valid_json()
    test_normalization()
    test_repair_scenario()

    print("=" * 60)
    print("Tests abgeschlossen!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
