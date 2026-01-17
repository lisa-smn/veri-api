"""Tests für deutsche Tausenderpunkt-Normalisierung."""

from app.services.agents.factuality.number_normalization import (
    normalize_text_for_number_extraction,
    normalize_thousands_separators_de,
)


class TestNormalizeThousandsSeparatorsDE:
    """Tests für Tausenderpunkt-Normalisierung."""

    def test_simple_thousands(self):
        """Einfache Tausenderzahlen."""
        assert normalize_thousands_separators_de("1.500 Stellplätze") == "1500 Stellplätze"
        assert normalize_thousands_separators_de("2.246 Beispiele") == "2246 Beispiele"
        assert normalize_thousands_separators_de("10.000 Einwohner") == "10000 Einwohner"

    def test_multiple_thousands(self):
        """Mehrere Tausenderzahlen im Text."""
        text = "Die Stadt hat 1.500 Stellplätze und 2.246 Beispiele."
        expected = "Die Stadt hat 1500 Stellplätze und 2246 Beispiele."
        assert normalize_thousands_separators_de(text) == expected

    def test_decimal_not_affected(self):
        """Dezimalzahlen sollen NICHT verändert werden."""
        assert normalize_thousands_separators_de("1.5 Millionen") == "1.5 Millionen"
        assert normalize_thousands_separators_de("3.141") == "3.141"
        assert normalize_thousands_separators_de("2.5 Prozent") == "2.5 Prozent"

    def test_mixed_thousands_and_decimals(self):
        """Gemischte Tausender und Dezimalzahlen."""
        text = "1.500 Stellplätze und 1.5 Millionen Einwohner"
        expected = "1500 Stellplätze und 1.5 Millionen Einwohner"
        assert normalize_thousands_separators_de(text) == expected

    def test_no_numbers(self):
        """Text ohne Zahlen."""
        text = "Dies ist ein normaler Text ohne Zahlen."
        assert normalize_thousands_separators_de(text) == text

    def test_empty_string(self):
        """Leerer String."""
        assert normalize_thousands_separators_de("") == ""

    def test_only_thousands(self):
        """Nur Tausenderzahl."""
        assert normalize_thousands_separators_de("1.500") == "1500"

    def test_thousands_at_start(self):
        """Tausenderzahl am Textanfang."""
        assert normalize_thousands_separators_de("1.500 Stellplätze") == "1500 Stellplätze"

    def test_thousands_at_end(self):
        """Tausenderzahl am Textende."""
        assert normalize_thousands_separators_de("Die Zahl ist 1.500") == "Die Zahl ist 1500"

    def test_four_digit_thousands(self):
        """4-stellige Tausenderzahlen."""
        assert normalize_thousands_separators_de("12.345 Einwohner") == "12345 Einwohner"

    def test_six_digit_thousands(self):
        """6-stellige Tausenderzahlen (zwei Trennzeichen)."""
        text = "1.234.567 Einwohner"
        expected = "1234567 Einwohner"
        assert normalize_thousands_separators_de(text) == expected


class TestNormalizeTextForNumberExtraction:
    """Tests für Text-Normalisierung vor Number-Extraction."""

    def test_applies_thousands_normalization(self):
        """Sollte Tausenderpunkt-Normalisierung anwenden."""
        text = "1.500 Stellplätze"
        assert normalize_text_for_number_extraction(text) == "1500 Stellplätze"

    def test_preserves_decimal_numbers(self):
        """Sollte Dezimalzahlen erhalten."""
        text = "1.5 Millionen"
        assert normalize_text_for_number_extraction(text) == "1.5 Millionen"


class TestNumberExtractionAfterNormalization:
    """Integrationstest: Number-Extraction nach Normalisierung."""

    def test_number_extraction_with_normalization(self):
        """Test, dass nach Normalisierung korrekte Zahlen extrahiert werden."""
        import re

        # Original (ohne Normalisierung)
        text_original = "1.500 Stellplätze"
        numbers_original = set(re.findall(r"\d{1,4}", text_original))
        # Erwartung: {"1", "500"} (falsch)
        assert "1" in numbers_original
        assert "500" in numbers_original

        # Nach Normalisierung
        text_normalized = normalize_thousands_separators_de(text_original)
        numbers_normalized = set(re.findall(r"\d{1,4}", text_normalized))
        # Erwartung: {"1500"} oder {"1", "500"} je nach Regex, aber "1500" sollte vorkommen
        # Besser: verwende \d+ für vollständige Zahlen
        numbers_full = set(re.findall(r"\d+", text_normalized))
        assert "1500" in numbers_full
        # "1" und "500" sollten NICHT mehr separat vorkommen
        assert not ("1" in numbers_full and "500" in numbers_full and "1500" in numbers_full)

    def test_claim_verifier_number_matching(self):
        """Simuliert Claim-Verifier Number-Matching nach Normalisierung."""
        import re

        # Claim mit Tausenderzahl
        claim = "1.500 Stellplätze"
        evidence = "Der Parkplatz hat 1.500 Stellplätze."

        # Ohne Normalisierung: Claim hat {"1", "500"}, Evidence hat {"1", "500"} -> Match
        # Aber das ist technisch korrekt, nur die Zahl ist falsch interpretiert

        # Mit Normalisierung:
        claim_norm = normalize_thousands_separators_de(claim)
        evidence_norm = normalize_thousands_separators_de(evidence)

        # Beide sollten "1500" enthalten
        claim_numbers = set(re.findall(r"\d+", claim_norm))
        evidence_numbers = set(re.findall(r"\d+", evidence_norm))

        # "1500" sollte in beiden vorkommen
        assert "1500" in claim_numbers
        assert "1500" in evidence_numbers
        assert claim_numbers & evidence_numbers  # Overlap vorhanden

    def test_real_world_example(self):
        """Test mit realistischem Beispiel: '1.500 Stellplätzen'."""
        import re

        from app.services.agents.factuality.number_normalization import (
            normalize_text_for_number_extraction,
        )

        summary = (
            "Hamburg eröffnete 2021 am Hauptbahnhof einen Parkplatz mit etwa 1.500 Stellplätzen."
        )
        article = (
            "Hamburg eröffnete 2020 am Hauptbahnhof einen Parkplatz mit etwa 1.500 Stellplätzen."
        )

        # Normalisiere beide Texte
        summary_norm = normalize_text_for_number_extraction(summary)
        article_norm = normalize_text_for_number_extraction(article)

        # Extrahiere Zahlen
        summary_numbers = set(re.findall(r"\d+", summary_norm))
        article_numbers = set(re.findall(r"\d+", article_norm))

        # "1500" sollte in beiden vorkommen (nicht "1" und "500" separat)
        assert "1500" in summary_numbers
        assert "1500" in article_numbers
        # "1" und "500" sollten NICHT separat vorkommen
        assert "1" not in summary_numbers or "500" not in summary_numbers
        # Oder präziser: "1500" ist vorhanden, und "1" und "500" sind nicht beide vorhanden
        assert not ({"1", "500"} <= summary_numbers and "1500" not in summary_numbers)
