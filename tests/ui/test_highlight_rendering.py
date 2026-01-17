"""Tests für Highlight-Rendering in UI."""

from ui.render import _deduplicate_and_merge_spans


class TestDeduplicateAndMergeSpans:
    """Tests für Span-Deduplizierung und Merge."""

    def test_no_overlaps(self):
        """Keine Überlappungen: alle Spans bleiben."""
        spans = [
            {"start": 0, "end": 10, "severity": "high", "dimension": "factuality", "message": ""},
            {"start": 20, "end": 30, "severity": "medium", "dimension": "coherence", "message": ""},
        ]
        result = _deduplicate_and_merge_spans(spans)
        assert len(result) == 2

    def test_identical_ranges(self):
        """Identische Ranges: höchste Severity gewinnt."""
        spans = [
            {"start": 0, "end": 10, "severity": "low", "dimension": "factuality", "message": ""},
            {"start": 0, "end": 10, "severity": "high", "dimension": "coherence", "message": ""},
        ]
        result = _deduplicate_and_merge_spans(spans)
        assert len(result) == 1
        assert result[0]["severity"] == "high"

    def test_overlapping_spans(self):
        """Überlappende Spans werden gemerged."""
        spans = [
            {"start": 0, "end": 20, "severity": "high", "dimension": "factuality", "message": ""},
            {"start": 15, "end": 35, "severity": "medium", "dimension": "coherence", "message": ""},
        ]
        result = _deduplicate_and_merge_spans(spans)
        assert len(result) == 1
        assert result[0]["start"] == 0
        assert result[0]["end"] == 35
        assert result[0]["severity"] == "high"  # max severity

    def test_nested_spans(self):
        """Verschachtelte Spans werden gemerged."""
        spans = [
            {"start": 0, "end": 50, "severity": "high", "dimension": "factuality", "message": ""},
            {"start": 10, "end": 30, "severity": "medium", "dimension": "coherence", "message": ""},
        ]
        result = _deduplicate_and_merge_spans(spans)
        assert len(result) == 1
        assert result[0]["start"] == 0
        assert result[0]["end"] == 50

    def test_no_duplicate_text(self):
        """Test, dass Text nicht doppelt gerendert wird."""
        summary = "Hamburg eröffnete 2021 am Hauptbahnhof einen neuen Parkplatz."
        spans = [
            {"start": 0, "end": 64, "severity": "high", "dimension": "factuality", "message": ""},
            {"start": 0, "end": 64, "severity": "high", "dimension": "coherence", "message": ""},
            {"start": 76, "end": 92, "severity": "high", "dimension": "factuality", "message": ""},
        ]

        # Deduplicate sollte identische Ranges zusammenführen
        merged = _deduplicate_and_merge_spans(spans)
        assert len(merged) == 2  # Zwei unique Spans

        # Prüfe, dass keine doppelten Ranges vorhanden sind
        ranges = [(s["start"], s["end"]) for s in merged]
        assert len(ranges) == len(set(ranges))  # Keine Duplikate

    def test_severity_max_on_overlap(self):
        """Bei Überlappung: max Severity wird verwendet."""
        spans = [
            {"start": 0, "end": 20, "severity": "low", "dimension": "factuality", "message": ""},
            {"start": 10, "end": 30, "severity": "high", "dimension": "coherence", "message": ""},
        ]
        result = _deduplicate_and_merge_spans(spans)
        assert result[0]["severity"] == "high"
