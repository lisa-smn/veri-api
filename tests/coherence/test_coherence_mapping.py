"""
Mapping-Tests: 1-5 -> 0-1 und 0-1 -> 1-5 (falls vorhanden).
"""

from app.services.agents.readability.normalization import (
    normalize_1_5_to_0_1,
    normalize_to_0_1,
)


def test_coherence_normalize_1_5_to_0_1_basic():
    """Prüft grundlegende Normalisierung 1-5 -> 0-1 (wiederverwendet von Readability)."""
    assert normalize_1_5_to_0_1(1.0) == 0.0
    assert normalize_1_5_to_0_1(5.0) == 1.0
    assert normalize_1_5_to_0_1(3.0) == 0.5


def test_coherence_normalize_to_0_1_basic():
    """Prüft allgemeine Normalisierung mit min/max (GT-Normalisierung)."""
    # Standard GT-Normalisierung: 1-5 -> 0-1
    assert normalize_to_0_1(1.0, min_v=1.0, max_v=5.0) == 0.0
    assert normalize_to_0_1(5.0, min_v=1.0, max_v=5.0) == 1.0
    assert normalize_to_0_1(3.0, min_v=1.0, max_v=5.0) == 0.5
