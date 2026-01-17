"""
Mapping-Tests: 1-5 -> 0-1 und 0-1 -> 1-5.
"""

import pytest

from app.services.agents.readability.normalization import (
    normalize_0_1_to_1_5,
    normalize_1_5_to_0_1,
    normalize_to_0_1,
)


def test_normalize_1_5_to_0_1_basic():
    """Prüft grundlegende Normalisierung 1-5 -> 0-1."""
    assert normalize_1_5_to_0_1(1.0) == 0.0
    assert normalize_1_5_to_0_1(5.0) == 1.0
    assert normalize_1_5_to_0_1(3.0) == 0.5


def test_normalize_1_5_to_0_1_clamp():
    """Prüft, dass Werte außerhalb [1, 5] geclamped werden."""
    assert normalize_1_5_to_0_1(0.0) == 0.0  # < 1.0 -> clamp auf 1.0 -> 0.0
    assert normalize_1_5_to_0_1(6.0) == 1.0  # > 5.0 -> clamp auf 5.0 -> 1.0
    assert normalize_1_5_to_0_1(-1.0) == 0.0


def test_normalize_0_1_to_1_5_basic():
    """Prüft grundlegende Normalisierung 0-1 -> 1-5."""
    assert normalize_0_1_to_1_5(0.0) == 1.0
    assert normalize_0_1_to_1_5(1.0) == 5.0
    assert normalize_0_1_to_1_5(0.5) == 3.0


def test_normalize_0_1_to_1_5_clamp():
    """Prüft, dass Werte außerhalb [0, 1] geclamped werden."""
    assert normalize_0_1_to_1_5(-0.5) == 1.0  # < 0.0 -> clamp auf 0.0 -> 1.0
    assert normalize_0_1_to_1_5(1.5) == 5.0  # > 1.0 -> clamp auf 1.0 -> 5.0


def test_normalize_to_0_1_basic():
    """Prüft allgemeine Normalisierung mit min/max."""
    # Standard GT-Normalisierung: 1-5 -> 0-1
    assert normalize_to_0_1(1.0, min_v=1.0, max_v=5.0) == 0.0
    assert normalize_to_0_1(5.0, min_v=1.0, max_v=5.0) == 1.0
    assert normalize_to_0_1(3.0, min_v=1.0, max_v=5.0) == 0.5


def test_normalize_to_0_1_clamp():
    """Prüft, dass Werte außerhalb [min_v, max_v] geclamped werden."""
    assert normalize_to_0_1(0.0, min_v=1.0, max_v=5.0) == 0.0  # clamp auf 1.0
    assert normalize_to_0_1(6.0, min_v=1.0, max_v=5.0) == 1.0  # clamp auf 5.0


def test_normalize_to_0_1_invalid_range():
    """Prüft, dass ungültiger Range einen ValueError wirft."""
    with pytest.raises(ValueError, match="gt_max muss > gt_min sein"):
        normalize_to_0_1(3.0, min_v=5.0, max_v=1.0)

    with pytest.raises(ValueError, match="gt_max muss > gt_min sein"):
        normalize_to_0_1(3.0, min_v=5.0, max_v=5.0)


def test_normalize_roundtrip():
    """Prüft, dass Normalisierung hin und zurück konsistent ist."""
    # 1-5 -> 0-1 -> 1-5
    for score_1_5 in [1.0, 2.0, 3.0, 4.0, 5.0]:
        score_0_1 = normalize_1_5_to_0_1(score_1_5)
        score_1_5_roundtrip = normalize_0_1_to_1_5(score_0_1)
        assert abs(score_1_5_roundtrip - score_1_5) < 1e-9
