"""
Normalisierungsfunktionen für Readability-Scores.

Diese Funktionen werden sowohl in der Evaluation als auch in Tests verwendet.
"""


def normalize_to_0_1(x: float, min_v: float, max_v: float) -> float:
    """
    Normalisiert einen Wert von [min_v, max_v] auf [0, 1].

    Args:
        x: Wert zum Normalisieren
        min_v: Minimum der Quellskala
        max_v: Maximum der Quellskala (muss > min_v)

    Returns:
        Normalisierter Wert in [0, 1]

    Raises:
        ValueError: Wenn max_v <= min_v
    """
    if max_v <= min_v:
        raise ValueError("gt_max muss > gt_min sein")
    # clamp then normalize
    if x < min_v:
        x = min_v
    if x > max_v:
        x = max_v
    return (x - min_v) / (max_v - min_v)


def normalize_1_5_to_0_1(score_1_5: float) -> float:
    """
    Normalisiert einen Score von Skala 1-5 auf 0-1.

    Args:
        score_1_5: Score auf Skala 1-5

    Returns:
        Normalisierter Score in [0, 1]
    """
    # Clamp auf [1, 5]
    if score_1_5 < 1.0:
        score_1_5 = 1.0
    if score_1_5 > 5.0:
        score_1_5 = 5.0
    # Normalisiere: (score - 1) / 4
    return (score_1_5 - 1.0) / 4.0


def normalize_0_1_to_1_5(score_0_1: float) -> float:
    """
    Normalisiert einen Score von Skala 0-1 auf 1-5.

    Args:
        score_0_1: Score auf Skala 0-1

    Returns:
        Normalisierter Score in [1, 5]
    """
    # Clamp auf [0, 1]
    if score_0_1 < 0.0:
        score_0_1 = 0.0
    if score_0_1 > 1.0:
        score_0_1 = 1.0
    # Map: 1.0 + 4.0 * score
    return 1.0 + 4.0 * score_0_1
