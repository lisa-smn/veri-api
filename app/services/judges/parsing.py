"""
Robustes Parsing von LLM-Judge-Outputs.

Features:
- Strict JSON parsing
- Fallback regex extraction
- Retry-Logik
- Degeneration-Checks
"""

import json
import re
from typing import Any, Literal


def parse_judge_json(
    raw_text: str,
    dimension: Literal["readability", "coherence", "factuality"],
    expected_schema: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """
    Parst JSON aus LLM-Output mit Fallback-Mechanismen.

    Returns:
        (parsed_data, flags): parsed_data ist None bei komplettem Fehler, flags enthalten Warnungen
    """
    flags = []

    # Versuch 1: Direktes JSON-Parsing
    try:
        # Suche JSON-Objekt im Text (kann von Markdown umgeben sein)
        start = raw_text.find("{")
        end = raw_text.rfind("}") + 1
        if start != -1 and end > start:
            json_str = raw_text[start:end]
            data = json.loads(json_str)
            return data, flags
    except (json.JSONDecodeError, ValueError):
        flags.append("parse_primary_failed")

    # Versuch 2: Regex-Extraktion (Fallback)
    try:
        data = _regex_extract_judge_output(raw_text, dimension)
        if data:
            flags.append("parse_fallback")
            return data, flags
    except Exception:
        flags.append("parse_fallback_failed")

    return None, flags


def _regex_extract_judge_output(
    text: str,
    dimension: Literal["readability", "coherence", "factuality"],
) -> dict[str, Any] | None:
    """
    Fallback: Extrahiert rating/score, rationale, confidence via Regex.
    """
    data: dict[str, Any] = {}

    # Rating (1-5 oder 0-100) oder Score (0.00-1.00)
    rating_match = re.search(r'"rating"\s*:\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if not rating_match:
        rating_match = re.search(r'rating["\']?\s*[:=]\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
    if rating_match:
        data["rating"] = float(rating_match.group(1))

    # Score (0.00-1.00) - für v2_float
    score_match = re.search(r'"score"\s*:\s*([\d.]+)', text, re.IGNORECASE)
    if not score_match:
        score_match = re.search(r'score["\']?\s*[:=]\s*([\d.]+)', text, re.IGNORECASE)
    if score_match:
        data["score"] = float(score_match.group(1))

    # error_present (binary) - für factuality v2_binary
    if dimension == "factuality":
        error_match = re.search(r'"error_present"\s*:\s*(true|false)', text, re.IGNORECASE)
        if not error_match:
            error_match = re.search(
                r'error_present["\']?\s*[:=]\s*(true|false)', text, re.IGNORECASE
            )
        if error_match:
            data["error_present"] = error_match.group(1).lower() == "true"

    # Rationale
    rationale_match = re.search(r'"rationale"\s*:\s*"([^"]+)"', text, re.IGNORECASE)
    if not rationale_match:
        rationale_match = re.search(r'rationale["\']?\s*[:=]\s*"([^"]+)"', text, re.IGNORECASE)
    if rationale_match:
        data["rationale"] = rationale_match.group(1)
    else:
        data["rationale"] = ""

    # Confidence (optional)
    conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', text, re.IGNORECASE)
    if not conf_match:
        conf_match = re.search(r'confidence["\']?\s*[:=]\s*([\d.]+)', text, re.IGNORECASE)
    if conf_match:
        data["confidence"] = float(conf_match.group(1))

    if "rating" in data or "score" in data or "error_present" in data:
        return data
    return None


def normalize_rating(
    rating_raw: float | int,
    rating_scale: Literal["1-5", "0-100", "0-1"] = "1-5",
) -> float:
    """
    Normalisiert Rating auf [0,1] Skala.

    Args:
        rating_raw: Roher Rating-Wert
        rating_scale: Skala des Ratings ("1-5", "0-100", "0-1")

    Returns:
        Normalisierter Wert [0,1]
    """
    if rating_scale == "1-5":
        # (x - 1) / 4
        normalized = (float(rating_raw) - 1.0) / 4.0
    elif rating_scale == "0-100":
        # x / 100
        normalized = float(rating_raw) / 100.0
    elif rating_scale == "0-1":
        # bereits normalisiert
        normalized = float(rating_raw)
    else:
        raise ValueError(f"Unbekannte Skala: {rating_scale}")

    # Clamp auf [0,1]
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return normalized


def check_degeneration(
    outputs: list[dict[str, Any]],
    dimension: Literal["readability", "coherence", "factuality"],
) -> list[str]:
    """
    Prüft auf Degeneration (z.B. immer gleiche Werte, leere Rationales).

    Returns:
        Liste von Flags (z.B. ["constant_output", "empty_rationale"])
    """
    flags = []

    if not outputs:
        return flags

    # Prüfe auf konstante Outputs
    ratings = [out.get("rating_raw") for out in outputs if "rating_raw" in out]
    if len(ratings) > 1:
        if len(set(ratings)) == 1:
            flags.append("constant_output")

    # Prüfe auf leere Rationales
    rationales = [out.get("rationale", "").strip() for out in outputs]
    if all(not r for r in rationales):
        flags.append("empty_rationale")

    return flags
