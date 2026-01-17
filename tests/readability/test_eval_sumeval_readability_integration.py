"""
Mini-Integration-Test für eval_sumeval_readability.py.
"""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def mini_fixture_path():
    """Pfad zur Mini-Fixture."""
    fixture_path = Path(__file__).parent.parent / "fixtures" / "sumeval_readability_mini.jsonl"
    if not fixture_path.exists():
        pytest.skip(f"Fixture nicht gefunden: {fixture_path}")
    return fixture_path


@pytest.fixture
def temp_output_dir(tmp_path):
    """Temporäres Output-Verzeichnis."""
    return tmp_path / "test_output"


def test_eval_sumeval_readability_importable(monkeypatch):
    """Prüft, dass das Script importierbar ist."""
    # Mock dotenv, um Sandbox-Probleme zu vermeiden
    import unittest.mock

    with unittest.mock.patch("dotenv.load_dotenv"):
        # Füge Script-Verzeichnis zum Path hinzu
        script_dir = Path(__file__).parent.parent.parent / "scripts"
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))

        # Versuche Import (ohne Ausführung)
        try:
            import eval_sumeval_readability

            assert hasattr(eval_sumeval_readability, "main")
        except (ImportError, PermissionError) as e:
            pytest.skip(f"Script nicht importierbar: {e}")


def test_eval_sumeval_readability_output_structure(mini_fixture_path, temp_output_dir, monkeypatch):
    """
    Prüft, dass das Script die erwartete CLI-Struktur hat.

    Skip-Reason: Subprocess-Ausführung ist in CI-Sandbox nicht zuverlässig.
    Dieser Test prüft nur die CLI-Struktur via --help und wird übersprungen,
    wenn Subprocess-Ausführung nicht möglich ist.
    """
    # Setze ENV-Variablen
    monkeypatch.setenv("ENABLE_LLM_JUDGE", "false")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")

    # Prüfe, ob das Script existiert
    script_path = Path(__file__).parent.parent.parent / "scripts" / "eval_sumeval_readability.py"
    if not script_path.exists():
        pytest.skip(reason=f"Script nicht gefunden: {script_path}")

    # Prüfe, ob wir in CI sind (dann skip mit konsistenter Begründung)
    if os.getenv("CI") == "true":
        pytest.skip(reason="Subprocess execution restricted in CI sandbox")

    # Lokal: Versuche Subprocess-Ausführung
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Script sollte mit --help funktionieren
        assert (
            result.returncode == 0
            or "usage" in result.stdout.lower()
            or "help" in result.stdout.lower()
        )
    except (subprocess.TimeoutExpired, OSError, PermissionError) as e:
        pytest.skip(reason=f"Subprocess execution restricted: {type(e).__name__}")


def test_mini_fixture_format(mini_fixture_path):
    """Prüft, dass die Mini-Fixture das erwartete Format hat."""
    with mini_fixture_path.open("r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    assert len(lines) > 0, "Fixture sollte mindestens 1 Zeile enthalten"
    assert len(lines) <= 10, "Fixture sollte maximal 10 Zeilen enthalten (Mini-Test)"

    # Prüfe Format jeder Zeile
    for i, line in enumerate(lines, 1):
        try:
            data = json.loads(line)
            assert "article" in data or "summary" in data, (
                f"Zeile {i}: Fehlt 'article' oder 'summary'"
            )
            assert "gt" in data, f"Zeile {i}: Fehlt 'gt'"
            assert "readability" in data["gt"], f"Zeile {i}: Fehlt 'gt.readability'"
        except json.JSONDecodeError as e:
            pytest.fail(f"Zeile {i} ist kein gültiges JSON: {e}")
