"""Markdown-Dokumente Viewer."""

from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DOCS_STATUS_DIR = ROOT / "docs" / "status"

DOCS_FILES = [
    ("agents_verification_audit.md", "Agent Verification Audit"),
    ("persistence_audit.md", "Persistence Audit"),
    ("explainability_persistence_proof.md", "Explainability Persistence Proof"),
    ("readability_status.md", "Readability Status"),
    ("factuality_status.md", "Factuality Status"),
    ("coherence_status.md", "Coherence Status"),
    ("fig_explainability_persistence_proof.md", "Explainability Persistence Proof (Figure)"),
]


def get_available_docs() -> list[tuple[str, str, bool]]:
    """Gibt Liste von (filename, title, exists) zurück."""
    return [
        (filename, title, (DOCS_STATUS_DIR / filename).exists()) for filename, title in DOCS_FILES
    ]


def render_doc(filename: str) -> None:
    """Rendert ein Markdown-Dokument."""
    filepath = DOCS_STATUS_DIR / filename
    if not filepath.exists():
        st.warning(f"Datei nicht gefunden: {filename}")
        return

    try:
        with filepath.open("r", encoding="utf-8") as f:
            content = f.read()
        st.markdown(content)
    except Exception as e:
        st.error(f"Fehler beim Lesen der Datei: {e}")
