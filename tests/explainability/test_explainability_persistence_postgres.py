"""
Persistence Tests für Explainability in Postgres (optional, skippbar).

Prüft:
- Save/Query funktioniert
- Keine dangling Nodes
- Cleanup funktioniert
"""

import os

import pytest

# Skip wenn DB nicht verfügbar
POSTGRES_AVAILABLE = os.getenv("POSTGRES_DSN") or os.getenv("DATABASE_URL") or False


@pytest.mark.skipif(
    not POSTGRES_AVAILABLE,
    reason="Postgres not available (POSTGRES_DSN or DATABASE_URL not set)",
)
def test_explainability_save_to_postgres():
    """Prüft, dass Explainability in Postgres gespeichert werden kann."""
    # TODO: Implementiere wenn Persistenz-Layer vorhanden
    pytest.skip("Postgres persistence not yet implemented in tests")


@pytest.mark.skipif(
    not POSTGRES_AVAILABLE,
    reason="Postgres not available (POSTGRES_DSN or DATABASE_URL not set)",
)
def test_explainability_query_from_postgres():
    """Prüft, dass Explainability aus Postgres abgefragt werden kann."""
    # TODO: Implementiere wenn Persistenz-Layer vorhanden
    pytest.skip("Postgres persistence not yet implemented in tests")


@pytest.mark.skipif(
    not POSTGRES_AVAILABLE,
    reason="Postgres not available (POSTGRES_DSN or DATABASE_URL not set)",
)
def test_explainability_postgres_cleanup():
    """Prüft, dass Test-Daten aus Postgres entfernt werden können."""
    # TODO: Implementiere wenn Persistenz-Layer vorhanden
    pytest.skip("Postgres persistence not yet implemented in tests")
