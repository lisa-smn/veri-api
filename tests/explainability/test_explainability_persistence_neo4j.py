"""
Persistence Tests für Explainability in Neo4j (optional, skippbar).

Prüft:
- Save/Query funktioniert
- Keine dangling Nodes
- Cleanup funktioniert
"""

import os

import pytest

# Skip wenn DB nicht verfügbar
NEO4J_AVAILABLE = (
    os.getenv("NEO4J_URI") or os.getenv("NEO4J_URL") or os.getenv("NEO4J_ENABLED") == "1" or False
)


@pytest.mark.skipif(
    not NEO4J_AVAILABLE,
    reason="Neo4j not available (NEO4J_URI or NEO4J_URL not set, or NEO4J_ENABLED != '1')",
)
def test_explainability_save_to_neo4j():
    """Prüft, dass Explainability in Neo4j gespeichert werden kann."""
    # TODO: Implementiere wenn Persistenz-Layer vorhanden
    pytest.skip("Neo4j persistence not yet implemented in tests")


@pytest.mark.skipif(
    not NEO4J_AVAILABLE,
    reason="Neo4j not available (NEO4J_URI or NEO4J_URL not set, or NEO4J_ENABLED != '1')",
)
def test_explainability_query_from_neo4j():
    """Prüft, dass Explainability aus Neo4j abgefragt werden kann."""
    # TODO: Implementiere wenn Persistenz-Layer vorhanden
    pytest.skip("Neo4j persistence not yet implemented in tests")


@pytest.mark.skipif(
    not NEO4J_AVAILABLE,
    reason="Neo4j not available (NEO4J_URI or NEO4J_URL not set, or NEO4J_ENABLED != '1')",
)
def test_explainability_neo4j_no_dangling_nodes():
    """Prüft, dass keine dangling Nodes in Neo4j existieren."""
    # TODO: Implementiere wenn Persistenz-Layer vorhanden
    pytest.skip("Neo4j persistence not yet implemented in tests")
