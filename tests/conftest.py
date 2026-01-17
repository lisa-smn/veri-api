import os

import pytest

os.environ.setdefault("TEST_MODE", "1")
os.environ.setdefault("NEO4J_ENABLED", "0")


@pytest.fixture(autouse=True)
def _disable_neo4j(monkeypatch):
    monkeypatch.setenv("NEO4J_ENABLED", "0")
