# M4 – Graph-Erweiterung (Neo4j-Integration)

## Ziel

M4 erweitert die bestehende M3-Verifikation um eine **Graphdarstellung in Neo4j**.
SQL bleibt das Hauptsystem, Neo4j bietet eine zusätzlich visualisierbare Struktur aus Artikeln, Summaries, Scores und Fehlern.

---

## Umsetzung

### Neo4j-Anbindung

* Neuer Client (`app/db/neo4j/neo4j_client.py`) mit dem offiziellen Neo4j-Python-Driver (vgl. `app/db/neo4j/neo4j_client.py:1-26`).
* Konfiguration über `.env` via `pydantic-settings`: `NEO4J_URL`, `NEO4J_USER`, `NEO4J_PASSWORD` werden in `app/core/config.py` geladen (vgl. `app/core/config.py:19-21`, `app/db/neo4j/neo4j_client.py:12-13`).
* Sessions zum Ausführen von Cypher-Queries (`execute_write`).

### Graph-Modell

**Knoten:**

* `Article` (vgl. `app/db/neo4j/graph_persistence.py:64-66`)
* `Summary` (vgl. `app/db/neo4j/graph_persistence.py:65-66`)
* `Run` (vgl. `app/db/neo4j/graph_persistence.py:73-83`)
* `Metric` (für jede Dimension + Overall) (vgl. `app/db/neo4j/graph_persistence.py:87-98`)
* `IssueSpan:Error` (Dual-Label: sowohl `IssueSpan` als auch `Error` für Abwärtskompatibilität) (vgl. `app/db/neo4j/graph_persistence.py:113`)

**Beziehungen:**

* `HAS_SUMMARY` (Article → Summary) (vgl. `app/db/neo4j/graph_persistence.py:66`)
* `EVALUATES` (Run → Summary) (vgl. `app/db/neo4j/graph_persistence.py:79`)
* `HAS_METRIC` (Summary → Metric) (vgl. `app/db/neo4j/graph_persistence.py:92`)
* `HAS_ISSUE_SPAN` (Metric → IssueSpan) (vgl. `app/db/neo4j/graph_persistence.py:124`)

Struktur entspricht der SQL-Logik, aber als navigierbarer Graph.

### Schreiblogik

* Implementiert in `graph_persistence.py`.
* `write_verification_graph(...)` erzeugt/aktualisiert (vgl. `app/db/neo4j/graph_persistence.py:12-49`):

  * Artikel- und Summary-Knoten
  * Run-Knoten
  * Metrics pro Dimension (factuality, coherence, readability, overall)
  * IssueSpan:Error-Knoten pro Agentenfehler (mit Dual-Label)
* Alle Operationen über `MERGE`, um Duplikate zu vermeiden.

### Integration in die Persistenz

* `store_verification_run(...)` wurde erweitert:

  * SQL-Transaktion bleibt unverändert.
  * Nach erfolgreichem Commit erfolgt ein „best-effort“-Graph-Update.
  * Neo4j-Fehler blockieren nicht das SQL-Speichern.

### API & Service

* `/verify` erhält per `Depends(get_db)` eine aktive DB-Session.
* `VerificationService.verify(db, ...)` gibt Session an die Persistenzschicht weiter.

---

## Ergebnis

Nach jedem `/verify`-Aufruf stehen die Daten zusätzlich als Neo4j-Graph bereit (vgl. `app/db/postgres/persistence.py:253-268`):

Article → Summary (via HAS_SUMMARY)  
Run → Summary (via EVALUATES)  
Summary → Metric (via HAS_METRIC)  
Metric → IssueSpan:Error (via HAS_ISSUE_SPAN)

Damit ist M4 abgeschlossen und das System bereit für Graphabfragen, Visualisierung und spätere Wissensgraph-Erweiterungen.
