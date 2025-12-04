# M4 – Graph-Erweiterung (Neo4j-Integration)

## Ziel

M4 erweitert die bestehende M3-Verifikation um eine **Graphdarstellung in Neo4j**.
SQL bleibt das Hauptsystem, Neo4j bietet eine zusätzlich visualisierbare Struktur aus Artikeln, Summaries, Scores und Fehlern.

---

## Umsetzung

### Neo4j-Anbindung

* Neuer Client (`neo4j_client.py`) mit dem offiziellen Neo4j-Python-Driver.
* Konfiguration über `.env`: `NEO4J_URL`, `NEO4J_USER`, `NEO4J_PASSWORD`.
* Sessions zum Ausführen von Cypher-Queries (`execute_write`).

### Graph-Modell

**Knoten:**

* `Article`
* `Summary`
* `Metric` (für jede Dimension + Overall)
* `Error`

**Beziehungen:**

* `HAS_SUMMARY`
* `HAS_METRIC`
* `HAS_ERROR`

Struktur entspricht der SQL-Logik, aber als navigierbarer Graph.

### Schreiblogik

* Implementiert in `graph_persistence.py`.
* `write_verification_graph(...)` erzeugt/aktualisiert:

  * Artikel- und Summary-Knoten
  * Metrics pro Dimension
  * Error-Knoten pro Agentenfehler
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

Nach jedem `/verify`-Aufruf stehen die Daten zusätzlich als Neo4j-Graph bereit:

Artikel → Summary → Metrics → Errors

Damit ist M4 abgeschlossen und das System bereit für Graphabfragen, Visualisierung und spätere Wissensgraph-Erweiterungen.
