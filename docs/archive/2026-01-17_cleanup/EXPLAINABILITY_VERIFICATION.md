# Explainability-Verifikation: Systematische Prüfung der Aussagen

**Datum:** 2026-01-03  
**Zweck:** Verifikation der Aussagen über Explainability-Integration im System

---

## 1) Repo-weite Suche nach Explainability-relevanten Begriffen

| Begriff | Datei | Zeilen | Kontext |
|---------|-------|--------|---------|
| **ExplainabilityService** | `app/pipeline/verification_pipeline.py` | 21, 39 | Import und Initialisierung in Pipeline |
| **explainability_service** | `app/pipeline/verification_pipeline.py` | 39, 58 | Service-Instanz wird erstellt und `build()` aufgerufen |
| **.build(** | `app/pipeline/verification_pipeline.py` | 58 | `self.explainability_service.build(result, summary_text=summary)` |
| **ExplainabilityResult** | `app/models/pydantic.py` | 4, 71, 84 | Import und Verwendung in PipelineResult/VerifyResponse |
| **explainability_reports** | `app/db/postgres/persistence.py` | 226 | SQL-Insert in `explainability_reports` Tabelle |
| **store_explainability** | - | - | **Nicht auffindbar** (Suche nach "store_explainability", "store.*explainability") |
| **report_json** | `app/db/postgres/persistence.py` | 226, 233 | Feld in `explainability_reports` Tabelle |

---

## 2) Pipeline-Integration

**Datei:** `app/pipeline/verification_pipeline.py`

### Reihenfolge in der Pipeline:

| Schritt | Zeilen | Beleg |
|---------|--------|-------|
| **a) Agenten laufen zuerst** | 42-44 | `factuality = self.factuality_agent.run(...)`, `coherence = self.coherence_agent.run(...)`, `readability = self.readability_agent.run(...)` |
| **b) Aggregation overall_score** | 46 | `overall = (factuality.score + coherence.score + readability.score) / 3.0` |
| **c) Explainability wird danach gebaut** | 58 | `result.explainability = self.explainability_service.build(result, summary_text=summary)` |

**Kurzbegründung:** Pipeline baut Explainability nach Agenten und Aggregation. Zeile 58 zeigt explizit, dass `build()` nach `PipelineResult`-Erstellung aufgerufen wird.

---

## 3) Determinismus-Prüfung

**Datei:** `app/services/explainability/explainability_service.py`

### Prüfung auf LLM/Client-Aufrufe:

**Suche nach:** `llm_client`, `OpenAIClient`, `chat`, `completions`, `random`, `numpy.random`

**Ergebnis:** ❌ **Keine Treffer** in `app/services/explainability/explainability_service.py`

**Code-Analyse:**
- Zeilen 259-289: `build()` Methode verwendet nur:
  - `_get_attr_or_key()` (Zeile 261-263) - Dictionary/Attribute-Zugriff
  - `_normalize_factuality()`, `_normalize_generic()` (Zeilen 266-268) - Regelbasierte Normalisierung
  - `_dedupe_and_cluster()` (Zeile 270) - Deterministisches Clustering
  - `_rank()` (Zeile 271) - Deterministisches Ranking (mathematische Formel)
  - `_top_spans()`, `_stats()`, `_executive_summary()` (Zeilen 273, 279, 280) - Regelbasierte Generierung

**Verwendete Bibliotheken:**
- `hashlib` (Zeile 32) - Deterministisches Hashing
- `math` (Zeile 33) - Mathematische Funktionen (deterministisch)
- Keine `random`, `numpy.random`, `time` (außer für Logging)

**Status:** ✅ **Deterministisch** (keine LLM-Calls, keine Randomness)

---

## 4) API-Output-Prüfung

### a) VerifyResponse enthält Feld "explainability"

**Datei:** `app/models/pydantic.py` (Zeilen 74-84)

```python
class VerifyResponse(BaseModel):
    run_id: int
    overall_score: float
    factuality: AgentResult
    coherence: AgentResult
    readability: AgentResult
    explainability: Optional[ExplainabilityResult] = None
```

**Beleg:** Zeile 84 - `explainability: Optional[ExplainabilityResult] = None`

### b) /verify Endpoint gibt explainability im Response zurück

**Datei:** `app/api/routes.py` (Zeilen 19-33)

```python
@router.post("/verify", response_model=VerifyResponse)
def verify(req: VerifyRequest, db: Session = Depends(get_db)):
    try:
        run_id, result = verification_service.verify(req, db)
        return VerifyResponse(
            run_id=run_id,
            overall_score=result.overall_score,
            factuality=result.factuality,
            coherence=result.coherence,
            readability=result.readability,
            explainability=result.explainability,  # <- Zeile 30
        )
```

**Beleg:** Zeile 30 - `explainability=result.explainability` wird explizit im Response zurückgegeben.

---

## 5) Postgres-Persistenz-Prüfung

### a) Wo wird run gespeichert?

**Datei:** `app/services/verification_service.py` (Zeilen 58-67)

```python
if not TEST_MODE:
    run_id = store_verification_run(
        db=db,
        article_id=article_id,
        summary_id=summary_id,
        overall_score=result.overall_score,
        factuality=result.factuality,
        coherence=result.coherence,
        readability=result.readability,
        explainability=result.explainability,  # <- Zeile 66
    )
```

**Beleg:** Zeile 58 - `store_verification_run()` wird in `verify()` aufgerufen (nur wenn nicht TEST_MODE).

### b) Wo wird explainability gespeichert?

**Datei:** `app/db/postgres/persistence.py` (Zeilen 219-235)

```python
# Explainability Report speichern (falls vorhanden)
if explainability is not None:
    report_payload = _pydantic_dump(explainability)

    db.execute(
        text(
            """
            INSERT INTO explainability_reports (run_id, version, report_json)
            VALUES (:run_id, :version, :report_json::jsonb)
            """
        ),
        {
            "run_id": run_id,
            "version": getattr(explainability, "version", "m9_v1"),
            "report_json": json.dumps(report_payload),
        },
    )
```

**Beleg:** Zeilen 220-235 - Explainability wird in `explainability_reports` Tabelle gespeichert (Feld `report_json` als JSONB).

### c) Wird das wirklich in verify(...) aufgerufen?

**Datei:** `app/services/verification_service.py` (Zeilen 33-72)

**Beleg:** 
- Zeile 50-54: Pipeline wird ausgeführt → `result` enthält `explainability`
- Zeile 58-67: `store_verification_run()` wird aufgerufen mit `explainability=result.explainability`
- `store_verification_run()` speichert Explainability in Zeilen 219-235 (siehe oben)

**Kurzbegründung:** Ja, `verify()` ruft `store_verification_run()` auf, welches Explainability in `explainability_reports` speichert.

---

## 6) Neo4j-Traceability-Prüfung

**Datei:** `app/db/neo4j/graph_persistence.py`

### a) Welche Nodes/Edges werden geschrieben?

**Zeilen 62-82: Core Nodes**
- `Article` (Zeile 64)
- `Summary` (Zeile 65)
- `Run` (Zeile 75)
- Edge: `(Article)-[:HAS_SUMMARY]->(Summary)` (Zeile 66)
- Edge: `(Run)-[:EVALUATES]->(Summary)` (Zeile 78)

**Zeilen 84-137: Metric Nodes + IssueSpans**
- `Metric` Nodes (Zeile 89) - pro Dimension (factuality, coherence, readability, overall)
- `IssueSpan` Nodes (Zeile 112) - Dual-Label: `IssueSpan:Error`
- Edge: `(Summary)-[:HAS_METRIC]->(Metric)` (Zeile 91)
- Edge: `(Metric)-[:HAS_ISSUE_SPAN]->(IssueSpan)` (Zeile 123)

**Beleg:** Zeilen 135-137 - `metric_with_issue_spans()` wird für factuality, coherence, readability aufgerufen.

### b) Wird Explainability-Report als Node/Property gespeichert?

**Suche nach:** "explainability", "Explainability", "report", "Report" in `app/db/neo4j/graph_persistence.py`

**Ergebnis:** ❌ **Nicht vorhanden**

**Funktionssignatur:** `write_verification_graph()` (Zeilen 12-20) akzeptiert nur:
- `article_id`, `summary_id`, `run_id`, `overall_score`
- `factuality`, `coherence`, `readability` (AgentResult)
- **Kein `explainability` Parameter**

**Beleg:** Zeilen 12-20, 51-60 - `explainability` wird nicht an `write_verification_graph()` übergeben und nicht in Neo4j geschrieben.

**Kurzbegründung:** Explainability-Report wird **nicht** in Neo4j gespeichert. Nur Agent-Issues (IssueSpans) werden als Nodes geschrieben.

---

## Zusammenfassung: Verifikationstabelle

| Claim | Status | Belegstellen | Kurzbegründung |
|-------|--------|--------------|----------------|
| **1) Explainability ist fest in der Pipeline integriert: Nach den Agenten wird ein Explainability-Report gebaut.** | ✅ | `app/pipeline/verification_pipeline.py:42-44,46,58` | Agenten laufen zuerst (42-44), dann Aggregation (46), dann Explainability.build() (58) |
| **2) Der Explainability-Report ist deterministisch (keine LLM-Calls/Randomness) und basiert auf Agent-Outputs.** | ✅ | `app/services/explainability/explainability_service.py:259-289` | Keine LLM/Client-Aufrufe, keine Randomness; nur regelbasierte Normalisierung, Clustering, Ranking |
| **3) Der Explainability-Report wird im API-Response zurückgegeben.** | ✅ | `app/models/pydantic.py:84`, `app/api/routes.py:30` | VerifyResponse enthält `explainability` Feld (84), `/verify` gibt es zurück (30) |
| **4) Der Explainability-Report wird persistiert (Postgres) und ist über Runs nachvollziehbar.** | ✅ | `app/services/verification_service.py:58-67`, `app/db/postgres/persistence.py:219-235` | `store_verification_run()` wird in `verify()` aufgerufen (58-67), speichert in `explainability_reports` Tabelle (219-235) |
| **5) Zusätzlich gibt es Traceability in Neo4j (mind. IssueSpans/Beziehungen). Prüfe, ob Explainability selbst in Neo4j landet oder nur Agent-Issues.** | ⚠️ | `app/db/neo4j/graph_persistence.py:12-151` | **Nur Agent-Issues (IssueSpans) werden in Neo4j geschrieben** (Zeilen 99-133). Explainability-Report wird **nicht** in Neo4j gespeichert (nicht in Funktionssignatur, keine Nodes/Properties für Explainability). |

---

## Fazit

**✅ Vollständig verifiziert:**
- Explainability ist fest in der Pipeline integriert (nach Agenten)
- Explainability ist deterministisch (keine LLM-Calls)
- Explainability wird im API-Response zurückgegeben
- Explainability wird in Postgres persistiert (`explainability_reports` Tabelle)

**⚠️ Teilweise:**
- Neo4j-Traceability: Nur Agent-Issues (IssueSpans) werden in Neo4j geschrieben, **nicht** der Explainability-Report selbst. Der Report ist nur in Postgres verfügbar.

**Empfehlung:**
- Wenn Explainability auch in Neo4j benötigt wird, müsste `write_verification_graph()` erweitert werden, um Explainability-Nodes/Properties zu schreiben.






