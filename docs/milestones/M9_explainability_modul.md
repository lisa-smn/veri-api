# M9 – Explainability-Modul

## Ziel

In M9 wurde ein **Explainability-Modul** umgesetzt, das die Roh-Ergebnisse der Agenten (Factuality, Coherence, Readability) zu einem **deterministischen, prüfbaren Report** zusammenführt.
Der Fokus liegt bewusst auf **nachvollziehbaren Befunden mit Belegen (Spans)**, klaren Regeln zur Normalisierung/Deduplikation und einer stabilen, versionierten Ausgabe.

**Ergebnis:** `/verify` liefert zusätzlich ein Feld `explainability` (JSON), das für Persistenz, Analytics und Demo/Swagger geeignet ist.

---

## Designentscheidung:

**Primäre Datenquelle für Explainability ist `AgentResult.issue_spans`.**
Das Projekt verwendet damit **ein** kanonisches Format für “Issues im Text”, statt parallel `details["issues"]` o.ä. verpflichtend einzuführen.

* Agenten markieren problematische Textstellen als `issue_spans` (`IssueSpan`) (vgl. `app/models/pydantic.py:8-30, 46-49`)
* Explainability normalisiert diese Spans zu `findings`, clustert/entdoppelt und priorisiert
* Factuality kann zusätzlich *optional* Hinweise aus `details` nutzen, aber nicht als Pflichtformat

---

## Umsetzung

### 1) Datenmodell (Pydantic)

**`IssueSpan`** (früher `ErrorSpan`) enthält `issue_type`, um strukturierte Fehlertypen (z.B. NUMBER/DATE) im Report sauber zu mappen (vgl. `app/models/pydantic.py:8-30`).

```python
class IssueSpan(BaseModel):
    start_char: int | None = None
    end_char: int | None = None
    message: str
    severity: Literal["low", "medium", "high"] | None = None
    issue_type: str | None = None
    # ... weitere Felder (confidence, mapping_confidence, evidence_found, verdict)
```

**Explainability-Report** (vereinfacht beschrieben) enthält u.a.:

* `version` (aktuell: `m9_v1`)
* `summary` (kurz, aus Report-Fakten abgeleitet)
* `findings` (normalisiert, dedupliziert, gerankt)
* `by_dimension` (Findings pro Dimension)
* `top_spans` (Top-K Evidence-Spans über alle Dimensionen)
* `stats` (Basisstatistiken zur Einordnung)

---

### 2) ExplainabilityService

**Ort:** `app/services/explainability/explainability_service.py`

Der Service erstellt einen Report deterministisch aus `PipelineResult` + `summary_text`.

**Pipeline:**

1. `PipelineResult` lesen (`factuality`, `coherence`, `readability`)
2. Findings normalisieren

   * Factuality: spezieller Normalizer (z.B. Issue-Types, Severity-Logik)
   * Coherence/Readability: generische Normalisierung aus `issue_spans`
3. Dedupe/Clustering (überlappende bzw. sehr ähnliche Befunde zusammenfassen)
4. Ranking (gewichtet nach Severity/Signals)
5. Top-Spans extrahieren
6. `by_dimension`, `stats`, `executive summary` erzeugen
7. Report mit `version="m9_v1"` zurückgeben

---

### 3) Pipeline-Integration

**Ort:** `app/pipeline/verification_pipeline.py`

* ExplainabilityService wird im Pipeline-Init erzeugt
* Nach Erstellung des `PipelineResult` wird Explainability deterministisch gebaut:

```python
result.explainability = self.explainability_service.build(result, summary_text=summary)
```

Damit ist Explainability automatisch Bestandteil jedes Verifikations-Runs.

---

### 4) API-Integration

**Ort:** `app/api/routes.py`

Der `/verify` Endpoint gibt den Explainability-Report direkt zurück:

```python
return VerifyResponse(
    ...,
    explainability=result.explainability,
)
```

---

## Persistenz (Postgres)

### Tabelle: `explainability_reports`

**Ort:** `app/db/postgres/schema.sql`

* `run_id` (FK)
* `version`
* `report_json` (JSONB)
* `created_at`

Die Speicherung erfolgt in `store_verification_run(...)` optional, wenn `explainability` übergeben wird.

**Ort:** `app/db/postgres/persistence.py`

* Report wird als JSONB gespeichert (Pydantic v1/v2 kompatibel via Dump)

---

## Neo4j (best-effort & Tests)

Das System schreibt weiterhin best-effort in Neo4j.
Für Tests wurde ein Guard eingebaut:

* `NEO4J_ENABLED=0` deaktiviert Neo4j-Schreiben, damit Tests nicht durch DNS/Container-Abhängigkeiten ausgebremst werden.

---

## Tests

### Unit-Tests (Explainability-Kernlogik)

* **Clustering/Dedupe:** zwei überlappende `issue_spans` → **ein Finding**
* **Severity-Mapping:** `issue_type="NUMBER"` → Finding-Severity **high**

### API-Test (`/verify`)

* Response enthält `explainability`
* `version == "m9_v1"`
* Strukturfelder wie `findings` und `top_spans` vorhanden

**Status:** Alle Tests grün (`14 passed`).

---

## Ergebnis von M9

* Explainability läuft **end-to-end**:

  * Pipeline → Explainability → API Response → optional Postgres Persistenz
* Report ist:

  * **deterministisch** (kein LLM-Narrativ erforderlich)
  * **maschinenlesbar** (JSON)
  * **versioniert** (`m9_v1`)
  * **auditierbar** (Findings + Spans als Evidenz)

---

## Definition of Done (erfüllt)

* `/verify` liefert `explainability` stabil aus (vgl. `app/api/routes.py:31`)
* `IssueSpan.issue_type` unterstützt strukturiertes Severity/Type-Mapping (vgl. `app/models/pydantic.py:22`)
* Postgres Persistenz für Explainability-Reports vorhanden
* Tests (Unit + API) laufen grün
* Neo4j-Schreiben bleibt best-effort, Tests sind entkoppelt (NEO4J_ENABLED)

---
