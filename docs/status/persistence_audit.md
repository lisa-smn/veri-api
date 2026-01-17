# Persistence Audit: Postgres + Neo4j

**Datum:** 2026-01-17 01:25:47
**Zweck:** Automatische Prüfung der Datenintegrität und Konsistenz zwischen Postgres und Neo4j

---

## Executive Summary

- **Postgres:** ⚠️  WARNINGS
- **Neo4j:** ⚠️  WARNINGS
- **Cross-Store Konsistenz:** ✅ PASS

---

## Postgres Findings

### Table Inventory

| Table | Row Count | Key Columns |
|-------|-----------|--------------|
| datasets | 10 | created_at |
| articles | 39 | created_at |
| errors | 0 | run_id, created_at |
| summaries | 39 | created_at |
| runs | 21 | - |
| verification_results | 76 | run_id, created_at, dimension, score |
| explanations | 57 | run_id, created_at |
| explainability_reports | 2 | run_id, created_at |

**Run-related Tables:** runs, verification_results

### Constraints

- **explainability_reports:** uq_explainability_reports_run_version

### Duplicates

- **verification_results:** 10 Duplikate gefunden ⚠️
- **explanations:** 10 Duplikate gefunden ⚠️

### Missingness (NULL in Key Columns)

- ✅ Keine NULL-Werte in Key Columns

### Sample Rows (Newest)

**datasets:**
  1. id=13, name=proof, split=None, description=Proof test dataset, source_url=None
  2. id=11, name=proof, split=None, description=Proof test dataset, source_url=None

**articles:**
  1. id=42, dataset_id=13, external_id=None, title=None, text=Die Studie zeigt, dass 50% der Teilnehmer positiv reagierten. Das Datum war 2020.
  2. id=40, dataset_id=11, external_id=None, title=None, text=Die Studie zeigt, dass 50% der Teilnehmer positiv reagierten. Das Datum war 2020.

**summaries:**
  1. id=42, article_id=42, source=llm, text=Die Studie zeigt, dass 50% der Teilnehmer positiv reagierten. Das Datum war 2020., llm_model=None
  2. id=40, article_id=40, source=llm, text=Die Studie zeigt, dass 50% der Teilnehmer positiv reagierten. Das Datum war 2020., llm_model=None


---

## Neo4j Findings

### Labels & Relationship Types

**Labels:** Article, Summary, Metric, Error, IssueSpan, Run, Explainability, Finding, Span
**Relationship Types:** HAS_SUMMARY, HAS_METRIC, HAS_ERROR, HAS_ISSUE_SPAN, EVALUATES, HAS_EXPLAINABILITY, HAS_FINDING, HAS_SPAN

### Label Counts

| Label | Count |
|-------|-------|
| Article | 15 |
| Error | 19 |
| Explainability | 2 |
| Finding | 4 |
| IssueSpan | 24 |
| Metric | 60 |
| Run | 8 |
| Span | 4 |
| Summary | 15 |

### Constraints & Indexes

**Constraints:** 0
**Indexes:** 2

### Run Nodes

**Labels mit run_id Property:** Metric (run_id), Error:IssueSpan (run_id), IssueSpan (run_id), Finding (run_id), Span (run_id), Error (run_id), Run (run_id), Explainability (run_id), Run (id)

### Dangling Nodes (isolated, keine Relationships)

- ✅ Keine isolierten Nodes gefunden

### Duplicates (same run_id >1 per label)

- **Metric:** 15 Duplikate gefunden ⚠️
- **Error:** 6 Duplikate gefunden ⚠️
- **IssueSpan:** 7 Duplikate gefunden ⚠️
- **Finding:** 2 Duplikate gefunden ⚠️
- **Span:** 2 Duplikate gefunden ⚠️

---

## Cross-Store Consistency

**Postgres run_ids (distinct):** 19
**Neo4j run_ids (distinct):** 17
**Overlap:** 11

**Nur in Postgres (Sample, max 20):**

- 12
- 9
- 3
- 22
- 23
- 25
- 24
- 14

**Nur in Neo4j (Sample, max 20):**

- 32
- 31
- 34
- proof-20260117_011800
- proof-test-001
- 33

### Sample Join Checks

| run_id | Postgres | Neo4j |
|--------|----------|-------|
| 20 | ✅ | ❌ |
| 4 | ✅ | ❌ |
| 21 | ✅ | ❌ |

---

## Recommendations

### P0 (Datenverlust / fehlende Verknüpfungen / Duplicates)

- Postgres Duplikate beheben (siehe 'Duplicates' Abschnitt)
- Neo4j Duplikate beheben (siehe 'Duplicates' Abschnitt)
- Cross-Store Konsistenz prüfen: fehlende run_ids in einem Store

### P1 (Constraints/Indexes ergänzen)

- Postgres: Unique Constraints für (run_id, dimension) prüfen
- Neo4j: Constraints für run_id uniqueness pro Label prüfen
- Indexes für häufige Queries prüfen (run_id, example_id)

### P2 (Redundanz reduzieren, Archivierung)

- Alte Runs archivieren (beide Stores)
- Redundante Daten zwischen Postgres und Neo4j dokumentieren
- Cleanup-Strategie für isolierte Nodes definieren
