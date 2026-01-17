# Persistence Audit: Postgres + Neo4j

**Datum:** 2026-01-17 01:01:16
**Zweck:** Automatische Prüfung der Datenintegrität und Konsistenz zwischen Postgres und Neo4j

---

## Executive Summary

- **Postgres:** ⚠️  WARNINGS
- **Neo4j:** ⚠️  WARNINGS
- **Cross-Store Konsistenz:** ⚠️  WARNINGS

---

## Postgres Findings

### Table Inventory

| Table | Row Count | Key Columns |
|-------|-----------|--------------|
| datasets | 8 | created_at |
| articles | 37 | created_at |
| errors | 0 | run_id, created_at |
| summaries | 37 | created_at |
| runs | 19 | - |
| verification_results | 76 | run_id, created_at, dimension, score |
| explanations | 57 | run_id, created_at |

**Run-related Tables:** runs, verification_results

### Constraints

- Keine Unique Constraints gefunden

### Duplicates

- **verification_results:** 10 Duplikate gefunden ⚠️
- **explanations:** 10 Duplikate gefunden ⚠️

### Missingness (NULL in Key Columns)

- ✅ Keine NULL-Werte in Key Columns

### Sample Rows (Newest)

**datasets:**
  1. id=8, name=frank, split=None, description=None, source_url=None
  2. id=7, name=manual, split=None, description=None, source_url=None

**articles:**
  1. id=37, dataset_id=3, external_id=None, title=None, text=Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner und ist seit 1990 wieder vereint. Berlin liegt im Nordosten Deutschlands an der Spree.
  2. id=36, dataset_id=3, external_id=None, title=None, text=Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner.

**summaries:**
  1. id=37, article_id=37, source=llm, text=Paris ist die Hauptstadt von Deutschland. Die Stadt hat 4 Millionen Einwohner., llm_model=gpt-4o-mini
  2. id=36, article_id=36, source=llm, text=Paris ist die Hauptstadt von Deutschland. Die Stadt hat 4 Millionen Einwohner., llm_model=gpt-4o-mini


---

## Neo4j Findings

### Labels & Relationship Types

**Labels:** Keine gefunden
**Relationship Types:** Keine gefunden

### Label Counts

| Label | Count |
|-------|-------|

### Constraints & Indexes

**Constraints:** 0
**Indexes:** 0

### Run Nodes

**Labels mit run_id Property:** Keine gefunden

### Dangling Nodes (isolated, keine Relationships)

- ✅ Keine isolierten Nodes gefunden

### Duplicates (same run_id >1 per label)

- ✅ Keine Duplikate gefunden

### Errors

- ⚠️  Neo4j nicht verfügbar

---

## Cross-Store Consistency

**Postgres run_ids (distinct):** 0
**Neo4j run_ids (distinct):** 0
**Overlap:** 0

### Sample Join Checks

| run_id | Postgres | Neo4j |
|--------|----------|-------|

### Errors

- ⚠️  Postgres oder Neo4j nicht verfügbar

---

## Recommendations

### P0 (Datenverlust / fehlende Verknüpfungen / Duplicates)

- Postgres Duplikate beheben (siehe 'Duplicates' Abschnitt)

### P1 (Constraints/Indexes ergänzen)

- Postgres: Unique Constraints für (run_id, dimension) prüfen
- Neo4j: Constraints für run_id uniqueness pro Label prüfen
- Indexes für häufige Queries prüfen (run_id, example_id)

### P2 (Redundanz reduzieren, Archivierung)

- Alte Runs archivieren (beide Stores)
- Redundante Daten zwischen Postgres und Neo4j dokumentieren
- Cleanup-Strategie für isolierte Nodes definieren
