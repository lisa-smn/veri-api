# Explainability-Persistenz: Proof

**Datum:** 2026-01-17 18:01:55
**Run-ID:** proof-final

---

## Executive Summary

- **Postgres:** ✅ PASS
- **Neo4j:** ❌ FAIL
  - Fehler: Neo4j driver nicht verfügbar
- **Cross-Store Match:** ❌ FAIL

- **Overall:** ❌ FAIL

---

## Postgres Verification

**Gespeichert:** ✅

| Check | Status |
|-------|--------|
| run_exists | ✅ |
| explainability_report_exists | ✅ |
| run_id_in_config | ✅ |
| report_json_valid | ✅ |

---

## Evidence

### Postgres Queries

```sql
-- Run mit proof_run_id finden
SELECT r.id, r.config FROM runs r WHERE r.config->>'proof_run_id' = 'proof-final';

-- Explainability Report finden
SELECT er.id, er.version, er.created_at
FROM explainability_reports er
JOIN runs r ON er.run_id = r.id
WHERE r.config->>'proof_run_id' = 'proof-final';
```

### Neo4j Queries

```cypher
-- Run-Node finden
MATCH (r:Run) WHERE r.id = 'proof-final' OR r.run_id = 'proof-final' RETURN r;

-- Explainability-Node finden
MATCH (e:Explainability {run_id: 'proof-final'}) RETURN e;

-- Findings finden
MATCH (f:Finding {run_id: 'proof-final'}) RETURN count(f) as findings_count;
```

---

## Reproduktion

### Command

```bash
python scripts/prove_explainability_persistence.py
  --run-id proof-final
  --fixture minimal
  --out docs/status/explainability_persistence_proof.md
```

### Erwartete PASS-Kriterien

1. **Postgres:**
   - Run mit `proof_run_id` in `config` JSONB gespeichert
   - Explainability Report in `explainability_reports` Tabelle vorhanden
   - Report JSON ist gültig und enthält `findings`

2. **Neo4j:**
   - Run-Node mit `run_id` Property existiert
   - Explainability-Node mit `run_id` existiert
   - Mindestens 1 Finding-Node vorhanden
   - Mindestens 1 Span-Node vorhanden (falls Findings Spans haben)

3. **Cross-Store:**
   - Dieselbe `run_id` existiert in Postgres (via `config->>'proof_run_id'`)
   - Dieselbe `run_id` existiert in Neo4j (via `Run.run_id` Property)

### Environment Variables

**Postgres:**
- `POSTGRES_DSN` oder
- `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`

**Neo4j:**
- `NEO4J_URI` (z.B. `bolt://localhost:7687`)
- `NEO4J_USER` (default: `neo4j`)
- `NEO4J_PASSWORD`

**Hinweis:** Falls eine DB nicht verfügbar ist, verwende `--skip-postgres` oder `--skip-neo4j`.
