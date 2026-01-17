# System-Overview – Verifikationssystem (veri-api)

**Zweck:** Kurze technische Übersicht über Architektur, Agenten und technische Raffinessen.

---

## Architektur (vereinfacht)

```
API (/verify)
  ↓
VerificationService
  ↓
VerificationPipeline
  ↓
  ├─ FactualityAgent → AgentResult (score + issue_spans)
  ├─ CoherenceAgent → AgentResult (score + issue_spans)
  └─ ReadabilityAgent → AgentResult (score + issue_spans)
  ↓
ExplainabilityService → ExplainabilityResult (summary + findings + top_spans + stats)
  ↓
Persistence (Postgres/Neo4j)
```

**Technologie-Stack:**
- FastAPI (REST-API)
- OpenAI GPT-4o-mini (LLM für Agenten)
- PostgreSQL (Persistenz)
- Neo4j (optional, Graph-Analytics)
- Pydantic (Datenmodelle)

---

## Agenten (3 Dimensionen)

### 1. Factuality-Agent
- **Zweck:** Prüft, ob Claims in der Summary durch den Artikel belegt sind.
- **Output:** Score [0,1] + `issue_spans` mit `issue_type` (NUMBER, DATE, ENTITY, OTHER) + Evidence-Quotes.
- **Besonderheit:** Evidence-Retrieval aus dem Artikel, Claim-Extraction, Verdict (correct/incorrect/uncertain).

### 2. Coherence-Agent
- **Zweck:** Prüft logischen Fluss, Widersprüche, Referenzklarheit.
- **Output:** Score [0,1] + `issue_spans` mit `issue_type` (CONTRADICTION, MISSING_LINK, JUMPY_ORDER, UNCLEAR_REFERENCE).
- **Besonderheit:** Analysiert Sätze und Übergänge, keine externe Evidence nötig.

### 3. Readability-Agent
- **Zweck:** Prüft Lesbarkeit (Satzlänge, Komplexität, Struktur).
- **Output:** Score [0,1] + `issue_spans` (optional).
- **Status:** Implementiert, aber noch nicht final evaluiert.

---

## Technische Raffinessen

1. **Deterministische Explainability-Struktur:**
   - Findings werden normalisiert, dedupliziert und gerankt (stabile IDs via Hash).
   - Versionierung (`m9_v1`) für Vergleichbarkeit über Zeit.

2. **Reproduzierbare Evaluation:**
   - Seed-basierte Runs (z.B. `seed=42`).
   - Run-IDs enthalten Timestamp, Modell, Prompt-Version, Seed.
   - Bootstrap-CIs (n=2000) für alle Metriken.

3. **Manifest-basierte Fairness (FRANK):**
   - `frank_subset_manifest.jsonl` definiert exakte Subset für Agent und Baselines.
   - `dataset_signature` (SHA256) verifiziert Konsistenz zwischen Runs.

4. **Dependency-Guards (Baselines):**
   - Fail-fast bei fehlenden Paketen (`rouge-score`, `bert-score`).
   - `dependencies_ok` in `run_metadata.json` markiert ungültige Runs.

5. **Stress-Tests (Coherence):**
   - Shuffle-Test: Permutiert Sätze und misst Score-Delta.
   - Injection-Test: Injiziert inkohärente Sätze und misst Reaktion.

6. **Aggregatoren:**
   - `aggregate_coherence_runs.py` und `aggregate_factuality_runs.py` erstellen Vergleichsmatrizen (CSV + MD).
   - Erkennt automatisch Agent/Baseline/Judge-Runs und warnt bei inkonsistenten `dataset_signature`s.

---

## Datenmodelle (Pydantic)

- **`AgentResult`:** `score`, `explanation`, `issue_spans[]`, `details{}`
- **`ErrorSpan`:** `start_char`, `end_char`, `message`, `severity`, `issue_type`
- **`ExplainabilityResult`:** `summary[]`, `findings[]`, `by_dimension{}`, `top_spans[]`, `stats{}`, `version`
- **`Finding`:** `id`, `dimension`, `severity`, `message`, `span`, `evidence[]`, `recommendation`

---

## Evaluation-Infrastruktur

- **Scripts:** `scripts/eval_*.py` (Coherence, Factuality, Baselines, Judge, Stress-Tests)
- **Artefakte:** `results/evaluation/<dimension>/<run_id>/` (summary.json, summary.md, predictions.jsonl, run_metadata.json)
- **Aggregation:** `scripts/aggregate_*.py` → `results/evaluation/summary_*_matrix.csv/.md`

---

**Details zu Explainability:** Siehe `02_explainability_audit.md`.  
**Details zu Evaluation:** Siehe `03_evaluation_results.md`.

