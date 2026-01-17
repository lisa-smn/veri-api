# Architecture Overview

**Datum:** 2026-01-16  
**Ziel:** High-level Übersicht für Betreuerinnen (10-Minuten-Verständnis)

---

## 1. System-Übersicht

`veri-api` ist ein Verifikationssystem für automatisch generierte Zusammenfassungen (Summaries). Es prüft drei Dimensionen:

- **Factuality:** Enthält die Summary faktische Fehler?
- **Coherence:** Ist die Summary kohärent und gut strukturiert?
- **Readability:** Ist die Summary gut lesbar?

**Output:** Scores (0-1) pro Dimension + Explainability (Findings mit Spans und Severity).

---

## 2. Pipeline (Textform)

```
Article (Input)
    ↓
┌─────────────────────────────────────┐
│  Verification Pipeline              │
│                                     │
│  1. Factuality Agent               │
│     - Claim Extraction (LLM)       │
│     - Evidence Retrieval (Sliding) │
│     - Claim Verification (LLM)      │
│     → IssueSpans + Score            │
│                                     │
│  2. Coherence Agent                 │
│     - Sentence-level Analysis      │
│     - Structure Scoring            │
│     → Score                         │
│                                     │
│  3. Readability Agent               │
│     - Sentence Complexity          │
│     - Normalization (1-5 → 0-1)    │
│     → Score                         │
│                                     │
│  4. Explainability Module           │
│     - Aggregiert Agent-Outputs     │
│     - Generiert Findings           │
│     - Top-Spans Selection          │
│     → ExplainabilityResult          │
└─────────────────────────────────────┘
    ↓
PipelineResult (Scores + Explainability)
    ↓
Optional: Persistence (Postgres + Neo4j)
```

---

## 3. Agents: Inputs/Outputs

### Factuality Agent

**Input:**
- `article_text: str` (Quelltext)
- `summary_text: str` (zu prüfende Summary)

**Prozess:**
1. **Claim Extraction:** LLM extrahiert Claims aus Summary-Sätzen
2. **Evidence Retrieval:** Sliding-Window-Suche im Article (Boosting für Zahlen/Daten)
3. **Claim Verification:** LLM prüft Claim gegen Evidence (mit Evidence-Gate: "uncertain" → "non_error")

**Output:**
- `score: float` (0-1, normalisiert aus Issue-Counts)
- `issue_spans: List[IssueSpan]` (Positionen + Verdicts)

**Normalisierung:**
- Issue-Counts → Score (0 = keine Issues, 1 = viele Issues)
- Evidence-Gate: "uncertain" Claims werden als "non_error" behandelt

### Coherence Agent

**Input:**
- `summary_text: str`

**Prozess:**
1. Sentence-level Analysis (Struktur, Übergänge)
2. Scoring (kontinuierlich, 0-1)

**Output:**
- `score: float` (0-1)

**Normalisierung:**
- Direkt 0-1 (keine weitere Normalisierung nötig)

### Readability Agent

**Input:**
- `summary_text: str`

**Prozess:**
1. Sentence Complexity Analysis
2. Scoring (ursprünglich 1-5, normalisiert zu 0-1)

**Output:**
- `score: float` (0-1)

**Normalisierung:**
- 1-5 Skala → 0-1 (1 → 0.0, 5 → 1.0, linear)

---

## 4. Explainability

**Input:**
- Agent-Outputs (Factuality: IssueSpans, Coherence/Readability: Scores)

**Prozess:**
1. **Aggregation:** IssueSpans → Findings (dedupliziert, gemerged)
2. **Top-Spans Selection:** Wichtigste Spans pro Dimension
3. **Severity Assignment:** `low`, `medium`, `high` basierend auf Agent-Scores

**Output:**
- `ExplainabilityResult`:
  - `findings: List[Finding]` (Dimension, Severity, Spans, Evidence)
  - `summary_highlights: List[Span]` (für UI-Rendering)

**Traceability:**
- Jedes Finding referenziert originale Agent-Outputs
- Spans sind exakt auf Summary-Text gemappt

---

## 5. Persistence

### Postgres Schema (High-Level)

**Tabellen:**
- `runs`: Run-Metadaten (`run_id`, `config`, `status`, `timestamp`)
- `articles`: Artikel-Text (`id`, `text`, `dataset`)
- `summaries`: Summary-Text (`id`, `text`, `article_id`)
- `explainability_reports`: Explainability-Output (`run_id`, `report_json`)

**Relationships:**
- `runs` → `articles`, `summaries` (via IDs)
- `runs` → `explainability_reports` (1:1)

### Neo4j Schema (High-Level)

**Labels:**
- `:Run` (`run_id`, `timestamp`, `config`)
- `:Example` (`example_id`, `run_id`, `article_text`, `summary_text`)
- `:Explainability` (`run_id`, `version`)
- `:Finding` (`dimension`, `severity`, `run_id`)
- `:Span` (`start_i`, `end_i`, `text`, `run_id`)

**Relationships:**
- `(:Run)-[:HAS_EXAMPLE]->(:Example)`
- `(:Run)-[:HAS_EXPLAINABILITY]->(:Explainability)`
- `(:Explainability)-[:HAS_FINDING]->(:Finding)`
- `(:Finding)-[:HAS_SPAN]->(:Span)`

**Cross-Store Consistency:**
- `run_id` ist identisch in Postgres und Neo4j
- Constraint: `(:Run {run_id})` ist UNIQUE

---

## 6. API & UI

### FastAPI (`/verify` Endpoint)

**Request:**
```json
{
  "article": "...",
  "summary": "...",
  "enable_explainability": true,
  "persist_to_db": false,
  "run_llm_judge": false
}
```

**Response:**
```json
{
  "overall_score": 0.85,
  "factuality": {"score": 0.9, "issue_count": 1},
  "coherence": {"score": 0.8},
  "readability": {"score": 0.85},
  "explainability": {
    "findings": [...],
    "summary_highlights": [...]
  },
  "judge": {"error_present": false, "confidence": 0.95}  // optional
}
```

### Streamlit UI

**Tabs:**
1. **Verify:** Article + Summary Input, Scores + Highlights
2. **Runs:** Liste aller Runs aus Postgres
3. **Status:** Markdown-Viewer für Status-Dokumente

**Features:**
- Dataset-Loading (FRANK, SummEval)
- Error Injection (Demo)
- Comparison Panel (Agent vs Judge vs Baselines)

---

## 7. Evaluation

### Datasets

- **FRANK:** Factuality (Binary: `has_error` true/false)
- **SummEval:** Coherence + Readability (Kontinuierlich: 1-5, normalisiert zu 0-1)

### Metrics

**Factuality:**
- Precision, Recall, F1, Balanced Accuracy, MCC, AUROC (wenn confidence vorhanden)

**Coherence/Readability:**
- Pearson r, Spearman ρ, MAE, RMSE, R² (vs. Human Ratings)

**Baselines:**
- **Classical:** Flesch Reading Ease, Flesch-Kincaid, Gunning Fog
- **LLM-as-a-Judge:** GPT-4o-mini als Baseline (optional, `ENABLE_LLM_JUDGE=true`)

### Evaluation Scripts

- `eval_sumeval_readability.py`: Readability Agent + Judge
- `eval_sumeval_coherence.py`: Coherence Agent
- `eval_frank_factuality_llm_judge.py`: Factuality Judge Baseline
- `eval_sumeval_baselines.py`: Classical Baselines

**Output:**
- `results/evaluation/<dimension>/<run_id>/`
  - `predictions.jsonl`
  - `summary.md` (Metrics + CIs)
  - `summary.json` (Structured)

---

## 8. Key Design Decisions

1. **Evidence-Gate:** "uncertain" Claims → "non_error" (verhindert False Positives)
2. **Normalization:** Alle Scores auf 0-1 (konsistent für UI + Evaluation)
3. **Explainability:** Aggregiert Agent-Outputs, keine zusätzlichen LLM-Calls
4. **Persistence:** Optional (nur wenn DB verfügbar)
5. **Judge:** Optional (nur wenn `ENABLE_LLM_JUDGE=true`)

---

## 9. Entry Points

**API:**
- `app/api/routes.py` → `VerificationService` → `VerificationPipeline`

**CLI (Scripts):**
- `scripts/eval_*.py`: Evaluation Scripts
- `scripts/audit_*.py`: Audit Scripts
- `scripts/verify_quality_*.py`: Quality Checks

**UI:**
- `ui/app.py`: Streamlit Dashboard

---

**Weitere Details:**
- **Status Docs:** `docs/status/*.md`
- **Tests:** `tests/readability/`, `tests/coherence/`, `tests/explainability/`
- **Status Pack:** `docs/status_pack/2026-01-08/`

