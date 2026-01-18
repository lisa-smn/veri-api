# Präsentations-Report: Veri-API System

**Zielgruppe:** Fachfremde Prüfer (Kolloquium, 12-15 Minuten)  
**Format:** Markdown mit klaren Überschriften und Belegen

---

## 1) Repo-Map

### Ordnerstruktur (relevante Teile)

```
veri-api/
├── app/
│   ├── api/routes.py              # FastAPI Endpoints
│   ├── services/
│   │   ├── agents/                 # 3 spezialisierte Agenten
│   │   │   ├── factuality/        # Faktenprüfung (Claim-basiert)
│   │   │   ├── coherence/         # Logik/Kohärenz
│   │   │   └── readability/       # Lesbarkeit
│   │   ├── explainability/        # Explainability-Report-Generator
│   │   └── verification_service.py # Orchestrator
│   ├── pipeline/verification_pipeline.py  # Agenten-Pipeline
│   ├── models/pydantic.py         # Request/Response-Modelle
│   └── db/
│       ├── postgres/              # SQL-Persistenz
│       └── neo4j/                 # Graph-Persistenz
├── scripts/
│   ├── run_m10_factuality.py      # Evaluation-Runner
│   └── test_evidence_gate_eval.py # Evidence-Gate-Tests
└── results/evaluation/             # Evaluations-Artefakte
```

### Top 15 wichtigste Dateien/Module

| Datei | Zweck |
|-------|-------|
| `app/api/routes.py` | FastAPI `/verify` Endpoint (Zeilen 19-33) |
| `app/services/verification_service.py` | Orchestrator: DB-Persistenz + Pipeline-Aufruf (Zeilen 33-72) |
| `app/pipeline/verification_pipeline.py` | Führt 3 Agenten aus, aggregiert Scores (Zeilen 41-60) |
| `app/services/agents/factuality/factuality_agent.py` | Factuality-Agent: Claim-Extraktion + Verifikation |
| `app/services/agents/factuality/claim_verifier.py` | Evidence-Gate: "incorrect nur mit Evidence" (Zeilen 312-362) |
| `app/services/agents/factuality/evidence_retriever.py` | Retriever: Sliding-Window-Passagen, Jaccard-Scoring |
| `app/services/explainability/explainability_service.py` | Baut Explainability-Report aus Agent-Ergebnissen (Zeilen 259-289) |
| `app/models/pydantic.py` | Pydantic-Modelle: VerifyRequest, VerifyResponse, AgentResult, IssueSpan |
| `app/db/postgres/persistence.py` | Speichert Runs, Results, Reports in Postgres |
| `app/db/neo4j/graph_persistence.py` | Schreibt Graph-Struktur in Neo4j (Zeilen 12-150) |
| `scripts/run_m10_factuality.py` | Evaluation-Runner: YAML-Config → Runs → Dokumentation |
| `scripts/test_evidence_gate_eval.py` | Evidence-Gate-Test: Coverage, Abstention, FP/FN-Analyse |
| `configs/m10_factuality_runs.yaml` | Run-Konfigurationen (Baseline, Tuned, Ablation) |
| `results/evaluation/summary.md` | Evaluations-Zusammenfassung (Metriken, Interpretation) |
| `results/evaluation/summary_matrix.csv` | Metriken-Matrix aller Runs (TP/FP/TN/FN, Recall, Precision, F1) |

---

## 2) Entry Point

### `/verify` Endpoint

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
            explainability=result.explainability,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### Request/Response Modelle

**Datei:** `app/models/pydantic.py`

**VerifyRequest** (Zeilen 50-58):
```python
class VerifyRequest(BaseModel):
    dataset: Optional[str] = None
    article_text: str
    summary_text: str
    llm_model: Optional[str] = None
    meta: Optional[Dict[str, str]] = None
```

**VerifyResponse** (Zeilen 74-84):
```python
class VerifyResponse(BaseModel):
    run_id: int
    overall_score: float
    factuality: AgentResult
    coherence: AgentResult
    readability: AgentResult
    explainability: Optional[ExplainabilityResult] = None
```

**AgentResult** (Zeilen 27-47):
```python
class AgentResult(BaseModel):
    name: str
    score: float
    explanation: str
    issue_spans: List[IssueSpan] = Field(default_factory=list)
    details: Optional[Dict[str, Any]] = None
```

**IssueSpan** (Zeilen 6-25):
```python
class IssueSpan(BaseModel):
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    message: str
    severity: Optional[Literal["low", "medium", "high"]] = None
    issue_type: Optional[str] = None
    confidence: Optional[float] = None
    evidence_found: Optional[bool] = None
    verdict: Optional[Literal["incorrect", "uncertain"]] = None
```

---

## 3) Methodik-Pipeline

### Orchestrierung

**Datei:** `app/services/verification_service.py` (Zeilen 33-72)

**Schrittfolge:**

1. **Input-Persistenz** (Zeilen 38-47):
   - Artikel + Summary in Postgres speichern
   - `store_article_and_summary()` → `(article_id, summary_id)`

2. **Pipeline-Ausführung** (Zeilen 50-54):
   - `self.pipeline.run(article, summary, meta)`
   - Gibt `PipelineResult` zurück

3. **Run-Persistenz** (Zeilen 57-67):
   - `store_verification_run()` → `run_id`
   - Speichert Scores, Agent-Results, Explainability

### Pipeline-Details

**Datei:** `app/pipeline/verification_pipeline.py` (Zeilen 41-60)

```python
def run(self, article: str, summary: str, meta: dict | None = None) -> PipelineResult:
    # 1. Agenten ausführen
    factuality = self.factuality_agent.run(article, summary, meta)
    coherence = self.coherence_agent.run(article, summary, meta)
    readability = self.readability_agent.run(article, summary, meta)
    
    # 2. Gesamt-Score aggregieren
    overall = (factuality.score + coherence.score + readability.score) / 3.0
    
    # 3. PipelineResult bauen
    result = PipelineResult(
        factuality=factuality,
        coherence=coherence,
        readability=readability,
        overall_score=overall,
        explainability=None,
    )
    
    # 4. Explainability deterministisch generieren
    result.explainability = self.explainability_service.build(result, summary_text=summary)
    
    return result
```

**Schrittfolge: Input → Agenten → Aggregation → Explainability → Persistenz**

1. **Input:** `article_text`, `summary_text`
2. **Agenten:** Factuality, Coherence, Readability (parallel ausführbar)
3. **Aggregation:** Durchschnitt der 3 Scores → `overall_score`
4. **Explainability:** `ExplainabilityService.build()` → strukturierter Report
5. **Persistenz:** Postgres (Runs, Results) + Neo4j (Graph)

---

## 4) USP 1: Evidence-Gate / Entscheidung

### Kernidee

**"incorrect" wird nur vergeben, wenn belastbare Evidence (wörtliches Zitat) gefunden wurde.**

**Datei:** `app/services/agents/factuality/claim_verifier.py` (Zeilen 312-362)

### Gate-Logik

```python
def _apply_gate(self, out: VerifierLLMOutput, selection: EvidenceSelection, ...) -> GateDecision:
    label_raw = out.label
    label_final = label_raw
    gate_reason = "ok"
    
    # Gate 1: correct ohne Evidence => uncertain
    if (label_raw == "correct" and self.require_evidence_for_correct 
        and not selection.evidence_found):
        label_final = "uncertain"
        conf = min(conf, 0.55)
        gate_reason = "no_evidence"
    
    # Gate 2: incorrect ohne Evidence => uncertain
    elif label_raw == "incorrect" and not selection.evidence_found:
        label_final = "uncertain"
        conf = min(conf, 0.5)
        gate_reason = "no_evidence"
    
    # Gate 3: incorrect mit Evidence aber Coverage-Fail => Confidence clamp
    elif label_raw == "incorrect" and selection.evidence_found and not coverage_ok:
        conf = min(conf, 0.5)
        gate_reason = "coverage_fail"
    
    return GateDecision(...)
```

### Entscheidungsparameter

**Datei:** `app/services/agents/factuality/factuality_agent.py`

- **`decision_mode`:** `"issues"` (Issue-basiert) oder `"score"` (Score-basiert)
- **`issue_threshold`:** Mindestanzahl Issues für `pred_has_error` (z.B. `1`)
- **`severity_min`:** Mindest-Severity (`"low"`, `"medium"`, `"high"`)
- **`uncertainty_policy`:** `"non_error"`, `"weight_0.5"`, `"count_as_error"`
- **`score_cutoff`:** Schwellwert für Score-Mode (optional)

**Beleg:** Evaluations-Configs in `configs/m10_factuality_runs.yaml` zeigen verschiedene Kombinationen.

### Invarianten

1. `selected_evidence_index == -1` → `evidence_quote = None` → `evidence_found = False`
2. `selected_evidence_index >= 0` → `evidence_quote` muss nicht-leer sein UND in Passage vorkommen → `evidence_found = True`
3. `label in {"correct", "incorrect"}` nur wenn `evidence_found == True`
4. Coverage-Fail führt nicht zu Label-Downgrade, sondern nur zu Confidence-Clamp

---

## 5) USP 2: Explainability

### Explainability-Report-Generator

**Datei:** `app/services/explainability/explainability_service.py` (Zeilen 259-289)

### Struktur

**Datei:** `app/services/explainability/explainability_models.py` (Zeilen 79-85)

```python
class ExplainabilityResult(BaseModel):
    summary: List[str]                    # Executive Summary (3-6 Sätze)
    findings: List[Finding]               # Alle Findings (normalisiert, dedupliziert, sortiert)
    by_dimension: Dict[Dimension, List[Finding]]  # Gruppiert nach factuality/coherence/readability
    top_spans: List[TopSpan]               # Top-K wichtigste Textstellen
    stats: ExplainabilityStats             # Anzahl Findings, Coverage
    version: str = "m9_v1"                 # Version für Vergleichbarkeit
```

### Finding-Struktur

**Datei:** `app/services/explainability/explainability_models.py` (Zeilen 51-59)

```python
class Finding(BaseModel):
    id: str
    dimension: Dimension                   # factuality/coherence/readability
    severity: Severity                     # low/medium/high
    message: str
    span: Optional[Span]                 # Textstelle (start_char, end_char, text)
    evidence: List[EvidenceItem]          # Evidence-Quotes, Claims, Raw-Daten
    recommendation: Optional[str]         # Handlungsempfehlung
    source: Dict[str, Any]                # Provenance (agent, issue_type, claim_id)
```

### Verarbeitungsschritte

1. **Normalisieren:** Agent-Outputs → einheitliches Finding-Format
2. **Deduplizieren & Clustern:** Überlappende Findings zusammenführen
3. **Ranking:** `score = severity_weight * dimension_weight * log(span_length)`
4. **Top-Spans:** Top-K wichtigste Textstellen
5. **Executive Summary:** Regelbasiert aus Findings abgeleitet

**Beleg:** `app/services/explainability/explainability_service.py` (Zeilen 259-289)

---

## 6) USP 3: Persistenz/Traceability

### Postgres: Modelle/Tables

**Datei:** `app/db/postgres/persistence.py` (Zeilen 34-166)

**Tables (via SQL):**

- **`datasets`:** Datensatz-Metadaten (`id`, `name`)
- **`articles`:** Artikel (`id`, `dataset_id`, `text`)
- **`summaries`:** Summaries (`id`, `article_id`, `source`, `text`, `llm_model`)
- **`verification_runs`:** Runs (`id`, `article_id`, `summary_id`, `overall_score`, `factuality_score`, `coherence_score`, `readability_score`, `created_at`)
- **`verification_results`:** Agent-Results als JSON (`run_id`, `agent_name`, `result_json`)
- **`explainability_reports`:** Explainability-Reports als JSON (`run_id`, `report_json`)

**Beleg:** `app/db/postgres/persistence.py` (Zeilen 68-166)

### Neo4j: Nodes/Edges

**Datei:** `app/db/neo4j/graph_persistence.py` (Zeilen 12-150)

**Graph-Struktur:**

```
(Article {id}) -[:HAS_SUMMARY]-> (Summary {id})
(Summary) -[:HAS_METRIC]-> (Metric {run_id, summary_id, dimension, score})
(Metric) -[:HAS_ISSUE_SPAN]-> (IssueSpan:Error {run_id, summary_id, dimension, span_index, message, severity, start_char, end_char})
(Run {id}) -[:EVALUATES]-> (Summary)
```

**Beleg:** `app/db/neo4j/graph_persistence.py` (Zeilen 51-150)

**Schreibstelle:** `app/db/postgres/persistence.py` (Zeilen 166-266) ruft `write_verification_graph()` auf.

---

## 7) Bestes Demo-Beispiel

### Request (gekürzt)

```json
{
  "dataset": "frank",
  "article_text": "The FBI has offered to help in a murder case in Arkansas...",
  "summary_text": "the fbi has said it will help the san bernardino killer to access iphones used by the san bernardino victims.",
  "llm_model": "gpt-4o-mini"
}
```

### Response (gekürzt)

```json
{
  "run_id": 123,
  "overall_score": 0.5,
  "factuality": {
    "name": "factuality",
    "score": 0.5,
    "explanation": "1 uncertain claim detected",
    "issue_spans": [
      {
        "start_char": 0,
        "end_char": 108,
        "message": "Satz 1: Claim 'the fbi has said it will help the san bernardino killer...' – Nicht sicher verifizierbar (Quelle zu vage/fehlend). Kein belastbares Evidence-Zitat für einen Widerspruch gefunden; daher 'uncertain'.",
        "severity": "low",
        "issue_type": "OTHER",
        "confidence": 0.5,
        "evidence_found": false,
        "verdict": "uncertain"
      }
    ],
    "details": {
      "num_claims": 1,
      "num_incorrect": 0,
      "num_uncertain": 1
    }
  },
  "coherence": {...},
  "readability": {...},
  "explainability": {
    "summary": [
      "In der Summary wurde 1 Finding identifiziert (0 high, 0 medium, 1 low).",
      "Der Schwerpunkt liegt bei **factuality** (meiste Findings in dieser Dimension)."
    ],
    "findings": [...],
    "top_spans": [...],
    "stats": {
      "num_findings": 1,
      "num_high_severity": 0,
      "num_medium_severity": 0,
      "num_low_severity": 1,
      "coverage_chars": 108,
      "coverage_ratio": 0.42
    }
  }
}
```

### Verweis: Wo werden die Felder erzeugt?

- **`issue_spans`:** `app/services/agents/factuality/factuality_agent.py` → `_build_issue_spans_from_claims()` (Zeilen 246-252)
- **`evidence_found`:** `app/services/agents/factuality/claim_verifier.py` → `_validate_evidence()` (Zeilen 221-294)
- **`verdict`:** `app/services/agents/factuality/factuality_agent.py` → `_build_issue_spans_from_claims()` (setzt `verdict` basierend auf `Claim.label`)
- **`explainability`:** `app/services/explainability/explainability_service.py` → `build()` (Zeilen 259-289)

**Beleg:** Real Example aus `results/evaluation/runs/results/factuality_combined_final_v1_examples.jsonl` (Zeile 1)

---

## 8) Evaluation

### Eval-Skripte

**Datei:** `scripts/run_m10_factuality.py` (Zeilen 1-780)

- **Input:** YAML-Config (`configs/m10_factuality_runs.yaml`)
- **Output:** Pro Run: `results_*.json`, `examples_*.jsonl`, `docs/<run_id>.md`

**Datei:** `scripts/test_evidence_gate_eval.py` (Zeilen 1-676)

- **Zweck:** Evidence-Gate-Test (Coverage, Abstention, FP/FN-Analyse)
- **Output:** `results_*.json`, `debug_claims.jsonl`, `error_cases.jsonl`

### Datensätze

1. **FRANK** (`data/frank/frank_clean.jsonl`): 300 Examples (Dev/Calibration)
2. **FineSumFact** (`data/finesumfact/human_label_test_clean.jsonl`): 200 Examples (Test)
3. **Combined:** 500 Examples (FRANK + FineSumFact)

### Metriken

**Datei:** `app/services/analysis/metrics.py`

- **Binary Classification:** TP, FP, TN, FN, Accuracy, Precision, Recall, F1, Specificity, Balanced Accuracy, AUROC, MCC
- **Coverage/Abstention:** `coverage = (num_correct + num_incorrect) / total`, `abstention_rate = num_uncertain / total`

**Beleg:** `results/evaluation/summary_matrix.csv` (Zeilen 1-22)

### Pfade zu Ergebnissen

- **Summary:** `results/evaluation/summary.md`
- **Metriken-Matrix:** `results/evaluation/summary_matrix.csv`
- **Run-Dokumentation:** `results/evaluation/runs/docs/<run_id>.md`
- **Example-Level:** `results/evaluation/runs/results/<run_id>_examples.jsonl`
- **Debug-Artefakte:** `results/evaluation/evidence_gate_test/debug_claims.jsonl`, `error_cases.jsonl`

### 3-5 wichtigste Befunde

**Beleg:** `results/evaluation/summary.md` (Zeilen 31-65)

1. **Evidence-Gate reduziert False Positives:**
   - **FRANK Baseline:** Specificity 0.057 (viele FP)
   - **Nach Evidence-Gate:** Specificity steigt (bei `uncertainty_policy=non_error`)
   - **Invariante:** 0 "incorrect" Claims ohne `evidence_found`

2. **Recall vs Specificity Trade-off:**
   - **FRANK Tuned:** Recall 0.958, Specificity 0.057
   - **System priorisiert Recall** (findet mehr Fehler, aber auch mehr False Positives)

3. **Balanced Accuracy als Optimierungsmetrik:**
   - **FRANK Baseline:** Balanced Acc 0.508
   - **FineSumFact Final:** Balanced Acc 0.523 (+0.015 Generalisierung)
   - **Optimierungsziel:** Balanced Accuracy (wegen unbalancierter Klassen)

4. **Coverage/Abstention:**
   - **Coverage:** ~0.80 (80% der Claims werden als correct/incorrect klassifiziert)
   - **Abstention:** ~0.20 (20% bleiben "uncertain" ohne Evidence)

5. **Ablation-Effekt:**
   - **Claim-Extraktion hat geringen Einfluss:** Balanced Acc-Drop durch Ablation = 0.000
   - **Schwacher Effekt:** Claim-Extraktion ist nicht kritisch für Performance

---

## Zusammenfassung

**Kern-USPs:**

1. **Evidence-Gate:** "incorrect" nur mit belastbarer Evidence → reduziert False Positives
2. **Explainability:** Strukturierter Report mit Findings, Top-Spans, Empfehlungen
3. **Persistenz:** Postgres (Runs, Results) + Neo4j (Graph) für Traceability

**Evaluation:**

- **FRANK (Dev):** Balanced Acc 0.508, Recall 0.958, Specificity 0.057
- **FineSumFact (Test):** Balanced Acc 0.523 (+0.015 Generalisierung)
- **Evidence-Gate:** Invariante eingehalten (0 incorrect ohne Evidence)

**Artefakte:**

- `results/evaluation/summary.md` (Interpretation)
- `results/evaluation/summary_matrix.csv` (Metriken-Matrix)
- `results/evaluation/runs/docs/` (Run-Dokumentation)






