# Coherence-Agent: Capability Matrix

**Datum:** 2026-01-03  
**Zweck:** Systematische Dokumentation der aktuellen Coherence-Agent-Fähigkeiten für Evaluationsplanung

---

## A) Was kann Coherence heute?

### ✅ Implementiert

- **LLM-basierte Kohärenzbewertung:** Nutzt `LLMCoherenceEvaluator` mit strukturiertem JSON-Output
- **Score-Berechnung:** Kontinuierlicher Score im Bereich [0, 1] (direkt vom LLM)
- **Issue-Erkennung:** 5 Issue-Typen (LOGICAL_INCONSISTENCY, CONTRADICTION, REDUNDANCY, ORDERING, OTHER)
- **Severity-Klassifikation:** low, medium, high (vom LLM bestimmt)
- **Span-Mapping:** Automatisches Mapping von `summary_span` auf Character-Offsets im Summary
- **Pipeline-Integration:** Vollständig integriert in `VerificationPipeline`
- **Postgres-Persistenz:** Speichert in `verification_results` Tabelle (dimension='coherence')
- **Neo4j-Traceability:** Schreibt Metric- und IssueSpan-Nodes in Graph
- **Explainability-Integration:** Fließt in `ExplainabilityService` ein (Dimension.coherence)
- **Evaluation-Script:** `scripts/eval_sumeval_coherence.py` existiert bereits
- **Unit-Tests:** `tests/unit/test_coherence_agent.py` vorhanden

### ❌ Nicht implementiert (im Vergleich zu Factuality)

- **Keine Decision Logic Parameter:** Kein `issue_threshold`, `score_cutoff`, `decision_mode`, `severity_min`, `ignore_issue_types`
- **Keine gewichtete Aggregation:** Keine `severity_weights`, `confidence_weights`, `decision_threshold_float`
- **Keine Evidence-Gate:** Keine `evidence_found`, `evidence_quote` Felder
- **Keine Confidence-Felder:** Kein `confidence`, `mapping_confidence` in IssueSpan
- **Keine Verdict-Trennung:** Kein `verdict` Feld (incorrect/uncertain) - nur `severity`
- **Keine Ablation-Flags:** Kein `use_claim_extraction`, `use_claim_verification`, `use_spans`

---

## B) Issue-Typen + Definition + Erkennung

**Datei:** `app/services/agents/coherence/coherence_models.py:7-13`

| Issue-Type | Definition | Erkennung |
|------------|------------|-----------|
| **LOGICAL_INCONSISTENCY** | Interne logische Widersprüche innerhalb der Summary (z.B. "X ist groß" und "X ist klein" im selben Kontext) | LLM-basiert (Prompt: "interne logische Konsistenz") |
| **CONTRADICTION** | Direkte Widersprüche zwischen Aussagen (stärker als INCONSISTENCY) | LLM-basiert (Prompt: "keine Selbstwidersprüche") |
| **REDUNDANCY** | Unnötige Wiederholungen von Informationen | LLM-basiert (Prompt: "keine unnötigen Wiederholungen") |
| **ORDERING** | Probleme mit der Reihenfolge/Informationsfluss (abrupte Sprünge ohne Übergang) | LLM-basiert (Prompt: "nachvollziehbarer Informationsfluss / Reihenfolge") |
| **OTHER** | Sonstige Kohärenzprobleme (z.B. unklare Referenzen, Pronomen ohne Bezug) | LLM-basiert (Fallback für nicht kategorisierbare Probleme) |

**Erkennungsmethode:** Alle Issue-Typen werden **vollständig LLM-basiert** erkannt. Keine regelbasierten Pre-Filter oder Post-Processing-Regeln.

**Beleg:** `app/services/agents/coherence/coherence_verifier.py:107-150` - Prompt fordert explizit JSON mit `type` Feld.

---

## C) Output-Felder + Beispiele

### AgentResult-Struktur

**Datei:** `app/models/pydantic.py:27-47`

```python
class AgentResult(BaseModel):
    name: str                    # "coherence"
    score: float                  # 0.0..1.0 (1 = sehr kohärent)
    explanation: str              # Globale Erklärung (1-2 Sätze)
    issue_spans: List[IssueSpan]  # Lokalisierte Probleme
    details: Dict[str, Any]       # {"issues": [...], "num_issues": int}
```

### IssueSpan-Struktur (für Coherence)

**Datei:** `app/models/pydantic.py:6-25`

```python
class IssueSpan(BaseModel):
    start_char: Optional[int]     # Character-Offset Start (oder None)
    end_char: Optional[int]       # Character-Offset Ende (oder None)
    message: str                  # Format: "[severity] type in 'span' – comment"
    severity: Optional[Literal["low", "medium", "high"]]  # Vom LLM bestimmt
    issue_type: Optional[str]     # Für Coherence: None (nicht gesetzt)
    # Felder, die Coherence NICHT nutzt:
    # confidence: None
    # mapping_confidence: None
    # evidence_found: None
    # verdict: None
```

**Beleg:** `app/services/agents/coherence/coherence_agent.py:76-104` - `_build_error_spans()` setzt nur `start_char`, `end_char`, `message`, `severity`.

### CoherenceIssue (intern)

**Datei:** `app/services/agents/coherence/coherence_models.py:5-19`

```python
@dataclass
class CoherenceIssue:
    type: Literal["LOGICAL_INCONSISTENCY", "CONTRADICTION", "REDUNDANCY", "ORDERING", "OTHER"]
    severity: Literal["low", "medium", "high"]
    summary_span: str      # Wörtlicher Textauszug aus Summary
    comment: str           # Kurze Beschreibung des Problems
    hint: Optional[str]    # Optional: Reparaturidee
```

**Beleg:** `app/services/agents/coherence/coherence_agent.py:48-51` - `details["issues"]` enthält `asdict(CoherenceIssue)`.

### Beispiel-Output (aus Test)

**Datei:** `tests/unit/test_coherence_agent.py:21-36`

```json
{
  "name": "coherence",
  "score": 0.2,
  "explanation": "Kurz erklärt.",
  "issue_spans": [
    {
      "start_char": 0,
      "end_char": 13,
      "message": "[high] CONTRADICTION in 'This is bad' – Widerspruch im Satz.",
      "severity": "high",
      "issue_type": null,
      "confidence": null,
      "mapping_confidence": null,
      "evidence_found": null,
      "verdict": null
    }
  ],
  "details": {
    "issues": [
      {
        "type": "CONTRADICTION",
        "severity": "high",
        "summary_span": "This is bad",
        "comment": "Widerspruch im Satz.",
        "hint": null
      }
    ],
    "num_issues": 1
  }
}
```

**Beleg:** `app/services/agents/coherence/coherence_agent.py:53-59` - Return-Statement zeigt exakte Struktur.

---

## D) Konfig-Parameter + Defaultwerte + Auswirkungen

### LLM-Parameter

**Datei:** `app/llm/openai_client.py:14-22`

| Parameter | Wert | Auswirkung |
|-----------|------|------------|
| `temperature` | `0.0` (hardcoded) | Deterministische LLM-Ausgaben (keine Randomness) |
| `max_tokens` | `800` (hardcoded) | Begrenzt LLM-Output-Länge |
| `model_name` | Konfigurierbar (Default: `"gpt-4o-mini"`) | LLM-Modell für Evaluation |

**Beleg:** `app/llm/openai_client.py:18-19` - `temperature=0.0` ist hardcoded.

### Evaluator-Parameter (intern)

**Datei:** `app/services/agents/coherence/coherence_verifier.py:34-35`

| Parameter | Wert | Auswirkung |
|-----------|------|------------|
| `ISSUE_REQUIRED_BELOW` | `0.7` (hardcoded) | Wenn Score < 0.7, wird Fallback-Issue erzwungen |
| `MAX_ISSUES` | `8` (hardcoded) | Maximale Anzahl Issues pro Summary |

**Beleg:** `app/services/agents/coherence/coherence_verifier.py:84-95` - Safety-Net: Wenn Score < 0.7 und keine Issues, wird Fallback-Issue erzeugt.

### Prompt-Version

**Datei:** `scripts/eval_sumeval_coherence.py:188` (Parameter `prompt_version`)

| Parameter | Default | Auswirkung |
|-----------|---------|------------|
| `prompt_version` | `"v1"` (in Config) | Wird für Cache-Key verwendet, aber aktuell nur eine Prompt-Version vorhanden |

**Beleg:** `scripts/eval_sumeval_coherence.py:137-144` - Cache-Key enthält `prompt_version`, aber Prompt selbst ist nicht versioniert.

### ❌ Fehlende Parameter (im Vergleich zu Factuality)

- **Kein `issue_threshold`:** Keine binäre Entscheidung basierend auf Anzahl Issues
- **Kein `score_cutoff`:** Keine binäre Entscheidung basierend auf Score
- **Kein `decision_mode`:** Keine Wahl zwischen "issues", "score", "either", "both"
- **Kein `severity_min`:** Keine Filterung nach minimaler Severity
- **Kein `ignore_issue_types`:** Keine Blocklist für Issue-Typen
- **Kein `allow_issue_types`:** Keine Allowlist für Issue-Typen
- **Kein `error_threshold`:** Keine gewichtete Aggregation
- **Kein `decision_threshold_float`:** Keine gewichtete Decision Logic
- **Kein `severity_weights`:** Keine Gewichtung nach Severity
- **Kein `uncertainty_policy`:** Keine Uncertainty-Behandlung (Coherence hat kein "uncertain")

**Beleg:** Suche nach `issue_threshold|score_cutoff|decision_mode|severity_min` in `app/services/agents/coherence/` → **Keine Treffer**.

---

## E) Was fehlt für eine echte Evaluation gegen SummEval

### ✅ Bereits vorhanden

- **Evaluation-Script:** `scripts/eval_sumeval_coherence.py` existiert
- **Metriken:** Pearson r, Spearman rho, MAE (bereits implementiert)
- **Normalisierung:** GT-Skala (1-5) → [0,1] (bereits implementiert)
- **Cache-System:** JSONL-basiertes Caching (bereits implementiert)
- **Datensatz:** `data/sumeval/sumeval_clean.jsonl` vorhanden

### ❌ Fehlend für vollständige Evaluation

#### 1) Mapping/Labeling

**Problem:** SummEval liefert kontinuierliche Scores (1-5), aber keine Issue-Level-Labels.

**Was fehlt:**
- Keine Gold-Labels für Issue-Typen (LOGICAL_INCONSISTENCY, CONTRADICTION, etc.)
- Keine Gold-Labels für Issue-Positionen (Spans)
- Keine Gold-Labels für Severity (low/medium/high)

**Auswirkung:** Nur Score-Korrelation möglich, keine Issue-Level-Evaluation (Precision/Recall pro Issue-Type).

#### 2) Binary Decision Logic

**Problem:** Coherence hat keine binäre Decision Logic (kein `issue_threshold`, `score_cutoff`).

**Was fehlt:**
- Keine Funktion `pred_has_error = f(score, issues, threshold)` wie bei Factuality
- Keine binären Metriken (TP/FP/TN/FN, Precision, Recall, F1)

**Auswirkung:** Nur kontinuierliche Metriken (Korrelation, MAE) möglich, keine binäre Klassifikation.

#### 3) Issue-Level-Evaluation

**Problem:** Keine Gold-Labels für Issues, daher keine Issue-Level-Metriken.

**Was fehlt:**
- Keine Span-Overlap-Metriken (Issue-Position vs. Gold-Position)
- Keine Issue-Type-Klassifikation (Precision/Recall pro Type)
- Keine Severity-Klassifikation (Precision/Recall pro Severity)

**Auswirkung:** Nur Score-Korrelation, keine detaillierte Issue-Analyse.

#### 4) Konfigurierbare Thresholds

**Problem:** Keine Parameter für Decision Logic, daher keine Threshold-Tuning.

**Was fehlt:**
- Kein `score_cutoff` für binäre Entscheidung
- Kein `issue_threshold` für binäre Entscheidung
- Keine `severity_min` Filterung

**Auswirkung:** Keine Threshold-Optimierung möglich, nur Score-Korrelation.

#### 5) Robustheitstests

**Problem:** Keine Ablation-Studien oder Stresstests.

**Was fehlt:**
- Keine Ablation-Flags (`use_claim_extraction`-Äquivalent)
- Keine Shuffle/Swap-Tests (Summary-Sätze vertauschen → sollte Score sinken)
- Keine Adversarial-Tests (bewusst inkohärente Summaries)

**Auswirkung:** Keine Robustheits-Validierung.

---

## F) Code-Referenzen (Zusammenfassung)

| Komponente | Datei | Zeilen | Zweck |
|------------|-------|--------|-------|
| **CoherenceAgent** | `app/services/agents/coherence/coherence_agent.py` | 13-105 | Hauptagent-Klasse |
| **LLMCoherenceEvaluator** | `app/services/agents/coherence/coherence_verifier.py` | 18-200 | LLM-basierte Evaluation |
| **CoherenceIssue** | `app/services/agents/coherence/coherence_models.py` | 5-19 | Interne Issue-Datenstruktur |
| **AgentResult** | `app/models/pydantic.py` | 27-47 | Standardisiertes Output-Format |
| **IssueSpan** | `app/models/pydantic.py` | 6-25 | Span-Modell (für Coherence: nur start/end/message/severity) |
| **Pipeline-Integration** | `app/pipeline/verification_pipeline.py` | 36, 43, 46, 51 | Aufruf, Aggregation, Persistenz |
| **Postgres-Persistenz** | `app/db/postgres/persistence.py` | 205 | Speichert in `verification_results` |
| **Neo4j-Persistenz** | `app/db/neo4j/graph_persistence.py` | 135-137 | Schreibt Metric + IssueSpan Nodes |
| **Explainability-Integration** | `app/services/explainability/explainability_service.py` | 59, 262, 267 | Fließt in Explainability-Report ein |
| **Evaluation-Script** | `scripts/eval_sumeval_coherence.py` | 1-407 | SummEval-Evaluation (Pearson, Spearman, MAE) |
| **Unit-Tests** | `tests/unit/test_coherence_agent.py` | 1-48 | Basis-Tests vorhanden |

---

## G) Prompt-Version

**Aktueller Prompt:** `app/services/agents/coherence/coherence_verifier.py:107-150`

**Version:** Nicht explizit versioniert (kein `VERSION = "v1"` Feld), aber Prompt ist stabil.

**Determinismus:** ✅ `temperature=0.0` (hardcoded in `OpenAIClient`)

**Beleg:** `app/llm/openai_client.py:18` - `temperature=0.0`.

---

## H) Span-Mapping-Logik

**Datei:** `app/services/agents/coherence/coherence_agent.py:76-104`

**Methode:** `_build_error_spans()`

**Logik:**
1. Versucht `summary_span` (aus LLM) als Substring in `summary_text` zu finden
2. Wenn gefunden: `start_char = summary_text.find(issue.summary_span)`, `end_char = start + len(span)`
3. Wenn nicht gefunden: `start_char = None`, `end_char = None`

**Fallback:** `app/services/agents/coherence/coherence_verifier.py:176-187` - `_ensure_spans_are_substrings()` setzt auf ersten 120 Zeichen, wenn Span nicht in Summary gefunden wird.

**Beleg:** `app/services/agents/coherence/coherence_agent.py:85-89` - `start = summary_text.find(issue.summary_span)`.

---

## I) Scoring-Logik

**Datei:** `app/services/agents/coherence/coherence_verifier.py:40-45`

**Methode:** Direkt vom LLM (keine Post-Processing-Formel)

**Logik:**
1. LLM gibt `score` im JSON-Output zurück (erwartet: float in [0,1])
2. `_clamp01()` stellt sicher, dass Score in [0,1] liegt
3. **Keine** Formel wie `1 - f(num_issues, severity)` oder gewichtete Aggregation

**Beleg:** `app/services/agents/coherence/coherence_verifier.py:45` - `score = self._clamp01(data.get("score", 0.0))`.

**Safety-Net:** Wenn Score < 0.7 und keine Issues, wird Fallback-Issue erzeugt (Zeilen 84-95).

---

## J) Pipeline-Integration (Details)

### Aufruf

**Datei:** `app/pipeline/verification_pipeline.py:43`

```python
coherence = self.coherence_agent.run(article, summary, meta)
```

### Aggregation

**Datei:** `app/pipeline/verification_pipeline.py:46`

```python
overall = (factuality.score + coherence.score + readability.score) / 3.0
```

**Beleg:** Coherence-Score geht zu 1/3 in `overall_score` ein.

### Persistenz

**Postgres:** `app/db/postgres/persistence.py:205`

```python
insert_dimension("coherence", coherence, "CoherenceAgent")
```

**Neo4j:** `app/db/neo4j/graph_persistence.py:135-137`

```python
metric_with_issue_spans("coherence", coherence)
```

### Explainability

**Datei:** `app/services/explainability/explainability_service.py:262,267`

```python
coherence = _get_attr_or_key(pipeline_result, "coherence", None)
findings += self._normalize_generic(Dimension.coherence, coherence, summary_text)
```

**Beleg:** Coherence-Issues werden in `ExplainabilityResult.findings` normalisiert und gerankt.

---

## K) Test-Status

**Datei:** `tests/unit/test_coherence_agent.py`

**Vorhandene Tests:**
1. `test_coherence_agent_run_returns_agentresult()` - Prüft, dass AgentResult zurückgegeben wird
2. `test_coherence_agent_span_mapping_sets_offsets_when_found()` - Prüft Span-Mapping

**Fehlende Tests:**
- Keine Tests für verschiedene Issue-Typen
- Keine Tests für Severity-Klassifikation
- Keine Tests für Score-Berechnung
- Keine Tests für Fallback-Logik (Score < 0.7 ohne Issues)
- Keine Tests für Span-Mapping-Fallbacks

---

## L) Evaluation-Script-Status

**Datei:** `scripts/eval_sumeval_coherence.py`

**Vorhanden:**
- ✅ SummEval-Datenladen (`load_jsonl()`)
- ✅ Cache-System (`load_cache()`, `append_cache()`)
- ✅ Metriken (Pearson, Spearman, MAE)
- ✅ Normalisierung (GT 1-5 → 0-1)
- ✅ JSONL-Output für Predictions

**Fehlend:**
- ❌ Keine YAML-Config-Unterstützung (wie bei Factuality `m10_factuality_runs.yaml`)
- ❌ Keine automatische Dokumentation (wie bei Factuality `docs/<run_id>.md`)
- ❌ Keine Run-Management (wie bei Factuality `RunManager`)
- ❌ Keine Summary-Matrix (wie bei Factuality `summary_matrix.csv`)

**Beleg:** Vergleich `scripts/eval_sumeval_coherence.py` vs. `scripts/run_m10_factuality.py` - Coherence-Script ist einfacher, ohne Run-Management.




