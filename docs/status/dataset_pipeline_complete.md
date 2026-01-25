# Dataset-Pipeline: Vollständige Dokumentation

**Datum:** 2026-01-17  
**Zweck:** Umfassende Übersicht über Dataset-Converter, unified Schema, Loader, Evaluations-Skripte und Configs

---

## 1. Dataset-Converter (convert_*.py)

### 1.1 convert_sumeval.py

**Inputquelle / Erwartetes Rohformat:**
- **Format:** JSON (nicht JSONL)
- **Strukturen:**
  - Liste: `[{...}, {...}, ...]`
  - Dict mit Container: `{"data1": [...], ...}` oder `{"examples": [...], ...}`
  - **HuggingFace-Dump-Format:**
    - `item["source"]` oder `item["article"]` → Artikel
    - `item["hyp"]` oder `item["summary"]` → Zusammenfassung
    - `item["expert_coherence"]`, `item["expert_fluency"]` → Expert-Scores (1-5)
    - `item["model_id"]` → System-Name

**Outputformat (JSONL-Schema):**
```python
{
  "article": str,           # Artikel-Text (gestrippt)
  "summary": str,           # Zusammenfassung (gestrippt)
  "gt": {                   # Gold-Labels (dict)
    "coherence": float,     # z.B. 1.33 (kann normalisiert sein)
    "readability": float,   # z.B. 3.0 (kann normalisiert sein)
    "fluency": float        # z.B. 3.0 (kann normalisiert sein)
  },
  "meta": {                 # Metadaten
    "doc_id": str | None,
    "system": str | None,
    "model_id": Any | None,
    "filepath": str | None,
    "readability_source": str | None  # Hinweis: "expert_fluency"
  }
}
```

**Vorverarbeitung:**
1. **JSON-Loading:** Defensives Laden (Liste oder Dict mit Container)
2. **Feld-Extraktion:** Flexible Suche nach verschiedenen Key-Namen
3. **Score-Extraktion:**
   - Bevorzugt: HF `expert_*` Felder (`expert_coherence`, `expert_fluency`)
   - Fallback: Generische Strukturen (`scores[dim]`, `annotations`, `human_scores[dim]`)
   - Alias-Mapping: `readability` → `expert_fluency` (falls vorhanden)
4. **Normalisierung (optional):** Wenn `--normalize_from` gesetzt (z.B. 5.0), wird Score durch diesen Wert geteilt und auf [0,1] geclampet
5. **Filtering:** Beispiele ohne `article` oder `summary` werden übersprungen
6. **Required-Dims-Check:** Beispiele ohne erforderliche Dimensionen werden übersprungen

**Keine Sentence-Splitting oder Label-Mapping** (Scores werden direkt übernommen)

**Beispiel-Output (1 JSONL-Zeile):**
```json
{"article": "Paul Merson has restarted his row with Andros Townsend...", "summary": "paul merson was brought on with only seven minutes remaining...", "gt": {"coherence": 1.3333333333, "readability": 3.0, "fluency": 3.0}, "meta": {"doc_id": "dm-test-8764fb95bfad8ee849274873a92fb8d6b400eee2", "system": "M11", "model_id": "M11", "filepath": "cnndm/dailymail/stories/8764fb95bfad8ee849274873a92fb8d6b400eee2.story", "readability_source": "expert_fluency"}}
```

---

### 1.2 convert_finesumfact.py

**Inputquelle / Erwartetes Rohformat:**
- **Format:** JSON oder JSONL (automatische Erkennung)
- **Erwartete Felder:**
  - **Artikel:** `doc`, `article`, `document`, `text` (kann String, Liste von Strings, oder verschachtelte Struktur sein)
  - **Summary:** `model_summary`, `summary`, `generated_summary`
  - **Labels:**
    - **Machine:** `pred_general_factuality_labels` (Liste von 0/1 pro Satz)
    - **Human:** `label` (Liste von 0/1 pro Satz)
  - **Metadaten:** `source`, `model`, `split`

**Outputformat (JSONL-Schema):**
```python
{
  "article": str,           # Artikel-Text (normalisiert, gestrippt)
  "summary": str,           # Zusammenfassung (normalisiert, gestrippt)
  "has_error": bool,       # Binäres Label: true wenn mindestens ein Satz Fehler hat
  "meta": {                 # Metadaten
    "dataset": str,         # Immer "finesumfact"
    "label_source": str,    # "human" | "machine" | "unknown"
    "source": str | None,
    "model": str | None,
    "split": str | None,
    "n_sent_labels": int,   # Anzahl Satz-Labels
    "n_error_sent": int,    # Anzahl Sätze mit Fehler (label=1)
    "sentence_labels": list[int] | None  # Vollständige Liste (nur wenn --store-sentence-labels)
  }
}
```

**Vorverarbeitung:**
1. **JSON/JSONL-Loading:** Automatische Erkennung
2. **Text-Normalisierung (`_as_text`):**
   - String → direkt übernommen (gestrippt)
   - Liste von Strings → Join mit Leerzeichen
   - Liste von Listen → Flatten + Join
   - Dict → Suche nach `text`, `doc`, `article`, `document`, `sentences`, `sents`
   - Sonst → `str(x).strip()`
3. **Label-Extraktion:**
   - Machine: `pred_general_factuality_labels` (Liste)
   - Human: `label` (Liste)
   - Normalisierung: `_to_int01()` konvertiert bool/int/str → 0/1
4. **Label-Source-Inferenz:** Heuristik (Feld-Name oder Dateiname)
5. **Aggregation:** `has_error = any(x == 1 for x in sent_labels)`
6. **Filtering:** Beispiele ohne `article`, `summary` oder `sentence_labels` werden übersprungen

**Keine Sentence-Splitting** (Labels sind bereits satzweise vorhanden)

**Beispiel-Output (1 JSONL-Zeile):**
```json
{"article": "MICHELE NORRIS, host: To learn a little bit more about the Quds Force...", "summary": "Quds Force is a branch of the Revolutionary Guards in Iran...", "has_error": false, "meta": {"dataset": "finesumfact", "label_source": "human", "source": "mediasum_test", "model": null, "split": null, "n_sent_labels": 2, "n_error_sent": 0}}
```

---

### 1.3 convert_frank.py

**Inputquelle / Erwartetes Rohformat:**
- **Format:** 2× JSON-Dateien
  1. **`benchmark_data.json`:** Liste von Dicts mit:
     - `article: str` - Artikel-Text
     - `summary: str` - Zusammenfassung
     - `hash: str` - Hash-Identifikator
     - `model_name: str` - Model-Name
  2. **`human_annotations.json`:** Liste von Dicts mit:
     - `hash: str` - Hash-Identifikator (muss zu benchmark passen)
     - `model_name: str` - Model-Name (muss zu benchmark passen)
     - `Factuality: float` - Factuality-Score (0.0-1.0)

**Outputformat (JSONL-Schema):**
```python
{
  "article": str,           # Artikel-Text (gestrippt)
  "summary": str,           # Zusammenfassung (gestrippt)
  "has_error": bool,        # Binäres Label: true wenn factuality < 1.0
  "meta": {                 # Metadaten
    "hash": str,            # Hash-Identifikator
    "model_name": str,      # Model-Name
    "factuality": float     # Original Factuality-Score (0.0-1.0)
  }
}
```

**Vorverarbeitung:**
1. **JSON-Loading:** Beide Dateien werden als Listen geladen
2. **Annotation-Index:** `human_annotations.json` wird in ein Dict indexiert: `{(hash, model_name): annotation}`
3. **Matching:** Für jeden Benchmark-Eintrag wird nach passender Annotation gesucht
4. **Label-Mapping:**
   - `factuality == 1.0` → `has_error = false` (kein Fehler)
   - `factuality < 1.0` → `has_error = true` (hat Fehler)
5. **Text-Cleaning:** `article.strip()`, `summary.strip()`
6. **Filtering:**
   - Beispiele ohne `article` oder `summary` werden übersprungen
   - Beispiele ohne `hash` oder `model_name` werden übersprungen
   - Beispiele ohne passende Annotation werden übersprungen

**Keine Sentence-Splitting** (nur Dokument-Level-Labels)

**Beispiel-Output (1 JSONL-Zeile):**
```json
{"article": "Police in Arkansas wish to unlock an iPhone and iPod belonging to two teenagers...", "summary": "the fbi has said it will help the san bernardino killer to access iphones...", "has_error": true, "meta": {"hash": "35933239", "model_name": "BERTS2S", "factuality": 0.0}}
```

---

## 2. Unified Schema

### 2.1 EvaluationExample (Dataclass)

**Code-Referenz:** `scripts/eval_unified.py:43-51`

```python
@dataclass
class EvaluationExample:
    """Einheitliches Format für Evaluation-Beispiele."""
    
    example_id: str | None
    article: str              # Pflicht
    summary: str              # Pflicht
    ground_truth: dict[str, Any]  # dimension -> value (Pflicht)
    meta: dict[str, Any] | None = None  # Optional
```

**Verwendung:**
- Wird in `eval_unified.py` verwendet
- Wird von `load_frank_examples()` und `load_sumeval_examples()` erzeugt
- **Ground-Truth-Struktur:**
  - **Factuality:** `{"factuality": bool}` (binär)
  - **Coherence/Readability:** `{"coherence": float}` oder `{"readability": float}` (kontinuierlich)

### 2.2 JSONL-Format (Converter-Output)

Alle Converter erzeugen JSONL mit folgender Struktur:

**Gemeinsame Felder:**
- `article: str` (Pflicht)
- `summary: str` (Pflicht)
- `meta: dict` (Optional, aber empfohlen)

**Dataset-spezifische Felder:**
- **SummEval:** `gt: dict[str, float]` (coherence, readability, fluency)
- **FineSumFact/FRANK:** `has_error: bool` (binär)

---

## 3. Dataset-Loader in der Pipeline

### 3.1 FRANK / FineSumFact

**Code-Referenz:** `scripts/eval_unified.py:108-142`

**Loader-Funktion:**
```python
def load_frank_examples(path: Path) -> list[EvaluationExample]:
    """Lädt FRANK/FineSumFact-Format (binär, has_error)."""
```

**Dateipfade:**
- **FRANK:** `data/frank/frank_clean.jsonl`
- **FineSumFact:** `data/finesumfact/human_label_test_clean.jsonl`

**Felder (Pflicht vs Optional):**
- **Pflicht:** `article`, `summary`, `has_error`
- **Optional:** `id`, `meta`

**Verarbeitung:**
- `has_error` wird robust geparst (bool/int/str → bool)
- `ground_truth = {"factuality": gt_bool}`
- Beispiele ohne `article` oder `summary` werden übersprungen

### 3.2 SummEval

**Code-Referenz:** `scripts/eval_unified.py:145-174`

**Loader-Funktion:**
```python
def load_sumeval_examples(path: Path, dimension: str) -> list[EvaluationExample]:
    """Lädt SummEval-Format (kontinuierlich, gt[dimension])."""
```

**Dateipfade:**
- **SummEval:** `data/sumeval/sumeval_clean.jsonl`

**Felder (Pflicht vs Optional):**
- **Pflicht:** `article`, `summary`, `gt[dimension]` (z.B. `gt["coherence"]` oder `gt["readability"]`)
- **Optional:** `id`, `meta`

**Verarbeitung:**
- `gt_value = gt_dict.get(dimension)` wird zu `float` konvertiert
- `ground_truth = {dimension: gt_float}`
- Beispiele ohne `article`, `summary` oder `gt[dimension]` werden übersprungen

---

## 4. Sentence Segmentation

### 4.1 Factuality Agent

**Code-Referenz:** `app/services/agents/factuality/factuality_agent.py:421-446`

**Implementierung:**
```python
def _split_into_sentences_with_spans(self, text: str) -> List[Tuple[str, int, int]]:
    """
    Liefert [(sentence_text, start_char, end_char)] im Originaltext.
    Stabil und span-freundlich: splittet an Satzendzeichen und Newlines.
    """
```

**Methode:**
- **Regex-basiert:** `re.finditer(r"[^.!?\n]+[.!?]?", text)`
- **Keine externe Library** (kein NLTK, kein spaCy)
- **Char-Spans:** Liefert `(sentence_text, start_char, end_char)` für Span-Mapping
- **Trimming:** Whitespace wird entfernt, aber Char-Indizes bleiben korrekt

**Verwendung:**
- Wird für Summary-Segmentierung verwendet
- Char-Spans werden für Issue-Span-Generierung benötigt

### 4.2 Coherence Agent

**Code-Referenz:** `app/services/agents/coherence/coherence_extractor.py:10-16`

**Implementierung:**
```python
def split_summary_into_sentences(summary_text: str) -> list[SentenceInfo]:
    """
    Sehr einfacher, naiver Satzsplitter.
    Reicht für M7 völlig aus. Später ggf. durch spaCy o.Ä. ersetzen.
    """
    raw_sentences = [s.strip() for s in summary_text.split(".") if s.strip()]
```

**Methode:**
- **Einfach:** `summary_text.split(".")`
- **Keine externe Library**
- **Keine Char-Spans** (nur Text)

### 4.3 Readability Agent

**Code-Referenz:** `app/services/agents/readability/readability_extractor.py:39-70`

**Implementierung:**
```python
def split_summary_into_sentences(summary_text: str) -> list[ReadabilitySentenceInfo]:
    """
    Sehr einfacher, naiver Satzsplitter für den Readability-Agent.
    """
    raw_sentences = [s.strip() for s in summary_text.split(".") if s.strip()]
```

**Methode:**
- **Einfach:** `summary_text.split(".")`
- **Keine externe Library**
- **Zusätzlich:** Berechnet `word_count`, `comma_count`, `avg_word_length` pro Satz

**Zusammenfassung:**
- **Factuality:** Regex-basiert mit Char-Spans (für Issue-Span-Mapping)
- **Coherence/Readability:** Einfaches `split(".")` ohne Char-Spans
- **Keine externe Library** (kein NLTK, kein spaCy)

---

## 5. Evaluations-Skripte

### 5.1 Factuality (eval_unified.py)

**Code-Referenz:** `scripts/eval_unified.py:273-341`

**Scores:**
- **Agent-Score:** `agent_result.score` (0-1, kontinuierlich)
- **Binary Prediction:** `pred_has_error = num_issues >= error_threshold` (binär)

**Aggregation:**
- **Satzweise → Summaryweise:**
  - Factuality Agent extrahiert Claims pro Satz
  - Claims werden zu Satz-Labels aggregiert (`correct=1.0`, `incorrect=0.0`, `uncertain=0.5`)
  - Score wird aus Satz-Labels berechnet: `score = mean(sentence_labels)`
  - Issue-Spans werden aus Claims generiert
  - **Binary Decision:** `has_error = (num_issues >= error_threshold)`

**Outputs:**
- **`predictions.jsonl`:** Pro Beispiel:
  ```json
  {
    "example_id": str,
    "gt_has_error": bool,
    "pred_has_error": bool,
    "score": float,
    "num_issues": int,
    "meta": dict
  }
  ```
- **Metriken:** TP/FP/TN/FN, Precision, Recall, F1, Balanced Accuracy, MCC, AUROC

**Speicherort:**
- `results/evaluation/factuality/<run_id>/`

### 5.2 Coherence (eval_sumeval_coherence.py)

**Code-Referenz:** `scripts/eval_sumeval_coherence.py`

**Scores:**
- **Agent-Score:** `agent_result.score` (0-1, direkt vom LLM, keine weitere Normalisierung)
- **Ground-Truth:** `gt["coherence"]` (1-5) → normalisiert zu 0-1: `(gt_raw - 1) / 4`

**Aggregation:**
- **Keine satzweise Aggregation:** Coherence Agent bewertet die gesamte Summary
- Score kommt direkt vom LLM (0-1)

**Outputs:**
- **`predictions.jsonl`:** Pro Beispiel:
  ```json
  {
    "example_id": str,
    "gt_raw": float,      # 1-5
    "gt_norm": float,     # 0-1
    "pred_score": float,  # 0-1
    "meta": dict
  }
  ```
- **`summary.json`:** Metriken + Bootstrap-CIs (Pearson r, Spearman ρ, MAE, RMSE, R²)
- **`summary.md`:** Human-readable Report
- **`run_metadata.json`:** Timestamp, Git-Commit, Python-Version, Seed, etc.

**Speicherort:**
- `results/evaluation/coherence/<run_id>/`

### 5.3 Readability (eval_sumeval_readability.py)

**Code-Referenz:** `scripts/eval_sumeval_readability.py`

**Scores:**
- **Agent-Score:** `agent_result.score` (0-1, normalisiert von ursprünglich 1-5)
- **Normalisierung:** `(score_1_5 - 1.0) / 4.0` (siehe `app/services/agents/readability/normalization.py:33-49`)
- **Ground-Truth:** `gt["readability"]` (1-5) → normalisiert zu 0-1: `(gt_raw - 1) / 4`

**Aggregation:**
- **Keine satzweise Aggregation:** Readability Agent bewertet die gesamte Summary
- Score wird vom LLM als 1-5 ausgegeben, dann zu 0-1 normalisiert

**Outputs:**
- **`predictions.jsonl`:** Pro Beispiel:
  ```json
  {
    "example_id": str,
    "gt_raw": float,      # 1-5
    "gt_norm": float,     # 0-1
    "pred_score": float,  # 0-1
    "meta": dict
  }
  ```
- **`summary.json`:** Metriken + Bootstrap-CIs (Pearson r, Spearman ρ, MAE, RMSE, R²)
- **`summary.md`:** Human-readable Report
- **`run_metadata.json`:** Timestamp, Git-Commit, Python-Version, Seed, etc.

**Speicherort:**
- `results/evaluation/readability/<run_id>/`

---

## 6. YAML Run-Configs

### 6.1 configs/m10_factuality_runs.yaml

**Code-Referenz:** `configs/m10_factuality_runs.yaml`

**Dataset-bezogene Parameter:**

```yaml
runs:
  - run_id: "factuality_frank_baseline_v1"
    dataset: "frank"                    # Dataset-Name
    dataset_path: "data/frank/frank_clean.jsonl"  # Pfad zum JSONL
    max_examples: 300                  # Limit (optional)
    # ... weitere Parameter
```

**Parameter-Übersicht:**

| Parameter | Typ | Beschreibung | Beispiel |
|---|---|---|---|
| `dataset` | str | Dataset-Name | `"frank"`, `"finesumfact"`, `"combined"` |
| `dataset_path` | str | Pfad zum JSONL (einzelnes Dataset) | `"data/frank/frank_clean.jsonl"` |
| `dataset_paths` | list | Pfade für combined (mehrere Datasets) | `[{"path": "...", "max_examples": 300}, ...]` |
| `max_examples` | int | Limit der Beispiele | `300`, `200` |
| `llm_model` | str | LLM-Modell | `"gpt-4o-mini"` |
| `llm_temperature` | float | LLM-Temperatur | `0.0` |
| `llm_seed` | int | Seed für Reproduzierbarkeit | `42` |
| `prompt_version` | str | Prompt-Version | `"v3_uncertain_spans"` |
| `decision_mode` | str | Decision-Mode | `"issues"`, `"score"`, `"either"`, `"both"` |
| `error_threshold` | int | Threshold für Binary-Entscheidung | `1` |
| `severity_min` | str | Minimale Severity | `"low"`, `"medium"`, `"high"` |
| `uncertainty_policy` | str | Policy für uncertain Claims | `"count_as_error"`, `"non_error"`, `"weight_0.5"` |
| `cache_enabled` | bool | Cache aktiviert | `true`, `false` |
| `ablation_mode` | str | Ablation-Mode | `"none"`, `"no_claims"`, `"sentence_only"`, `"no_spans"` |
| `use_claim_extraction` | bool | Claim-Extraktion aktiviert | `true`, `false` |
| `use_claim_verification` | bool | Claim-Verifikation aktiviert | `true`, `false` |
| `use_spans` | bool | Issue-Spans generieren | `true`, `false` |

**Preprocessing Flags:**
- Keine expliziten Preprocessing-Flags in der Config
- Preprocessing erfolgt in den Converter-Skripten

**Output Directories:**
- Werden automatisch generiert: `results/evaluation/factuality/<run_id>/`
- Nicht in Config spezifiziert

### 6.2 BA-Evaluation relevante Configs

**Final Runs (für BA-Evaluation):**

1. **FRANK Final:**
   - `factuality_frank_tuned_v1` (n=300, best tuned config)
   - `factuality_frank_baseline_v1` (n=300, baseline)
   - `factuality_frank_ablation_v1` (n=300, ablation)

2. **FineSumFact Final:**
   - `factuality_finesumfact_final_v1` (n=200, best FRANK config, unverändert)
   - `factuality_finesumfact_ablation_v1` (n=200, ablation)

3. **Combined Final:**
   - `factuality_combined_final_v1` (FRANK n=300 + FineSumFact n=200)

**Tuning-Runs (für Analyse):**
- Verschiedene `factuality_frank_tune_*_v1` Runs (severity, uncertainty_policy, thresholds)

**Wichtig:**
- Alle Final-Runs verwenden `seed=42` für Reproduzierbarkeit
- `cache_enabled: true` für Effizienz
- `llm_temperature: 0.0` für Determinismus

---

## 7. Zusammenfassung

### 7.1 Converter → Loader → Evaluation Pipeline

```
Raw Dataset (JSON/JSONL)
    ↓
Converter (convert_*.py)
    ↓
Clean JSONL (unified format)
    ↓
Loader (load_*_examples)
    ↓
EvaluationExample (dataclass)
    ↓
Agent.run() → AgentResult
    ↓
Metrics Calculation
    ↓
Output (predictions.jsonl, summary.json, summary.md)
```

### 7.2 Score-Aggregation Übersicht

| Dimension | Satzweise → Summaryweise | Score-Range | Normalisierung |
|---|---|---|---|
| **Factuality** | ✅ Ja (Claims → Satz-Labels → Score) | 0-1 | Aggregation aus Satz-Labels |
| **Coherence** | ❌ Nein (ganze Summary) | 0-1 | Direkt vom LLM |
| **Readability** | ❌ Nein (ganze Summary) | 0-1 | Normalisiert von 1-5 |

### 7.3 Sentence Segmentation Übersicht

| Agent | Methode | Library | Char-Spans |
|---|---|---|---|
| **Factuality** | Regex (`re.finditer`) | Keine | ✅ Ja |
| **Coherence** | `split(".")` | Keine | ❌ Nein |
| **Readability** | `split(".")` | Keine | ❌ Nein |

### 7.4 Output-Formate

| Skript | predictions.jsonl | summary.json | summary.md | run_metadata.json |
|---|---|---|---|---|
| **Factuality** | ✅ | ✅ | ✅ | ✅ |
| **Coherence** | ✅ | ✅ | ✅ | ✅ |
| **Readability** | ✅ | ✅ | ✅ | ✅ |

**Speicherort:**
- `results/evaluation/<dimension>/<run_id>/`

