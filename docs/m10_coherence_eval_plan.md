# M10 Coherence Evaluation Plan

## Übersicht

Dieses Dokument beschreibt die Evaluation-Infrastruktur für den Coherence-Agenten auf SummEval (Human Coherence Ratings). Die Evaluation umfasst:

1. **Agent-Evaluation:** Vergleich von Agent-Scores gegen Human Coherence Ratings
2. **Baselines:** ROUGE-L und BERTScore als Vergleichsmetriken
3. **Robustheitstests:** Shuffle- und Injection-Tests zur Validierung der Sensitivität
4. **Aggregation:** Vergleichstabelle für alle Runs

---

## 1. Setup

### Datenformat

**Input:** `data/sumeval/sumeval_clean.jsonl`

Jede Zeile enthält:
```json
{
  "article": "...",
  "summary": "...",
  "gt": {
    "coherence": 3.7  // Likert 1-5
  },
  "meta": {
    "doc_id": "...",
    "system": "...",
    ...
  }
}
```

### Dependencies

**Erforderlich:**
- `openai` (für LLM-Calls)
- `pydantic` (für Models)
- Standard-Bibliotheken: `json`, `pathlib`, `random`, `math`, etc.

**Optional (für Baselines):**
- `rouge-score`: `pip install rouge-score`
- `bert-score`: `pip install bert-score`
- `pandas`: `pip install pandas` (für Aggregator)

---

## 2. Agent-Evaluation

### Script: `scripts/eval_sumeval_coherence.py`

**Zweck:** Evaluiert den CoherenceAgent gegen Human Coherence Ratings.

**Metriken:**
- **Pearson r** (mit Bootstrap-CI)
- **Spearman ρ** (mit Bootstrap-CI)
- **MAE** (Mean Absolute Error, mit Bootstrap-CI)
- **RMSE** (Root Mean Squared Error, mit Bootstrap-CI)
- **R²** (Coefficient of Determination)
- **n_used, n_failed, n_skipped**

**Ground Truth Normalization:**
- Raw: Likert 1-5 (`gt.coherence`)
- Normalized: `gt_norm = (gt_raw - 1) / 4` → [0, 1]
- Prediction: Agent-Score in [0, 1]
- Prediction (1-5 scale): `pred_1_5 = 1 + 4 * pred`

**Bootstrap-Konfidenzintervalle:**
- Standard: 2000 Resamples
- 95% CI: 2.5th und 97.5th Percentile
- Seedbar für Reproduzierbarkeit

**Artefakte:**
- `results/evaluation/coherence/<run_id>/`
  - `predictions.jsonl`: Pro Beispiel (example_id, pred, pred_1_5, gt_raw, gt_norm, num_issues, max_severity, issue_types_counts, top_issues)
  - `summary.json`: Alle Metriken + CIs + Metadaten
  - `summary.md`: Human-readable Zusammenfassung
  - `run_metadata.json`: timestamp, git_commit, python_version, seed, dataset_path, n_total/n_used/n_failed, config params
  - `cache.jsonl`: Optional (bei --cache)

**CLI:**
```bash
python scripts/eval_sumeval_coherence.py \
    --data data/sumeval/sumeval_clean.jsonl \
    --model gpt-4o-mini \
    --prompt_version v1 \
    --max_examples 100 \
    --seed 42 \
    --bootstrap_n 2000 \
    --cache
```

**Backward Compatibility:**
- `--data1` ist Alias für `--data`
- `--llm-model` ist Alias für `--model`
- `--max` ist Alias für `--max_examples`

---

## 3. Baselines

### Script: `scripts/eval_sumeval_coherence_baselines.py`

**Zweck:** Berechnet ROUGE-L und BERTScore als Baselines.

**Voraussetzung:** Referenzen müssen im Datensatz vorhanden sein (Feld `ref`, `reference` oder `references`).

**Hinweis:** SummEval clean enthält aktuell keine Referenzen. Das Script gibt eine Warnung aus und überspringt Beispiele ohne Referenz.

**Metriken:** Gleiche wie beim Agent (Pearson, Spearman, MAE, RMSE, Bootstrap-CIs)

**Artefakte:**
- `results/evaluation/coherence_baselines/<run_id>/`
  - Gleiche Struktur wie Agent-Evaluation
  - `baseline_type`: "rouge_l" oder "bertscore"

**CLI:**
```bash
# ROUGE-L
python scripts/eval_sumeval_coherence_baselines.py \
    --data data/sumeval/sumeval_clean.jsonl \
    --baseline rouge_l \
    --seed 42 \
    --bootstrap_n 2000

# BERTScore
python scripts/eval_sumeval_coherence_baselines.py \
    --data data/sumeval/sumeval_clean.jsonl \
    --baseline bertscore \
    --seed 42 \
    --bootstrap_n 2000
```

---

## 3.5 LLM-as-a-Judge Baseline

### Script: `scripts/eval_sumeval_coherence_llm_judge.py`

**Zweck:** LLM-as-a-Judge als moderne Baseline für Coherence-Bewertung.

**Ansatz:**
- **Rubrik-basiert:** Strukturierte Bewertungskriterien (1-5 Skala)
- **Striktes JSON:** Output muss exakt JSON sein (keine Prosa)
- **Temperature=0:** Determinismus für Reproduzierbarkeit
- **Mehrfachurteile:** `n_judgments` (default: 3) für Self-Consistency
- **Mean-Aggregation:** Finaler Score = Mean(normalisierte Scores)

**Rubrik (v1):**
- **Kriterien:**
  - a) Logischer Fluss/Übergänge zwischen Sätzen
  - b) Keine Widersprüche innerhalb der Summary
  - c) Klare Referenzen (Pronomen/Bezüge eindeutig)
  - d) Keine abrupten Sprünge/fehlende Verknüpfungen
- **Skala:** 1 (sehr inkohärent) bis 5 (sehr kohärent)

**JSON-Output Schema:**
```json
{
  "coherence_score_1_to_5": <integer 1-5>,
  "main_issue": "<none|missing_link|contradiction|jumpy_order|unclear_reference|other>",
  "explanation_short": "<max 2 Sätze>",
  "confidence_0_to_1": <float 0.0-1.0>
}
```

**Robustes Parsing:**
- Extrahiert JSON auch aus Markdown-Code-Blöcken
- Bei Parse-Fehler: 1 Retry mit Repair-Prompt
- Bei weiterem Fehler: Beispiel als `failed` markiert (nicht stilles Dummy)

**Metriken:**
- Gleiche wie Agent: Pearson r, Spearman ρ, MAE, RMSE, R² (mit Bootstrap-CIs)
- `n_used`, `n_skipped`, `n_failed`

**Artefakte:**
- `results/evaluation/coherence_judge/<run_id>/`
  - `predictions.jsonl`: Pro Beispiel (example_id, gt, gt_norm, pred, pred_raw_scores_1_to_5, pred_norm_per_judge, main_issue_mode, confidence_mean, judge_model, rubric_version, n_judgments, meta)
  - `summary.json`: Alle Metriken + CIs + Metadaten + `method: "llm_judge"` + `judge_config`
  - `summary.md`: Human-readable (mit Judge-Config)
  - `run_metadata.json`: timestamp, git_commit, python_version, seed, dataset_path, judge_model, rubric_version, n_judgments, temperature, etc.
  - `cache.jsonl`: Optional (bei --cache)

**CLI:**
```bash
python scripts/eval_sumeval_coherence_llm_judge.py \
    --data data/sumeval/sumeval_clean.jsonl \
    --judge_model gpt-4o-mini \
    --rubric_version v1 \
    --n_judgments 3 \
    --temperature 0 \
    --max_examples 200 \
    --seed 42 \
    --bootstrap_n 2000 \
    --cache
```

**Interpretation:**
- Judge ist **Baseline/Comparator**, nicht Ground Truth
- Standardisierung (Rubrik + JSON + temp=0 + Mehrfachurteile) statt "freie Meinung"
- Wichtig für Thesis: Judge als moderne Baseline, nicht als Kernlogik des Systems

---

## 4. Robustheitstests (Stress-Tests)

### Script: `scripts/stress_test_coherence.py`

**Zweck:** Prüft, ob der Agent "merkt", wenn Text kaputt gemacht wird.

**Modi:**

#### 4.1 Shuffle-Test
- **Methode:** Permutiert Satzreihenfolge (seedbar)
- **Erwartung:** `score_original > score_shuffled` (höherer Score = besser)
- **Metrik:** `delta = score_original - score_shuffled`
- **Success:** `delta > 0`

#### 4.2 Injection-Test
- **Methode:** Injiziert klar inkohärenten Satz an zufälliger Position
- **Erwartung:** `score_original > score_injected`
- **Metrik:** `delta = score_original - score_injected`
- **Success:** `delta > 0`

**Artefakte:**
- `results/evaluation/coherence_stress/<run_id>/`
  - `stress_results.jsonl`: Pro Beispiel (example_id, mode, score_original, score_perturbed, delta, success)
  - `summary.json`: success_rate, mean_delta, median_delta, min_delta, max_delta
  - `summary.md`: Human-readable Zusammenfassung
  - `run_metadata.json`: timestamp, git_commit, mode, llm_model, seed, etc.

**CLI:**
```bash
# Shuffle-Test
python scripts/stress_test_coherence.py \
    --data data/sumeval/sumeval_clean.jsonl \
    --mode shuffle \
    --model gpt-4o-mini \
    --seed 42 \
    --max_examples 50

# Injection-Test
python scripts/stress_test_coherence.py \
    --data data/sumeval/sumeval_clean.jsonl \
    --mode inject \
    --model gpt-4o-mini \
    --seed 42 \
    --max_examples 50
```

**Interpretation:**
- **Success rate > 0.5:** Agent erkennt Perturbationen in der Mehrzahl
- **Mean Δ > 0:** Original im Durchschnitt besser als perturbed (erwartet)
- **Median Δ > 0:** Median zeigt konsistentes Verhalten

---

## 5. Aggregation

### Script: `scripts/aggregate_coherence_runs.py`

**Zweck:** Erstellt Vergleichstabelle für mehrere Runs (Agent + Baselines + Stress optional).

**Input:** Verzeichnisse mit `summary.json`

**Output:**
- `summary_matrix.csv`: Tabellarischer Vergleich (pandas DataFrame)
- `summary_matrix.md`: Human-readable Markdown-Tabelle

**Spalten:**
- `run_id`, `system` (agent/rouge_l/bertscore/llm_judge/stress_shuffle/stress_inject)
- `model`, `prompt_version` (bzw. `rubric_version` für Judge), `seed`
- `pearson`, `spearman`, `mae`, `rmse`, `r_squared` (mit CIs)
- `n_used`, `n_failed`

**CLI:**
```bash
python scripts/aggregate_coherence_runs.py \
    results/evaluation/coherence/coherence_20240101_120000_gpt-4o-mini_v1 \
    results/evaluation/coherence_baselines/coherence_rouge_l_20240101_130000 \
    results/evaluation/coherence_baselines/coherence_bertscore_20240101_140000 \
    --out results/evaluation/coherence/summary_matrix
```

---

## 6. Issue-Struktur

### Issue-Typen

Der CoherenceAgent setzt `issue_type` explizit in `IssueSpan`:

- `LOGICAL_INCONSISTENCY`: Logische Widersprüche
- `CONTRADICTION`: Direkte Widersprüche
- `REDUNDANCY`: Überflüssige Wiederholungen
- `ORDERING`: Probleme mit Informationsfluss/Reihenfolge
- `OTHER`: Sonstige Kohärenzprobleme

**Code-Stelle:** `app/services/agents/coherence/coherence_agent.py:101`

```python
IssueSpan(
    ...
    issue_type=issue.type,  # Explizit gesetzt
    ...
)
```

**Verwendung in Evaluation:**
- `issue_types_counts`: Häufigkeit pro Typ
- `max_severity`: Höchste Severity (high > medium > low)
- `top_issues`: Top-N Issues sortiert nach Severity

---

## 7. Reproduzierbarkeit

### Seed-Management

Alle Scripts unterstützen `--seed` für deterministische Ausführung:

- **Bootstrap-CIs:** Seed für Resampling
- **Stress-Tests:** Seed für Perturbationen (Shuffle, Injection)
- **Caching:** Deterministische Cache-Keys (hash-basiert)

### Git-Integration

`run_metadata.json` enthält:
- `git_commit`: Aktueller Commit-Hash (falls verfügbar)
- `python_version`: Python-Version
- `timestamp`: ISO-Format

---

## 8. Workflow-Beispiel

### Vollständige Evaluation

```bash
# 1. Agent-Evaluation
python scripts/eval_sumeval_coherence.py \
    --data data/sumeval/sumeval_clean.jsonl \
    --model gpt-4o-mini \
    --prompt_version v1 \
    --seed 42 \
    --bootstrap_n 2000 \
    --cache

# 2. Baselines (falls Referenzen vorhanden)
python scripts/eval_sumeval_coherence_baselines.py \
    --data data/sumeval/sumeval_clean.jsonl \
    --baseline rouge_l \
    --seed 42

python scripts/eval_sumeval_coherence_baselines.py \
    --data data/sumeval/sumeval_clean.jsonl \
    --baseline bertscore \
    --seed 42

# 3. Stress-Tests
python scripts/stress_test_coherence.py \
    --data data/sumeval/sumeval_clean.jsonl \
    --mode shuffle \
    --model gpt-4o-mini \
    --seed 42 \
    --max_examples 50

python scripts/stress_test_coherence.py \
    --data data/sumeval/sumeval_clean.jsonl \
    --mode inject \
    --model gpt-4o-mini \
    --seed 42 \
    --max_examples 50

# 4. LLM-Judge Baseline
python scripts/eval_sumeval_coherence_llm_judge.py \
    --data data/sumeval/sumeval_clean.jsonl \
    --judge_model gpt-4o-mini \
    --rubric_version v1 \
    --n_judgments 3 \
    --temperature 0 \
    --seed 42

# 5. Aggregation
python scripts/aggregate_coherence_runs.py \
    results/evaluation/coherence/coherence_* \
    results/evaluation/coherence_baselines/coherence_* \
    results/evaluation/coherence_judge/coherence_judge_* \
    --out results/evaluation/coherence/summary_matrix
```

---

## 9. Acceptance Criteria

✅ **Definition of Done:**

1. ✅ `python scripts/eval_sumeval_coherence.py --data ...` läuft und erzeugt Artefakte
2. ✅ `python scripts/eval_sumeval_coherence_baselines.py --data ...` läuft und erzeugt Artefakte
3. ✅ `python scripts/eval_sumeval_coherence_llm_judge.py --data ...` läuft und erzeugt Artefakte
4. ✅ `python scripts/stress_test_coherence.py --data ... --mode shuffle` und `--mode inject` laufen
5. ✅ `python scripts/aggregate_coherence_runs.py` schreibt eine vergleichbare Matrix (inkl. Judge-Runs)
6. ✅ Alles seedbar, alles geloggt, alles nachvollziehbar

---

## 10. Troubleshooting

### Fehler: "rouge-score nicht installiert"
```bash
pip install rouge-score
```

### Fehler: "bert-score nicht installiert"
```bash
pip install bert-score
```

### Fehler: "Keine Referenz gefunden"
- SummEval clean enthält aktuell keine Referenzen
- Baselines können nur mit Datensätzen mit Referenzen laufen
- Alternative: Verwende SummEval raw oder anderen Datensatz mit Referenzen

### Fehler: "KeyError: 'data'"
- Verwende `--data` statt `--data1` (oder nutze `--data1` als Alias)

### Fehler: "ModuleNotFoundError: pandas"
```bash
pip install pandas
```

---

## 11. Weitere Dokumentation

- `docs/factuality_agent.md`: Detaillierte Beschreibung des Factuality-Agents (analog für Coherence)
- `M10_EVALUATION.md`: Evaluationsworkflow und Ergebnisse (allgemein)
- `PROJECT_STATUS.md`: Aktueller Projektstatus

