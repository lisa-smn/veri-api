# M10 Factuality Baselines (ROUGE-L, BERTScore)

**Datum:** 2025-01-07  
**Zweck:** Vergleich des Factuality-Agenten mit klassischen Metriken (ROUGE-L, BERTScore)

---

## 1. Übersicht

### Fairen Vergleich sicherstellen

**Wichtig:** Agent und Baselines müssen auf **exakt demselben FRANK-Subset** evaluiert werden.

**Lösung:** Gemeinsames Manifest (`frank_subset_manifest.jsonl`)

1. **Manifest erstellen:** `scripts/build_frank_subset_manifest.py`
   - Definiert gemeinsames Subset mit allen benötigten Feldern
   - Berechnet Dataset-Signature (SHA256-Hash) für Konsistenz-Prüfung

2. **Agent evaluieren:** `scripts/eval_frank_factuality_agent_on_manifest.py`
   - Nutzt Manifest als Input
   - Speichert `dataset_signature` in `summary.json`

3. **Baselines evaluieren:** `scripts/eval_frank_factuality_baselines.py`
   - Unterstützt `--manifest` (empfohlen) oder Legacy-Modus (`--benchmark` + `--annotations`)
   - Speichert `dataset_signature` in `summary.json`

4. **Aggregation:** `scripts/aggregate_factuality_runs.py`
   - Prüft automatisch, ob alle Runs dieselbe `dataset_signature` haben
   - Warnt bei unterschiedlichen `n_used` Werten

### Datensatz

**FRANK (Original-Format):**
- `data/frank/benchmark_data.json` - Enthält `article`, `summary`, `reference`
- `data/frank/human_annotations.json` - Enthält `Factuality` (via `hash` + `model_name`)

**Manifest (empfohlen):**
- `data/frank/frank_subset_manifest.jsonl` - Gemeinsames Subset für fairen Vergleich

**Schema (Manifest):**
- `hash`, `model_name`: Identifikation
- `article_text`, `summary_text`, `reference`: Texte
- `has_error`: Gold-Label (bool)
- `meta`: Optional (factuality score, etc.)

**Hinweis:** `frank_clean.jsonl` enthält **keine Referenzen**. Das Baseline-Script nutzt daher `benchmark_data.json` direkt oder das Manifest.

---

## 2. Manifest erstellen (Schritt 1)

### Script: `scripts/build_frank_subset_manifest.py`

**Funktion:**
- Lädt FRANK `benchmark_data.json` und `human_annotations.json`
- Filtert Beispiele mit: `article` + `summary` + `reference` + valid annotation
- Erstellt Manifest-JSONL mit allen benötigten Feldern
- Berechnet Dataset-Signature (SHA256-Hash) für Konsistenz-Prüfung

**CLI:**
```bash
python3 scripts/build_frank_subset_manifest.py \
    --benchmark data/frank/benchmark_data.json \
    --annotations data/frank/human_annotations.json \
    --output data/frank/frank_subset_manifest.jsonl \
    --max_examples 200 \
    --seed 42
```

**Output:**
- `data/frank/frank_subset_manifest.jsonl` (eine Zeile pro Beispiel)
- Manifest-Hash wird ausgegeben (für Verifikation)

---

## 3. Agent-Evaluation (Schritt 2)

### Script: `scripts/eval_frank_factuality_agent_on_manifest.py`

**Funktion:**
- Lädt Manifest-JSONL
- Evaluiert FactualityAgent auf allen Beispielen
- Speichert binäre Metriken + `dataset_signature`

**CLI:**
```bash
python3 scripts/eval_frank_factuality_agent_on_manifest.py \
    --manifest data/frank/frank_subset_manifest.jsonl \
    --model gpt-4o-mini \
    --issue_threshold 1 \
    --max_examples 200
```

**Artefakte:**
- `results/evaluation/factuality/<run_id>/`
  - `predictions.jsonl`
  - `summary.json` (mit `dataset_signature`)
  - `summary.md`
  - `run_metadata.json`

---

## 4. Baseline-Script (Schritt 3)

### Script: `scripts/eval_frank_factuality_baselines.py`

**Funktion:**
- **Modus 1 (empfohlen):** Lädt Manifest-JSONL (`--manifest`)
- **Modus 2 (Legacy):** Lädt `benchmark_data.json` + `human_annotations.json`
- Berechnet pro Beispiel:
  - `rouge_l`: ROUGE-L F1-Score (summary vs reference)
  - `bertscore_f1`: BERTScore F1-Score (summary vs reference)
- Skaliert Gold: `has_error=False` → 1.0, `has_error=True` → 0.0
- Berechnet Metriken: Pearson r, Spearman ρ, MAE, RMSE, R² (mit Bootstrap-CIs)

**Metriken:**
- **Pearson r** (mit Bootstrap-CI)
- **Spearman ρ** (mit Bootstrap-CI)
- **MAE** (mit Bootstrap-CI)
- **RMSE** (mit Bootstrap-CI)
- **R²** (Coefficient of Determination)
- **n_total, n_used, n_skipped**
- **dataset_signature** (für Konsistenz-Prüfung)

**Bootstrap-Konfidenzintervalle:**
- Standard: 2000 Resamples
- 95% CI: 2.5th und 97.5th Percentile
- Seedbar für Reproduzierbarkeit

**Artefakte:**
- `results/evaluation/factuality_baselines/<run_id>/`
  - `predictions.jsonl`: Pro Beispiel (example_id, baseline, pred, gt, has_error, meta)
  - `summary.json`: Alle Metriken + CIs + Metadaten + `dataset_signature`
  - `summary.md`: Human-readable Zusammenfassung (mit kompakter Vergleichstabelle)
  - `run_metadata.json`: timestamp, git_commit, python_version, seed, dataset_path, n_total/n_used/n_skipped, config params

**CLI (Manifest-Modus, empfohlen):**
```bash
# ROUGE-L
python3 scripts/eval_frank_factuality_baselines.py \
    --manifest data/frank/frank_subset_manifest.jsonl \
    --baseline rouge_l \
    --seed 42 \
    --bootstrap_n 2000

# BERTScore
python3 scripts/eval_frank_factuality_baselines.py \
    --manifest data/frank/frank_subset_manifest.jsonl \
    --baseline bertscore \
    --seed 42 \
    --bootstrap_n 2000
```

**CLI (Legacy-Modus):**
```bash
python3 scripts/eval_frank_factuality_baselines.py \
    --benchmark data/frank/benchmark_data.json \
    --annotations data/frank/human_annotations.json \
    --baseline rouge_l \
    --seed 42
```

**Dependencies:**
- `rouge-score`: `pip install rouge-score`
- `bert-score`: `pip install bert-score transformers torch`
- `pandas`: `pip install pandas` (für Aggregator)

### Dependency-Schutz / Fail-fast

**Wichtig:** Das Script bricht standardmäßig ab, wenn benötigte Dependencies fehlen. Dies verhindert stille 0.0-Placeholder und erhöht die Reproduzierbarkeit.

**Standardverhalten:**
- Bei fehlenden Dependencies: Script bricht mit `SystemExit(1)` ab
- Klare Fehlermeldung mit Installationsanweisungen
- `run_metadata.json` enthält `dependencies_ok: false` und `missing_packages: [...]`

**Debug-Modus (nur für Entwicklung):**
- `--allow_dummy_baseline`: Erlaubt 0.0 als Dummy-Wert bei fehlenden Dependencies
- **Nicht für Thesis-Evaluation nutzen!**
- `run_metadata.json` markiert solche Runs als `dependencies_ok: false`

**Beispiele:**

**Normaler Run (empfohlen):**
```bash
# Script bricht ab, wenn rouge-score fehlt
python3 scripts/eval_frank_factuality_baselines.py \
    --manifest data/frank/frank_subset_manifest.jsonl \
    --baseline rouge_l \
    --seed 42
```

**Dummy-Run (nur Debugging, nicht für Thesis):**
```bash
# ⚠️ Nur für Debugging! Ergebnisse sind nicht vergleichbar!
python3 scripts/eval_frank_factuality_baselines.py \
    --manifest data/frank/frank_subset_manifest.jsonl \
    --baseline rouge_l \
    --seed 42 \
    --allow_dummy_baseline
```

**run_metadata.json enthält:**
```json
{
  "dependencies_ok": true,  // oder false
  "missing_packages": []    // oder ["rouge-score"]
}
```

---

## 5. Aggregator (Schritt 4)

### Script: `scripts/aggregate_factuality_runs.py`

**Funktion:**
- Liest `summary.json` aus:
  - `results/evaluation/factuality/<run_id>/summary.json` (Agent-Runs)
  - `results/evaluation/factuality_baselines/<run_id>/summary.json` (Baseline-Runs)
- **Prüft automatisch Dataset-Konsistenz:**
  - Warnt, wenn unterschiedliche `dataset_signature` gefunden werden
  - Warnt, wenn unterschiedliche `n_used` Werte gefunden werden
- Erstellt Vergleichstabelle:
  - `results/evaluation/summary_factuality_matrix.csv`
  - `results/evaluation/summary_factuality_matrix.md`

**Spalten:**
- `run_id`, `method` (agent/rouge_l/bertscore)
- `model`, `prompt_version`, `seed`
- **Dataset-Info:** `n_total`, `dataset_signature`
- **Binäre Metriken (Agent):** `tp`, `fp`, `tn`, `fn`, `precision`, `recall`, `f1`, `balanced_accuracy`, `auroc`
- **Kontinuierliche Metriken (Baselines):** `pearson`, `spearman`, `mae`, `rmse`, `r_squared` (mit CIs)
- `n`, `n_failed`

**CLI:**
```bash
python3 scripts/aggregate_factuality_runs.py \
    results/evaluation/factuality/factuality_agent_manifest_* \
    results/evaluation/factuality_baselines/factuality_rouge_l_* \
    results/evaluation/factuality_baselines/factuality_bertscore_* \
    --out results/evaluation/summary_factuality_matrix
```

**Dataset-Konsistenz-Prüfung:**
- Automatisch aktiviert (kann mit `--skip-consistency-check` übersprungen werden)
- Gibt Warnungen aus, wenn:
  - Unterschiedliche `dataset_signature` gefunden werden
  - Unterschiedliche `n_used` Werte gefunden werden

---

## 4. Darstellung

### summary.md (Baselines)

Enthält oben eine kompakte Vergleichstabelle:

| Metric | ROUGE-L | BERTScore |
|--------|---------|-----------|
| Pearson r | 0.XXXX [CI] | 0.XXXX [CI] |
| Spearman ρ | 0.XXXX [CI] | 0.XXXX [CI] |
| MAE | 0.XXXX [CI] | 0.XXXX [CI] |
| RMSE | 0.XXXX [CI] | 0.XXXX [CI] |

### summary_matrix.md (Aggregation)

Enthält:
1. **Quick Comparison:** Kompakte Tabelle mit allen Methoden
2. **Detaillierte Tabellen:** Pro Method-Typ (agent, rouge_l, bertscore)
3. **Interpretation:**
   - "Baselines messen Ähnlichkeit zur Referenz, nicht Faktentreue. Vergleich dient als Proxy-Baseline."
   - "Factuality-Agent misst explizit Faktentreue durch Evidence-Gate und Claim-Verifikation."

---

## 6. Workflow-Beispiel (Vollständig)

### Faire Evaluation mit Manifest

```bash
# 1. Manifest erstellen (einmalig)
python3 scripts/build_frank_subset_manifest.py \
  --benchmark data/frank/benchmark_data.json \
  --annotations data/frank/human_annotations.json \
  --out data/frank/frank_subset_manifest.jsonl \
  --max_examples 200 \
  --seed 42

# 2. Agent auf Manifest
python3 scripts/eval_frank_factuality_agent_on_manifest.py \
  --manifest data/frank/frank_subset_manifest.jsonl \
  --model gpt-4o-mini \
  --seed 42

# 3. Baselines auf Manifest
python3 scripts/eval_frank_factuality_baselines.py \
  --manifest data/frank/frank_subset_manifest.jsonl \
  --baseline rouge_l \
  --seed 42

python3 scripts/eval_frank_factuality_baselines.py \
  --manifest data/frank/frank_subset_manifest.jsonl \
  --baseline bertscore \
  --seed 42

# 4. Aggregation
python3 scripts/aggregate_factuality_runs.py \
  results/evaluation/factuality/*manifest* \
  results/evaluation/factuality_baselines/* \
  --out results/evaluation/summary_factuality_matrix
```

**Erwartete Ausgabe (Aggregator):**
```
Aggregiere 3 Runs...
============================================================
DATASET-KONSISTENZ-PRÜFUNG
============================================================
✅ Alle Runs nutzen dieselbe Dataset-Signature: abc123...
✅ Alle Runs haben n_used=200
============================================================
```

**Bei Inkonsistenz:**
```
============================================================
DATASET-KONSISTENZ-PRÜFUNG
============================================================
⚠️  WARNUNG: Unterschiedliche Dataset-Signatures gefunden!
- Signature abc123...: 1 Runs
- Signature def456...: 2 Runs
→ Agent und Baselines nutzen möglicherweise unterschiedliche Datensätze!
→ Vergleich ist nicht fair!
============================================================
```

---

## 7. Artefaktstruktur

### Baseline-Run

```
results/evaluation/factuality_baselines/<run_id>/
├── predictions.jsonl      # Pro Beispiel: example_id, baseline, pred, gt, has_error, meta
├── summary.json            # Metriken + CIs + Metadaten
├── summary.md              # Human-readable (mit Vergleichstabelle)
└── run_metadata.json       # timestamp, git_commit, python_version, seed, etc.
```

### Aggregation

```
results/evaluation/
├── summary_factuality_matrix.csv  # Tabellarischer Vergleich (pandas DataFrame)
└── summary_factuality_matrix.md   # Human-readable Markdown (mit Interpretation)
```

---

## 8. Interpretation

### Vergleich Agent vs. Baselines

**Baselines (ROUGE-L, BERTScore):**
- Messen **Ähnlichkeit zur Referenz**, nicht direkt Faktentreue
- Höhere Ähnlichkeit korreliert oft mit besserer Faktentreue, ist aber kein direkter Maßstab
- Proxy-Baseline für Vergleich

**Factuality-Agent:**
- Misst **explizit Faktentreue** durch:
  - Evidence-Gate (nur mit Evidence als "incorrect" markiert)
  - Claim-Verifikation gegen Artikel
  - Strukturierte Issue-Spans

**Erwartung:**
- Agent sollte **bessere Korrelation** mit Human-Factuality zeigen als Baselines
- Baselines können als **Untergrenze** dienen (wenn Agent schlechter ist, gibt es Probleme)

---

## 9. Troubleshooting

### Fehler: "rouge-score nicht installiert"
```bash
pip install rouge-score
```

### Fehler: "bert-score nicht installiert"
```bash
pip install bert-score
```

### Fehler: "Keine Referenzen gefunden"
- Prüfe, ob `benchmark_data.json` das Feld `reference` enthält
- Falls nicht: Nutze Original-FRANK-Datensatz (nicht `frank_clean.jsonl`)

### Fehler: "ModuleNotFoundError: pandas"
```bash
pip install pandas
```

---

## 10. Acceptance Criteria

✅ **Definition of Done:**

1. ✅ `python3 scripts/eval_frank_factuality_baselines.py --baseline rouge_l ...` läuft und erzeugt Artefakte
2. ✅ `python3 scripts/eval_frank_factuality_baselines.py --baseline bertscore ...` läuft und erzeugt Artefakte
3. ✅ `python3 scripts/aggregate_factuality_runs.py` schreibt eine vergleichbare Matrix
4. ✅ Metriken/CIs sind reproduzierbar bei gleichem Seed
5. ✅ Vergleich ist fair: identische Filterlogik (nur Beispiele mit gültigem gold + reference)

---

## 11. Weitere Dokumentation

- `docs/m10_coherence_eval_plan.md`: Coherence-Evaluation (analog)
- `M10_EVALUATION.md`: Evaluationsworkflow und Ergebnisse (allgemein)
- `DATASET_REFERENCE_CHECK.md`: Prüfung auf Referenzen in Datensätzen

