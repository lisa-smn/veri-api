# M10 Factuality Tuning Workflow

## Übersicht

Dieses Dokument beschreibt den vollständigen Workflow für das Tuning des FactualityAgents mit zusätzlichen Strategien über `issue_threshold` hinaus.

## Workflow-Schritte

### 1. Baseline-Analyse

```bash
# Baseline-Run ausführen (falls noch nicht vorhanden)
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml --run-id factuality_frank_baseline_v1

# Baseline-Analyse mit FP-Issue-Types
python3 scripts/tune_from_baseline.py --baseline-run-id factuality_frank_baseline_v1
```

**Output:**
- Top 10 FP Issue Types
- Uncertainty-Anteil in FPs
- Vorschläge für Tuning-Strategien

### 2. Config anpassen (optional)

Basierend auf der Baseline-Analyse:
- `ignore_issue_types` in `factuality_frank_tune_ignore_types_v1` anpassen
- Top FP Types aus der Analyse übernehmen

### 3. FRANK Tuning-Runs ausführen

```bash
# Alle Tuning-Varianten auf FRANK (Dev/Calibration)
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml --run-id factuality_frank_tune_severity_v1
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml --run-id factuality_frank_tune_ignore_types_v1
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml --run-id factuality_frank_tune_uncertain_policy_v1
```

**Oder alle auf einmal:**
```bash
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml --skip-baseline
```

### 4. Ergebnisse aggregieren

```bash
python3 scripts/aggregate_m10_results.py
```

Erstellt `results/evaluation/summary_matrix.csv` und `summary.md`.

### 5. Best Tuned Run auswählen

```bash
python3 scripts/select_best_tuned_run.py --recall-min 0.90 --specificity-min 0.20 --target mcc
```

**Robuste Kriterien (neu implementiert):**
- Gate 1: **recall >= 0.90** (Constraint - verhindert zu niedrigen Recall)
- Gate 2: **specificity >= 0.20** (NEU - verhindert katastrophal niedrige Specificity)
- Optimierungsziel: **mcc** (Matthews Correlation Coefficient) ODER **balanced_accuracy** (wählbar)
- Tie-breaker: precision, dann f1

**Warum diese Gates wichtig sind:**
- **Recall-Gate:** Verhindert, dass wichtige Fehler übersehen werden
- **Specificity-Gate:** Verhindert, dass fast alle Summaries als fehlerhaft markiert werden (wie beim alten Best-Run mit Specificity 0.057)
- **MCC:** Besser als Balanced Accuracy für unbalanced Klassen, da es alle 4 Confusion-Matrix-Werte berücksichtigt

**Output:**
- Confusion Matrix (TP/TN/FP/FN)
- Alle Metriken (Recall, Specificity, Balanced Accuracy, MCC, Precision, F1)
- Begründung warum der Run gewählt wurde (welche Gates erfüllt, welcher Score max)
- Top-5 Tabelle aus Kandidatenmenge (für Transparenz)
- YAML-Snippet für FineSumFact final (inkl. `decision_threshold_float`, `severity_weights` falls vorhanden)

**Artefakte (Reproduzierbarkeit):**
- `results/evaluation/<dataset>/best_run_selection.json`: Vollständige Metadaten (Selection Criteria, Filtering Stats, Candidate Pool, Ranking Details)
- `results/evaluation/<dataset>/best_run_top5.csv`: Top-5 Runs aus Kandidatenmenge
- Selection speichert Kandidatenmenge, Filterstats und Top-K, um Entscheidungen nachvollziehbar zu machen

### 6. FineSumFact Final (Test-Set)

**WICHTIG:** FineSumFact ist ein reines Test-Set. Keine Parameteränderungen nach FRANK!

```bash
# Config manuell anpassen: factuality_finesumfact_final_v1 mit best-tuned Config
# Dann ausführen:
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml --run-id factuality_finesumfact_final_v1
```

### 7. Combined Final

```bash
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml --run-id factuality_combined_final_v1
```

## Tuning-Strategien

### Haupttreiber: uncertainty_policy und error_threshold

**Warum uncertainty_policy und threshold die Haupttreiber sind:**
- Viele False Positives entstehen durch "uncertain" Issues, die als Fehler gezählt werden
- `uncertainty_policy=count_as_error` + `error_threshold=1` + `severity_min=low` → sehr viele FPs
- Lösung: `uncertainty_policy=non_error` oder `weight_0.5` reduziert FPs deutlich
- Alternative: `error_threshold=2` erfordert 2 Issues statt 1

### 1. severity_min

**Run:** `factuality_frank_tune_severity_v1`

- Filtert nur Issues mit `severity >= medium`
- Reduziert False Positives (viele FPs haben nur "low" severity)
- **Problem:** Kann Recall zu stark reduzieren (wenn wichtige Issues "low" sind)
- **Ergebnis:** Recall < 0.10 bei severity_min=medium (zu aggressiv)

### 2. ignore_issue_types

**Run:** `factuality_frank_tune_ignore_types_v1`

- Ignoriert bestimmte Issue Types (z.B. "OTHER")
- Basierend auf Baseline-Analyse: Top FP Types identifizieren
- Reduziert False Positives für "noisy" Types
- **Problem:** Kann Recall reduzieren, wenn wichtige Issues ignoriert werden

### 3. uncertainty_policy (Haupttreiber)

**Runs:**
- `factuality_frank_tune_uncertain_policy_v1` (mit severity_min=medium)
- `factuality_frank_tune_simple_non_error_v1` (mit severity_min=low) - **Gruppe A**
- `factuality_frank_tune_simple_weight05_v1` (mit severity_min=low) - **Gruppe B**

**Optionen:**
- `count_as_error`: Uncertain zählt wie incorrect (Standard) → viele FPs
- `non_error`: Uncertain zählt NICHT für error_threshold → reduziert FPs deutlich
- `weight_0.5`: Uncertain zählt als 0.5 "Issue-Punkte" → reduziert FPs moderat

**Ziel:** Reduziert False Positives, da viele FPs "uncertain" Issues haben.

**Erwartete Effekte:**
- **Gruppe A (non_error):** Sollte Specificity deutlich erhöhen, Recall sollte hoch bleiben (>= 0.90)
- **Gruppe B (weight_0.5):** Sollte Specificity moderat erhöhen, Recall sollte hoch bleiben

### 3.5. Evidence-Gate (zentrale FP-Reduktion) ✅

**Implementiert in ClaimVerifier:**
- **Neue Regel:** "incorrect" darf nur zurückgegeben werden, wenn `evidence_found==True` UND die Evidenz einen klaren Widerspruch trägt.
- Wenn keine passende Evidenz gefunden wird: `verdict="uncertain"` (nicht "incorrect").
- **Strukturierte Evidence-Felder:**
  - `evidence_found: bool` - Wurde belastbare Evidence gefunden?
  - `evidence_spans_structured: List[EvidenceSpan]` - Strukturierte Evidence-Spans mit Position
  - `evidence_quote: str` - Kurzer Textauszug (max 1-2 Sätze)
  - `rationale: str` - Kurzer Grund für verdict

**Safety-Downgrade in FactualityAgent:**
- Falls `verdict=="incorrect"` aber `evidence_found==False`, dann downgrade zu "uncertain".
- Ziel: Selbst wenn das LLM mal "incorrect" halluziniert, kann es nicht mehr als harter Fehler durchrutschen.

**Ziel:** Reduziert False Positives signifikant, da "incorrect" nur noch mit belegter Evidenz markiert wird.

### 4. error_threshold (Haupttreiber)

**Run:** `factuality_frank_tune_simple_threshold2_v1` - **Gruppe C**

- Erhöht `error_threshold` von 1 auf 2
- Erfordert 2 Issues statt 1 für "has_error" Entscheidung
- Reduziert False Positives (weniger "single-issue" FPs)
- **Erwarteter Effekt:** Specificity sollte steigen, Recall sollte leicht sinken (aber >= 0.90)

### 5. Gewichtete Decision Logic (erweitert)

**Runs:** `factuality_frank_tune_weighted_*`

- Verwendet `decision_threshold_float` statt `error_threshold`
- Gewichtet Issues nach Severity, Confidence, Uncertainty-Policy
- Ermöglicht feinere Kontrolle über Entscheidungsschwelle
- **Ergebnis:** Bisherige gewichtete Runs haben sehr niedrigen Recall (< 0.05), da Threshold zu hoch

### 4. score_cutoff (optional)

**Nur wenn AUROC > 0.55 auf FRANK**

- `decision_mode: "score"` oder `"either"` / `"both"`
- Score-Richtung dokumentieren (score high=good oder high=bad)

## Akzeptanzkriterien

- ✅ Mindestens 3 zusätzliche FRANK-Tuning-Runs existieren
- ✅ Für jeden Run existieren metrics JSON + docs/<run_id>.md
- ✅ `summary_matrix.csv` enthält Balanced Accuracy
- ✅ FineSumFact wird erst nach Auswahl der besten FRANK-Konfig ausgeführt
- ✅ Uncertainty-Policy ist implementiert und wirkt messbar

## Dateien

- **Config:** `configs/m10_factuality_runs.yaml`
- **Runner:** `scripts/run_m10_factuality.py`
- **Baseline-Analyse:** `scripts/tune_from_baseline.py`
- **Best-Run-Auswahl:** `scripts/select_best_tuned_run.py`
- **Aggregator:** `scripts/aggregate_m10_results.py`
- **Results:** `results/evaluation/runs/results/`
- **Docs:** `results/evaluation/runs/docs/`
- **Summary:** `results/evaluation/summary_matrix.csv` + `summary.md`

