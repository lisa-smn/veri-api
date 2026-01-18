# Projekt-Status: M10 Factuality Evaluation

**Stand:** 28. Dezember 2025

## ✅ Implementiert

### 1. M10-Evaluation-Infrastruktur

**Scripts:**
- ✅ `scripts/run_m10_factuality.py` - Haupt-Runner für alle M10-Runs
- ✅ `scripts/aggregate_m10_results.py` - Aggregiert Ergebnisse zu Summary-Matrix
- ✅ `scripts/tune_from_baseline.py` - Baseline-Analyse mit FP-Issue-Types
- ✅ `scripts/select_best_tuned_run.py` - Wählt besten Tuned Run (robuste Gates: recall + specificity, MCC/Balanced Acc)
  - Persistiert Auswahl-Metadaten als JSON (`best_run_selection.json`)
  - Generiert Top-5 Tabelle für Transparenz (`best_run_top5.csv`)
  - Fallback-Begründung explizit (Kandidatenmenge, Zielmetrik, Tie-Breaker)
  - Gate-Status niemals gelogen (nur ✅ wenn wirklich erfüllt)
- ✅ `scripts/run_m10_complete.sh` - Kompletter Workflow-Script

**Config:**
- ✅ `configs/m10_factuality_runs.yaml` - Alle 6+ Runs definiert (Baseline, Tuned, Ablation, Tuning-Varianten)

**Dokumentation:**
- ✅ `M10_EVALUATION.md` - Hauptdokumentation
- ✅ `M10_TUNING_WORKFLOW.md` - Tuning-Workflow
- ✅ `M10_IMPLEMENTATION_SUMMARY.md` - Implementierungs-Übersicht
- ✅ `QUICKSTART_M10.md` - Schnellstart
- ✅ `results/README.md` - Results-Struktur

### 2. Tuning-Strategien implementiert

**Erweiterte Tuning-Parameter:**
- ✅ `severity_min` - Filtert nach Issue-Severity (low/medium/high)
- ✅ `ignore_issue_types` - Ignoriert "noisy" Issue Types
- ✅ `uncertainty_policy` - 3 Modi:
  - `count_as_error` - Uncertain zählt wie incorrect
  - `non_error` - Uncertain zählt NICHT
  - `weight_0.5` - Uncertain zählt als 0.5 Issue-Punkte
- ✅ `decision_mode` - issues/score/either/both

**Evidence-Gate (zentrale FP-Reduktion):**
- ✅ **ClaimVerifier:** "incorrect" nur wenn `evidence_found==True` UND klarer Widerspruch
- ✅ **Safety-Downgrade:** Falls `verdict=="incorrect"` aber `evidence_found==False` → downgrade zu "uncertain"
- ✅ **Strukturierte Evidence-Felder:** `evidence_spans_structured`, `evidence_quote`, `rationale`
- ✅ **Ziel:** Reduziert False Positives signifikant, da "incorrect" nur noch mit belegter Evidenz markiert wird

**Ablation-Studien:**
- ✅ `use_claim_extraction` - Claim-Extraktion deaktivierbar
- ✅ `use_claim_verification` - Claim-Verifikation deaktivierbar
- ✅ `use_spans` - Issue-Spans deaktivierbar

### 3. Automatische Dokumentation

- ✅ Pro Run: Metrics JSON + Examples JSONL + Markdown-Doku
- ✅ Summary-Matrix (CSV) mit allen Metriken
- ✅ Summary-MD mit Interpretation
- ✅ Reproduzierbarkeit: Commit Hash, Timestamp, Config-Dump

### 4. Results-Verzeichnis aufgeräumt

- ✅ Alte Test-Runs archiviert (`results/archive/pre_m10_runs/`)
- ✅ Struktur klar: `results/evaluation/runs/` für M10-Runs
- ✅ Cache-Dateien organisiert

## 📊 Aktuelle Ergebnisse

### Abgeschlossene Runs

**FRANK (Dev/Calibration):**
1. ✅ `factuality_frank_baseline_v1` - Baseline (abgeschlossen)
2. ✅ `factuality_frank_tuned_v1` - Tuned (abgeschlossen)
3. ✅ `factuality_frank_ablation_v1` - Ablation (abgeschlossen)
4. ✅ `factuality_frank_tune_severity_v1` - severity_min=medium (abgeschlossen)
5. ✅ `factuality_frank_tune_ignore_types_v1` - ignore_issue_types (abgeschlossen)
6. ✅ `factuality_frank_tune_uncertain_policy_v1` - uncertainty_policy=non_error (abgeschlossen)

**FineSumFact (Test):**
7. ✅ `factuality_finesumfact_final_v1` - Final (abgeschlossen)
8. ✅ `factuality_finesumfact_ablation_v1` - Ablation (abgeschlossen)

**Combined:**
9. ✅ `factuality_combined_final_v1` - Combined Final (abgeschlossen)

### Beste Ergebnisse (FRANK)

**Baseline:**
- Balanced Accuracy: 0.508
- Recall: 0.958
- Specificity: 0.057 (sehr niedrig!)

**Tuned (aktuell selektiert, aber problematisch):**
- Balanced Accuracy: 0.508
- Recall: 0.958
- Specificity: 0.057 (katastrophal niedrig!)
- **Problem:** Praktisch "alles ist ein Fehler" - Specificity von 0.057 bedeutet, dass nur 5.7% der korrekten Summaries als korrekt erkannt werden.

**Warum der alte Best-Run unsinnig war:**
- `factuality_frank_tuned_v1` wurde basierend nur auf Balanced Accuracy + Recall-Constraint ausgewählt
- Kein Specificity-Gate → Run mit extrem niedriger Specificity (0.057) wurde gewählt
- Folge: System markiert fast alle Summaries als fehlerhaft, auch korrekte

**Neue Auswahlregel (implementiert):**
- Gate 1: `recall >= 0.90` (wie bisher)
- Gate 2: `specificity >= 0.20` (NEU - verhindert katastrophal niedrige Specificity)
- Optimierungsziel: `mcc` (Matthews Correlation Coefficient) ODER `balanced_accuracy` (wählbar)
- Tie-breaker: precision, dann f1

**Tuning-Varianten (alte):**
- `severity_min=medium`: Balanced Acc 0.489, Recall 0.064 (zu niedrig!)
- `ignore_types`: Balanced Acc 0.484, Recall 0.053 (zu niedrig!)
- `uncertainty_policy=non_error` (mit severity_min=medium): Balanced Acc 0.475, Recall 0.064 (zu niedrig!)

**Neue gezielte Tuning-Runs (Gruppe A/B/C):**
- Gruppe A: `uncertainty_policy=non_error`, `severity_min=low`, `error_threshold=1`
- Gruppe B: `uncertainty_policy=weight_0.5`, `severity_min=low`, `error_threshold=1`
- Gruppe C: `uncertainty_policy=count_as_error`, `severity_min=low`, `error_threshold=2`

### FineSumFact (Test-Set)

**Final:**
- Balanced Accuracy: 0.523
- Recall: 1.0
- Specificity: 0.046 (sehr niedrig!)

**Ablation:**
- Balanced Accuracy: 0.505
- Recall: 1.0
- Specificity: 0.009 (extrem niedrig!)

## 🔄 Nächste Schritte

### 1. Tuning-Strategien anpassen

**Problem identifiziert:**
- `severity_min=medium` filtert zu aggressiv → Recall bricht ein
- Viele False Positives haben "low" severity Issues
- Uncertainty-Policy allein hilft nicht genug

**Mögliche Lösungen:**
- ✅ Kombination: `severity_min=low` + `uncertainty_policy=non_error`
- ✅ Kombination: `severity_min=low` + selektive `ignore_issue_types`
- ✅ `uncertainty_policy=weight_0.5` testen
- ✅ Score-basierte Entscheidungen (wenn AUROC > 0.55)

### 2. Weitere Tuning-Runs

**✅ Implementiert (Gruppe A/B/C):**
```yaml
- factuality_frank_tune_simple_non_error_v1
  severity_min: "low"
  uncertainty_policy: "non_error"
  error_threshold: 1
  
- factuality_frank_tune_simple_weight05_v1
  severity_min: "low"
  uncertainty_policy: "weight_0.5"
  error_threshold: 1
  
- factuality_frank_tune_simple_threshold2_v1
  severity_min: "low"
  uncertainty_policy: "count_as_error"
  error_threshold: 2
```

### 3. Best Run auswählen

```bash
python3 scripts/select_best_tuned_run.py --recall-min 0.90 --specificity-min 0.20 --target mcc
```

**Neue Kriterien (robust):**
- Gate 1: `recall >= 0.90` (Constraint)
- Gate 2: `specificity >= 0.20` (NEU - verhindert katastrophal niedrige Specificity)
- Optimierungsziel: `mcc` (Matthews Correlation Coefficient) ODER `balanced_accuracy` (wählbar)
- Tie-breaker: precision, dann f1

**Output:**
- Confusion Matrix (TP/TN/FP/FN)
- Begründung warum der Run gewählt wurde
- YAML-Snippet für FineSumFact final (inkl. `decision_threshold_float`, `severity_weights` falls vorhanden)

### 4. FineSumFact Final anpassen

Nach Auswahl des besten FRANK-Runs:
- Config `factuality_finesumfact_final_v1` mit best-tuned Config aktualisieren
- FineSumFact Final erneut ausführen (falls nötig)

## 📁 Projekt-Struktur

```
veri-api/
├── configs/
│   └── m10_factuality_runs.yaml      # Alle Run-Configs
├── scripts/
│   ├── run_m10_factuality.py         # Haupt-Runner
│   ├── aggregate_m10_results.py      # Aggregator
│   ├── tune_from_baseline.py         # Baseline-Analyse
│   ├── select_best_tuned_run.py      # Best-Run-Auswahl
│   └── run_m10_complete.sh           # Kompletter Workflow
├── results/
│   ├── evaluation/                   # Aktuelle M10-Evaluation
│   │   ├── runs/                     # Run-Management
│   │   ├── summary_matrix.csv        # Aggregierte Metriken
│   │   └── summary.md                # Summary-Dokumentation
│   └── archive/                      # Archivierte Daten
└── docs/                             # Dokumentation
```

## 🎯 Akzeptanzkriterien Status

- ✅ M10-Infrastruktur vollständig implementiert
- ✅ Tuning-Strategien implementiert (severity, ignore_types, uncertainty_policy)
- ✅ Automatische Dokumentation pro Run
- ✅ Summary-Matrix mit Balanced Accuracy
- ✅ FineSumFact als reines Test-Set (keine Parameteränderungen nach FRANK)
- ✅ **Robuste Best-Run-Auswahl implementiert** - Specificity-Gate verhindert katastrophal niedrige Specificity
- ✅ **Neue gezielte Tuning-Runs** - Gruppe A/B/C (einfache Parameter-Kombinationen)
- 🔄 **Evaluation läuft** - Neue Runs werden ausgeführt und aggregiert

## 📝 Wichtige Hinweise

1. **FRANK = Dev/Calibration** - Hier wird getuned
2. **FineSumFact = Test-Set** - Keine Parameteränderungen nach FRANK!
3. **Optimierungsmetrik:** Balanced Accuracy (wegen unbalanced Klassen)
4. **Constraint:** Recall >= 0.90 (nicht komplett implodieren)
5. **Problem:** Specificity sehr niedrig (viele False Positives)

## 🚀 Quick Start

```bash
# 1. Baseline-Analyse
python3 scripts/tune_from_baseline.py

# 2. Neue Tuning-Runs ausführen
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml --skip-baseline

# 3. Aggregation
python3 scripts/aggregate_m10_results.py

# 4. Best Run auswählen
python3 scripts/select_best_tuned_run.py
```

