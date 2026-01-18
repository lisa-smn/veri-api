# M10 Factuality Evaluation - Komplett-Setup

## Übersicht

Strukturierte, reproduzierbare Evaluation für Bachelorarbeit mit 6 Runs:
1. FRANK Baseline (Dev/Calibration)
2. FRANK Tuned (nach Baseline-Analyse)
3. FRANK Ablation (Claim-Extraktion deaktiviert)
4. FineSumFact Final (Test, keine Änderungen)
5. FineSumFact Ablation
6. Combined Final (FRANK + FineSumFact)

## Workflow

### Option 1: Kompletter Workflow (Empfohlen)

```bash
./scripts/run_m10_complete.sh
```

Dieses Script führt automatisch aus:
1. Baseline-Run
2. Baseline-Analyse & Tuning-Vorschläge
3. (Pause für manuelle Config-Anpassung)
4. Alle weiteren Runs
5. Aggregation

### Option 2: Schrittweise

**1. Baseline ausführen:**
```bash
python3 scripts/run_m10_factuality.py --run-id factuality_frank_baseline_v1
```

**2. Baseline analysieren:**
```bash
python3 scripts/tune_from_baseline.py --baseline-run-id factuality_frank_baseline_v1
```

**3. Config anpassen:**
Bearbeite `configs/m10_factuality_runs.yaml`:
- Setze `factuality_frank_tuned_v1.error_threshold` basierend auf Vorschlägen
- Optional: `decision_mode` anpassen

**4. Alle Runs ausführen:**
```bash
python3 scripts/run_m10_factuality.py
```

**5. Aggregation:**
```bash
python3 scripts/aggregate_m10_results.py
```

## Konfiguration

Alle Runs sind in `configs/m10_factuality_runs.yaml` definiert.

**Wichtige Parameter:**
- `error_threshold`: Anzahl Issues für "has_error" Prediction
- `decision_mode`: "issues" | "score" | "either" | "both"
- `ablation_mode`: "none" | "no_claims" | "sentence_only" | "no_spans"
- `use_claim_extraction`: Ablation-Flag
- `use_claim_verification`: Ablation-Flag
- `use_spans`: Ablation-Flag

## Ablation-Modi

- **none**: Vollständige Pipeline (Claim-Extraktion + Verifikation + Spans)
- **no_claims**: Keine LLM-basierte Claim-Extraktion (nur Satz-Fallback)
- **sentence_only**: Nur Satz-basierte Claims (noch nicht implementiert)
- **no_spans**: Keine Issue-Spans (nur Scores)

## Ergebnisse

### Pro Run

- **Metrics JSON:** `results/evaluation/runs/results/<run_id>.json`
- **Examples JSONL:** `results/evaluation/runs/results/<run_id>_examples.jsonl`
- **Dokumentation MD:** `results/evaluation/runs/docs/<run_id>.md`

Jede Dokumentation enthält:
- Dataset, N, Label-Verteilung
- Confusion Matrix
- Metriken (Precision, Recall, F1, Specificity, Accuracy, AUROC, Balanced Accuracy)
- Top-5 False Positives + Top-5 False Negatives
- Failure Pattern Analysis
- Reproducibility-Info (Commit Hash, Model, Prompt Version)

### Gesamtübersicht

- **Summary Matrix CSV:** `results/evaluation/summary_matrix.csv`
  - Alle Runs in tabellarischer Form
  - Spalten: run_id, dataset, n, pos_rate, tp, fp, tn, fn, precision, recall, f1, specificity, accuracy, auroc, balanced_accuracy

- **Summary MD:** `results/evaluation/summary.md`
  - Vergleichstabelle
  - Interpretation (Generalization, Trade-offs, Ablation-Effekt)

## Reproduzierbarkeit

Jeder Run enthält:
- Git Commit Hash (falls verfügbar)
- Vollständige Config (YAML-Dump)
- Timestamp
- Model + Prompt Version
- Alle Parameter (Temperature, Seed, Thresholds)

## Akzeptanzkriterien

✅ Ein Befehl startet alle 6 Runs nacheinander  
✅ Für jeden Run existiert metrics JSON + MD-Doku  
✅ summary_matrix.csv und summary.md werden automatisch erstellt  
✅ FineSumFact Final verwendet exakt die tuned-Konfiguration aus FRANK  

## Nächste Schritte

1. **Baseline ausführen:** `python3 scripts/run_m10_factuality.py --run-id factuality_frank_baseline_v1`
2. **Tuning analysieren:** `python3 scripts/tune_from_baseline.py`
3. **Config anpassen:** Bearbeite `configs/m10_factuality_runs.yaml`
4. **Alle Runs:** `python3 scripts/run_m10_factuality.py`
5. **Ergebnisse prüfen:** `results/evaluation/summary.md`






