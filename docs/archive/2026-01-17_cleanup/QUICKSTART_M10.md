# Quickstart: M10 Evaluation

## Schnellstart (3 Schritte)

### 1. Dependencies installieren

```bash
pip3 install -r requirements.txt
```

### 2. Kompletten Workflow starten

```bash
./scripts/run_m10_complete.sh
```

Das Script führt automatisch aus:
- Baseline-Run
- Baseline-Analyse
- (Pause für Config-Anpassung)
- Alle weiteren Runs
- Aggregation

### 3. Ergebnisse ansehen

```bash
# Summary-Matrix (CSV)
cat results/evaluation/summary_matrix.csv

# Summary-Report (Markdown)
cat results/evaluation/summary.md

# Einzelne Run-Dokumentationen
ls results/evaluation/runs/docs/
```

## Was wird erstellt?

### Pro Run (6 Runs)
- `results/evaluation/runs/results/<run_id>.json` - Metriken
- `results/evaluation/runs/results/<run_id>_examples.jsonl` - Example-Level
- `results/evaluation/runs/docs/<run_id>.md` - Vollständige Dokumentation

### Gesamtübersicht
- `results/evaluation/summary_matrix.csv` - Alle Metriken tabellarisch
- `results/evaluation/summary.md` - Interpretation & Vergleich

## Run-Übersicht

1. **factuality_frank_baseline_v1** - FRANK Baseline
2. **factuality_frank_tuned_v1** - FRANK Tuned (nach Baseline)
3. **factuality_frank_ablation_v1** - FRANK Ablation
4. **factuality_finesumfact_final_v1** - FineSumFact Final
5. **factuality_finesumfact_ablation_v1** - FineSumFact Ablation
6. **factuality_combined_final_v1** - Combined Final

## Für BA-Text

Die automatisch generierten Dokumente enthalten:
- ✅ Alle Metriken (tabellarisch)
- ✅ Confusion Matrices
- ✅ Top FP/FN Beispiele
- ✅ Failure Pattern Analysis
- ✅ Interpretation (Generalization, Trade-offs, Ablation)

**Verwendung:**
- `summary.md` → Haupt-Ergebnisse für BA
- `summary_matrix.csv` → Tabellen für BA
- Einzelne Run-Dokus → Details & Fallstudien






