# Quickstart: Factuality-Evaluation

## Schnellstart

### 1. Run definieren

Erstelle oder verwende eine Config-Datei in `evaluation_configs/`:

```bash
# Beispiel-Config bereits vorhanden:
cat evaluation_configs/factuality_frank_test_v1.json
```

### 2. Run ausführen

```bash
python scripts/eval_factuality_structured.py evaluation_configs/factuality_frank_test_v1.json
```

**Mit Explainability:**
```bash
python scripts/eval_factuality_structured.py evaluation_configs/factuality_frank_test_v1.json --include-explainability
```

### 3. Ergebnisse ansehen

**Vollständige Dokumentation:**
```bash
cat results/evaluation/runs/docs/factuality_frank_test_v1.md
```

**Metriken:**
```bash
cat results/evaluation/runs/results/factuality_frank_test_v1.json | jq .metrics
```

**Example-Level Results:**
```bash
head -1 results/evaluation/runs/results/factuality_frank_test_v1_examples.jsonl | jq .
```

### 4. Runs vergleichen

```bash
# Alle Runs auflisten
python scripts/compare_runs.py --list

# Zwei Runs vergleichen
python scripts/compare_runs.py --run1 factuality_frank_test_v1 --run2 factuality_frank_test_v2
```

## Was wird automatisch dokumentiert?

✅ **Run-Definition** - Alle Parameter (frozen)  
✅ **Execution-Log** - Status, Progress, Fehler  
✅ **Ergebnisse** - Metriken, Example-Level  
✅ **Auswertung** - Robustheit, Subsets, Fehleranalyse  
✅ **Interpretation** - Automatisch generiert  

## Nächste Schritte

1. **Parameter anpassen** → Neue Config mit `v2`
2. **Neuen Run ausführen** → Vergleichbare Ergebnisse
3. **Dokumentation erweitern** → Fallstudien hinzufügen
4. **Wiederholen** → Iterative Verbesserung

Siehe `EVALUATION_WORKFLOW.md` für Details.

