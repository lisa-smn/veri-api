# Evaluation Workflow - Strukturierte Runs

## Workflow-Übersicht

Jeder Run folgt diesem strukturierten Workflow mit automatischer Dokumentation:

```
1. Run definieren (Freeze)
   ↓
2. Run ausführen (Agenten + Explainability + Baselines)
   ↓
3. Ergebnisse speichern (Example-Level + Run-Summary + Log)
   ↓
4. Auswertung berechnen (Quant + Robustheit + Subsets)
   ↓
5. Interpretieren und dokumentieren (Fallstudien + Template)
   ↓
6. Änderungen versionieren und als neuer Run wiederholen
```

## 1. Run definieren (Freeze)

**Ziel:** Alle Parameter für Reproduzierbarkeit festlegen.

**Aktion:** Erstelle oder bearbeite Config in `evaluation_configs/`:

```json
{
  "run_id": "factuality_frank_test_v1",
  "dataset": "frank",
  "split": "test",
  "dimension": "factuality",
  "llm_model": "gpt-4o-mini",
  "llm_temperature": 0.0,
  "llm_seed": 42,
  "prompt_versions": {
    "factuality": "v3_uncertain_spans"
  },
  "thresholds": {
    "error_threshold": 1
  },
  "max_examples": null,
  "cache_enabled": true,
  "description": "Beschreibung des Runs"
}
```

**Automatisch gespeichert:**
- `results/evaluation/runs/definitions/<run_id>.json` - Frozen Definition
- Definition-Hash für Vergleichbarkeit

## 2. Run ausführen

**Ziel:** Evaluation durchführen mit automatischem Logging.

**Aktion:**
```bash
python scripts/eval_factuality_structured.py evaluation_configs/factuality_frank_test_v1.json
```

**Optional mit Explainability:**
```bash
python scripts/eval_factuality_structured.py evaluation_configs/factuality_frank_test_v1.json --include-explainability
```

**Automatisch erfasst:**
- Execution-Log (`executions/<run_id>.json`)
- Progress-Tracking (processed/failed/cached)
- Fehler und Warnings

## 3. Ergebnisse speichern

**Automatisch gespeichert:**

- **Run-Summary:** `results/<run_id>.json`
  - Aggregierte Metriken
  - Anzahl Beispiele
  - Baselines (falls berechnet)
  
- **Example-Level:** `results/<run_id>_examples.jsonl`
  - Pro Beispiel: GT, Prediction, Score, Issue Spans, Explainability

- **Execution-Log:** `executions/<run_id>.json`
  - Status, Timestamps, Fehler

## 4. Auswertung berechnen

**Automatisch berechnet:**

- **Primary Metrics:** Accuracy, Precision, Recall, F1, AUROC
- **Robustness:** Threshold-Sweep, bester Threshold
- **Subsets:** Metriken pro Subset (falls Meta-Daten vorhanden)
- **Error Analysis:** FP/FN Patterns, Issue-Type-Verteilung

**Gespeichert in:** `analyses/<run_id>.json`

## 5. Interpretieren und dokumentieren

**Automatisch generiert:**

- **Vollständige Dokumentation:** `docs/<run_id>.md`
  - Run-Definition
  - Execution-Log
  - Ergebnisse
  - Analyse
  - Interpretation (automatisch generiert)

**Manuell ergänzen:**

- **Fallstudien:** Beispiele für FP/FN in Dokumentation
- **BA-Text:** Interpretation erweitern

## 6. Versionierung und Wiederholung

**Für Änderungen:**

1. Erstelle neue Config mit neuem `run_id` (z.B. `v2`)
2. Ändere Parameter (z.B. `error_threshold`, `prompt_version`)
3. Führe neuen Run aus
4. Vergleiche Runs über Definition-Hashes

**Run-Vergleich:**

```python
from app.services.run_manager import RunManager

manager = RunManager("results/evaluation")
run1 = manager.load_run("factuality_frank_test_v1")
run2 = manager.load_run("factuality_frank_test_v2")

# Vergleiche Definition-Hashes
print(f"Run 1 hash: {run1.definition.hash()}")
print(f"Run 2 hash: {run2.definition.hash()}")
```

## Best Practices

1. **Jeder Run hat eindeutige ID:** `dimension_dataset_split_v<version>`
2. **Beschreibung dokumentieren:** Was wurde geändert/untersucht?
3. **Cache nutzen:** `cache_enabled: true` für schnelle Wiederholungen
4. **Subset-Analyse:** Meta-Daten in Examples für Subset-Analyse
5. **Dokumentation erweitern:** Automatische Interpretation durch Fallstudien ergänzen

## Beispiel-Workflow

```bash
# 1. Erste Evaluation
python scripts/eval_factuality_structured.py evaluation_configs/factuality_frank_test_v1.json

# 2. Ergebnisse prüfen
cat results/evaluation/runs/docs/factuality_frank_test_v1.md

# 3. Parameter anpassen (z.B. error_threshold)
# Bearbeite evaluation_configs/factuality_frank_test_v2.json

# 4. Neuen Run ausführen
python scripts/eval_factuality_structured.py evaluation_configs/factuality_frank_test_v2.json

# 5. Vergleichen
# Öffne beide Dokumentationen und vergleiche Metriken
```

