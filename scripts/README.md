# Evaluation Scripts

## Übersicht

Dieses Verzeichnis enthält Evaluationsskripte für das Verifikationssystem.

## Neue Evaluationsskripte (M10+)

### `eval_unified.py` ⭐ **Empfohlen**

Einheitliches Evaluationsskript für alle Dimensionen:
- **Factuality**: Binäre Klassifikation (FRANK, FineSumFact)
- **Coherence**: Kontinuierliche Scores (SummEval)
- **Readability**: Kontinuierliche Scores (SummEval)
- **Explainability**: Vollständiges System (alle Dimensionen)

**Verwendung:**
```bash
python scripts/eval_unified.py evaluation_configs/factuality_frank_test.json
```

Das Skript:
- Lädt Run-Config aus `evaluation_configs/`
- Führt Evaluation durch
- Speichert Ergebnisse in `results/evaluation/<dimension>/`
- Generiert Run-Summary und Predictions

## Legacy Skripte (vor M10)

### `eval_factuality_binary_v2.py`
Erweiterte Factuality-Evaluation mit vielen Features (Caching, Threshold-Sweeps, etc.)
- Wird weiterhin unterstützt für spezielle Use Cases
- Neue Evaluationen sollten `eval_unified.py` verwenden

### `eval_sumeval_coherence.py`
Coherence-Evaluation auf SummEval
- Wird durch `eval_unified.py` ersetzt

### `eval_sumeval_readability.py`
Readability-Evaluation auf SummEval
- Wird durch `eval_unified.py` ersetzt

### `archive_eval_factuality_binary_v1.py`
Veraltete Version, nur zu Archivierungszwecken

## Weitere Skripte

### `convert_*.py`
Konvertierungsskripte für Datensätze (FRANK, FineSumFact, SummEval)

### `print_runs.py`, `report_predictions.py`
Hilfsskripte für Analyse und Reporting

