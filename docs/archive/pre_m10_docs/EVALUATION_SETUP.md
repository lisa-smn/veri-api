# Evaluation Setup - M10

## ✅ Abgeschlossen: Projektstruktur aufgeräumt

### 1. Archivierung alter Evaluation-Runs
- Alle alten Runs wurden nach `results/archive/pre_m10_evaluation/` verschoben
- Enthält: frank/, finesumfact/, sumeval/, other/, tables/
- README dokumentiert den Inhalt

### 2. Projektstruktur bereinigt
- Doppelte `dashboard/` Verzeichnis entfernt (nur `app/dashboard/` bleibt)
- Veraltete Skripte archiviert (`eval_factuality_binary.py` → `archive_eval_factuality_binary_v1.py`)

### 3. Neue Evaluationsstruktur erstellt

#### Verzeichnisse
```
results/
├── archive/                    # Alte Runs (vor M10)
│   └── pre_m10_evaluation/
└── evaluation/                 # Neue strukturierte Evaluation
    ├── factuality/
    ├── coherence/
    ├── readability/
    └── explainability/

evaluation_configs/             # Run-Konfigurationen
├── factuality_frank_test.json
├── coherence_sumeval_test.json
├── readability_sumeval_test.json
└── explainability_full_test.json
```

#### Neues einheitliches Evaluationsskript
- `scripts/eval_unified.py`: Unterstützt alle Dimensionen
  - Factuality (binär)
  - Coherence (kontinuierlich)
  - Readability (kontinuierlich)
  - Explainability (vollständiges System)

## 🚀 Nächste Schritte: Evaluation durchführen

### 1. Einzelne Dimensionen evaluieren

#### Factuality (FRANK)
```bash
python scripts/eval_unified.py evaluation_configs/factuality_frank_test.json
```

#### Coherence (SummEval)
```bash
python scripts/eval_unified.py evaluation_configs/coherence_sumeval_test.json
```

#### Readability (SummEval)
```bash
python scripts/eval_unified.py evaluation_configs/readability_sumeval_test.json
```

### 2. Vollständiges System mit Explainability evaluieren

```bash
python scripts/eval_unified.py evaluation_configs/explainability_full_test.json
```

### 3. Run-Configs anpassen

Die Config-Dateien in `evaluation_configs/` können angepasst werden:
- `max_examples`: Anzahl Beispiele (null = alle)
- `llm_model`: Modell (z.B. "gpt-4o-mini")
- `thresholds`: Entscheidungskriterien
- `cache_enabled`: Caching aktivieren/deaktivieren

## 📊 Ergebnisse

Ergebnisse werden gespeichert in:
- `results/evaluation/<dimension>/run_<run_id>_<timestamp>.json` - Run-Summary mit Metriken
- `results/evaluation/<dimension>/predictions_<run_id>_<timestamp>.jsonl` - Per-Example Predictions
- `results/evaluation/<dimension>/cache_<model>_<version>.jsonl` - LLM-Cache (optional)

## 📝 Notizen

- Alle Runs sind reproduzierbar durch Run-Configs
- Cache-Dateien ermöglichen schnelle Wiederholungen ohne neue LLM-Calls
- Legacy-Skripte bleiben verfügbar für spezielle Use Cases

