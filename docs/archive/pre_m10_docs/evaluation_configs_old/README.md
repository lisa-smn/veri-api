# Evaluation Configurations

Dieses Verzeichnis enthält Run-Konfigurationen für die strukturierte M10-Evaluation.

## Struktur

Jede Config-Datei definiert:
- Dataset und Split
- LLM-Modell und Settings
- Prompt-Versionen pro Agent
- Explainability-Version
- Thresholds und Entscheidungskriterien
- Anzahl Beispiele

## Format

JSON-Format mit folgender Struktur:

```json
{
  "run_id": "unique_run_identifier",
  "dataset": "frank|finesumfact|sumeval",
  "split": "test|train|dev",
  "dimension": "factuality|coherence|readability|all",
  "llm_model": "gpt-4o-mini",
  "llm_temperature": 0.0,
  "llm_seed": 42,
  "prompt_versions": {
    "factuality": "v3_uncertain_spans",
    "coherence": "v1",
    "readability": "v1"
  },
  "explainability_version": "m9_v1",
  "thresholds": {
    "error_threshold": 1,
    "score_cutoff": null,
    "severity_min": "low"
  },
  "max_examples": null,
  "cache_enabled": true,
  "description": "Beschreibung des Runs"
}
```

## Verwendung

Configs werden von den Evaluationsskripten geladen und bestimmen alle Parameter eines Runs.

