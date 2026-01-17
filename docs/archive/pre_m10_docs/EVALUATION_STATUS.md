# Evaluation Status

## Aktueller Run

**Run ID:** `factuality_frank_test_v1`

**Status:** Läuft im Hintergrund

**Config:** `evaluation_configs/factuality_frank_test_v1.json`

**Dataset:** `data/frank/frank_clean.jsonl`

## Fortschritt prüfen

```bash
# Log ansehen
tail -f evaluation_run.log

# Oder direkt im Terminal
python3 scripts/eval_factuality_structured.py evaluation_configs/factuality_frank_test_v1.json --dataset-path data/frank/frank_clean.jsonl
```

## Ergebnisse

Nach Abschluss findest du die Ergebnisse in:

- **Dokumentation:** `results/evaluation/runs/docs/factuality_frank_test_v1.md`
- **Metriken:** `results/evaluation/runs/results/factuality_frank_test_v1.json`
- **Examples:** `results/evaluation/runs/results/factuality_frank_test_v1_examples.jsonl`
- **Analyse:** `results/evaluation/runs/analyses/factuality_frank_test_v1.json`

## Run abbrechen

```bash
# Finde den Prozess
ps aux | grep eval_factuality_structured

# Beende den Prozess
kill <PID>
```

