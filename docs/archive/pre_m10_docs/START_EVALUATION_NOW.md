# Evaluation jetzt starten

## Schnellstart (3 Schritte)

### 1. Dependencies installieren

```bash
cd /Users/lisasimon/PycharmProjects/veri-api

# Installiere alle Dependencies
pip3 install -r requirements.txt

# Oder falls du ein venv verwendest:
# python3 -m venv venv
# source venv/bin/activate
# pip install -r requirements.txt
```

### 2. Environment-Variablen prüfen

Stelle sicher, dass deine `.env` Datei existiert und `OPENAI_API_KEY` enthält:

```bash
# Prüfe ob .env existiert
ls -la .env

# Falls nicht, erstelle sie:
# echo "OPENAI_API_KEY=your_key_here" > .env
```

### 3. Evaluation starten

```bash
# Starte die Evaluation
python3 scripts/eval_factuality_structured.py \
  evaluation_configs/factuality_frank_test_v1.json \
  --dataset-path data/frank/frank_clean.jsonl
```

## Was passiert?

1. ✅ Run wird erstellt und dokumentiert
2. ✅ Dataset wird geladen (FRANK Test-Set)
3. ✅ Factuality-Agent evaluiert jedes Beispiel
4. ✅ Ergebnisse werden automatisch gespeichert
5. ✅ Analyse wird berechnet (Metriken, Robustheit, Subsets)
6. ✅ Vollständige Dokumentation wird generiert

## Ergebnisse

Nach Abschluss findest du alles in:

```
results/evaluation/runs/
├── docs/factuality_frank_test_v1.md          # Vollständige Dokumentation
├── results/factuality_frank_test_v1.json      # Metriken
├── results/factuality_frank_test_v1_examples.jsonl  # Example-Level
└── analyses/factuality_frank_test_v1.json     # Analyse
```

## Fortschritt

Die Evaluation zeigt Progress-Updates alle 10 Beispiele:
```
INFO: Processed 10/2247 examples
INFO: Processed 20/2247 examples
...
```

## Dauer

Abhängig von:
- Anzahl Beispiele (FRANK hat ~2247)
- Cache-Hits (schneller bei wiederholten Runs)
- LLM-Response-Zeit

**Geschätzt:** 30-60 Minuten für vollständigen Run (ohne Cache)

## Mit Cache (schneller)

Falls du bereits einen Cache hast:
- Cache wird automatisch verwendet
- Nur neue/geänderte Beispiele werden neu evaluiert
- Deutlich schneller bei wiederholten Runs

## Abbrechen und Fortsetzen

- **Abbrechen:** `Ctrl+C`
- **Fortsetzen:** Einfach erneut starten - Cache wird verwendet

## Hilfe

Bei Problemen siehe:
- `SETUP_EVALUATION.md` - Detaillierte Setup-Anleitung
- `EVALUATION_WORKFLOW.md` - Workflow-Dokumentation
- `QUICKSTART_EVALUATION.md` - Schnellstart-Guide

