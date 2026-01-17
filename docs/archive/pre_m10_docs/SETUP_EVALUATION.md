# Setup für Evaluation

## Dependencies installieren

Bevor du die Evaluation starten kannst, müssen die Python-Dependencies installiert sein:

```bash
# Im Projektverzeichnis
cd /Users/lisasimon/PycharmProjects/veri-api

# Dependencies installieren
python3 -m pip install -r requirements.txt

# Oder mit venv (empfohlen)
python3 -m venv venv
source venv/bin/activate  # Auf Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Environment-Variablen

Stelle sicher, dass deine `.env` Datei die OpenAI API-Keys enthält:

```bash
# .env Datei sollte enthalten:
OPENAI_API_KEY=your_api_key_here
```

## Evaluation starten

Nach Installation der Dependencies:

```bash
# Option 1: Direkt
python3 scripts/eval_factuality_structured.py evaluation_configs/factuality_frank_test_v1.json --dataset-path data/frank/frank_clean.jsonl

# Option 2: Mit Script
./START_EVALUATION.sh
```

## Troubleshooting

**ModuleNotFoundError:**
- Stelle sicher, dass alle Dependencies installiert sind: `pip install -r requirements.txt`
- Prüfe ob du im richtigen venv bist (falls verwendet)

**API Key Error:**
- Prüfe `.env` Datei
- Stelle sicher, dass `OPENAI_API_KEY` gesetzt ist

**Dataset nicht gefunden:**
- Prüfe ob `data/frank/frank_clean.jsonl` existiert
- Verwende `--dataset-path` um den Pfad anzugeben

