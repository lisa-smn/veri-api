# Evaluation starten - Einfachste Methode

## Option 1: Script ausführen (Empfohlen)

Einfach im Terminal:

```bash
cd /Users/lisasimon/PycharmProjects/veri-api
./START_EVALUATION_BA.sh
```

Das Script macht alles automatisch:
- Prüft Python
- Prüft Config und Dataset
- Zeigt Cache-Status
- Startet Evaluation
- Zeigt Ergebnisse am Ende

## Option 2: Befehl direkt kopieren

Falls du den Befehl direkt ausführen möchtest:

```bash
cd /Users/lisasimon/PycharmProjects/veri-api

python3 scripts/eval_factuality_structured.py \
  evaluation_configs/factuality_frank_test_v1.json \
  --dataset-path data/frank/frank_clean.jsonl
```

**Oder in einer Zeile:**
```bash
python3 scripts/eval_factuality_structured.py evaluation_configs/factuality_frank_test_v1.json --dataset-path data/frank/frank_clean.jsonl
```

## Was ist einfacher?

**Script (Option 1):**
- ✅ Automatische Prüfungen
- ✅ Zeigt Cache-Status
- ✅ Zeigt Ergebnisse am Ende
- ✅ Fehlerbehandlung

**Direkter Befehl (Option 2):**
- ✅ Einfacher Copy-Paste
- ✅ Direkter Start

**Empfehlung:** Option 1 (Script) - mehr Komfort und Info

