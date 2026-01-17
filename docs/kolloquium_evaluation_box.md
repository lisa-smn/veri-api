# Evaluation-Box für Factuality (Kolloquium)

## Schritt 1: Gefundene Dateien

### FRANK (Dev/Calibration)
**Datei:** `results/evaluation/runs/results/factuality_frank_tuned_v1.json`
- Enthält: Metriken, n=300, dataset="frank"
- **Alternative:** `results/evaluation/selection/best_run_frank_mcc.json` (best_run.best_run enthält Metriken)

### FineSumFact (Test)
**Datei:** `results/evaluation/runs/results/factuality_finesumfact_final_v1.json`
- Enthält: Metriken, n=200, dataset="finesumfact"

**Empfehlung:** Verwende die Run-Results JSONs (direkter Zugriff auf `metrics`), nicht die Selection-JSON (nur Best-Run-Auswahl).

---

## Schritt 2: Werte extrahieren

### Option A: jq One-Liner

**FRANK:**
```bash
cd /Users/lisasimon/PycharmProjects/veri-api
jq '{
  dataset: .config.dataset,
  n: .n,
  precision: .metrics.precision,
  recall: .metrics.recall,
  f1: .metrics.f1,
  balanced_accuracy: .metrics.balanced_accuracy,
  tp: .metrics.tp,
  fp: .metrics.fp,
  tn: .metrics.tn,
  fn: .metrics.fn
}' results/evaluation/runs/results/factuality_frank_tuned_v1.json
```

**FineSumFact:**
```bash
jq '{
  dataset: .config.dataset,
  n: .n,
  precision: .metrics.precision,
  recall: .metrics.recall,
  f1: .metrics.f1,
  balanced_accuracy: .metrics.balanced_accuracy,
  tp: .metrics.tp,
  fp: .metrics.fp,
  tn: .metrics.tn,
  fn: .metrics.fn
}' results/evaluation/runs/results/factuality_finesumfact_final_v1.json
```

### Option B: Python-Script (empfohlen für Screenshot)

**Ausführen:**
```bash
cd /Users/lisasimon/PycharmProjects/veri-api
python3 scripts/print_factuality_metrics.py
```

**Oder einzeln:**
```bash
python3 scripts/print_factuality_metrics.py frank
python3 scripts/print_factuality_metrics.py finesumfact
```

**Output-Format:**
```
============================================================
Factuality Evaluation: FRANK
============================================================
Dataset:     frank
Run ID:      factuality_frank_tuned_v1
n:           300

Metrics:
  Precision:          0.885
  Recall:             0.958
  F1:                 0.920
  Balanced Accuracy:  0.508

Confusion Matrix:
  TP: 254  FP: 33
  FN: 11   TN: 2
============================================================
```

---

## Schritt 3: Screenshot-Plan

### Empfohlener Screenshot-Typ: Terminal-Ausgabe (Option 1)

**Warum:** Sauber formatiert, laienverständlich, zeigt alle wichtigen Werte auf einen Blick.

**Was sichtbar sein muss:**
- Titelzeile: "Factuality Evaluation: [DATASET]"
- Dataset-Name (FRANK oder FineSumFact)
- n (Anzahl Beispiele, z.B. 300)
- 4 Kennzahlen: Precision, Recall, F1, Balanced Accuracy (alle mit 3 Dezimalstellen)
- Optional: Confusion Matrix (TP/FP/FN/TN)

**Dateiname:**
```
99_backup_factuality_metrics_frank.png
99_backup_factuality_metrics_finesumfact.png
```

**Alternative (Option 2): JSON im Editor**
- Öffne `results/evaluation/runs/results/factuality_frank_tuned_v1.json`
- Markiere nur den `metrics`-Block (Zeilen 3-14)
- Dateiname: `99_backup_factuality_metrics_json.png`

---

## Schritt 4: Backup-Folie (Anhang)

### Titel:
**"Factuality Evaluation: Ergebnisse"**

### Bullets (max 4):

1. **Factuality evaluiert auf 2 Datensätzen:**
   - FRANK (Dev/Calibration): n=300, Recall 0.958, F1 0.920, Balanced Accuracy 0.508
   - FineSumFact (Test): n=200, Recall 1.0, F1 0.641, Balanced Accuracy 0.523

2. **Stärken:** Hoher Recall (0.958-1.0) – System findet fast alle Fehler. F1-Score auf FRANK sehr gut (0.920).

3. **Schwächen:** Niedrige Specificity (0.057 auf FRANK, 0.046 auf FineSumFact) – viele False Positives. Precision auf FineSumFact niedrig (0.472) aufgrund unbalancierter Klassen.

4. **Nächste Schritte:** Coherence/Readability Evaluation auf SummEval, Baseline-Vergleich (ROUGE/BERTScore), Error-Profile-Analyse zur FP-Reduktion.

---

## Zusammenfassung

**Gefundene Dateien:**
1. `results/evaluation/runs/results/factuality_frank_tuned_v1.json` (FRANK)
2. `results/evaluation/runs/results/factuality_finesumfact_final_v1.json` (FineSumFact)

**Extraktions-Commands:**
- Python: `python3 scripts/print_factuality_metrics.py` (beide) oder `python3 scripts/print_factuality_metrics.py frank`
- jq: Siehe Option A oben

**Empfohlener Screenshot:**
- Terminal-Ausgabe von `print_factuality_metrics.py`
- Zeigt: Dataset, n, Precision, Recall, F1, Balanced Accuracy, Confusion Matrix

**Backup-Folie:**
- Titel + 4 Bullets (siehe oben)





