# Datensatz-Prüfung: Referenz-Zusammenfassungen für ROUGE/BERTScore

**Datum:** 2026-01-03  
**Zweck:** Prüfen, ob die M10-Evaluations-Datensätze Referenz-Zusammenfassungen enthalten, die für ROUGE/BERTScore benötigt werden.

---

## Verwendete Datendateien (M10)

**Quelle:** `configs/m10_factuality_runs.yaml` + `scripts/run_m10_factuality.py`

1. **FRANK:** `data/frank/frank_clean.jsonl` (2246 Zeilen)
2. **FineSumFact:** `data/finesumfact/human_label_test_clean.jsonl` (693 Zeilen)

---

## Übersichtstabelle

| Datei | Total | With Reference | Reference Key | Beispiel-Keys |
|-------|-------|----------------|---------------|---------------|
| `data/frank/frank_clean.jsonl` | 2246 | 0/100 | ❌ None | article, summary, has_error, meta |
| `data/finesumfact/human_label_test_clean.jsonl` | 693 | 0/100 | ❌ None | article, summary, has_error, meta |

---

## Detaillierte Analyse

### FRANK

**Datei:** `data/frank/frank_clean.jsonl`  
**Total Examples:** 2246  
**Keys:** `['article', 'summary', 'has_error', 'meta']`  
**Reference Key:** ❌ `None` (keine Referenz gefunden)

**Beispiel (erste Zeile, gekürzt):**
```json
{
  "article": "Police in Arkansas wish to unlock an iPhone and iPod belonging to two teenagers accused of killing a couple...",
  "summary": "the fbi has said it will help the san bernardino killer to access iphones used by the san bernardino victims.",
  "has_error": true,
  "meta": {
    "hash": "35933239",
    "model_name": "BERTS2S",
    "factuality": 0.0
  }
}
```

**Befund:** FRANK enthält nur die System-Summary (`summary`), keine Referenz-Zusammenfassung.

---

### FineSumFact

**Datei:** `data/finesumfact/human_label_test_clean.jsonl`  
**Total Examples:** 693  
**Keys:** `['article', 'summary', 'has_error', 'meta']`  
**Reference Key:** ❌ `None` (keine Referenz gefunden)

**Beispiel (erste Zeile, gekürzt):**
```json
{
  "article": "MICHELE NORRIS, host: To learn a little bit more about the Quds Force that President Bush referred to...",
  "summary": "Quds Force is a branch of the Revolutionary Guards in Iran responsible for intelligence and military operations in Iraq...",
  "has_error": false,
  "meta": {
    "dataset": "finesumfact",
    "label_source": "human",
    "source": "mediasum_test",
    "model": null,
    "split": null,
    "n_sent_labels": 2,
    "n_error_sent": 0
  }
}
```

**Befund:** FineSumFact enthält nur die System-Summary (`summary`), keine Referenz-Zusammenfassung.

---

## Fazit

❌ **ROUGE/BERTScore nicht möglich ohne zusätzlichen Datensatz**

**Aktueller Stand:**
- **FRANK:** Enthält nur `article`, `summary` (System-Summary), `has_error` (Label), `meta`
- **FineSumFact:** Enthält nur `article`, `summary` (System-Summary), `has_error` (Label), `meta`
- **Beide Datensätze haben keine Referenz-Zusammenfassungen (Gold-Standard)**

**Was fehlt für ROUGE/BERTScore:**
- ROUGE benötigt: `summary` (System) + `reference_summary` (Gold)
- BERTScore benötigt: `summary` (System) + `reference_summary` (Gold)

**Optionen:**

1. **Zusätzlichen Datensatz mit Referenzen verwenden:**
   - SummEval (enthält Referenzen)
   - XSum (enthält Referenzen)
   - CNN/DailyMail (enthält Highlights als Referenzen)

2. **Referenzen manuell erstellen:**
   - Aufwändig, nicht praktikabel für 2246+693 Beispiele

3. **ROUGE/BERTScore nur für Datensätze mit Referenzen berechnen:**
   - Falls ein zusätzlicher Datensatz mit Referenzen verwendet wird
   - Aktuelle M10-Datensätze (FRANK, FineSumFact) können nicht verwendet werden

**Empfehlung:**  
Für einen vollständigen Vergleich mit ROUGE/BERTScore sollte ein zusätzlicher Datensatz mit Referenz-Zusammenfassungen in die Evaluation aufgenommen werden (z.B. SummEval), oder die Evaluation fokussiert sich auf die binäre Klassifikation (has_error), die bereits implementiert ist.






