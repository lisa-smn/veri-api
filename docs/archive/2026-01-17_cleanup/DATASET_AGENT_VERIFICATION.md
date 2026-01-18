# Datensatz- und Baseline-Verifikation: Welche Agenten nutzen welche Datensätze?

**Datum:** 2026-01-03  
**Zweck:** Systematische Prüfung, welche Datensätze/Baselines für welche Agenten (Factuality/Coherence/Readability) tatsächlich vorgesehen oder bereits implementiert sind.

---

## 1) Repo-weite Suche nach Datensatznamen

| Begriff | Datei | Zeilenbereich | Kontext |
|---------|-------|---------------|---------|
| **frank** | `configs/m10_factuality_runs.yaml` | 10, 32, 54, 78, 100, 122, 146, 168, 190, 214, 240, 266, 292, 318, 344, 370, 396, 496, 518, 540 | Dataset-Pfade für Factuality-Evaluation: `data/frank/frank_clean.jsonl` |
| **frank** | `scripts/run_m10_factuality.py` | 154-500+ | Haupt-Runner für FRANK-Evaluation |
| **frank** | `EVALUATION_DATASETS.md` | 1-76 | Dokumentation: FRANK für Factuality (binär, has_error) |
| **finesumfact** | `configs/m10_factuality_runs.yaml` | 424, 446, 473 | Dataset-Pfade für FineSumFact-Evaluation: `data/finesumfact/human_label_test_clean.jsonl` |
| **finesumfact** | `scripts/run_m10_factuality.py` | 154-500+ | Haupt-Runner für FineSumFact-Evaluation |
| **finesumfact** | `EVALUATION_DATASETS.md` | 14-76 | Dokumentation: FineSumFact für Factuality (binär, has_error) |
| **summeval** | `docs/milestones/M10_evaluation_setup.md` | 58, 122, 214 | Planung: SummEval für Human-Ratings (coherence/consistency/fluency/relevance) → Korrelation/Regression |
| **summeval** | `scripts/eval_unified.py` | 5-7, 144-148, 536, 541 | Implementiert: `load_sumeval_examples()` für Coherence/Readability (kontinuierlich) |
| **summeval** | `scripts/eval_sumeval_coherence.py` | 1-326 | Dediziertes Script: Coherence-Evaluation auf SummEval |
| **summeval** | `scripts/eval_sumeval_readability.py` | 1-324 | Dediziertes Script: Readability-Evaluation auf SummEval |
| **summeval** | `scripts/convert_sumeval.py` | 1-100+ | Konvertierungsskript für SummEval-Format |
| **summeval** | `scripts/README.md` | 13-14, 35-40 | Dokumentation: Coherence/Readability auf SummEval (kontinuierlich) |
| **summac** | `docs/milestones/M10_evaluation_setup.md` | 9, 112, 214 | Planung: SummaC als Baseline (Konsistenz/Entailment-Check) |
| **summacoz** | `docs/milestones/M10_evaluation_setup.md` | 214-215 | Planung: SummaCoz als optionaler Benchmark (Add-on) |

---

## 2) Aktuelle Datenpfade in der Evaluation

### Factuality-Agent

**Config:** `configs/m10_factuality_runs.yaml` (Zeilen 10, 424, 471-473)

**Datensätze:**
- **FRANK:** `data/frank/frank_clean.jsonl` (2246 Beispiele)
- **FineSumFact:** `data/finesumfact/human_label_test_clean.jsonl` (693 Beispiele)

**Gelesene Felder:**
- `article` (Quelltext)
- `summary` (System-Summary, die geprüft wird)
- `has_error` (Binäres Label: true/false)
- `meta` (Metadaten: hash, model_name, factuality)

**Runner:** `scripts/run_m10_factuality.py` (Zeilen 154-500+)
- Lädt JSONL, extrahiert `article`, `summary`, `has_error`
- Keine Referenz-Zusammenfassungen (siehe `DATASET_REFERENCE_CHECK.md`)

**Metriken:**
- Binäre Klassifikation: TP/FP/TN/FN, Precision, Recall, F1, Specificity, Balanced Accuracy, AUROC, MCC
- Output: `results/evaluation/runs/results/<run_id>.json` + `summary_matrix.csv`

### Coherence-Agent

**Scripts:**
- `scripts/eval_sumeval_coherence.py` (Zeilen 1-326)
- `scripts/eval_unified.py` (Zeilen 144-148, 339-370)

**Datensatz:**
- **SummEval:** `data/sumeval/sumeval_clean.jsonl` (existiert im Repo)

**Gelesene Felder:**
- `article` (Quelltext)
- `summary` (System-Summary)
- `gt.coherence` (Ground-Truth: kontinuierlicher Score, typisch 1-5)
- `meta` (Metadaten)

**Metriken:**
- Kontinuierlich: Pearson r, Spearman rho, MAE, RMSE
- Normalisierung: GT-Skala (1-5) → [0,1] für Vergleich mit Agent-Score

**Status:** ✅ **Implementiert** (Scripts vorhanden, Datensatz vorhanden)

### Readability-Agent

**Scripts:**
- `scripts/eval_sumeval_readability.py` (Zeilen 1-324)
- `scripts/eval_unified.py` (Zeilen 144-148, 339-370)

**Datensatz:**
- **SummEval:** `data/sumeval/sumeval_clean.jsonl` (existiert im Repo)

**Gelesene Felder:**
- `article` (Quelltext)
- `summary` (System-Summary)
- `gt.readability` (Ground-Truth: kontinuierlicher Score, typisch 1-5)
- `meta` (Metadaten)

**Metriken:**
- Kontinuierlich: Pearson r, Spearman rho, MAE, RMSE
- Normalisierung: GT-Skala (1-5) → [0,1] für Vergleich mit Agent-Score

**Status:** ✅ **Implementiert** (Scripts vorhanden, Datensatz vorhanden)

---

## 3) Pro-Agent Evaluation-Status

| Agent | Eval-Skript(e) | Datensatz(e) | Metriken | Status | Belege |
|-------|----------------|--------------|----------|--------|--------|
| **Factuality** | `scripts/run_m10_factuality.py` | FRANK (`data/frank/frank_clean.jsonl`), FineSumFact (`data/finesumfact/human_label_test_clean.jsonl`) | Binär: TP/FP/TN/FN, Precision, Recall, F1, Specificity, Balanced Accuracy, AUROC, MCC | ✅ **Implementiert** | `configs/m10_factuality_runs.yaml:10,424`, `scripts/run_m10_factuality.py:154-500+`, `results/evaluation/summary_matrix.csv` |
| **Coherence** | `scripts/eval_sumeval_coherence.py`, `scripts/eval_unified.py` | SummEval (`data/sumeval/sumeval_clean.jsonl`) | Kontinuierlich: Pearson r, Spearman rho, MAE, RMSE | ✅ **Implementiert** | `scripts/eval_sumeval_coherence.py:1-326`, `scripts/eval_unified.py:144-148,339-370`, `data/sumeval/sumeval_clean.jsonl` |
| **Readability** | `scripts/eval_sumeval_readability.py`, `scripts/eval_unified.py` | SummEval (`data/sumeval/sumeval_clean.jsonl`) | Kontinuierlich: Pearson r, Spearman rho, MAE, RMSE | ✅ **Implementiert** | `scripts/eval_sumeval_readability.py:1-324`, `scripts/eval_unified.py:144-148,339-370`, `data/sumeval/sumeval_clean.jsonl` |

---

## 4) SummaC / SummaCoz Anbindung

### Suche nach Imports/Abhängigkeiten

**Gefundene Treffer:**
- ❌ Keine Imports von `summac`, `summa_coz`, `NLI`, `entailment`, `consistency`, `deberta`, `roberta` in Python-Dateien
- ❌ Keine Abhängigkeiten in `requirements.txt`

**Planung in Dokumentation:**
- ✅ `docs/milestones/M10_evaluation_setup.md:9,112,214`:
  - SummaC als Baseline geplant (Konsistenz/Entailment-Check, artikel↔summary)
  - SummaCoz als optionaler Benchmark (Add-on, erst nach M10 stabil)

**Status:** ❌ **Nicht implementiert** (nur Planung in Dokumentation)

**Belegstellen:**
- `docs/milestones/M10_evaluation_setup.md:106-115`:
  ```markdown
  ## 4) Klassische Metriken als Baselines hinzufügen
  
  Für jedes Beispiel (wo möglich, abhängig von Referenzen):
  - **ROUGE** (Overlap mit Referenz)
  - **BERTScore** (semantische Ähnlichkeit mit Referenz)
  - **SummaC** (Konsistenz/Entailment-Check, artikel↔summary)
  ```
- `docs/milestones/M10_evaluation_setup.md:212-215`:
  ```markdown
  ## Optional: Zusätzliche Datenquelle (SummaCoz)
  
  Empfehlung: erst integrieren, wenn M10 mit FRANK + SummEval + FineSumFact stabil läuft.
  SummaCoz kann als Add-on sinnvoll sein, weil es Konsistenzfälle oft klarer testbar macht.
  ```

---

## 5) Zusammenfassung

### ✅ Was ist sicher belegt?

#### A) Factuality-Agent nutzt FRANK und FineSumFact (binary labels)

**Belege:**
- `configs/m10_factuality_runs.yaml:10,424` - Dataset-Pfade definiert
- `scripts/run_m10_factuality.py:154-500+` - Runner implementiert
- `data/frank/frank_clean.jsonl` - Datensatz vorhanden (2246 Zeilen)
- `data/finesumfact/human_label_test_clean.jsonl` - Datensatz vorhanden (693 Zeilen)
- `results/evaluation/summary_matrix.csv` - Metriken werden berechnet (TP/FP/TN/FN, Recall, Precision, F1, Balanced Accuracy, MCC)
- `EVALUATION_DATASETS.md:1-76` - Dokumentation bestätigt binäres Format (`has_error: true/false`)

**Status:** ✅ **Vollständig implementiert und aktiv genutzt**

#### B) Coherence-Agent soll mit SummEval evaluiert werden

**Belege:**
- `scripts/eval_sumeval_coherence.py:1-326` - Dediziertes Evaluations-Script vorhanden
- `scripts/eval_unified.py:144-148,339-370` - Einheitliches Script unterstützt Coherence auf SummEval
- `data/sumeval/sumeval_clean.jsonl` - Datensatz vorhanden
- `docs/milestones/M10_evaluation_setup.md:58,122` - Planung dokumentiert
- `scripts/README.md:13,35-36` - Dokumentation: "Coherence: Kontinuierliche Scores (SummEval)"

**Status:** ✅ **Implementiert** (Scripts vorhanden, Datensatz vorhanden, Metriken: Pearson r, Spearman rho, MAE, RMSE)

#### C) Readability-Agent soll mit SummEval evaluiert werden

**Belege:**
- `scripts/eval_sumeval_readability.py:1-324` - Dediziertes Evaluations-Script vorhanden
- `scripts/eval_unified.py:144-148,339-370` - Einheitliches Script unterstützt Readability auf SummEval
- `data/sumeval/sumeval_clean.jsonl` - Datensatz vorhanden
- `docs/milestones/M10_evaluation_setup.md:58,122` - Planung dokumentiert
- `scripts/README.md:14,39-40` - Dokumentation: "Readability: Kontinuierliche Scores (SummEval)"

**Status:** ✅ **Implementiert** (Scripts vorhanden, Datensatz vorhanden, Metriken: Pearson r, Spearman rho, MAE, RMSE)

---

### ⚠️ Was ist geplant aber noch nicht implementiert?

#### D) SummaC und SummaCoz sind als Baselines/Anbindung geplant

**Belege:**
- `docs/milestones/M10_evaluation_setup.md:9,112,214` - Planung dokumentiert:
  - SummaC: "Konsistenz/Entailment-Check, artikel↔summary"
  - SummaCoz: "Optional (Add-on), erst nach M10 stabil"
- `docs/milestones/M10_evaluation_setup.md:106-115` - Baselines sollen in Ergebnisdatei gespeichert werden

**Fehlende Implementierung:**
- ❌ Keine Imports/Abhängigkeiten in `requirements.txt`
- ❌ Keine Python-Module/Stubs für SummaC/SummaCoz
- ❌ Keine Integration in `scripts/run_m10_factuality.py` oder `scripts/eval_unified.py`
- ❌ Keine Berechnung von SummaC-Scores in Evaluations-Scripts

**Status:** ❌ **Nur Planung, nicht implementiert**

**Empfehlung:** 
- SummaC/SummaCoz sind als "klassische Metriken als Baselines" geplant, aber aktuell nicht integriert.
- Für M10-Factuality-Evaluation werden aktuell nur Agent-Scores gegen binäre Labels (FRANK/FineSumFact) evaluiert.
- SummaC würde zusätzliche Baseline-Metriken liefern, ist aber nicht zwingend für die Factuality-Evaluation.

---

### ❓ Was ist nicht auffindbar / unklar?

#### ROUGE/BERTScore als Baselines

**Status:** ❌ **Nicht implementiert** (siehe `DATASET_REFERENCE_CHECK.md`)

**Problem:**
- FRANK und FineSumFact enthalten keine Referenz-Zusammenfassungen
- ROUGE/BERTScore benötigen `summary` (System) + `reference_summary` (Gold)
- Aktuell nur `article`, `summary`, `has_error`, `meta` vorhanden

**Beleg:**
- `DATASET_REFERENCE_CHECK.md` - Vollständige Analyse: Beide Datensätze haben keine Referenz-Felder

**Optionen:**
1. Zusätzlichen Datensatz mit Referenzen verwenden (z.B. SummEval für ROUGE/BERTScore)
2. Evaluation fokussiert sich auf binäre Klassifikation (bereits implementiert)
3. ROUGE/BERTScore nur für Datensätze mit Referenzen berechnen (falls vorhanden)

---

## 6) Fazit

### Implementiert und aktiv:

1. ✅ **Factuality-Agent:** FRANK + FineSumFact (binär, `has_error`)
2. ✅ **Coherence-Agent:** SummEval (kontinuierlich, `gt.coherence`)
3. ✅ **Readability-Agent:** SummEval (kontinuierlich, `gt.readability`)

### Geplant aber nicht implementiert:

4. ❌ **SummaC/SummaCoz:** Nur in Dokumentation erwähnt, keine Code-Integration
5. ❌ **ROUGE/BERTScore:** Nicht implementiert (Datensätze enthalten keine Referenzen)

### Empfehlung:

- **Für M10-Factuality-Evaluation:** Aktueller Stand (FRANK/FineSumFact, binäre Metriken) ist vollständig.
- **Für Coherence/Readability:** SummEval-Evaluation ist implementiert, kann ausgeführt werden.
- **Für Baselines:** SummaC/ROUGE/BERTScore sind geplant, aber nicht zwingend für M10-Factuality erforderlich.






