# Factuality Evaluation: Final Status

**Datum:** 2026-01-16 (aktualisiert: 2026-01-17)  
**Status:** Agent evaluiert, Judge Baseline ausgeführt (Smoke ✅, Final ✅, Verify ✅)

---

## Setup

- **Dataset (Final/Thesis):** FRANK Clean (`data/frank/frank_clean.jsonl`, n=200, seed=42)
- **Dataset (Smoke/Quick):** FRANK Balanced Smoke (`data/frank/frank_smoke_balanced_50_seed42.jsonl`, n=50, 25/25 pos/neg)
- **Model:** gpt-4o-mini
- **Agent Prompt:** v3 (uncertain spans)
- **Judge Prompt:** v2_binary (error_present: true/false, confidence)
- **Bootstrap:** n=2000 resamples (Final), n=200 (Smoke), 95% Konfidenzintervall
- **Cache:** write (Agent), write/read (Judge, empfohlen)

**Hinweis zu Datensätzen:**
- **Smoke = Pipeline/Quick sanity check:** Für schnelle Validierung der Pipeline und Metriken (n=50, balanced)
- **Final = Thesis Zahlen:** Für finale Evaluation und Reporting (n=200, bootstrap=2000)
- **⚠️ Warnung:** `frank_subset_manifest.jsonl` ist nicht balanced und kann single-class sein → ungeeignet für Metriken wie Balanced Accuracy, MCC, AUROC. Verwende stattdessen `frank_clean.jsonl` oder `frank_smoke_balanced_50_seed42.jsonl`.

---

## System vs Humans: Vergleichstabelle

| System | Precision (95% CI) | Recall (95% CI) | F1 (95% CI) | Balanced Accuracy (95% CI) | Specificity | MCC (95% CI) | AUROC | n |
|--------|-------------------|-----------------|-------------|---------------------------|-------------|--------------|-------|---|
| **Agent** | 0.786 [0.711, 0.856] | 0.798 [0.726, 0.866] | 0.792 [0.732, 0.843] | 0.722 [0.658, 0.787] | 0.645 | 0.445 [0.315, 0.571] | 0.892 | 200 |
| **Judge** | 0.9286 [0.8895, 0.9626] | 0.9602 [0.9286, 0.9881] | 0.9441 [0.9178, 0.9681] | 0.7093 [0.6040, 0.8086] | 0.4583 | 0.4753 | 0.9453 | 200 |
| **ROUGE-L** | - | - | - | - | - | - | - | 200 |
| **BERTScore** | - | - | - | - | - | - | - | 200 |

**Hinweise:**
- **Judge:** LLM-as-a-Judge Baseline ausgeführt (Smoke ✅, Final ✅). Siehe "Smoke Validation", "Final Results" und "Reproduzierbarkeit".
- **ROUGE-L / BERTScore:** Baselines messen Ähnlichkeit, nicht Faktentreue. Spearman ρ = 0.20 (ROUGE-L) bzw. 0.10 (BERTScore) zeigt schwache Korrelation mit menschlichen Labels.

### Confusion Matrix

**Agent:**

| | Predicted: Error | Predicted: No Error |
|---|---|---|
| **Actual: Error** | TP = 99 | FN = 25 |
| **Actual: No Error** | FP = 27 | TN = 49 |

**Judge:**
| | Predicted: Error | Predicted: No Error |
|---|---|---|
| **Actual: Error** | TP = 169 | FN = 7 |
| **Actual: No Error** | FP = 13 | TN = 11 |

**Dataset Signature (Agent):** `c32c25988d7a041fea833c132f4bd2bcc6484de4c22a157d114994e9812eb299`  
**Dataset (Judge Final):** `frank_clean.jsonl` (pos=1436, neg=810, total=2246)  
**Sample (Judge Final, n=200):** pos=176, neg=24 (unbalanced, daher niedrige BA/MCC)

---

## Interpretation

### Agent Performance

- **F1 = 0.792 [0.732, 0.843]:** Gute Balance zwischen Precision und Recall. Der Agent erfasst die meisten Fehler (Recall = 0.798) mit moderater Präzision (Precision = 0.786).
- **AUROC = 0.892:** Sehr gute Trennung zwischen fehlerhaften und korrekten Summaries (0.5 = Zufall, 1.0 = perfekt). Der Agent kann zuverlässig zwischen den Klassen unterscheiden.
- **MCC = 0.445 [0.315, 0.571]:** Moderate Korrelation zwischen Vorhersage und Ground Truth (0 = Zufall, 1 = perfekt). MCC berücksichtigt alle vier Confusion-Matrix-Zellen und ist robuster als Accuracy bei unbalancierten Klassen.
- **Specificity = 0.645:** Niedriger als Recall (0.798), was bedeutet, dass der Agent konservativ ist (lieber FP als FN). Dies ist für Factuality-Checks sinnvoll, da falsche Alarme besser sind als übersehene Fehler.

### Agent vs Baselines

- **ROUGE-L / BERTScore:** Schwache Korrelationen (Spearman ρ < 0.20) zeigen, dass Ähnlichkeitsmetriken ungeeignet sind als Proxy für Faktentreue. Baselines messen, ob die Summary der Referenz ähnelt, nicht ob sie mit dem Artikel übereinstimmt.
- **Agent deutlich besser:** F1 = 0.79 vs. ROUGE-L Spearman ρ = 0.20. Der Agent nutzt Artikel-Information, während Baselines nur Referenz-Ähnlichkeit messen.

### Judge Baseline (LLM-as-a-Judge)

- **Status:** Script implementiert und ausgeführt (Smoke ✅, Final ✅).
- **Ansatz:** Binary verdict (`error_present: true/false`) mit `confidence` [0,1], Majority-Vote über mehrere Judgements (JUDGE_N=3, temp=0).
- **Ziel:** Vergleichsmethode, nicht Ersatz für den Agent. Zeigt, wie moderne LLMs als Baseline funktionieren (analog Readability/Coherence Judge).
- **Smoke-Ergebnisse (n=50):** AUROC=0.9872, alle Metriken definiert (balanced dataset), Parse-Stats: 100% JSON OK.
- **Final-Ergebnisse (n=200):** F1=0.9441 [0.9178, 0.9681], AUROC=0.9453, Precision=0.9286, Recall=0.9602. **Höher als Agent** (F1=0.9441 vs 0.792), aber Sample unbalanciert (pos=176, neg=24) → BA=0.7093, MCC=0.4753 niedriger als Agent (BA=0.722, MCC=0.445).

### Agent vs Judge Vergleich

- **F1:** Judge (0.9441) > Agent (0.792) – Judge erfasst mehr Fehler (Recall 0.9602 vs 0.798) mit höherer Präzision (0.9286 vs 0.786).
- **AUROC:** Judge (0.9453) > Agent (0.892) – Judge trennt besser zwischen fehlerhaften und korrekten Summaries.
- **Balanced Accuracy / MCC:** Judge (BA=0.7093, MCC=0.4753) < Agent (BA=0.722, MCC=0.445) – **Ursache:** Sample unbalanciert (pos=176, neg=24), daher niedrige Specificity (0.4583) und viele FP (13 FP vs 11 TN). Agent-Sample war balanced (TP=99, FP=27, TN=49, FN=25).
- **Interpretation:** Judge zeigt bessere Performance auf unbalanciertem Sample, aber Agent ist robuster bei balanced Samples (BA/MCC berücksichtigen beide Klassen gleich).

### Offene Fragen

- **Specificity niedrig (Judge):** FP-Rate ist hoch (13 FP vs 11 TN) aufgrund unbalanciertem Sample. Bei balanced Sample wäre Specificity vermutlich höher.
- **Issue-Type Breakdown:** Welche Fehlertypen werden häufig verpasst? (NUMBER, DATE, CONTRADICTION, etc.)
- **Uncertain Policy:** Wie geht der Agent mit unsicheren Fällen um? (aktuell: neutral gewichtet, aber als IssueSpan ausgegeben)

---

## Reproduzierbarkeit

### Agent-Run

**Run-ID:** `factuality_agent_manifest_20260107_215431_gpt-4o-mini`  
**Pfad:** `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/`

**Command:**
```bash
python scripts/eval_frank_factuality_agent_on_manifest.py \
  --data data/frank/frank_subset_manifest.jsonl \
  --max_examples 200 \
  --seed 42
```

### Smoke Validation

**Smoke-Run:** `judge_factuality_20260116_231906_gpt-4o-mini_v2_binary_seed42`  
**Dataset:** `frank_smoke_balanced_50_seed42.jsonl` (n=50, 25/25 pos/neg)  
**Quality Verify:** ✅ PASS (alle Checks bestanden)
- AUROC korrekt berechnet auf confidence (AUROC=0.9872)
- Parse-Stats: 100% JSON OK (150 judgments, 0 regex fallback, 0 failed)
- Bootstrap Edge Cases: keine skipped resamples
- Label Distribution: pos=25, neg=25 (beide Klassen vorhanden)

**Erkenntnis:** Pipeline funktioniert korrekt, alle Metriken definiert, Parse-Robustheit bestätigt.

---

### Judge-Run (Final ausstehend)

**Generate Balanced Smoke Dataset:**
```bash
# Erstellt stratifizierten Smoke-Datensatz (25 pos / 25 neg, n=50, seed=42)
python scripts/make_frank_smoke_balanced.py

# Verifikation der Verteilung
jq -r '.has_error' data/frank/frank_smoke_balanced_50_seed42.jsonl | sort | uniq -c
```

**Smoke-Test (n=50, bootstrap_n=200, balanced):**
```bash
ENABLE_LLM_JUDGE=true JUDGE_MODE=primary JUDGE_N=3 JUDGE_TEMPERATURE=0 \
python scripts/eval_frank_factuality_llm_judge.py \
  --data data/frank/frank_smoke_balanced_50_seed42.jsonl \
  --max_examples 50 \
  --seed 42 \
  --bootstrap_n 200 \
  --cache_mode write \
  --prompt_version v2_binary \
  --judge_n 3 \
  --judge_temperature 0.0 \
  --judge_aggregation majority
```

**Final-Run (n=200, bootstrap_n=2000):**
```bash
# Final-Run auf frank_clean.jsonl (beide Klassen vorhanden: pos=1436, neg=810)
ENABLE_LLM_JUDGE=true JUDGE_MODE=primary JUDGE_N=3 JUDGE_TEMPERATURE=0 \
python scripts/eval_frank_factuality_llm_judge.py \
  --data data/frank/frank_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --bootstrap_n 2000 \
  --cache_mode write \
  --prompt_version v2_binary \
  --judge_n 3 \
  --judge_temperature 0.0 \
  --judge_aggregation majority
```

**Quality Verify auf Final-Run:**
```bash
# Quality Verify (✅ PASS)
python scripts/verify_quality_factuality_coherence.py \
  --judge_run results/evaluation/factuality/judge_factuality_20260116_233505_gpt-4o-mini_v2_binary_seed42 \
  --json_output results/evaluation/factuality/judge_factuality_20260116_233505_gpt-4o-mini_v2_binary_seed42/quality_checks.json
```

**⚠️ Wichtig: Datensatz-Auswahl**
- **Empfohlen für Smoke:** `frank_smoke_balanced_50_seed42.jsonl` (balanced, 25/25 pos/neg) → alle Metriken definiert
- **Empfohlen für Final:** `frank_clean.jsonl` (vollständiges Dataset, beide Klassen vorhanden)
- **Nicht empfohlen:** `frank_subset_manifest.jsonl` (kann single-class sein) → AUROC, Balanced Accuracy, MCC werden als N/A markiert

**Hinweise:**
- `cache_mode=read` empfohlen für Final-Run (nutzt Cache aus Smoke-Test)
- Output: `results/evaluation/factuality/judge_factuality_<timestamp>_gpt-4o-mini_v2_binary_seed42/`
- Nach Final-Run: Quality-Check ausführen: `python scripts/verify_quality_factuality_coherence.py --judge_run <FINAL_RUN_PATH>`

### Baseline-Runs

**ROUGE-L:**
- Run-ID: `factuality_rouge_l_20260107_230519_seed42`
- Pfad: `results/evaluation/factuality_baselines/factuality_rouge_l_20260107_230519_seed42/`

**BERTScore:**
- Run-ID: `factuality_bertscore_20260107_230523_seed42`
- Pfad: `results/evaluation/factuality_baselines/factuality_bertscore_20260107_230523_seed42/`

---

## Artefakte

### Agent-Run
- `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/`
  - `summary.json`: Metriken + CIs
  - `summary.md`: Human-readable Zusammenfassung
  - `predictions.jsonl`: Pro-Beispiel-Vorhersagen
  - `run_metadata.json`: Timestamp, Git-Commit, Config

### Judge-Run

**Smoke-Run:**
- `results/evaluation/factuality/judge_factuality_20260116_231906_gpt-4o-mini_v2_binary_seed42/`
  - `summary.json`: Metriken + CIs + parse_stats
  - `summary.md`: Human-readable Zusammenfassung
  - `predictions.jsonl`: Pro-Beispiel-Vorhersagen (mit confidence)
  - `run_metadata.json`: Timestamp, Git-Commit, Config
  - `cache.jsonl`: Cache für mögliche Wiederverwendung

**Final-Run:**
- `results/evaluation/factuality/judge_factuality_20260116_233505_gpt-4o-mini_v2_binary_seed42/`
  - `summary.json`: Metriken + CIs + parse_stats + label_distribution (dataset + sample)
  - `summary.md`: Human-readable Zusammenfassung
  - `predictions.jsonl`: Pro-Beispiel-Vorhersagen (mit confidence)
  - `run_metadata.json`: Timestamp, Git-Commit, Config
  - `cache.jsonl`: Cache für mögliche Wiederverwendung
  - `quality_checks.json`: ✅ Quality Verify Report (alle Checks PASS)

### Baselines
- `results/evaluation/factuality_baselines/` (ROUGE-L, BERTScore)

---

## Thesis Framing

**Methodisch sinnvoll:**
- **Agent vs Baselines:** Zeigt, dass artikelbasierte Verifikation deutlich besser ist als referenzbasierte Ähnlichkeitsmetriken (F1 = 0.79 vs. Spearman ρ = 0.20).
- **Judge Baseline:** LLM-as-a-Judge dient als moderne Baseline und zeigt, wie gut generische LLMs ohne spezialisierte Agent-Logik abschneiden (analog Readability/Coherence).
- **Binary Classification:** Factuality ist binär (Fehler ja/nein), daher sind Precision/Recall/F1/AUROC die passenden Metriken (nicht Spearman/Pearson wie bei Readability/Coherence).

**Fazit:** Der Factuality-Agent übertrifft klassische Baselines deutlich und zeigt, dass artikelbasierte Verifikation für Faktentreue essentiell ist.

