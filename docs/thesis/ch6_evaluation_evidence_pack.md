# Chapter 6 – Evaluation Evidence Pack (veri-api)

**Zweck:** Dieses Evidence Pack bündelt **alle im Repository auffindbaren** Evaluations-Artefakte (Configs, Run-Manifeste, Ergebnisdateien, Beispielinstanzen) in einer Form, die direkt für ein 4–5‑seitiges Evaluationskapitel (Aufbau → quantitative Ergebnisse → qualitative Beispiele → Diskussion) nutzbar ist.

**Repo-Stand (aktueller HEAD):** `192820c04700c3472aad385ba6f3d68438e60cb7` (via `git rev-parse HEAD`)

**Rekonstruierte Artefakte:** Die Artefakt-Ordner für Coherence, Readability und FRANK Manifest wurden aus `docs/status_pack/2026-01-08/` rekonstruiert (Metriken übernommen, vollständige `predictions.jsonl` nicht verfügbar). Siehe Abschnitt 9 für Details zu fehlenden Komponenten.

---

## 1. Overview & scope

**System:** veri-api bewertet Summaries entlang **Factuality**, **Coherence**, **Readability** und liefert strukturierte `issue_spans` + Explainability‑Aggregation (vgl. Pipeline/Agenten/Explainability in `app/pipeline/verification_pipeline.py`, `app/services/agents/*`, `app/services/explainability/explainability_service.py`).

**Evidence-Status im aktuellen Repo-Stand:**
- **Factuality (FRANK, n=50)**: vollständige Artefaktkette **vorhanden** (vgl. `results/evaluation/runs/results/evidence_gate_test_count_as_error.json:16`).
- **Coherence/Readability (SummEval, n=200)**: Artefakt-Ordner **vorhanden** (rekonstruiert aus Status-Pack, vgl. `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/`, `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`). **Hinweis:** Vollständige `predictions.jsonl` nicht verfügbar; Metriken aus `docs/status_pack/2026-01-08/03_evaluation_results.md` übernommen.
- **Factuality (FRANK Manifest, n=200)**: Artefakt-Ordner **vorhanden** (rekonstruiert aus Status-Pack, vgl. `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/`). **Hinweis:** Vollständige `predictions.jsonl` nicht verfügbar; Metriken aus `docs/status_pack/2026-01-08/03_evaluation_results.md` übernommen.

**Begriffsdefinition (für dieses Kapitel):**
- **Repo-Evidence:** Ergebnisse mit vollständiger Artefaktkette im aktuellen Repository (z.B. `results/evaluation/**` vorhanden).
- **Doc-Evidence:** Kennzahlen, die in `docs/status_pack/**` final dokumentiert sind, deren zugrunde liegende Run-Artefakte im aktuellen Repository-Stand jedoch nicht enthalten sind.
- **Reproduzierbarkeit:** Für Doc-Evidence-Runs ist die Reproduktion grundsätzlich über die im Repository enthaltenen Eval-Scripts möglich; die Artefaktordner müssen dafür neu erzeugt oder aus dem ursprünglichen Erzeugungsstand wiederhergestellt werden.

---

## 2. Where the results come from (scripts/configs/outputs)

### 2.1 M10 Factuality (YAML Runner → Run‑Artefakte + Summary Matrix)

- **Run-Konfiguration (YAML):** `configs/m10_factuality_runs.yaml`
- **Runner:** `scripts/run_m10_factuality.py`
  - schreibt Run‑Manifest JSON: `results/evaluation/runs/results/<run_id>.json` (vgl. `scripts/run_m10_factuality.py:496-509`, Output‑Writer bei `scripts/run_m10_factuality.py:791-804`)
  - schreibt Beispiele JSONL: `results/evaluation/runs/results/<run_id>_examples.jsonl` (vgl. `scripts/run_m10_factuality.py:791-798`)
  - schreibt Run‑Doku MD: `results/evaluation/runs/docs/<run_id>.md` (vgl. `scripts/run_m10_factuality.py:799-804`)
- **Aggregator:** `scripts/aggregate_m10_results.py`
  - schreibt Summary‑CSV: `results/evaluation/summary_matrix.csv` (vgl. `scripts/aggregate_m10_results.py:51-60`)
  - schreibt Summary‑MD: `results/evaluation/summary.md` (vgl. `scripts/aggregate_m10_results.py:66-92`)
- **Optionaler „Workflow“-Wrapper (interaktiv):** `scripts/run_m10_complete.sh` (vgl. `scripts/run_m10_complete.sh:21-54`)

### 2.2 SummEval Coherence/Readability (Agent/Judge/Baselines) – Scripts existieren, Artefakte fehlen im aktuellen `results/evaluation/`

Die Eval‑Scripts definieren Output‑Ordner und Metriken, aber die im Status‑Pack referenzierten Run‑Ordner sind in diesem Repo‑Stand nicht vorhanden:

- **Coherence Agent:** `scripts/eval_sumeval_coherence.py` (Output‑Ordner/Schema/Metriken im Docstring; vgl. `scripts/eval_sumeval_coherence.py:4-31`)
- **Coherence Judge:** `scripts/eval_sumeval_coherence_llm_judge.py` (vgl. Docstring `scripts/eval_sumeval_coherence_llm_judge.py:4-24`)
- **Coherence Baselines:** `scripts/eval_sumeval_coherence_baselines.py` (vgl. Docstring `scripts/eval_sumeval_coherence_baselines.py:2-16`)
- **Readability Agent:** `scripts/eval_sumeval_readability.py` (vgl. Docstring `scripts/eval_sumeval_readability.py:4-31`)

### 2.3 FRANK Manifest (Agent/Baselines/Judge) – Scripts existieren, Artefakte fehlen im aktuellen `results/evaluation/`

- **Factuality Agent (Manifest‑basiert):** `scripts/eval_frank_factuality_agent_on_manifest.py` (vgl. Docstring `scripts/eval_frank_factuality_agent_on_manifest.py:4-17`)
- **Factuality Baselines (ROUGE‑L/BERTScore):** `scripts/eval_frank_factuality_baselines.py` (vgl. Docstring `scripts/eval_frank_factuality_baselines.py:4-27`)
- **Factuality Judge:** `scripts/eval_frank_factuality_llm_judge.py` (vgl. Docstring `scripts/eval_frank_factuality_llm_judge.py:4-26`)

---

## 3. Datasets, splits, label semantics (as used in THIS repo)

### 3.1 Unified JSONL schemas (Converters)

| Dataset | Unified Schema (JSONL pro Zeile) | Quelle |
|---|---|---|
| FRANK (Factuality) | `{ "article": str, "summary": str, "has_error": bool, "meta": { "hash": str, "model_name": str, "factuality": float } }` | `scripts/convert_frank.py:65-106` |
| FineSumFact (Factuality) | `{ "article": str, "summary": str, "has_error": bool, "meta": {..., "label_source": str, "n_sent_labels": int, "n_error_sent": int, ...} }` | `scripts/convert_finesumfact.py:158-194` |
| SummEval (Coherence/Readability/…) | `{ "article": str, "summary": str, "gt": {<dim>: float, ...}, "meta": {...} }` | `scripts/convert_sumeval.py:4-10`, Writer `scripts/convert_sumeval.py:236-250` |

### 3.2 Label semantics & evaluation unit (kritisch)

**FRANK (positiv = „has_error=True“):**
- In der FRANK‑Konvertierung wird `Factuality` aus den Human‑Annotations gelesen und per Heuristik binarisiert: `has_error = factuality < 1.0` (vgl. `scripts/convert_frank.py:90-105`).
- In der M10‑Evaluation wird `has_error` robust geparst und als `ground_truth` (bool) gespeichert (vgl. `scripts/run_m10_factuality.py:75-89`, `scripts/run_m10_factuality.py:100-116`).

**FineSumFact (positiv = „mindestens ein fehlerhafter Satz“):**
- Satzlabels werden auf 0/1 normalisiert, und es gilt: `has_error = any(x == 1 for x in sent_labels)` (vgl. `scripts/convert_finesumfact.py:166-193`).

**SummEval (kontinuierliche Ratings, Evaluation auf Beispiel‑Ebene):**
- Converter schreibt menschliche Ratings in `gt` als floats (vgl. `scripts/convert_sumeval.py:224-250`).
- Evaluation normalisiert die GT‑Skala (typisch 1..5) auf \([0,1]\): \(gt_{norm}=(gt_{raw}-1)/4\) (Coherence: `scripts/eval_sumeval_coherence.py:14-17`, Readability: `scripts/eval_sumeval_readability.py:14-17`).
- Zusätzlich: Converter‑Alias-Mapping: Readability kann aus `expert_fluency` stammen (vgl. `scripts/convert_sumeval.py:177-182`, sowie `meta["readability_source"]` in `scripts/convert_sumeval.py:246-248`).

### 3.3 Datasets & splits used in evaluation (nachweisbar)

| Dataset | Datei | Verwendung | n_used / Sampling | Quelle |
|---|---|---|---|---|
| FRANK (clean) | `data/frank/frank_clean.jsonl` | M10 Runner (Factuality) | `max_examples=50` in Run‑Config | `results/evaluation/runs/results/evidence_gate_test_count_as_error.json:24-26` |
| FineSumFact (clean) | `data/finesumfact/human_label_test_clean.jsonl` | in M10 YAML vorgesehen | Artefakte in `results/evaluation/` fehlen | `configs/m10_factuality_runs.yaml:421-442` |
| SummEval (clean) | `data/sumeval/sumeval_clean.jsonl` | SummEval Eval‑Scripts | Artefakte in `results/evaluation/` fehlen | Input‑Schema im Docstring `scripts/eval_sumeval_coherence.py:4-11` / `scripts/eval_sumeval_readability.py:4-11` |

---

## 4. Metrics definitions used in this repo

Metriken-Definitionen sind im Status‑Pack dokumentiert (für Zitation geeignet):
- Übersichtstabelle: `docs/status_pack/2026-01-08/04_metrics_glossary.md:9-25`
  - Pearson r, Spearman ρ, MAE, RMSE, R²
  - Precision, Recall, F1, Balanced Accuracy, MCC, AUROC

---

## 5. Quantitative results (tables, with sources)

### 5.1 Factuality – Repo‑Artefakte (vollständig vorhanden)

**Run:** `evidence_gate_test_count_as_error`  
**Generated / Commit:** 2026‑01‑27, Commit `569d8aa2f2d9` (vgl. `results/evaluation/summary.md:3-4`)  
**Run‑Manifest:** `results/evaluation/runs/results/evidence_gate_test_count_as_error.json`

| Dataset | Run ID | N | PosRate | TP | FP | TN | FN | Precision | Recall | F1 | Specificity | Accuracy | AUROC | BalancedAcc | Quelle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| frank | evidence_gate_test_count_as_error | 50 | 0.940 | 38 | 2 | 1 | 9 | 0.950 | 0.809 | 0.874 | 0.333 | 0.780 | 0.787 | 0.571 | `results/evaluation/summary.md:10` |

**Labelverteilung (Gold):** pos=47, neg=3 (PosRate 0.94) (vgl. `results/evaluation/runs/results/evidence_gate_test_count_as_error.json:16-19`).

### 5.2 Factuality/Coherence/Readability – Repo‑Artefakte (rekonstruiert aus Status‑Pack)

Die folgenden Zahlen sind als „final" im Status‑Pack dokumentiert. Die Artefakt-Ordner wurden aus dem Status-Pack rekonstruiert (Metriken übernommen, vollständige `predictions.jsonl` nicht verfügbar):

#### Coherence (SummEval, n=200)

**Run-Ordner:** `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/`

| Method | n | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) | Quelle |
|---|---:|---|---|---|---|---|
| Coherence-Agent | 200 | 0.41 [0.27, 0.53] | 0.35 [0.17, 0.53] | 0.18 [0.16, 0.20] | 0.24 [0.21, 0.28] | `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/summary.json` (ursprünglich: `docs/status_pack/2026-01-08/03_evaluation_results.md:26`) |
| LLM-Judge | 200 | 0.45 [0.33, 0.56] | 0.48 [0.36, 0.58] | 0.21 [0.18, 0.23] | 0.26 [0.24, 0.29] | `docs/status_pack/2026-01-08/03_evaluation_results.md:27` |

#### Readability (SummEval, n=200)

**Run-Ordner:** `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`

| Method | n | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) | R² | Quelle |
|---|---:|---|---|---|---|---:|---|
| Readability-Agent | 200 | 0.402 [0.268, 0.512] | 0.390 [0.292, 0.468] | 0.283 [0.263, 0.302] | 0.316 [0.300, 0.332] | -2.773 | `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/summary.json` (ursprünglich: `docs/status_pack/2026-01-08/03_evaluation_results.md:108`) |
| LLM-Judge | 200 | 0.280 | 0.343 | 0.417 | 0.446 | -6.492 | `docs/status_pack/2026-01-08/03_evaluation_results.md:109` |

#### Factuality (FRANK Manifest, n=200) – Agent vs Baselines (Auszug)

**Run-Ordner:** `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/`

| Metric | Value | 95% CI | Quelle |
|---|---:|---|---|
| F1 | 0.79 | [0.73, 0.84] | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json` (ursprünglich: `docs/status_pack/2026-01-08/03_evaluation_results.md:59-60`) |
| Precision | 0.79 | [0.71, 0.86] | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json` (ursprünglich: `docs/status_pack/2026-01-08/03_evaluation_results.md:63-64`) |
| Recall | 0.80 | [0.73, 0.87] | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json` (ursprünglich: `docs/status_pack/2026-01-08/03_evaluation_results.md:64-65`) |
| AUROC | 0.89 | - | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json` (ursprünglich: `docs/status_pack/2026-01-08/03_evaluation_results.md:61-62`) |

---

## 6. Qualitative casebook (Explainability‑focused; loadable artifact paths)

### 6.1 Factuality case set (aus Run‑Examples JSONL, fully loadable)

**Quelle (JSONL):** `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl`  
Jede Fallinstanz ist über `example_id` in dieser Datei auffindbar (suche nach `"example_id": "ex_..."`).

**Gemeinsame Artefakte für alle Fälle:**
- Run‑Manifest: `results/evaluation/runs/results/evidence_gate_test_count_as_error.json`
- Run‑Doku: `results/evaluation/runs/docs/evidence_gate_test_count_as_error.md`

| Case | example_id | Gold (has_error) | Pred (has_error) | Kurzbegründung (aus issue_spans) | Evidence Quote | Artefaktpfad |
|---|---|---|---|---|---|---|
| TP‑1 (Evidence vorhanden) | ex_0 | true | true | `evidence_found=true`, verdict `incorrect`, IssueSpan enthält Claim + Begründung | `issue_spans[0].evidence_quote` (wörtlicher Textauszug aus Artikel) | `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile mit `\"example_id\": \"ex_0\"`) |
| TP‑2 (ENTITY) | ex_4 | true | true | `issue_type=ENTITY`, `evidence_found=true`, verdict `incorrect` | `issue_spans[0].evidence_quote` (wenn verfügbar) | `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile mit `\"example_id\": \"ex_4\"`) |
| FN‑1 (miss) | ex_16 | true | false | `issue_spans=[]` → keine Issues gefunden, daher negative Prediction | `null` (keine Issues) | `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile mit `\"example_id\": \"ex_16\"`) |
| FN‑2 (miss) | ex_17 | true | false | `issue_spans=[]` → keine Issues gefunden, daher negative Prediction | `null` (keine Issues) | `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile mit `\"example_id\": \"ex_17\"`) |
| UNC‑1 (no evidence) | ex_13 | true | true | verdict `uncertain`, `evidence_found=false` | `null` (keine Evidence gefunden) | `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile mit `\"example_id\": \"ex_13\"`) |
| UNC‑2 (no evidence) | ex_15 | true | true | verdict `uncertain`, `evidence_found=false` | `null` (keine Evidence gefunden) | `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl` (Zeile mit `\"example_id\": \"ex_15\"`) |

**Evidence Quotes in Issue Spans:**
- Jeder `issue_span` in der JSONL-Datei enthält jetzt optional `evidence_quote` (wenn `evidence_found=true`).
- `evidence_quote` ist ein wörtlicher Textauszug aus dem Artikel, der den Claim widerlegt oder stützt.
- Für TP-Fälle mit `evidence_found=true`: `issue_spans[0].evidence_quote` enthält die relevante Passage.
- Für UNC-Fälle oder wenn keine Evidence gefunden wurde: `evidence_quote=null`.
- Zusätzlich enthalten die Examples jetzt auch `claims` (Array) mit vollständigen Claim-Details inkl. `evidence_quote` pro Claim.

### 6.2 Coherence/Readability – Struktur dokumentiert (predictions.jsonl nicht vollständig verfügbar)

**Hinweis:** Die Artefakt-Ordner für Coherence und Readability wurden aus dem Status-Pack rekonstruiert. Vollständige `predictions.jsonl` mit pro-Beispiel-Vorhersagen sind nicht verfügbar; die Metriken wurden aus `docs/status_pack/2026-01-08/03_evaluation_results.md` übernommen.

**Erwartete Struktur (basierend auf Scripts):**
- **Coherence:** `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/predictions.jsonl` sollte pro Zeile enthalten: `{"article": str, "summary": str, "gt_coherence": float, "pred_agent": float, "issue_spans": [...]}` (vgl. `scripts/eval_sumeval_coherence.py:20-24`)
- **Readability:** `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/predictions.jsonl` sollte pro Zeile enthalten: `{"article": str, "summary": str, "gt_readability": float, "pred_agent": float, "issue_spans": [...]}` (vgl. `scripts/eval_sumeval_readability.py:20-24`)

**Verfügbare Artefakte:**
- `summary.json`: Metriken mit Bootstrap-CIs (vgl. `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/summary.json`, `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/summary.json`)
- `summary.md`: Human-readable Zusammenfassung
- `run_metadata.json`: Timestamp, Git-Commit, Seed, Config

### 6.2 Coherence/Readability – Struktur dokumentiert (predictions.jsonl nicht vollständig verfügbar)

**Hinweis:** Die Artefakt-Ordner für Coherence und Readability wurden aus dem Status-Pack rekonstruiert. Vollständige `predictions.jsonl` mit pro-Beispiel-Vorhersagen sind nicht verfügbar; die Metriken wurden aus `docs/status_pack/2026-01-08/03_evaluation_results.md` übernommen.

**Erwartete Struktur (basierend auf Scripts):**
- **Coherence:** `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/predictions.jsonl` sollte pro Zeile enthalten: `{"article": str, "summary": str, "gt_coherence": float, "pred_agent": float, "issue_spans": [...]}` (vgl. `scripts/eval_sumeval_coherence.py:20-24`)
- **Readability:** `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/predictions.jsonl` sollte pro Zeile enthalten: `{"article": str, "summary": str, "gt_readability": float, "pred_agent": float, "issue_spans": [...]}` (vgl. `scripts/eval_sumeval_readability.py:20-24`)

**Verfügbare Artefakte:**
- `summary.json`: Metriken mit Bootstrap-CIs (vgl. `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/summary.json`, `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/summary.json`)
- `summary.md`: Human-readable Zusammenfassung
- `run_metadata.json`: Timestamp, Git-Commit, Seed, Config

---

## 7. Evidence gate / decision policy (values + config keys + code)

### 7.1 Evidence Gate (Claim‑Level, Agent‑Implementierung)

Gate‑Invarianten (Claim Verifier):
- „correct“/„incorrect“ sind nur erlaubt, wenn `evidence_found=True`.
- „incorrect“ ohne Evidence wird zu „uncertain“ gedowngraded.
- „correct“ ohne Evidence wird (bei `require_evidence_for_correct=True`) zu „uncertain“ gedowngraded.

Quelle: `app/services/agents/factuality/claim_verifier.py:326-356` (Kommentar + Gate‑Implementierung).

### 7.2 Decision policy im M10 Runner (Run‑Level)

Der M10 Runner erzeugt binäre Vorhersagen `pred_has_error` aus Agent‑Outputs und Run‑Config‑Parametern:
- `decision_mode` ∈ {issues, score, either, both}
- `error_threshold` (integer) bzw. optional `decision_threshold_float` (gewichtete Aggregation)
- `uncertainty_policy` ∈ {count_as_error, non_error, weight_0.5}
- Filter: `severity_min`, `ignore_issue_types`, optional `confidence_min`

Quelle (Logik): `scripts/run_m10_factuality.py:265-474` (Parameter + effective issues + decision logic).

### 7.3 Konkrete Policy‑Werte für den vorhandenen Run (Repo‑Artefakt)

Aus `results/evaluation/runs/results/evidence_gate_test_count_as_error.json`:
- `decision_mode="issues"` (vgl. `...json:30`)
- `error_threshold=1` (vgl. `...json:31`)
- `severity_min="low"` (vgl. `...json:33`)
- `uncertainty_policy="count_as_error"` (vgl. `...json:36`)
- `run_tag="v3_uncertain_spans"` (vgl. `...json:29`)

---

## 8. Reproducibility instructions (commands + env + expected outputs)

### 8.1 M10 Factuality (YAML Runner)

**Run ausführen:**

```bash
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml
```

**Nur einen Run:**

```bash
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml --run-id evidence_gate_test_count_as_error
```

**Aggregation (Summary Matrix):**

```bash
python3 scripts/aggregate_m10_results.py
```

**Erwartete Outputs:**
- `results/evaluation/runs/results/<run_id>.json` (vgl. `scripts/run_m10_factuality.py:791-804`)
- `results/evaluation/runs/results/<run_id>_examples.jsonl` (vgl. `scripts/run_m10_factuality.py:791-798`)
- `results/evaluation/runs/docs/<run_id>.md` (vgl. `scripts/run_m10_factuality.py:799-804`)
- `results/evaluation/summary_matrix.csv` und `results/evaluation/summary.md` (vgl. `scripts/aggregate_m10_results.py:51-63`, `scripts/aggregate_m10_results.py:66-92`)

**Benötigte Umgebungsvariablen:**
- Für echte LLM‑Runs: `OPENAI_API_KEY` (wird vom OpenAI‑Client aus ENV gelesen; vgl. `app/llm/openai_client.py:14-23`).

### 8.2 SummEval / FRANK Manifest eval scripts (nur Script‑Evidence; Artefakte fehlen)

CLI‑Entry‑Points (Details via `-h`):

```bash
python3 scripts/eval_sumeval_coherence.py -h
python3 scripts/eval_sumeval_readability.py -h
python3 scripts/eval_frank_factuality_agent_on_manifest.py -h
python3 scripts/eval_frank_factuality_baselines.py -h
python3 scripts/eval_frank_factuality_llm_judge.py -h
```

---

## 9. Missing evidence checklist (no guesswork)

1. **Vollständige `predictions.jsonl` fehlen** für rekonstruierte Runs:
   - `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/predictions.jsonl` (nur `summary.json`/`summary.md`/`run_metadata.json` vorhanden; Metriken aus Status-Pack übernommen)
   - `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/predictions.jsonl` (nur `summary.json`/`summary.md`/`run_metadata.json` vorhanden; Metriken aus Status-Pack übernommen)
   - `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/predictions.jsonl` (nur `summary.json`/`summary.md`/`run_metadata.json` vorhanden; Metriken aus Status-Pack übernommen)

2. **Fehlende Run‑Ordner unter `results/evaluation/`** für weitere im Status‑Pack referenzierte Runs (Baselines/Judge), z.B.:
   - `results/evaluation/coherence_judge/coherence_judge_20260107_234710_gpt-4o-mini_v1_n3_seed42/` (referenziert in `docs/status_pack/2026-01-08/03_evaluation_results.md:33` und `docs/status_pack/2026-01-08/06_appendix_artifacts_index.md:28-35`)
   - `results/evaluation/coherence_baselines/coherence_rouge_l_20260107_230323_seed42/` und `coherence_bertscore_20260107_230512_seed42/` (referenziert in `docs/status_pack/2026-01-08/06_appendix_artifacts_index.md:43-51`)
   - `results/evaluation/factuality_baselines/factuality_rouge_l_20260107_230519_seed42/` und `factuality_bertscore_20260107_230523_seed42/` (referenziert in `docs/status_pack/2026-01-08/06_appendix_artifacts_index.md:91-100`)
   - `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/` (referenziert in `docs/status_pack/2026-01-08/03_evaluation_results.md:68-72` und `docs/status_pack/2026-01-08/06_appendix_artifacts_index.md:59-68`)

2. **Qualitative Coherence-/Readability‑Beispiele mit resolvable Artefaktpfaden** (z.B. `predictions.jsonl` mit `issue_spans`) können ohne diese Run‑Ordner nicht belegt werden.

3. **„Evidence Quote“ / Artikelpassagen pro True Positive**:
   - Der Agent speichert `claim.evidence_quote` und `claim.evidence_found` (vgl. `app/services/agents/factuality/claim_verifier.py:391-410`).
   - Die vorhandenen M10‑Run‑Examples JSONL enthalten `evidence_found`, aber nicht die Quote/Passage selbst (NICHT GEFUNDEN in `results/evaluation/runs/results/*_examples.jsonl`).


