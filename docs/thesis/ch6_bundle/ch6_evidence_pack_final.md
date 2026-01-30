# Chapter 6 – Evaluation Evidence Pack (veri-api) – Final

**Zweck:** Dieses Evidence Pack bündelt **alle im Repository auffindbaren** Evaluations-Artefakte (Configs, Run-Manifeste, Ergebnisdateien, Beispielinstanzen) in einer Form, die direkt für ein 4–5‑seitiges Evaluationskapitel (Aufbau → quantitative Ergebnisse → qualitative Beispiele → Diskussion) nutzbar ist.

**Repo-Stand (aktueller HEAD):** `f0dc39532bad0e760bb8aedc8db16550efb025bd` (via `git rev-parse HEAD`)  
**Bundle-Datum:** 2026-01-30

---

## 1. Overview & scope

**System:** veri-api bewertet Summaries entlang **Factuality**, **Coherence**, **Readability** und liefert strukturierte `issue_spans` + Explainability‑Aggregation (vgl. Pipeline/Agenten/Explainability in `app/pipeline/verification_pipeline.py`, `app/services/agents/*`, `app/services/explainability/explainability_service.py`).

**Evidence-Status im aktuellen Repo-Stand:**
- **Factuality (FRANK, n=50)**: ✅ **Repo-Evidence** – vollständige Artefaktkette vorhanden (vgl. `results/evaluation/runs/results/evidence_gate_test_count_as_error.json`)
- **Coherence (SummEval, n=200)**: ✅ **Repo-Evidence** – vollständige Artefaktkette vorhanden (vgl. `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/`)
- **Readability (SummEval, n=200)**: ✅ **Repo-Evidence** – vollständige Artefaktkette vorhanden (vgl. `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`)
- **Factuality (FRANK Manifest, n=200)**: ⚠️ **Doc-Evidence** – Metriken vorhanden, `predictions.jsonl` fehlt (vgl. `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/`)

**Begriffsdefinition (für dieses Kapitel):**
- **Repo-Evidence:** Ergebnisse mit vollständiger Artefaktkette im aktuellen Repository (z.B. `results/evaluation/**` vorhanden, inkl. `predictions.jsonl` oder `*_examples.jsonl`).
- **Doc-Evidence:** Kennzahlen, die in `docs/status_pack/**` final dokumentiert sind, deren zugrunde liegende Run-Artefakte im aktuellen Repository-Stand jedoch nicht vollständig enthalten sind (z.B. nur `summary.json`, keine `predictions.jsonl`).
- **Reproduzierbarkeit:** Für Doc-Evidence-Runs ist die Reproduktion grundsätzlich über die im Repository enthaltenen Eval-Scripts möglich; die Artefaktordner müssen dafür neu erzeugt oder aus dem ursprünglichen Erzeugungsstand wiederhergestellt werden.

---

## 2. Where the results come from (scripts/configs/outputs)

### 2.1 M10 Factuality (YAML Runner → Run‑Artefakte + Summary Matrix)

- **Run-Konfiguration (YAML):** `configs/m10_factuality_runs.yaml`
- **Runner:** `scripts/run_m10_factuality.py`
  - schreibt Run‑Manifest JSON: `results/evaluation/runs/results/<run_id>.json`
  - schreibt Beispiele JSONL: `results/evaluation/runs/results/<run_id>_examples.jsonl`
  - schreibt Run‑Doku MD: `results/evaluation/runs/docs/<run_id>.md`
- **Aggregator:** `scripts/aggregate_m10_results.py`
  - schreibt Summary‑CSV: `results/evaluation/summary_matrix.csv`
  - schreibt Summary‑MD: `results/evaluation/summary.md`

### 2.2 SummEval Coherence/Readability (Agent Scripts)

- **Coherence Agent:** `scripts/eval_sumeval_coherence.py`
  - Output: `results/evaluation/coherence/<run_id>/`
  - Dateien: `predictions.jsonl`, `summary.json`, `summary.md`, `run_metadata.json`, `errors.jsonl` (optional)
- **Readability Agent:** `scripts/eval_sumeval_readability.py`
  - Output: `results/evaluation/readability/<run_id>/`
  - Dateien: `predictions.jsonl`, `summary.json`, `summary.md`, `run_metadata.json`, `errors.jsonl` (optional)

### 2.3 FRANK Manifest (Agent Script)

- **Factuality Agent (Manifest‑basiert):** `scripts/eval_frank_factuality_agent_on_manifest.py`
  - Output: `results/evaluation/factuality/<run_id>/`
  - Dateien: `predictions.jsonl`, `summary.json`, `summary.md`, `run_metadata.json`

---

## 3. Datasets, splits, label semantics

### 3.1 Unified JSONL schemas

| Dataset | Unified Schema (JSONL pro Zeile) | Quelle |
|---|---|---|
| FRANK (Factuality) | `{ "article": str, "summary": str, "has_error": bool, "meta": {...} }` | `scripts/convert_frank.py:65-106` |
| SummEval (Coherence/Readability) | `{ "article": str, "summary": str, "gt": {<dim>: float, ...}, "meta": {...} }` | `scripts/convert_sumeval.py:4-10` |

### 3.2 Label semantics

**FRANK (positiv = „has_error=True"):**
- `has_error = factuality < 1.0` (vgl. `scripts/convert_frank.py:90-105`)

**SummEval (kontinuierliche Ratings):**
- GT-Skala (1..5) wird normalisiert: \(gt_{norm}=(gt_{raw}-1)/4\)
- Coherence: `scripts/eval_sumeval_coherence.py:14-17`
- Readability: `scripts/eval_sumeval_readability.py:14-17`

### 3.3 Datasets & splits

| Dataset | Datei | Verwendung | n_used | Quelle |
|---|---|---|---|---|
| FRANK (clean) | `data/frank/frank_clean.jsonl` | M10 Runner | 50 | `results/evaluation/runs/results/evidence_gate_test_count_as_error.json:24-26` |
| SummEval (clean) | `data/sumeval/sumeval_clean.jsonl` | Coherence/Readability Eval | 200 | `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/run_metadata.json:9` |

---

## 4. Metrics definitions

Metriken-Definitionen: `docs/status_pack/2026-01-08/04_metrics_glossary.md:9-25`
- **Regression:** Pearson r, Spearman ρ, MAE, RMSE, R²
- **Klassifikation:** Precision, Recall, F1, Balanced Accuracy, MCC, AUROC

---

## 5. Quantitative results (tables, with sources)

### 5.1 Factuality (FRANK, n=50) – Repo-Evidence

**Run:** `evidence_gate_test_count_as_error`  
**Run‑Manifest:** `results/evaluation/runs/results/evidence_gate_test_count_as_error.json`

| Dataset | Run ID | N | PosRate | TP | FP | TN | FN | Precision | Recall | F1 | Specificity | Accuracy | AUROC | BalancedAcc | Quelle |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| frank | evidence_gate_test_count_as_error | 50 | 0.940 | 38 | 2 | 1 | 9 | 0.950 | 0.809 | 0.874 | 0.333 | 0.780 | 0.787 | 0.571 | `results/evaluation/runs/results/evidence_gate_test_count_as_error.json:4-14` |

**Labelverteilung (Gold):** pos=47, neg=3 (vgl. `results/evaluation/runs/results/evidence_gate_test_count_as_error.json:16-19`)

### 5.2 Coherence (SummEval, n=200) – Repo-Evidence

**Run-Ordner:** `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/`

| Method | n | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) | Quelle |
|---|---:|---|---|---|---|---|
| Coherence-Agent | 200 | 0.451 [0.314, 0.575] | 0.371 [0.197, 0.557] | 0.174 [0.152, 0.197] | 0.237 [0.203, 0.273] | `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/summary.json:11-25` |

**Hinweis:** LLM-Judge und Baselines sind Doc-Evidence (vgl. `docs/status_pack/2026-01-08/03_evaluation_results.md:27-28`)

### 5.3 Readability (SummEval, n=200) – Repo-Evidence

**Run-Ordner:** `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`

| Method | n | Spearman ρ (95% CI) | Pearson r (95% CI) | MAE (95% CI) | RMSE (95% CI) | R² | Quelle |
|---|---:|---|---|---|---|---:|---|
| Readability-Agent | 200 | 0.424 [0.297, 0.532] | 0.401 [0.312, 0.478] | 0.285 [0.265, 0.305] | 0.318 [0.303, 0.334] | -2.824 | `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/summary.json:11-26` |

**Hinweis:** LLM-Judge und Baselines sind Doc-Evidence (vgl. `docs/status_pack/2026-01-08/03_evaluation_results.md:109-112`)

### 5.4 Factuality (FRANK Manifest, n=200) – Doc-Evidence

**Run-Ordner:** `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/`

| Metric | Value | 95% CI | Quelle |
|---|---:|---|---|
| F1 | 0.79 | [0.73, 0.84] | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:13-17` |
| Precision | 0.79 | [0.71, 0.86] | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:29-33` |
| Recall | 0.80 | [0.73, 0.87] | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:34-38` |
| AUROC | 0.89 | - | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/summary.json:23` |

**Hinweis:** `predictions.jsonl` fehlt; Metriken aus Status-Pack übernommen (vgl. `docs/status_pack/2026-01-08/03_evaluation_results.md:59-65`)

---

## 6. Qualitative casebook

### 6.1 Factuality case set (Repo-Evidence)

**Quelle:** `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl`

Siehe `ch6_qual_cases.md` für detaillierte Fallbeschreibungen mit:
- Exact JSONL line numbers
- Gold/pred labels
- Issue spans summary
- Evidence quotes (wenn verfügbar)
- Article/summary excerpts

### 6.2 Coherence/Readability – Repo-Evidence

**Coherence:**
- `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/predictions.jsonl` (200 Zeilen)
- Enthält: `example_id`, `pred`, `gt_norm`, `issue_spans`, `top_issues`

**Readability:**
- `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/predictions.jsonl` (200 Zeilen)
- Enthält: `example_id`, `pred`, `gt_norm`, `issue_spans`, `top_issues`

Siehe `ch6_qual_cases.md` für qualitative Beispiele.

---

## 7. Evidence gate / decision policy

### 7.1 Evidence Gate (Claim‑Level)

Gate‑Invarianten:
- „correct"/„incorrect" nur wenn `evidence_found=True`
- „incorrect" ohne Evidence → „uncertain"
- Quelle: `app/services/agents/factuality/claim_verifier.py:326-356`

### 7.2 Decision policy (Run‑Level)

Aus `results/evaluation/runs/results/evidence_gate_test_count_as_error.json:30-36`:
- `decision_mode="issues"`
- `error_threshold=1`
- `severity_min="low"`
- `uncertainty_policy="count_as_error"`

---

## 8. Reproducibility instructions

Siehe `ch6_run_repro.md` für vollständige Reproduktionsanweisungen.

---

## 9. Newly generated runs on HEAD

**Coherence Run (regenerated):**
- Run-ID: `coherence_20260107_205123_gpt-4o-mini_v1_seed42`
- Date: 2026-01-30 (predictions.jsonl regenerated)
- Artefakte: `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/`
- Status: ✅ Vollständig (predictions.jsonl, summary.json, summary.md, run_metadata.json)

**Readability Run (regenerated):**
- Run-ID: `readability_20260116_170832_gpt-4o-mini_v1_seed42`
- Date: 2026-01-30 (predictions.jsonl regenerated)
- Artefakte: `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`
- Status: ✅ Vollständig (predictions.jsonl, summary.json, summary.md, run_metadata.json)

**Hinweis:** Diese Runs wurden aus historischen Metadaten rekonstruiert und mit aktuellen Scripts neu generiert, um vollständige `predictions.jsonl` zu erhalten.

---

## 10. Missing evidence checklist

1. **FRANK Manifest `predictions.jsonl` fehlt:**
   - `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/predictions.jsonl`
   - Status: Nur `summary.json`/`summary.md`/`run_metadata.json` vorhanden

2. **Evidence Quotes in Factuality Examples:**
   - `evidence_quote` Feld wurde zu `IssueSpan` hinzugefügt (vgl. `app/models/pydantic.py:27`)
   - Aktuelle Examples-Datei enthält noch keine `evidence_quote` (Run wurde vor Code-Änderung erstellt)
   - Nach Re-Run: `issue_spans[].evidence_quote` wird verfügbar sein

3. **Baseline/Judge Runs:**
   - Coherence Judge, Coherence Baselines, Factuality Baselines sind Doc-Evidence
   - Referenzen: `docs/status_pack/2026-01-08/06_appendix_artifacts_index.md`
