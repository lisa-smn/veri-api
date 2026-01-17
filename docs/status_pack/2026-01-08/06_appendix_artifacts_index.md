# Appendix: Artefakte-Index

**Datum:** 2026-01-08  
**Zweck:** Vollständige Liste aller Run-Ordner, die in den Evaluation-Ergebnissen verwendet wurden.

---

## Coherence-Evaluation

### Agent-Run

| Run-ID | Pfad | Modell | Prompt | Seed | Dataset | n_used | Wichtigste Datei |
|--------|------|--------|--------|------|---------|--------|------------------|
| `coherence_20260107_205123_gpt-4o-mini_v1_seed42` | `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/` | gpt-4o-mini | v1 | 42 | sumeval_clean.jsonl | 200 | `summary.md` |

**Artefakte:**
- `summary.json`: Metriken + CIs
- `summary.md`: Human-readable Zusammenfassung
- `predictions.jsonl`: Pro-Beispiel-Vorhersagen
- `run_metadata.json`: Timestamp, Git-Commit, Python-Version, Seed, Config

---

### LLM-Judge-Run

| Run-ID | Pfad | Modell | Rubric | Judgments | Temp | Seed | Dataset | n_used | Wichtigste Datei |
|--------|------|--------|--------|-----------|------|------|---------|--------|------------------|
| `coherence_judge_20260107_234710_gpt-4o-mini_v1_n3_seed42` | `results/evaluation/coherence_judge/coherence_judge_20260107_234710_gpt-4o-mini_v1_n3_seed42/` | gpt-4o-mini | v1 | 3 | 0.0 | 42 | sumeval_clean.jsonl | 200 | `summary.md` |

**Artefakte:**
- `summary.json`: Metriken + CIs + Judge-Config
- `summary.md`: Human-readable Zusammenfassung
- `predictions.jsonl`: Pro-Beispiel-Vorhersagen (inkl. `pred_raw_scores_1_to_5`, `pred_norm_per_judge`, `main_issue_mode`, `confidence_mean`)
- `run_metadata.json`: Timestamp, Git-Commit, Python-Version, Seed, Judge-Config
- `cache.jsonl`: Optional (wenn Cache aktiv)

---

### Baseline-Runs

| Run-ID | Pfad | Baseline | Seed | Dataset | n_used | n_no_ref | Wichtigste Datei |
|--------|------|----------|------|---------|--------|----------|------------------|
| `coherence_rouge_l_20260107_230323_seed42` | `results/evaluation/coherence_baselines/coherence_rouge_l_20260107_230323_seed42/` | rouge_l | 42 | sumeval_clean.jsonl | 0 | 1700 | `summary.md` |
| `coherence_bertscore_20260107_230512_seed42` | `results/evaluation/coherence_baselines/coherence_bertscore_20260107_230512_seed42/` | bertscore | 42 | sumeval_clean.jsonl | 0 | 1700 | `summary.md` |

**Hinweis:** Beide Baseline-Runs haben `n_used=0` und `n_no_ref=1700`, da SummEval keine Referenzsummaries enthält. Daher sind ROUGE-L und BERTScore nicht auswertbar.

**Artefakte:**
- `summary.json`: Metriken (alle 0.0) + Warnung `n_no_ref`
- `summary.md`: Human-readable Zusammenfassung + Warnung
- `run_metadata.json`: Timestamp, Git-Commit, Python-Version, Seed, Config

---

## Factuality-Evaluation

### Agent-Run

| Run-ID | Pfad | Modell | Prompt | Manifest | Dataset Signature | n_used | Wichtigste Datei |
|--------|------|--------|--------|----------|-------------------|--------|------------------|
| `factuality_agent_manifest_20260107_215431_gpt-4o-mini` | `results/evaluation/factuality/factuality_agent_manifest_20260107_215431_gpt-4o-mini/` | gpt-4o-mini | v1 | frank_subset_manifest.jsonl | c32c25988d7a041fea833c132f4bd2bcc6484de4c22a157d114994e9812eb299 | 200 | `summary.md` |

**Artefakte:**
- `summary.json`: Binäre Metriken (F1, BalAcc, AUROC, MCC, Precision, Recall, Accuracy, Specificity) + CIs + Confusion Matrix
- `summary.md`: Human-readable Zusammenfassung
- `predictions.jsonl`: Pro-Beispiel-Vorhersagen (inkl. `gt_has_error`, `gt_score`, `pred_has_error`, `agent_score`, `num_issues`, `max_severity`)
- `run_metadata.json`: Timestamp, Git-Commit, Python-Version, Manifest-Pfad, Dataset-Signature, Config

---

### LLM-Judge-Runs

| Run-ID | Pfad | Modell | Prompt | Judgments | Temp | Seed | Dataset | n_used | Wichtigste Datei |
|--------|------|--------|--------|-----------|------|------|---------|--------|------------------|
| `judge_factuality_20260116_231906_gpt-4o-mini_v2_binary_seed42` | `results/evaluation/factuality/judge_factuality_20260116_231906_gpt-4o-mini_v2_binary_seed42/` | gpt-4o-mini | v2_binary | 3 | 0.0 | 42 | frank_smoke_balanced_50_seed42.jsonl | 50 | `summary.md` (Smoke, ✅ verifiziert) |
| `judge_factuality_20260116_233505_gpt-4o-mini_v2_binary_seed42` | `results/evaluation/factuality/judge_factuality_20260116_233505_gpt-4o-mini_v2_binary_seed42/` | gpt-4o-mini | v2_binary | 3 | 0.0 | 42 | frank_clean.jsonl | 200 | `summary.md` (Final, ✅ verifiziert) |

**Artefakte:**
- `summary.json`: Binäre Metriken (F1, BalAcc, AUROC, MCC, Precision, Recall, Accuracy, Specificity) + CIs
- `summary.md`: Human-readable Zusammenfassung
- `predictions.jsonl`: Pro-Beispiel-Vorhersagen (inkl. `gt_has_error`, `judge_verdict`, `judge_confidence`, `judge_score_norm`)
- `run_metadata.json`: Timestamp, Git-Commit, Python-Version, Seed, Judge-Config
- `cache.jsonl`: Optional (wenn Cache aktiv)

---

### Baseline-Runs

| Run-ID | Pfad | Baseline | Seed | Manifest | Dataset Signature | n_used | Wichtigste Datei |
|--------|------|----------|------|----------|-------------------|--------|------------------|
| `factuality_rouge_l_20260107_230519_seed42` | `results/evaluation/factuality_baselines/factuality_rouge_l_20260107_230519_seed42/` | rouge_l | 42 | frank_subset_manifest.jsonl | c32c25988d7a041fea833c132f4bd2bcc6484de4c22a157d114994e9812eb299 | 200 | `summary.md` |
| `factuality_bertscore_20260107_230523_seed42` | `results/evaluation/factuality_baselines/factuality_bertscore_20260107_230523_seed42/` | bertscore | 42 | frank_subset_manifest.jsonl | c32c25988d7a041fea833c132f4bd2bcc6484de4c22a157d114994e9812eb299 | 200 | `summary.md` |

**Hinweis:** Beide Baseline-Runs haben die gleiche `dataset_signature` wie der Agent-Run, daher sind sie direkt vergleichbar.

**Artefakte:**
- `summary.json`: Regression-Metriken (Pearson, Spearman, MAE, RMSE, R²) + CIs + AUROC + Best-F1 (falls berechnet)
- `summary.md`: Human-readable Zusammenfassung + Warnung (falls `dependencies_ok=false`)
- `predictions.jsonl`: Pro-Beispiel-Vorhersagen (inkl. `gold`, `pred`, `has_error`)
- `run_metadata.json`: Timestamp, Git-Commit, Python-Version, Seed, Manifest-Pfad, Dataset-Signature, Config, `dependencies_ok`, `missing_packages`

---

## Readability-Evaluation

### Agent-Run

| Run-ID | Pfad | Modell | Prompt | Seed | Dataset | n_used | Wichtigste Datei |
|--------|------|--------|--------|------|---------|--------|------------------|
| `readability_20260116_170832_gpt-4o-mini_v1_seed42` | `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/` | gpt-4o-mini | v1 | 42 | sumeval_clean.jsonl | 200 | `summary.md` |

**Artefakte:**
- `summary.json`: Metriken + CIs
- `summary.md`: Human-readable Zusammenfassung
- `predictions.jsonl`: Pro-Beispiel-Vorhersagen (inkl. `pred_agent`, `pred_judge`, `gt_norm`)
- `run_metadata.json`: Timestamp, Git-Commit (`558e17442542d9a1d5034895c7afb1b35f2d675b`), Python-Version, Seed, Config

**Metriken:**
- Spearman ρ = 0.402 [0.268, 0.512]
- Pearson r = 0.390 [0.292, 0.468]
- MAE = 0.283 [0.263, 0.302]
- RMSE = 0.316 [0.300, 0.332]
- R² = -2.773

---

### Baseline-Runs

| Run-ID | Pfad | Baselines | Seed | Dataset | n_used | Wichtigste Datei |
|--------|------|-----------|------|---------|--------|------------------|
| `baselines_readability_flesch_fk_fog_20260116_175246_seed42` | `results/evaluation/baselines/baselines_readability_flesch_fk_fog_20260116_175246_seed42/` | Flesch, Flesch-Kincaid, Gunning Fog | 42 | sumeval_clean.jsonl | 200 | `summary.md` |
| `baselines_readability_coherence_flesch_fk_fog_20260116_175253_seed42` | `results/evaluation/baselines/baselines_readability_coherence_flesch_fk_fog_20260116_175253_seed42/` | Flesch, Flesch-Kincaid, Gunning Fog | 42 | sumeval_clean.jsonl | 200 | `summary.md` |

**Hinweis:** ROUGE/BERTScore/BLEU/METEOR nicht berechenbar, da SummEval keine Referenz-Zusammenfassungen enthält.

**Artefakte:**
- `summary.json`: Baseline-Metriken pro Target (readability/coherence)
- `summary.md`: Human-readable Zusammenfassung
- `predictions.jsonl`: Pro-Beispiel-Vorhersagen (inkl. `flesch`, `flesch_kincaid`, `gunning_fog`, `gt_readability_norm`, `gt_coherence_norm`)
- `run_metadata.json`: Timestamp, Git-Commit, Python-Version, Seed, Config

---

## Aggregations-Matrizen

### Coherence

**Pfad:** `results/evaluation/summary_coherence_matrix.csv` und `summary_coherence_matrix.md` (falls vorhanden)

**Inhalt:**
- Vergleichstabelle: Agent, LLM-Judge, Baselines (ROUGE-L, BERTScore)
- Spalten: `run_id`, `method`, `pearson`, `spearman`, `mae`, `rmse`, `r_squared`, `n_used`, `seed`, `model`, `prompt_version`

**Generierung:**
```bash
python3 scripts/aggregate_coherence_runs.py \
  --inputs results/evaluation/coherence \
           results/evaluation/coherence_judge \
           results/evaluation/coherence_baselines
```

---

### Factuality

**Pfad:** `results/evaluation/summary_factuality_matrix.csv` und `summary_factuality_matrix.md` (falls vorhanden)

**Inhalt:**
- Vergleichstabelle: Agent (binäre Metriken), Baselines (Regression-Metriken)
- Spalten: `run_id`, `method`, `f1`, `balanced_accuracy`, `auroc`, `mcc`, `pearson`, `spearman`, `mae`, `rmse`, `n_used`, `dataset_signature`, `dependencies_ok`, `allow_dummy`, `missing_packages`

**Generierung:**
```bash
python3 scripts/aggregate_factuality_runs.py \
  --inputs results/evaluation/factuality \
           results/evaluation/factuality_baselines
```

---

### Baselines (Readability/Coherence)

**Pfad:** `results/evaluation/baselines/summary_matrix.csv` und `summary_matrix.md`

**Inhalt:**
- Vergleichstabelle: Baselines (Flesch, Flesch-Kincaid, Gunning Fog) pro Target (readability, coherence)
- Spalten: `run_id`, `baseline`, `target`, `pearson`, `spearman`, `mae`, `rmse`, `r_squared`, `n`, `seed`, `has_references`

**Generierung:**
```bash
python3 scripts/aggregate_baseline_runs.py \
  results/evaluation/baselines/* \
  --out results/evaluation/baselines/summary_matrix
```

---

## Manifest-Dateien

### FRANK Subset Manifest

**Pfad:** `data/frank/frank_subset_manifest.jsonl`  
**Meta:** `data/frank/frank_subset_manifest.meta.json`

**Inhalt:**
- Standardisiertes Subset für faire Agent/Baseline-Vergleiche
- Felder: `id`, `hash`, `model_name`, `article_text`, `summary_text`, `reference_text`, `gold_has_error`, `gold_score`, `meta`
- `dataset_signature`: SHA256-Hash der sortierten Example-IDs

**Generierung:**
```bash
python3 scripts/build_frank_subset_manifest.py \
  --benchmark data/frank/benchmark_data.json \
  --annotations data/frank/human_annotations.json \
  --out data/frank/frank_subset_manifest.jsonl
```

---

## Neo4j-Screenshots (optional)

**Falls vorhanden:** `docs/neo4j_screenshots/` oder `results/neo4j/`

**Inhalt:**
- Graph-Visualisierungen von Verification-Runs
- Claim-Evidence-Verbindungen
- Issue-Span-Überschneidungen

---

## Vollständige Artefakt-Struktur

Jeder Run-Ordner enthält:

```
<run_id>/
  ├── summary.json          # Metriken + CIs + Metadaten (JSON)
  ├── summary.md            # Human-readable Zusammenfassung (Markdown)
  ├── predictions.jsonl     # Pro-Beispiel-Vorhersagen (JSONL)
  ├── run_metadata.json     # Timestamp, Git-Commit, Python-Version, Seed, Config (JSON)
  └── cache.jsonl           # Optional (wenn Cache aktiv, JSONL)
```

**Zugriff:**
- Alle Runs: `results/evaluation/**/summary.json`
- Aggregation: `scripts/aggregate_*.py`
- Vergleich: `results/evaluation/summary_*_matrix.csv/.md`

---

**Details zu Evaluation:** Siehe `03_evaluation_results.md`.  
**Details zu Metriken:** Siehe `04_metrics_glossary.md`.

---

## Status-Dokumente

**Pfad:** `docs/status/`

**Inhalt:**
- `readability_status.md`: ✅ Final (Setup, Vergleichstabelle, Interpretation, Reproduzierbarkeit, Artefakte)
- `coherence_status.md`: ✅ Final (Setup, Vergleichstabelle, Interpretation, Reproduzierbarkeit, Artefakte)
- `factuality_status.md`: ✅ Final (Setup, Vergleichstabelle, Interpretation, Reproduzierbarkeit, Artefakte)

**Zweck:** Kompakte, supervisor-lesbare Status-Reports für jede Dimension mit Setup, Ergebnissen, Interpretation und Reproduzierbarkeit.

---

## Persistence & Explainability Proof

**Pfad:** `docs/status/`

**Inhalt:**
- `fig_explainability_persistence_proof.md`: ✅ Mini-Abbildung (Quality Gate / Proof Summary für Methodik/Anhang)
- `explainability_persistence_proof.md`: ✅ Proof-Report (run_id=proof-test-001, Postgres 4/4, Neo4j 5/5, Cross-Store 3/3)
- `persistence_audit.md`: ✅ Cross-Store Konsistenz-Audit (Postgres + Neo4j, run_id overlap)

**Zweck:** Validierung der Persistenz-Korrektheit für Explainability-Ergebnisse in beiden Stores (Postgres + Neo4j) mit run_id-basierter Verknüpfung.

