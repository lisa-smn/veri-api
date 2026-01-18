# Veri-API Dokumentation

**Zweck:** Übersicht über die wichtigsten Dokumente für schnellen Einstieg (10-Minuten-Verständnis)

---

## Einstieg

**Für Betreuerinnen (Supervisor):**
1. [Architecture Overview](status/architecture_overview.md) - System-Übersicht (Pipeline, Agents, Persistence)
2. [Evaluation Results (M10)](milestones/M10_evaluation_setup.md) - Finale Evaluationsergebnisse
3. [UI Dashboard (M12)](milestones/M12_streamlit_interface.md) - Wie starten + was kann es

**Für Entwickler:**
1. [Architecture Overview](status/architecture_overview.md) - System-Design
2. [Explainability Spec](status/explainability_spec.md) - Contracts + Aggregationsregeln
3. [Persistence Audit](status/persistence_audit.md) - Postgres + Neo4j Schema

---

## Wichtigste Dokumente

### System & Architektur

- **[Architecture Overview](status/architecture_overview.md)** - High-level Übersicht (Pipeline, Agents, Persistence, API)
- **[Explainability Spec](status/explainability_spec.md)** - Was macht Explainability + warum (Input/Output Contracts)

### Evaluation (M10)

- **[Final Evaluation Stand](status/evaluation_final.md)** ⭐ **Canonical Final-Dokument** - Kern-Ergebnisse, 1 Tabelle, 5 Bullet-Kernaussagen
- **[M10 Evaluation Setup](milestones/M10_evaluation_setup.md)** - Vollständige Evaluationsansätze, Methoden, detaillierte Ergebnisse
- **[Readability Status](status/readability_status.md)** - Detail-Report Readability (Agent vs Judge vs Baselines)
- **[Factuality Status](status/factuality_status.md)** - Detail-Report Factuality (Agent vs Judge)
- **[Coherence Status](status/coherence_status.md)** - Detail-Report Coherence (Agent vs Judge)

### UI & Demo (M12)

- **[M12 Streamlit Interface](milestones/M12_streamlit_interface.md)** - Dashboard Start + Features + Grenzen

### Persistence

- **[Persistence Audit](status/persistence_audit.md)** - Postgres + Neo4j Schema + Cross-Store Consistency
- **[Explainability Persistence Proof](status/explainability_persistence_proof.md)** - Proof: Explainability in beiden DBs

### Glossar & Metriken

- **[Metrics Glossary](status_pack/2026-01-08/04_metrics_glossary.md)** - Begriffe kurz erklärt (Spearman, R², AUROC, etc.)

### Repo-Inventar

- **[Repo-Inventar & Dokumentations-Audit](status/repo_inventory.md)** - Vollständige Übersicht über Repo-Struktur, Entry Points, Sprachkonsistenz

---

## Milestones

**Vollständige Übersicht:** [Milestones README](milestones/README.md) - Chronologische Dokumentation der System-Entstehung (M1-M12)

**Kern-Milestones:**
- **M10:** [Evaluation Setup](milestones/M10_evaluation_setup.md) ✅ Abgeschlossen
- **M11:** [Orchestrierung & Integration](milestones/M11_orchestrierung_und_integration.md) ❌ Entfällt
- **M12:** [Streamlit Interface](milestones/M12_streamlit_interface.md) ✅ Abgeschlossen

**Reihenfolge:** M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9 → M10 → M12

---

## Weitere Details

**Status-Dokumente:** `docs/status/*.md` (Detail-Reports, Audits, Specs)  
**Status Pack:** `docs/status_pack/2026-01-08/` (Executive Summary, Evaluation Results, Glossary)  
**Milestones:** `docs/milestones/` (M1-M12) - Vollständige Dokumentation der System-Entstehung

---

## Reproduzierbarkeit

**Git Tags:**
- `readability-final-2026-01-16`
- `thesis-snapshot-2026-01-17`

**Run-Artefakte:**
- `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`
- `results/evaluation/factuality/judge_factuality_*/`
- `results/evaluation/baselines/baselines_readability_flesch_fk_fog_*/`

**Detail-Reports:** Siehe "Wichtigste Dokumente" oben.

---

