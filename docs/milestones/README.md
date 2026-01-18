# Milestones: System-Entstehung

**Zweck:** Chronologische Übersicht über die Entwicklung des Verifikationssystems von M1 bis M12.

---

## Übersicht

| Milestone | Thema | Status | Beschreibung |
|-----------|-------|--------|--------------|
| **M1** | Projekt-Setup & Minimal-API | ✅ | Grundstruktur, FastAPI, Konfiguration |
| **M2** | Datenmodell & Persistenzschicht (Postgres) | ✅ | Postgres-Schema, SQLAlchemy, Testskript |
| **M3** | Eval-Core-Skelett | ✅ | Pipeline-Grundgerüst, Dummy-Agenten |
| **M4** | Graph-Modell (Neo4j) | ✅ | Neo4j-Integration, Graph-Persistenz |
| **M5** | Factuality-Agent & Verifikationslogik | ✅ | Erster echter Agent, LLM-Abstraktion |
| **M6** | Claim-basierter Factuality-Agent & Evaluationsinfrastruktur | ✅ | Claim-Extraction, Evidence-Retrieval, Evaluationsskripte |
| **M7** | Kohärenz-Agent | ✅ | Coherence-Agent, Satzstruktur-Analyse |
| **M8** | Readability-Agent | ✅ | Readability-Agent, Satzkomplexität, Struktur |
| **M9** | Explainability-Modul | ✅ | Findings-Aggregation, Severity-Levels, Top-Spans |
| **M10** | Evaluation & Vergleich mit klassischen Metriken | ✅ | Systematische Evaluation, Baselines, LLM-as-a-Judge |
| **M11** | Orchestrierung & Integration | ❌ | Entfällt (nicht im Thesis-Scope) |
| **M12** | Streamlit Dashboard (UI) | ✅ | Interaktives Dashboard, Demo, Inspektion |

**Reihenfolge:** M1 → M2 → M3 → M4 → M5 → M6 → M7 → M8 → M9 → M10 → M12

---

## Detaillierte Milestones

### M1: Projekt-Setup & Minimal-API
**Ziel:** Grundstruktur des Projekts mit lauffähiger FastAPI.

**Ergebnis:**
- Projektstruktur (`app/`, `api/`, `core/`, `services/`, `models/`, `db/`)
- FastAPI mit `/health` und `/verify` Endpunkten
- Konfiguration über `.env` und `pydantic-settings`
- Basis-Abhängigkeiten (FastAPI, Uvicorn, Pydantic, SQLAlchemy, Neo4j-Treiber)

**Dokumentation:** [M1_setup.md](M1_setup.md)

---

### M2: Datenmodell & Persistenzschicht (Postgres)
**Ziel:** Datenhaltung mit Postgres-Schema und SQLAlchemy-Integration.

**Ergebnis:**
- Postgres 16 als Docker-Container
- Schema mit Tabellen: `datasets`, `articles`, `summaries`, `runs`, `verification_results`, `errors`, `explanations`
- SQLAlchemy-Session-Management
- Testskript für End-to-End-Persistenz

**Dokumentation:** [M2_datenmodell.md](M2_datenmodell.md)

---

### M3: Eval-Core-Skelett
**Ziel:** Pipeline-Grundgerüst mit Dummy-Agenten.

**Ergebnis:**
- `VerificationPipeline` mit Agent-Orchestrierung
- Dummy-Agenten für Factuality, Coherence, Readability
- Pydantic-Modelle für `AgentResult`
- Integration in `/verify` Endpunkt

**Dokumentation:** [M3_eval_core_skelett.md](M3_eval_core_skelett.md)

---

### M4: Graph-Modell (Neo4j)
**Ziel:** Neo4j-Integration für Graph-Persistenz.

**Ergebnis:**
- Neo4j als Docker-Container
- Graph-Persistenz für Runs, Examples, Issues
- Cross-Store-Konsistenz (Postgres + Neo4j)

**Dokumentation:** [M4_graph_modell.md](M4_graph_modell.md)

---

### M5: Factuality-Agent & Verifikationslogik
**Ziel:** Erster echter Agent mit LLM-Abstraktion.

**Ergebnis:**
- `LLMClient` Interface mit `OpenAIClient` und `FakeLLMClient`
- Factuality-Agent mit Satz-für-Satz-Analyse
- JSON-Parsing, Score-Berechnung, Issue-Extraktion
- Integration in Pipeline

**Dokumentation:** [M5_eval_core_factuality_agent.md](M5_eval_core_factuality_agent.md)

---

### M6: Claim-basierter Factuality-Agent & Evaluationsinfrastruktur
**Ziel:** Strukturierte Claim-Extraction und Evidence-Retrieval.

**Ergebnis:**
- Claim-Extraction aus Summaries
- Evidence-Retrieval aus Artikeln
- Claim-Verification gegen Evidence
- Evaluationsskripte für FRANK-Datensatz

**Dokumentation:** [M6_claim_basierter_factuality_agent_und_evaluationsinfrastruktur.md](M6_claim_basierter_factuality_agent_und_evaluationsinfrastruktur.md)

---

### M7: Kohärenz-Agent
**Ziel:** Coherence-Agent für logische Konsistenz.

**Ergebnis:**
- Coherence-Agent mit Satzstruktur-Analyse
- Logische Übergänge, Referenzklarheit
- Issue-Typen: CONTRADICTION, MISSING_TRANSITION, UNCLEAR_REFERENCE

**Dokumentation:** [M7_kohärenz_agent.md](M7_kohärenz_agent.md)

---

### M8: Readability-Agent
**Ziel:** Readability-Agent für Lesbarkeitsbewertung.

**Ergebnis:**
- Readability-Agent mit Satzkomplexität-Analyse
- Struktur-Bewertung, Lesbarkeits-Score
- Issue-Typen: COMPLEX_SENTENCE, POOR_STRUCTURE, UNCLEAR_REFERENCE

**Dokumentation:** [M8_readability_agent.md](M8_readability_agent.md)

---

### M9: Explainability-Modul
**Ziel:** Aggregation von Agent-Outputs zu erklärbaren Findings.

**Ergebnis:**
- Explainability-Modul mit Findings-Aggregation
- Severity-Levels (low, medium, high)
- Top-Spans, Deduplizierung, Merge-Logik
- Persistenz in Postgres + Neo4j

**Dokumentation:** [M9_explainability_modul.md](M9_explainability_modul.md)

---

### M10: Evaluation & Vergleich mit klassischen Metriken
**Ziel:** Systematische Evaluation gegen Human Ratings und Baselines.

**Ergebnis:**
- Evaluationsskripte für FRANK (Factuality) und SummEval (Coherence, Readability)
- Klassische Baselines: Flesch, Flesch-Kincaid, Gunning Fog
- LLM-as-a-Judge als Vergleichsmethode
- Metriken: Spearman ρ, Pearson r, MAE, RMSE, R², F1, Balanced Accuracy, AUROC
- **Agent übertrifft klassische Baselines** (Readability: ρ = 0.402 vs -0.05)
- **Agent vergleichbar mit LLM-as-a-Judge** (Readability: ρ = 0.402 vs 0.280)

**Dokumentation:** [M10_evaluation_setup.md](M10_evaluation_setup.md)

---

### M11: Orchestrierung & Integration
**Status:** ❌ Entfällt (nicht im Thesis-Scope)

**Grund:** Fokus liegt auf Evaluation (M10) und UI-Demo (M12). Erweiterte Orchestrierung (LangChain, LangGraph, n8n) ist optional und wird nicht umgesetzt.

**Dokumentation:** [M11_orchestrierung_und_integration.md](M11_orchestrierung_und_integration.md)

---

### M12: Streamlit Dashboard (UI)
**Ziel:** Interaktives Dashboard für Demo und Inspektion.

**Ergebnis:**
- Streamlit-UI mit drei Tabs: Verify, Runs, Status
- Interaktive Verifikation (Article + Summary → Scores + Findings)
- Run-Anzeige aus Postgres
- Explainability-Visualisierung
- Dataset-Auswahl (FRANK, SummEval)
- Error-Injection für Demo-Zwecke

**Dokumentation:** [M12_streamlit_interface.md](M12_streamlit_interface.md)

---

## System-Architektur (Final)

Nach M12 besteht das System aus:

1. **API-Layer:** FastAPI mit `/verify` Endpunkt
2. **Pipeline:** `VerificationPipeline` orchestriert drei Agenten (Factuality, Coherence, Readability)
3. **Agenten:** Strukturierte Analyse mit IssueSpans, Scores, Verdicts
4. **Explainability:** Aggregation zu Findings mit Severity-Levels
5. **Persistence:** Postgres (relational) + Neo4j (graph)
6. **UI:** Streamlit-Dashboard für Demo und Inspektion
7. **Evaluation:** Reproduzierbare Evaluationsskripte mit Baselines und LLM-as-a-Judge

---

## Weitere Dokumentation

- **Architecture Overview:** `docs/status/architecture_overview.md`
- **Evaluation Results:** `docs/milestones/M10_evaluation_setup.md`
- **Status Reports:** `docs/status/readability_status.md`, `docs/status/factuality_status.md`, `docs/status/coherence_status.md`
- **Persistence:** `docs/status/persistence_audit.md`
- **Explainability:** `docs/status/explainability_spec.md`

---

## Reproduzierbarkeit

**Git Tags:**
- `readability-final-2026-01-16`
- `thesis-snapshot-2026-01-17`

**Run-Artefakte:**
- `results/evaluation/readability/readability_20260116_170832_gpt-4o-mini_v1_seed42/`
- `results/evaluation/factuality/judge_factuality_*/`
- `results/evaluation/baselines/baselines_readability_flesch_fk_fog_*/`

**Detail-Reports:** Siehe `docs/README.md` für vollständige Übersicht.

