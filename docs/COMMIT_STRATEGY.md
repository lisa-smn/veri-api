# Git-Commit-Strategie für Bachelorarbeit

**Datum:** 2025-01-07  
**Zweck:** Kuratierte Auswahl der wichtigsten Änderungen für Bachelorarbeit

---

## 1. Prioritäten-Übersicht

### 🔴 Top-Priorität (muss rein)

**A) M10 Coherence Evaluation**
- `scripts/eval_sumeval_coherence.py` - Haupt-Eval-Script (RMSE, Bootstrap-CIs, Artefakte)
- `scripts/eval_sumeval_coherence_baselines.py` - ROUGE-L/BERTScore Baselines
- `scripts/stress_test_coherence.py` - Shuffle/Injection Tests
- `scripts/aggregate_coherence_runs.py` - Vergleichstabelle
- `app/services/agents/coherence/coherence_agent.py` - issue_type explizit gesetzt
- `docs/m10_coherence_eval_plan.md` - Vollständige Dokumentation

**B) M9 Explainability**
- `app/services/explainability/__init__.py` - Neues Modul
- `app/services/explainability/explainability_models.py` - Pydantic Models
- `app/services/explainability/explainability_service.py` - Service-Logik
- `app/pipeline/verification_pipeline.py` - Integration
- `app/api/routes.py` - API-Endpoint
- `app/models/pydantic.py` - Response-Models
- `tests/api/test_verify_explainability.py` - API-Tests
- `tests/unit/test_explainability_service.py` - Unit-Tests

**C) Agenten-Code (kritische Fixes)**
- `app/services/agents/coherence/coherence_agent.py` - issue_type Fix
- `app/services/agents/factuality/claim_verifier.py` - Evidence-Gate Logik
- `app/services/agents/factuality/factuality_agent.py` - Issue-Span Aggregation

---

### 🟡 Gute Ergänzung (sollte rein)

**D) Evaluation-Infrastruktur**
- `configs/m10_factuality_runs.yaml` - M10 Evaluation Config
- `scripts/run_m10_factuality.py` - M10 Runner
- `scripts/select_best_tuned_run.py` - Tuning-Workflow
- `scripts/aggregate_m10_results.py` - M10 Aggregation
- `scripts/eval_factuality_binary_v2.py` - Factuality Eval (verbessert)

**E) Tests (Abdeckung)**
- `tests/unit/test_coherence_agent.py` - Coherence Tests
- `tests/unit/test_factuality_agent_unit.py` - Factuality Tests
- `tests/unit/test_readability_agent.py` - Readability Tests
- `tests/api/test_verify_route.py` - API Tests
- `tests/conftest.py` - Test-Fixtures

**F) Dokumentation (thesis-relevant)**
- `docs/milestones/M9_explainability_modul.md` - M9 Dokumentation
- `docs/milestones/M10_evaluation_setup.md` - M10 Dokumentation
- `docs/factuality_agent.md` - Factuality Agent Doku
- `README.md` - Haupt-README (falls aktualisiert)

---

### 🟢 Optional (nice-to-have)

**G) Tooling/Utilities**
- `app/services/run_manager.py` - Run-Management
- `app/services/analysis/metrics.py` - Metriken-Helpers
- `scripts/print_factuality_eval_bullets.py` - Bullet-Generator
- `scripts/print_runs.py` - Run-Printer
- `scripts/report_predictions.py` - Prediction-Reporter

**H) Weitere Dokumentation**
- `docs/finesumfact_audit.md` - FineSumFact Audit
- `docs/kolloquium_beweis_einfach.md` - Kolloquium-Beweis
- `docs/kolloquium_links.md` - Kolloquium-Links

---

### ⚠️ Explizit ausschließen (nicht committen)

**I) Outputs/Artefakte**
- `results/` - Alle Ergebnisse (bereits in .gitignore)
- `*.jsonl` in `results/` - Predictions/Caches
- `demo_payload.json`, `demo_request.py`, `test_beweis.py` - Demo-Scripts (nicht thesis-relevant)

**J) Root-Markdowns (verschieben oder ignorieren)**
- `M10_EVALUATION.md`, `M10_IMPLEMENTATION_SUMMARY.md`, `M10_TUNING_WORKFLOW.md` → `docs/notes/`
- `PROJECT_STATUS.md`, `QUICKSTART_M10.md` → `docs/notes/`
- `PRESENTATION_*.md` → `docs/notes/` (Präsentations-Notizen)
- `EVIDENCE_GATE_*.md`, `CACHE_FIX.md`, etc. → `docs/notes/` (Entwicklungs-Notizen)

**K) Archiv/Backup**
- `docs/archive/` - Bereits archiviert
- `scripts/archive_*.py` - Archiv-Scripts

**L) Alte/Deprecated**
- `scripts/eval_factuality_binary.py` - Gelöscht (gut, nicht committen)
- `results/finesumfact/run_*.json` - Gelöscht (gut, nicht committen)

---

## 2. .gitignore Prüfung

**Aktueller Status:**
- ✅ `results/` ist bereits ignoriert (Zeile 92-94)
- ✅ `data/` ist bereits ignoriert (Zeile 82)
- ⚠️ **Fehlt:** Explizite Ignorierung von `*predictions*.jsonl`, `*cache*.jsonl` in Scripts-Verzeichnissen

**Empfehlung (minimal-invasiv):**
```gitignore
# In .gitignore ergänzen (nach Zeile 94):
# Evaluation outputs / results
results/
!results/README.md
!results/.gitkeep

# Evaluation caches and predictions (in scripts dirs)
**/cache*.jsonl
**/predictions*.jsonl
```

**Hinweis:** Aktuell werden diese Dateien wahrscheinlich nicht getrackt (wegen `results/`), aber explizit ist besser.

---

## 3. Root-Markdowns Entscheidung

**Empfehlung:** Verschieben nach `docs/notes/` (nicht committen als Root-Files)

**Begründung:**
- Root sollte sauber bleiben (nur README.md)
- `docs/notes/` ist passender Ort für Entwicklungs-Notizen
- Thesis-relevante Docs sind bereits in `docs/milestones/`

**Dateien zum Verschieben:**
```bash
mkdir -p docs/notes
mv M10_*.md PROJECT_STATUS.md QUICKSTART_M10.md PRESENTATION_*.md \
   EVIDENCE_GATE_*.md CACHE_FIX.md DATASET_*.md EVALUATION_DATASETS.md \
   EXPLAINABILITY_VERIFICATION.md IMPROVEMENTS_*.md PROMPT_AND_FUZZY_IMPROVEMENTS.md \
   ARCHITECTURE_DIAGRAM.md docs/notes/
```

---

## 4. Git-Kommandos (Copy-Paste Ready)

### Schritt 1: Staged Files zurücksetzen (falls nötig)
```bash
cd /Users/lisasimon/PycharmProjects/veri-api
git restore --staged .
```

### Schritt 2: Root-Markdowns verschieben (optional, aber empfohlen)
```bash
mkdir -p docs/notes
mv M10_*.md PROJECT_STATUS.md QUICKSTART_M10.md PRESENTATION_*.md \
   EVIDENCE_GATE_*.md CACHE_FIX.md DATASET_*.md EVALUATION_DATASETS.md \
   EXPLAINABILITY_VERIFICATION.md IMPROVEMENTS_*.md PROMPT_AND_FUZZY_IMPROVEMENTS.md \
   ARCHITECTURE_DIAGRAM.md docs/notes/ 2>/dev/null || true
```

### Schritt 3: Commit 1 - M10 Coherence Evaluation
```bash
# M10 Coherence Evaluation Scripts
git add scripts/eval_sumeval_coherence.py
git add scripts/eval_sumeval_coherence_baselines.py
git add scripts/stress_test_coherence.py
git add scripts/aggregate_coherence_runs.py

# Coherence Agent Fix
git add app/services/agents/coherence/coherence_agent.py

# M10 Dokumentation
git add docs/m10_coherence_eval_plan.md

# Commit
git commit -m "M10: Coherence Evaluation Infrastructure

- Add eval_sumeval_coherence.py with RMSE, Bootstrap-CIs, and artifacts
- Add eval_sumeval_coherence_baselines.py (ROUGE-L, BERTScore)
- Add stress_test_coherence.py (shuffle/injection tests)
- Add aggregate_coherence_runs.py for comparison matrix
- Fix coherence_agent.py: set issue_type explicitly in IssueSpans
- Add comprehensive documentation (m10_coherence_eval_plan.md)"
```

### Schritt 4: Commit 2 - M9 Explainability
```bash
# Explainability Service
git add app/services/explainability/__init__.py
git add app/services/explainability/explainability_models.py
git add app/services/explainability/explainability_service.py

# Pipeline Integration
git add app/pipeline/verification_pipeline.py

# API Integration
git add app/api/routes.py
git add app/models/pydantic.py

# Tests
git add tests/api/test_verify_explainability.py
git add tests/unit/test_explainability_service.py
git add tests/conftest.py

# Dokumentation
git add docs/milestones/M9_explainability_modul.md

# Commit
git commit -m "M9: Explainability Module

- Add explainability service with structured reports
- Integrate explainability into verification pipeline
- Add /verify endpoint with explainability support
- Add comprehensive tests (API + unit)
- Update M9 milestone documentation"
```

### Schritt 5: Commit 3 - M10 Evaluation Config & Scripts
```bash
# M10 Config
git add configs/m10_factuality_runs.yaml

# M10 Scripts
git add scripts/run_m10_factuality.py
git add scripts/select_best_tuned_run.py
git add scripts/aggregate_m10_results.py
git add scripts/eval_factuality_binary_v2.py

# M10 Dokumentation
git add docs/milestones/M10_evaluation_setup.md

# Commit
git commit -m "M10: Factuality Evaluation Infrastructure

- Add m10_factuality_runs.yaml configuration
- Add run_m10_factuality.py runner
- Add select_best_tuned_run.py for tuning workflow
- Add aggregate_m10_results.py for result aggregation
- Update eval_factuality_binary_v2.py
- Update M10 milestone documentation"
```

### Schritt 6: Commit 4 - Agent Fixes & Tests
```bash
# Agent Fixes
git add app/services/agents/factuality/claim_verifier.py
git add app/services/agents/factuality/factuality_agent.py
git add app/services/agents/coherence/coherence_verifier.py
git add app/services/agents/readability/readability_agent.py
git add app/services/agents/readability/readability_verifier.py

# Tests
git add tests/unit/test_coherence_agent.py
git add tests/unit/test_factuality_agent_unit.py
git add tests/unit/test_readability_agent.py
git add tests/api/test_verify_route.py
git add tests/agents/test_claim_extractor_unit.py

# Commit
git commit -m "Fix: Agent improvements and test coverage

- Improve claim_verifier.py (evidence-gate logic)
- Update factuality_agent.py (issue span aggregation)
- Update coherence_agent.py and verifier
- Update readability_agent.py and verifier
- Add/update unit tests for all agents
- Add API route tests"
```

### Schritt 7: Commit 5 - Database & Infrastructure
```bash
# Database
git add app/db/postgres/persistence.py
git add app/db/postgres/schema.sql
git add app/db/neo4j/graph_persistence.py

# Service
git add app/services/verification_service.py

# Pipeline
git add app/pipeline/verification_pipeline.py

# Models
git add app/models/pydantic.py

# Commit
git commit -m "Infrastructure: Database and service updates

- Update postgres persistence and schema
- Update neo4j graph persistence
- Update verification service
- Update pipeline integration
- Update pydantic models"
```

### Schritt 8: Commit 6 - Documentation & Misc
```bash
# Dokumentation
git add docs/milestones/M5_eval_core_factuality_agent.md
git add docs/milestones/M6_claim_basierter_factuality_agent_und_evaluationsinfrastruktur.md
git add docs/milestones/M7_kohärenz_agent.md
git add docs/milestones/M8_readability_agent.md
git add docs/factuality_agent.md
git add docs/finesumfact_audit.md
git add README.md

# Scripts
git add scripts/convert_finesumfact.py
git add scripts/convert_frank.py
git add scripts/convert_sumeval.py

# Dashboard (optional)
git add app/dashboard/__init__.py
git add app/dashboard/streamlit_app.py

# .gitignore
git add .gitignore

# Requirements
git add requirements.txt

# Commit
git commit -m "Docs: Update milestone documentation and utilities

- Update M5-M8 milestone documentation
- Add factuality_agent.md documentation
- Add finesumfact_audit.md
- Update README.md
- Update data conversion scripts
- Add Streamlit dashboard
- Update .gitignore and requirements.txt"
```

### Schritt 9: Final Check
```bash
# Prüfe staged files
git diff --cached --name-only

# Warnung falls results/ oder große Dateien staged sind
if git diff --cached --name-only | grep -E "(results/|\.jsonl$|\.json$)" | grep -vE "(configs/|evaluation_configs/|\.gitignore)"; then
    echo "⚠️  WARNUNG: Ergebnisse oder große Dateien sind staged!"
    echo "Bitte prüfen und ggf. entfernen: git restore --staged <file>"
fi

# Zeige Status
git status --short
```

---

## 5. Zusammenfassung

**Commits:**
1. ✅ M10 Coherence Evaluation (6 Dateien)
2. ✅ M9 Explainability (9 Dateien)
3. ✅ M10 Factuality Evaluation (5 Dateien)
4. ✅ Agent Fixes & Tests (9 Dateien)
5. ✅ Database & Infrastructure (5 Dateien)
6. ✅ Documentation & Misc (10+ Dateien)

**Nicht committen:**
- ❌ `results/` (bereits ignoriert)
- ❌ `demo_payload.json`, `demo_request.py`, `test_beweis.py`
- ❌ Root-Markdowns (verschoben nach `docs/notes/`)
- ❌ `docs/archive/` (bereits archiviert)

**Gesamt:** ~44 Dateien für Bachelorarbeit (fokussiert, keine Artefakte)

