# Factuality-Agent Commit Scope (Bachelorarbeit)

**Datum:** 2025-01-07  
**Methode:** Datenbasierte Analyse (Imports, Entry Points, Abhängigkeiten)

---

## 1. Entry Points & Abhängigkeitsgraph

### Core Agent Module (app/services/agents/factuality/)

**Haupt-Agent:**
- `factuality_agent.py` → Importiert: `claim_models.Claim`, `claim_extractor.ClaimExtractor`, `claim_verifier.ClaimVerifier`, `app.models.pydantic.AgentResult`, `app.models.pydantic.IssueSpan`

**Claim-Extraktion:**
- `claim_extractor.py` → Importiert: `claim_models.Claim`, `app.llm.llm_client.LLMClient`

**Claim-Verifikation:**
- `claim_verifier.py` → Importiert: `claim_models.Claim`, `evidence_retriever.EvidenceRetriever`, `verifier_models.*`, `app.llm.llm_client.LLMClient`

**Evidence-Retrieval:**
- `evidence_retriever.py` → Keine internen Imports (nur stdlib)

**Models:**
- `claim_models.py` → Keine internen Imports (nur stdlib)
- `verifier_models.py` → Keine internen Imports (nur stdlib, pydantic)

**Nicht verwendet (Ablation/Deprecated):**
- `ablation_extractor.py` → ❌ Nicht in factuality_agent.py importiert
- `ablation_verifier.py` → ❌ Nicht in factuality_agent.py importiert

### Pipeline Integration

**Verification Pipeline:**
- `app/pipeline/verification_pipeline.py` → Importiert: `factuality_agent.FactualityAgent`
- `app/services/verification_service.py` → Nutzt: `VerificationPipeline` (enthält FactualityAgent)
- `app/api/routes.py` → Nutzt: `VerificationService` (enthält Pipeline mit FactualityAgent)

### Models & Schemas

**Pydantic Models:**
- `app/models/pydantic.py` → Enthält: `AgentResult`, `IssueSpan`, `PipelineResult`, `VerifyRequest`, `VerifyResponse` (alle für Factuality relevant)

---

## 2. Evaluation Scripts (aktuell vs. archiviert)

### ✅ Aktuell (verwendet)

**M10 Evaluation:**
- `scripts/run_m10_factuality.py` (29K, Jan 7) → **Haupt-Runner für M10**
- `scripts/select_best_tuned_run.py` (22K, Jan 7) → **Tuning-Workflow**
- `scripts/tune_from_baseline.py` (8.2K, Jan 7) → **Tuning-Script**
- `scripts/aggregate_m10_results.py` → **Result-Aggregation**

**Binary Evaluation:**
- `scripts/eval_factuality_binary_v2.py` (24K, Dec 24) → **Aktuelle Version (v2)**
- `scripts/eval_factuality_structured.py` (17K, Dec 28) → **Strukturierte Eval (neuer)**

### ❌ Archiviert/Deprecated

- `scripts/archive_eval_factuality_binary_v1.py` → **Archiviert (v1, nicht aktuell)**

**Entscheidung:** `eval_factuality_binary_v2.py` ist die aktuelle Version (v2, neuer als v1). `eval_factuality_structured.py` ist zusätzlich aktuell (neueste, Dec 28).

---

## 3. Tests

**Unit Tests:**
- `tests/unit/test_factuality_agent_unit.py` → **Factuality Agent Unit Tests**
- `tests/agents/test_claim_extractor_unit.py` → **Claim Extractor Tests**
- `tests/agents/test_factuality_agent_meta_sentence.py` → **Meta-Sentence Tests**

**API Tests:**
- `tests/api/test_verify_route.py` → **API-Tests (factuality betroffen)**

---

## 4. Configs & Dokumentation

### Configs

**M10 Config:**
- `configs/m10_factuality_runs.yaml` → **M10 Factuality Evaluation Config (21 Runs)**

**Evaluation Configs:**
- `evaluation_configs/factuality_finesumfact_test_v1.json` → **FineSumFact Test Config**
- `evaluation_configs/factuality_combined_test_v1.json` → **Combined Test Config**

### Dokumentation (thesis-relevant)

**Milestones:**
- `docs/milestones/M5_eval_core_factuality_agent.md` → **M5: Core Factuality Agent**
- `docs/milestones/M6_claim_basierter_factuality_agent_und_evaluationsinfrastruktur.md` → **M6: Claim-basierter Agent**
- `docs/milestones/M10_evaluation_setup.md` → **M10: Evaluation Setup (factuality-relevant)**

**Agent-Doku:**
- `docs/factuality_agent.md` → **Factuality Agent Dokumentation**

---

## 5. Kuratierte Datei-Liste nach Priorität

### 🔴 MUST (Core Agent + Models + Pipeline Hooks)

**Core Agent (6 Dateien):**
1. `app/services/agents/factuality/factuality_agent.py` → Haupt-Agent (Entry Point)
2. `app/services/agents/factuality/claim_extractor.py` → Claim-Extraktion
3. `app/services/agents/factuality/claim_verifier.py` → Claim-Verifikation (Evidence-Gate)
4. `app/services/agents/factuality/evidence_retriever.py` → Evidence-Retrieval
5. `app/services/agents/factuality/claim_models.py` → Claim-Dataclass
6. `app/services/agents/factuality/verifier_models.py` → Verifier-Models

**Pipeline Integration (3 Dateien):**
7. `app/pipeline/verification_pipeline.py` → Pipeline (nutzt FactualityAgent)
8. `app/services/verification_service.py` → Service (nutzt Pipeline)
9. `app/api/routes.py` → API (nutzt Service)

**Models (1 Datei):**
10. `app/models/pydantic.py` → AgentResult, IssueSpan, PipelineResult, VerifyRequest, VerifyResponse

**Begründung:** Ohne diese Dateien funktioniert der Factuality-Agent nicht. Core-Dependencies für Bachelorarbeit.

---

### 🟡 SHOULD (Eval Scripts + Runner + Selection Logic)

**M10 Evaluation (4 Dateien):**
11. `scripts/run_m10_factuality.py` → M10 Haupt-Runner
12. `scripts/select_best_tuned_run.py` → Tuning-Workflow
13. `scripts/tune_from_baseline.py` → Tuning-Script
14. `scripts/aggregate_m10_results.py` → Result-Aggregation

**Binary Evaluation (2 Dateien):**
15. `scripts/eval_factuality_binary_v2.py` → Aktuelle Binary Eval (v2)
16. `scripts/eval_factuality_structured.py` → Strukturierte Eval (neueste)

**Begründung:** Evaluation-Scripts sind essentiell für Reproduzierbarkeit. M10-Scripts sind aktuell (Jan 7), v2 ist aktueller als v1.

---

### 🟢 NICE (Docs/Configs)

**Configs (3 Dateien):**
17. `configs/m10_factuality_runs.yaml` → M10 Config (21 Runs)
18. `evaluation_configs/factuality_finesumfact_test_v1.json` → FineSumFact Config
19. `evaluation_configs/factuality_combined_test_v1.json` → Combined Config

**Dokumentation (4 Dateien):**
20. `docs/milestones/M5_eval_core_factuality_agent.md` → M5 Doku
21. `docs/milestones/M6_claim_basierter_factuality_agent_und_evaluationsinfrastruktur.md` → M6 Doku
22. `docs/milestones/M10_evaluation_setup.md` → M10 Doku (factuality-relevant)
23. `docs/factuality_agent.md` → Agent-Doku

**Begründung:** Configs und Doku machen Evaluation reproduzierbar. Thesis-relevant.

---

### ❌ EXCLUDE (nicht committen)

**Outputs/Artefakte:**
- `results/` → Bereits in .gitignore
- `**/predictions*.jsonl` → Evaluation-Outputs
- `**/cache*.jsonl` → Cache-Dateien

**Archiviert:**
- `scripts/archive_eval_factuality_binary_v1.py` → Archiviert (v1, nicht aktuell)

**Nicht verwendet:**
- `app/services/agents/factuality/ablation_extractor.py` → Nicht importiert
- `app/services/agents/factuality/ablation_verifier.py` → Nicht importiert

**Begründung:** Artefakte sind nicht versionierbar, archivierte/ungenutzte Dateien sind nicht relevant.

---

## 6. Git-Kommandos (Copy-Paste Ready)

### Commit 1: Factuality Agent Core

```bash
cd /Users/lisasimon/PycharmProjects/veri-api

# Core Agent
git add app/services/agents/factuality/factuality_agent.py
git add app/services/agents/factuality/claim_extractor.py
git add app/services/agents/factuality/claim_verifier.py
git add app/services/agents/factuality/evidence_retriever.py
git add app/services/agents/factuality/claim_models.py
git add app/services/agents/factuality/verifier_models.py

# Pipeline Integration
git add app/pipeline/verification_pipeline.py
git add app/services/verification_service.py
git add app/api/routes.py

# Models
git add app/models/pydantic.py

git commit -m "Factuality Agent: Core implementation with evidence-gate

- Add factuality_agent.py (sentence-based, claim extraction, verification)
- Add claim_extractor.py (LLM-based claim extraction with substring constraint)
- Add claim_verifier.py (evidence-gate logic, coverage validation)
- Add evidence_retriever.py (deterministic sliding-window retrieval)
- Add claim_models.py and verifier_models.py (structured data models)
- Integrate into verification pipeline, service, and API
- Update pydantic models (AgentResult, IssueSpan, PipelineResult)"
```

### Commit 2: Factuality Evaluation Scripts

```bash
# M10 Evaluation
git add scripts/run_m10_factuality.py
git add scripts/select_best_tuned_run.py
git add scripts/tune_from_baseline.py
git add scripts/aggregate_m10_results.py

# Binary Evaluation
git add scripts/eval_factuality_binary_v2.py
git add scripts/eval_factuality_structured.py

git commit -m "Factuality: Evaluation scripts and M10 infrastructure

- Add run_m10_factuality.py (M10 evaluation runner)
- Add select_best_tuned_run.py (tuning workflow and selection)
- Add tune_from_baseline.py (baseline tuning script)
- Add aggregate_m10_results.py (result aggregation)
- Add eval_factuality_binary_v2.py (binary evaluation v2)
- Add eval_factuality_structured.py (structured evaluation)"
```

### Commit 3: Factuality Tests + Docs/Configs

```bash
# Tests
git add tests/unit/test_factuality_agent_unit.py
git add tests/agents/test_claim_extractor_unit.py
git add tests/agents/test_factuality_agent_meta_sentence.py
git add tests/api/test_verify_route.py

# Configs
git add configs/m10_factuality_runs.yaml
git add evaluation_configs/factuality_finesumfact_test_v1.json
git add evaluation_configs/factuality_combined_test_v1.json

# Dokumentation
git add docs/milestones/M5_eval_core_factuality_agent.md
git add docs/milestones/M6_claim_basierter_factuality_agent_und_evaluationsinfrastruktur.md
git add docs/milestones/M10_evaluation_setup.md
git add docs/factuality_agent.md

git commit -m "Factuality: Tests, configs, and documentation

- Add unit tests for factuality agent and claim extractor
- Add API tests for verification route
- Add M10 factuality evaluation config (21 runs)
- Add evaluation configs for FineSumFact and combined datasets
- Update milestone documentation (M5, M6, M10)
- Add factuality_agent.md documentation"
```

### Safety Check

```bash
# Prüfe staged files
echo "=== Staged Files ==="
git diff --cached --name-only

# Warnung falls results/ oder große Dateien staged sind
echo ""
echo "=== Safety Check ==="
if git diff --cached --name-only | grep -E "(results/|predictions.*\.jsonl|cache.*\.jsonl|\.json$)" | grep -vE "(configs/|evaluation_configs/|\.gitignore)"; then
    echo "⚠️  WARNUNG: Ergebnisse oder große Dateien sind staged!"
    echo "Bitte prüfen und ggf. entfernen: git restore --staged <file>"
else
    echo "✅ Keine Ergebnisse oder großen Dateien staged"
fi

# Zeige finalen Status
echo ""
echo "=== Final Status ==="
git status --short
```

---

## 7. Zusammenfassung

**Total: 23 Dateien für Factuality-Agent**

| Kategorie | Dateien | Priorität |
|-----------|---------|-----------|
| Core Agent | 6 | 🔴 MUST |
| Pipeline Integration | 3 | 🔴 MUST |
| Models | 1 | 🔴 MUST |
| Eval Scripts | 6 | 🟡 SHOULD |
| Tests | 4 | 🟢 NICE |
| Configs | 3 | 🟢 NICE |
| Dokumentation | 4 | 🟢 NICE |

**Commits:**
1. Core Agent (10 Dateien)
2. Evaluation Scripts (6 Dateien)
3. Tests + Docs/Configs (11 Dateien)

**Ausschlüsse:**
- ❌ `results/`, `predictions*.jsonl`, `cache*.jsonl`
- ❌ `archive_eval_factuality_binary_v1.py` (archiviert)
- ❌ `ablation_extractor.py`, `ablation_verifier.py` (nicht verwendet)

**Entscheidungen:**
- ✅ `eval_factuality_binary_v2.py` ist aktuell (v2, neuer als v1)
- ✅ `eval_factuality_structured.py` ist aktuell (neueste, Dec 28)
- ✅ `ablation_*.py` werden nicht committet (nicht importiert)

