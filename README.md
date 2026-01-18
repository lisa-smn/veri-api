# Veri-API: Automatische Verifikation von LLM-generierten Zusammenfassungen

## Thesis Snapshot

**Tag:** `thesis-snapshot-2026-01-17`  
**Commit:** `558e17442542d9a1d5034895c7afb1b35f2d675b`  
**Scope Freeze:** No further feature changes after snapshot  
**Known non-blocking lint warnings:** `ruff check` WARN (style-only: Unicode in strings, import order in scripts, unused imports in non-core files)

---

## Quickstart (für neue Nutzer)

**Voraussetzungen:**
- Docker Desktop installiert (oder Docker + Docker Compose)
- Git installiert
- OpenAI API Key (kostenpflichtig, [hier anfordern](https://platform.openai.com/api-keys))

**Schritte:**

```bash
# 1. Repository klonen
git clone https://github.com/lisa-smn/veri-api.git
cd veri-api

# 2. Environment-Variablen vorbereiten
cp .env.example .env
# Bearbeite .env und setze OPENAI_API_KEY=sk-...

# 3. Alle Services starten (API, Postgres, Neo4j, UI)
docker compose up --build

# 4. Öffne im Browser:
# - UI: http://localhost:8501
# - API: http://localhost:8000
# - API Health: http://localhost:8000/health
```

**Services stoppen:**
```bash
docker compose down
```

**Daten zurücksetzen (⚠️ löscht alle Daten):**
```bash
docker compose down -v
```

**Troubleshooting:**
- **Port belegt:** Prüfe ob Ports 8000, 8501, 5433, 7474, 7687 frei sind
- **Kein OPENAI_API_KEY:** Setze `OPENAI_API_KEY=sk-...` in `.env`
- **Docker läuft nicht:** Starte Docker Desktop
- **API nicht erreichbar:** Warte 10-20 Sekunden nach `docker compose up`, prüfe Logs mit `docker compose logs api`

**Wichtig:** `.env` enthält Secrets und wird **nicht** ins Repo committed. Nur `.env.example` ist versioniert.

---

## Projektzweck

Veri-API ist ein System zur automatischen Verifikation von LLM-generierten Zusammenfassungen gegen Originalartikel. Das System prüft drei Dimensionen:

- **Factuality (Faktenprüfung)**: Sind die Behauptungen im Summary durch den Artikel belegt?
- **Coherence (Kohärenz)**: Ist der Summary logisch zusammenhängend?
- **Readability (Lesbarkeit)**: Ist der Summary gut lesbar?

Der Factuality-Agent verwendet ein **Evidence-Gate**: Behauptungen werden nur als "incorrect" markiert, wenn belastbare Evidence (wörtliche Zitate) aus dem Artikel gefunden wurde. Dies reduziert False Positives erheblich.

---

## Setup (Lokaler Start, optional)

1. **Python-Umgebung erstellen:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # oder: venv\Scripts\activate  # Windows
   ```

2. **Dependencies installieren:**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Hinweis:** `requirements.txt` enthält auch Development-Tools (ruff, pytest). Für Production können diese optional sein.

3. **Umgebungsvariablen setzen:**
   ```bash
   cp .env.example .env
   # Bearbeite .env und setze OPENAI_API_KEY
   ```

4. **Optional: Server starten:**
   ```bash
   python app/server.py
   ```

## Persistenz (Postgres + Neo4j)

**Was wird gespeichert:**
- **Postgres:** Runs, Articles, Summaries, Explainability Reports
- **Neo4j:** Graph-Struktur (Run → Example → Explainability → Finding → Span)

**Aktivierung:**
- Docker Compose: Automatisch aktiv (Volumes: `postgres-data`, `neo4j-data`)
- Lokaler Start: Setze `POSTGRES_DSN` und `NEO4J_URI` in `.env`
- Optional: Setze `persist_to_db=false` in API-Request

**Daten bleiben erhalten:** Nach `docker compose down` bleiben Daten in Docker-Volumes erhalten.

---

## Dashboard (Streamlit)

Ein interaktives Dashboard für die Verifikation, Run-Verwaltung und Status-Dokumentation.

### Lokaler Start

```bash
# Dependencies installieren
pip install -r ui/requirements-ui.txt

# Dashboard starten
streamlit run ui/app.py
```

Das Dashboard ist dann unter `http://localhost:8501` erreichbar.

### Environment Variables

**Für lokalen Start (Host-Mode):**
```bash
export VERI_API_BASE_URL=http://localhost:8000
export POSTGRES_DSN=postgresql://veri:veri@localhost:5433/veri_db
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=changeme
```

**Oder einzelne ENV-Vars für Postgres:**
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5433
export POSTGRES_DB=veri_db
export POSTGRES_USER=veri
export POSTGRES_PASSWORD=veri
```

### Docker Compose

```bash
# Alle Services starten (inkl. UI)
docker compose up --build

# UI ist dann unter http://localhost:8501 erreichbar
# API unter http://localhost:8000
```

Das UI-Service ist bereits in `docker-compose.yml` konfiguriert und verbindet sich automatisch mit den anderen Services im Docker-Netzwerk.

### Features

- **Verify Tab:** Sende Article + Summary zur Verifikation, zeige Scores und Explainability-Highlights
- **Runs Tab:** Liste aller Runs aus Postgres mit Detailansicht
- **Status Tab:** Markdown-Viewer für Status-Dokumente (`docs/status/*.md`)

---

## Wichtigste Commands

### Unit Tests
```bash
# Alle Tests
pytest -q

# Spezifische Test-Suites
pytest tests/unit/test_evidence_gate_refactored.py -v
pytest tests/unit/test_issue_span_verdict.py -v

# Readability-Tests (ohne LLM-Calls, gemockt)
pytest tests/readability/ -v
```

**Hinweis:** Readability-Tests verwenden gemockte LLM-Calls (keine echten API-Aufrufe). Sie testen Contract-Verhalten, Normalisierung und Judge-Secondary-Mode.

### Readability: Vollständige Checks
```bash
# Alle Readability-Checks (Tests + Sanity-Checks + Status-Report-Check)
./scripts/check_readability_all.sh

# Oder manuell:
python -m pytest -q tests/readability/ && \
python scripts/check_readability_package.py && \
python scripts/build_readability_status.py --check  # (nur wenn Artefakte vorhanden)
```

### Coherence: Tests
```bash
# Alle Coherence-Tests (ohne LLM-Calls, gemockt)
pytest tests/coherence/ -v
```

### Factuality: LLM-as-a-Judge Baseline

#### Datasets: Smoke vs Final

- **Final (Thesis):** `data/frank/frank_clean.jsonl` (n=200, bootstrap=2000) - für finale Zahlen
- **Metrics Smoke (empfohlen):** `data/frank/frank_smoke_balanced_50_seed42.jsonl` (n=50, balanced 25/25) - für schnelle, sinnvolle Metrik-Checks
- **Pipeline Smoke (optional):** `data/frank/frank_pipeline_smoke_10.jsonl` (n=10) - nur Technik-Test, nicht für Metriken
- **⚠️ Warnung:** `frank_subset_manifest.jsonl` ist nicht balanced und kann single-class sein → nicht für Metriken wie Balanced Accuracy, MCC, AUROC geeignet

#### Generate Balanced Smoke Dataset
```bash
# Erstellt stratifizierten Smoke-Datensatz (25 pos / 25 neg, n=50, seed=42)
python scripts/make_frank_smoke_balanced.py

# Falls Datei bereits existiert, mit --force überschreiben
python scripts/make_frank_smoke_balanced.py --force

# Verifikation der Verteilung
jq -r '.has_error' data/frank/frank_smoke_balanced_50_seed42.jsonl | sort | uniq -c
```

#### Smoke Judge Run (billig, für schnelle Checks)
```bash
ENABLE_LLM_JUDGE=true JUDGE_MODE=primary JUDGE_N=3 JUDGE_TEMPERATURE=0 \
python scripts/eval_frank_factuality_llm_judge.py \
  --data data/frank/frank_smoke_balanced_50_seed42.jsonl \
  --max_examples 50 --seed 42 --bootstrap_n 200 --cache_mode write \
  --prompt_version v2_binary --judge_n 3 --judge_temperature 0.0 --judge_aggregation majority
```

#### Final Judge Run (Thesis)
```bash
ENABLE_LLM_JUDGE=true JUDGE_MODE=primary JUDGE_N=3 JUDGE_TEMPERATURE=0 \
python scripts/eval_frank_factuality_llm_judge.py \
  --data data/frank/frank_clean.jsonl \
  --max_examples 200 --seed 42 --bootstrap_n 2000 --cache_mode write \
  --prompt_version v2_binary --judge_n 3 --judge_temperature 0.0 --judge_aggregation majority
```

# Quality-Check für Factuality Judge + Coherence Tests
# Standard: Auto-detect latest Judge-Run (Checks 1-3: NOT RUN wenn kein Run vorhanden)
python scripts/verify_quality_factuality_coherence.py

# Fixture-Modus: Verwendet Mini-Fixture für alle Checks (CI-tauglich, keine LLM-Calls)
python scripts/verify_quality_factuality_coherence.py --use_fixture

# Full: Explizit Judge-Run angeben
python scripts/verify_quality_factuality_coherence.py --judge_run results/evaluation/factuality/judge_factuality_<timestamp>_<model>_v2_binary_seed42
```

### Persistence Audit (Postgres + Neo4j)
```bash
# Prüft Datenintegrität und Konsistenz zwischen Postgres und Neo4j
python scripts/audit_persistence.py

# Output: docs/status/persistence_audit.md

# Host mode (Neo4j läuft lokal):
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=changeme
python scripts/audit_persistence.py

# Docker mode (Neo4j läuft im Container):
export NEO4J_URI=bolt://neo4j:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=changeme
python scripts/audit_persistence.py
```

### Explainability-Modul Tests
```bash
# Alle Explainability-Tests ausführen (keine LLM-Calls)
pytest -q tests/explainability

# Mini-Audit (Contract + Determinism Check)
python scripts/audit_explainability.py

# Mit spezifischem Fixture
python scripts/audit_explainability.py --fixture mixed

# Output: docs/status/explainability_audit.md
```

### Evidence-Gate Evaluation
```bash
python3 scripts/test_evidence_gate_eval.py \
    --uncertainty-policy non_error \
    --debug \
    --no-cache \
    --max-examples 50
```

### Bullets Generator (für Evaluationskapitel)
```bash
python3 scripts/print_factuality_eval_bullets.py \
    --results results/evaluation/evidence_gate_test/results_non_error.json \
    --debug results/evaluation/evidence_gate_test/debug_claims.jsonl \
    --format md \
    --out docs/factuality_eval_bullets.md
```

## Evaluation: Ergebnisse & Status-Dokumente

**Ergebnisse:**
- `results/evaluation/readability/` - Readability Agent + Judge Runs
- `results/evaluation/factuality/` - Factuality Judge Runs
- `results/evaluation/coherence/` - Coherence Agent Runs
- `results/evaluation/baselines/` - Classical Baseline Runs

**Status-Dokumente:**
- `docs/status/readability_status.md` - Readability: Agent vs Judge vs Baselines
- `docs/status/factuality_status.md` - Factuality: Judge Baseline
- `docs/status/coherence_status.md` - Coherence: Agent Results
- `docs/status/agents_verification_audit.md` - Vollständige Verifikations-Matrix
- `docs/status/thesis_ready_checklist.md` - Thesis-Ready Checklist

**Status Pack:**
- `docs/status_pack/2026-01-08/` - Executive Summary, Evaluation Results, Artifacts Index

---

## Projektstruktur

```
app/services/agents/factuality/
├── factuality_agent.py      # Haupt-Agent (Sätze → Claims → Verifikation → IssueSpans)
├── claim_models.py          # Claim-Dataclass
├── claim_extractor.py       # Claim-Extraktion aus Sätzen
├── claim_verifier.py        # Claim-Verifikation gegen Artikel (Evidence-Gate)
├── evidence_retriever.py   # Evidence-Retrieval (sliding windows)
└── verifier_models.py       # Pydantic-Models für Verifier-Output

scripts/
├── test_evidence_gate_eval.py        # Evidence-Gate Evaluation
└── print_factuality_eval_bullets.py  # Stichpunkte-Generator

tests/unit/
├── test_evidence_gate_refactored.py  # Unit Tests für Gate-Logik
└── test_issue_span_verdict.py        # Unit Tests für Verdict vs Severity
```

## Clean Checkout Test

**Zum Nachweis der Vollständigkeit:**

```bash
# 1. Clean Clone (ohne lokale Restdateien)
git clone https://github.com/lisa-smn/veri-api.git /tmp/veri-api-test
cd /tmp/veri-api-test

# 2. Environment vorbereiten
cp .env.example .env
# Setze OPENAI_API_KEY=sk-... in .env

# 3. Services starten
docker compose up --build

# 4. Health-Check (in neuem Terminal)
curl http://localhost:8000/health
# Erwartet: {"status": "ok"}

# 5. UI prüfen
# Öffne http://localhost:8501 im Browser
```

**Erwartetes Ergebnis:**
- ✅ Alle Services starten ohne Fehler
- ✅ API Health-Endpoint antwortet
- ✅ UI ist erreichbar und kann Verify-Requests senden

---

## Weitere Dokumentation

- `docs/factuality_agent.md`: Detaillierte Beschreibung des Factuality-Agents und Evidence-Gates
- `M10_EVALUATION.md`: Evaluationsworkflow und Ergebnisse
- `PROJECT_STATUS.md`: Aktueller Projektstatus






