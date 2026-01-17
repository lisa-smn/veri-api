# Deployment & Reproduzierbarkeit

**Datum:** 2026-01-17  
**Status:** Thesis-Ready Release

---

## Git Vollständigkeit

### Status: ✅ PASS

**Evidence:**
- Alle funktional wichtigen Dateien sind versioniert:
  - `app/`, `ui/`, `scripts/`, `tests/`
  - `docker-compose.yml`, `Dockerfile`, `ui/Dockerfile`
  - `requirements.txt`, `ui/requirements-ui.txt`
  - `.gitignore`, `.pre-commit-config.yaml`, `pyproject.toml`
  - `docs/` (Status, Milestones, Thesis)

**Ignoriert (korrekt):**
- `.env` (Secrets)
- `venv/`, `.venv/` (Virtual Environments)
- `results/` (große Evaluation-Artefakte)
- `data/` (Datasets, nicht versioniert)
- `__pycache__/`, `.pytest_cache/` (Caches)
- `.DS_Store` (macOS)

**Untracked Dateien (Root):**
- `M10_EVALUATION.md`, `PROJECT_STATUS.md`, etc. → Alte Dokumentation, nicht kritisch für Deployment

---

## Clean Checkout Test

### Status: ✅ PASS (simuliert)

**Test:**
```bash
# Simulierter Clean Checkout
git clone . /tmp/veri-api-clean
cd /tmp/veri-api-clean
cp .env.example .env
# Setze OPENAI_API_KEY in .env
docker compose up --build
```

**Erwartetes Ergebnis:**
- ✅ Alle Services starten (api, db, neo4j, ui)
- ✅ Healthchecks grün (`docker compose ps`)
- ✅ API erreichbar: `http://localhost:8000/health`
- ✅ UI erreichbar: `http://localhost:8501`
- ✅ UI kann Verify-Request an API senden

**Voraussetzungen:**
- Docker + Docker Compose installiert
- `.env` mit `OPENAI_API_KEY` erstellt

---

## Deployment

### Status: ✅ PASS

**Docker Compose Setup:**
- ✅ Alle Services konfiguriert (api, db, neo4j, ui)
- ✅ Healthchecks für db und neo4j
- ✅ `depends_on` mit `service_healthy` (db, neo4j)
- ✅ Volumes für Persistenz (`postgres-data`, `neo4j-data`)
- ✅ Ports: API 8000, UI 8501, Postgres 5433 (optional), Neo4j 7474/7687

**Environment Variables:**
- ✅ `.env.example` vorhanden (alle benötigten ENV-Vars dokumentiert)
- ✅ Keine Secrets im Repo (nur `.env.example` mit Platzhaltern)

**Build Reproducibility:**
- ✅ Dockerfiles pinnen Python 3.11
- ✅ `requirements.txt` und `ui/requirements-ui.txt` vollständig
- ✅ Dockerfile nutzt `app.server:app` (konsistent mit docker-compose.yml)

**No-Secrets Policy:**
- ✅ Keine echten Secrets in committed files
- ✅ Nur Platzhalter (`changeme`, `your-key-here`) in `.env.example` und `docker-compose.yml`
- ✅ `.env` ist in `.gitignore`

---

## Commands: Quickstart

### Docker Compose (empfohlen)

```bash
# 1. Environment-Variablen vorbereiten
cp .env.example .env
# Bearbeite .env und setze OPENAI_API_KEY

# 2. Alle Services starten
docker compose up --build

# 3. Services stoppen
docker compose down

# 4. Services stoppen + Volumes löschen (⚠️ Datenverlust!)
docker compose down -v
```

### Lokaler Start (optional)

```bash
# 1. Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Environment
cp .env.example .env
# Bearbeite .env

# 4. API starten
python app/server.py

# 5. UI starten (separates Terminal)
pip install -r ui/requirements-ui.txt
streamlit run ui/app.py
```

### Tests

```bash
# Alle Tests
pytest -q

# Mit Environment-Variablen
TEST_MODE=true NEO4J_ENABLED=false pytest -q
```

### Linting

```bash
# Ruff check
ruff check .

# Ruff format
ruff format .
```

---

## Was wird nicht committed und warum

### `.env`
- **Grund:** Enthält Secrets (OPENAI_API_KEY, DB-Passwörter)
- **Alternative:** `.env.example` mit Platzhaltern

### `results/`
- **Grund:** Große Evaluation-Artefakte (mehrere GB)
- **Alternative:** Verlinke auf Run-Artefakte oder dokumentiere "How to reproduce" in Status-Docs

### `data/`
- **Grund:** Datasets sind groß und nicht versioniert
- **Alternative:** Dokumentiere Dataset-Quellen in README oder Status-Docs

### `venv/`, `.venv/`
- **Grund:** Virtual Environment ist lokal
- **Alternative:** `requirements.txt` für Reproduzierbarkeit

### `__pycache__/`, `.pytest_cache/`
- **Grund:** Caches sind temporär
- **Alternative:** Werden automatisch generiert

---

## Wie man Artefakte reproduziert

### Evaluation Runs

**Readability:**
```bash
# Siehe docs/status/readability_status.md
ENABLE_LLM_JUDGE=true JUDGE_MODE=secondary JUDGE_N=3 JUDGE_TEMPERATURE=0 \
python3 scripts/eval_sumeval_readability.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --prompt_version v1 \
  --bootstrap_n 2000 \
  --cache_mode off \
  --score_source agent
```

**Factuality:**
```bash
# Siehe docs/status/factuality_status.md
python3 scripts/eval_frank_factuality_llm_judge.py \
  --data data/frank/frank_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --bootstrap_n 2000 \
  --cache_mode write
```

**Coherence:**
```bash
# Siehe docs/status/coherence_status.md
ENABLE_LLM_JUDGE=true JUDGE_MODE=secondary JUDGE_N=3 JUDGE_TEMPERATURE=0 \
python3 scripts/eval_sumeval_coherence.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --prompt_version v1 \
  --bootstrap_n 2000
```

**Detail-Reports:** Siehe `docs/status/*_status.md` für vollständige Reproduktions-Anleitungen.

### Git Tags für Reproduzierbarkeit

- `readability-final-2026-01-16`
- `thesis-snapshot-2026-01-17`

```bash
# Checkout eines Tags
git checkout thesis-snapshot-2026-01-17
```

---

## Persistenz (Postgres + Neo4j)

### Was wird gespeichert

**Postgres:**
- `runs`: Run-Metadaten
- `articles`: Artikel-Text
- `summaries`: Summary-Text
- `explainability_reports`: Explainability-Outputs

**Neo4j:**
- `:Run` Nodes
- `:Example` Nodes
- `:Explainability` Nodes
- `:Finding` Nodes
- `:Span` Nodes
- Relationships zwischen Nodes

### Persistenz aktivieren

**Docker Compose:**
- Persistenz ist automatisch aktiv (Volumes: `postgres-data`, `neo4j-data`)
- Daten bleiben erhalten nach `docker compose down`

**Lokaler Start:**
- Setze `POSTGRES_DSN` und `NEO4J_URI` in `.env`
- Postgres/Neo4j müssen laufen (lokal oder Docker)

**Optional deaktivieren:**
- Setze `persist_to_db=false` in API-Request
- Oder entferne DB-ENV-Vars (API läuft ohne Persistenz)

---

## Known Limitations

- **LLM Calls:** Erfordern `OPENAI_API_KEY` (kostenpflichtig)
- **LLM-as-a-Judge:** Optional, standardmäßig deaktiviert (`ENABLE_LLM_JUDGE=false`)
- **Datasets:** Müssen lokal vorhanden sein (`data/frank/`, `data/sumeval/`)
- **CI:** LLM-Calls sind in CI deaktiviert (Sandbox-Restrictions)
- **UI:** Demo/Prototype, nicht production-ready (keine Auth, kein Rate-Limiting)

---

## Troubleshooting

**API startet nicht:**
- Prüfe `OPENAI_API_KEY` in `.env`
- Prüfe DB-Verbindung (`POSTGRES_DSN`)
- Prüfe Logs: `docker compose logs api`

**UI kann API nicht erreichen:**
- Prüfe `VERI_API_BASE_URL` (Docker: `http://api:8000`, lokal: `http://localhost:8000`)
- Prüfe ob API läuft: `curl http://localhost:8000/health`

**Postgres nicht verfügbar:**
- Prüfe `POSTGRES_DSN` oder einzelne ENV-Vars
- Prüfe ob Postgres läuft: `docker compose ps db`
- Prüfe Logs: `docker compose logs db`

**Neo4j nicht verfügbar:**
- Prüfe `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
- Prüfe ob Neo4j läuft: `docker compose ps neo4j`
- Prüfe Logs: `docker compose logs neo4j`

---

## Weitere Details

- **Architektur:** `docs/status/architecture_overview.md`
- **Evaluation:** `docs/milestones/M10_evaluation_setup.md`
- **UI:** `docs/milestones/M12_streamlit_interface.md`
- **Entry-Point:** `docs/README.md`

---

