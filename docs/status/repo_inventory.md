# Repo-Inventar

**Datum:** 2026-01-17  
**Zweck:** Übersicht über Repo-Struktur, Entry Points und Sprachkonsistenz

---

## Übersicht

**Gesamt:** 320 getrackte Dateien (gezählt via `git ls-files`)
- **163 Python-Dateien** (app: 57, ui: 9, scripts: 52, tests: 43)
- **106 Markdown-Dateien** (Dokumentation)

---

## Struktur

| Ordner/Datei | Zweck | Entry Points |
|--------------|-------|--------------|
| **app/** | Backend (FastAPI, Agents, Pipeline) | `app/server.py` → FastAPI App |
| **ui/** | Streamlit Dashboard | `ui/app.py` → Streamlit App |
| **scripts/** | Evaluation & Utility Scripts | CLI-Scripts für Evaluation |
| **tests/** | Unit & Integration Tests | `pytest` |
| **docs/** | Dokumentation | `docs/README.md` → Entry-Point |
| **docker-compose.yml** | Multi-Container Setup | `docker compose up --build` |
| **README.md** | Root-Dokumentation | Quickstart, Projektzweck |

---

## Entry Points & Ports

**Backend (API):**
- Start: `docker compose up --build` (alle Services) oder `python app/server.py` (nur API)
- Entry: `app/server.py` → FastAPI App mit `/verify` Endpunkt
- Port: **8000** (Health: `/health`)

**UI (Dashboard):**
- Start: `streamlit run ui/app.py` (lokal) oder via Docker Compose
- Port: **8501**

**Datenbanken:**
- Postgres: Port **5433**
- Neo4j: Ports **7474** (HTTP) / **7687** (Bolt)

**Evaluation:**
- Readability: `scripts/eval_sumeval_readability.py`
- Factuality: `scripts/eval_frank_factuality_llm_judge.py`
- Coherence: `scripts/eval_sumeval_coherence.py`

---

## Dokumentations-Entry Points

**Root:**
- `README.md` - Quickstart, Projektzweck, Commands

**docs/:**
- `docs/README.md` - Entry-Point für Dokumentation (10-Minuten-Übersicht)
- `docs/status/evaluation_final.md` ⭐ **Canonical Final Evaluation Stand** - Kern-Ergebnisse, 1 Tabelle, 5 Bullet-Kernaussagen

**Detail-Reports:**
- `docs/status/readability_status.md` - Readability: Agent vs Judge vs Baselines
- `docs/status/factuality_status.md` - Factuality: Agent vs Judge
- `docs/status/coherence_status.md` - Coherence: Agent vs Judge
- `docs/milestones/M10_evaluation_setup.md` - Vollständige Evaluationsansätze, Methoden, detaillierte Ergebnisse

---

## KEEP / ARCHIVE

**KEEP (Essentials):**
- Code: `app/`, `ui/`, `tests/`, `scripts/`
- Konfiguration: `docker-compose.yml`, `Dockerfile`, `requirements*.txt`, `pyproject.toml`
- Dokumentation: `README.md`, `docs/README.md`, `docs/milestones/`, `docs/status/evaluation_final.md`
- CI/CD: `.github/workflows/ci.yml`

**ARCHIVE (bereits verschoben):**
- `docs/archive/2026-01-17_cleanup/` - 20 Root-Markdown-Dateien
- `docs/archive/pre_m10_docs/` - Veraltete Evaluations-Dokumente

**Nicht getrackt (korrekt ignoriert):**
- `results/` - Evaluation outputs
- `data/` - Lokale Datensätze
- `.env` - Secrets
- `docs/thesis/` - Thesis-Dokumente

---

## Sprachkonsistenz

**Code-Dokumentation:**
- **app/**, **ui/**, **scripts/**: Docstrings und Kommentare überwiegend **Deutsch**
- **tests/**: Englisch ist akzeptabel (pytest-Konvention)

**Status:** ✅ Bereits konsistent (keine Änderungen nötig)

---

**Vollständige Details:** Siehe `docs/README.md` für Entry-Point und `docs/status/evaluation_final.md` für Final Evaluation Stand.
