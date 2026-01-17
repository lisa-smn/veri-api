---
# M12 – Streamlit Dashboard (UI)

## Ziel

Ein interaktives Dashboard für Demo und Inspektion des Verifikationssystems. Das UI dient als **Demo/Inspector-Tool**, nicht als Production-System.

---

## Was ist das Dashboard?

Das Streamlit Dashboard ist eine Web-Oberfläche, die es ermöglicht:
- Summaries interaktiv zu verifizieren (Article + Summary eingeben → Scores + Findings sehen)
- Verifikations-Runs aus Postgres anzuzeigen
- Status-Dokumentation zu durchsuchen

**Zweck:** Demo für Betreuerinnen, schnelle Inspektion von Ergebnissen, interaktive Exploration.

---

## Start (lokal)

### Voraussetzungen

- Python 3.11+
- Virtuelle Umgebung aktiviert
- API läuft (optional, aber empfohlen)

### Installation

```bash
pip install -r ui/requirements-ui.txt
```

### Start

```bash
streamlit run ui/app.py
```

Dashboard ist dann unter `http://localhost:8501` erreichbar.

---

## Start (Docker)

```bash
docker-compose up ui
```

Oder direkt:

```bash
docker build -t veri-ui -f ui/Dockerfile .
docker run -p 8501:8501 veri-ui
```

---

## Features (Tabs)

### 1. Verify Tab

**Was passiert:**
- Article- und Summary-Text eingeben (manuell oder via Dataset-Loader)
- Verifikation starten → API-Call an `/verify`
- Ergebnisse anzeigen: Scores (Overall, Factuality, Coherence, Readability) + Explainability (Findings, Highlights)

**Features:**
- **Dataset-Auswahl:** Lädt Beispieltexte aus FRANK oder SummEval in die Textfelder (nur zum Füllen, Verify bleibt manuell)
- **Error Injection:** Deterministische Fehler in Summary injizieren (Factuality/Coherence/Readability, low/medium/high)
- **Comparison Panel:** Zeigt Agent vs LLM-as-a-Judge vs Classical Baselines (Flesch Reading Ease)
- **Explainability:** Findings mit Severity (low/medium/high) und Highlights im Summary-Text

**Grenzen:**
- Keine Batch-Verarbeitung (nur einzelne Beispiele)
- Keine Authentifizierung
- Keine Rate-Limiting

### 2. Runs Tab

**Was passiert:**
- Zeigt alle Runs aus Postgres (neueste 50)
- Run auswählen → Details anzeigen (Scores, Explainability, Metadaten)

**Voraussetzung:** Postgres muss konfiguriert sein (`POSTGRES_DSN` oder einzelne ENV-Vars)

### 3. Status Tab

**Was passiert:**
- Markdown-Viewer für Status-Dokumente (`docs/status/*.md`)
- Zeigt Readability/Factuality/Coherence Status-Reports

---

## Explainability im UI

**Wo wird es angezeigt:**
- Verify Tab: Nach Verifikation werden Findings als Liste angezeigt
- Summary-Text wird mit Highlights gerendert (Farben nach Severity: low=gelb, medium=orange, high=rot)

**Was bedeuten Findings:**
- **Finding:** Ein Problem im Summary (z.B. "Falsche Zahl", "Inkohärenter Satz")
- **Severity:** `low` (kleines Problem), `medium` (moderates Problem), `high` (kritisches Problem)
- **Span:** Exakte Text-Position im Summary (Start/End-Char)
- **Dimension:** Factuality, Coherence oder Readability

**Top-Spans:** Die wichtigsten 5 Findings werden hervorgehoben.

---

## Dataset-Auswahl

**Was passiert:**
- Dropdown mit verfügbaren Datasets (FRANK, SummEval)
- "Load random example" Button → lädt zufälliges Beispiel aus Dataset
- Article- und Summary-Felder werden automatisch gefüllt
- Ground Truth/Metadata wird als Badge angezeigt

**Hinweis:** Dataset-Loading dient nur zum Füllen der Felder. Die Verifikation muss manuell gestartet werden.

---

## Grenzen/Limitations

- **Demo/Prototype:** Nicht production-ready (keine Auth, kein Rate-Limiting)
- **Single-Example:** Nur ein Beispiel pro Verifikation (keine Batch-Verarbeitung)
- **API-Abhängigkeit:** UI benötigt laufende API (lokal oder remote)
- **Postgres optional:** Runs-Tab funktioniert nur, wenn Postgres konfiguriert ist
- **Neo4j optional:** Neo4j wird nicht im UI angezeigt (nur Postgres)

---

## Environment Variables

**Für lokalen Start (Host-Mode):**
```bash
export VERI_API_BASE_URL=http://localhost:8000
export POSTGRES_DSN=postgresql://user:pass@localhost:5432/veri_db
export NEO4J_URI=bolt://localhost:7687  # optional
```

**Für Docker:**
- Werden via `docker-compose.yml` gesetzt
- API-URL: `http://api:8000` (interne Docker-Netzwerk)

---

## Troubleshooting

**API nicht erreichbar:**
- Prüfe `VERI_API_BASE_URL` (Standard: `http://localhost:8000`)
- Stelle sicher, dass API läuft (`python app/server.py`)

**Postgres nicht verfügbar:**
- Prüfe `POSTGRES_DSN` oder einzelne ENV-Vars (`POSTGRES_HOST`, `POSTGRES_PORT`, etc.)
- UI funktioniert auch ohne Postgres (nur Runs-Tab ist dann leer)

**Dataset-Loading fehlgeschlagen:**
- Prüfe, ob Dataset-Dateien existieren (`data/frank/frank_clean.jsonl`, `data/sumeval/sumeval_clean.jsonl`)
- Fehler werden im UI angezeigt

---

## Weitere Details

- **Code:** `ui/app.py` (Hauptdatei)
- **API Client:** `ui/api_client.py`
- **Rendering:** `ui/render.py` (Explainability, Highlights)
- **Dataset Loader:** `ui/dataset_loader.py`
- **Dockerfile:** `ui/Dockerfile`

---

