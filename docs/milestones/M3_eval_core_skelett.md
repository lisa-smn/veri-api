# M3 – Evaluations-Kern (Pipeline & Agenten)

## Ziel
In M3 sollte die technische Verifikationspipeline entstehen, die Anfragen verarbeiten, Artikel/Summaries speichern, einfache Agenten ausführen und Ergebnisse zurückgeben kann. Inhaltliche Intelligenz war noch nicht das Ziel – nur ein funktionsfähiges Grundgerüst.

## Umsetzung

### Pipeline & Agenten
- Drei Dummy-Agenten erstellt: Factuality, Coherence, Readability  
- Einheitliche Modelle für Input/Output (`AgentResult`, `PipelineResult`)  
- Pipeline gebaut, die alle Agenten ausführt und einen `overall_score` berechnet  

### Persistenz
- Nutzt Persistenz-Funktionen aus M2 (`store_article_and_summary`, `store_verification_run` in `app/db/postgres/persistence.py:33-271`)
- Speichert:
  - Artikeln (via `store_article_and_summary`)
  - Summaries (via `store_article_and_summary`)
  - Runs (via `store_verification_run`)
  - Agenten-Ergebnisse (via `store_verification_run`, in `verification_results` Tabelle)
  - Erklärungen (optional, via `store_verification_run`, in `explanations` Tabelle)
- JSON-Daten werden als JSONB-Felder gespeichert (vgl. `app/db/postgres/persistence.py:149-166`)

### Service & API
- `VerificationService`: verbindet Pipeline und Datenbank  
- `/verify`-Endpoint aktualisiert, um:
  - Request entgegenzunehmen  
  - Pipeline auszuführen  
  - Ergebnisse zurückzugeben  

### Docker
- API und Postgres laufen komplett in Docker  
- Kommunikation über Service-Namen (`db`)  
- Projekt per `docker compose up --build` startbar

## Ergebnis
Die komplette technische Verifikationskette funktioniert:

Request → Pipeline → Dummy-Agenten → Aggregation → Speicherung → Response.

M3 ist damit abgeschlossen und die Basis für „echte“ Agenten in den nächsten Schritten steht.
