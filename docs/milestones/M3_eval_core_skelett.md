# M3 – Evaluations-Kern (Pipeline & Agenten)

## Ziel
In M3 sollte die technische Verifikationspipeline entstehen, die Anfragen verarbeiten, Artikel/Summaries speichern, einfache Agenten ausführen und Ergebnisse zurückgeben kann. Inhaltliche Intelligenz war noch nicht das Ziel – nur ein funktionsfähiges Grundgerüst.

## Umsetzung

### Pipeline & Agenten
- Drei Dummy-Agenten erstellt: Factuality, Coherence, Readability  
- Einheitliche Modelle für Input/Output (`AgentResult`, `PipelineResult`)  
- Pipeline gebaut, die alle Agenten ausführt und einen `overall_score` berechnet  

### Persistenz
- Funktionen implementiert zum Speichern von:
  - Artikeln  
  - Summaries  
  - Runs  
  - Agenten-Ergebnissen  
  - Erklärungen  
- JSON-Daten werden als Strings in JSONB-Felder geschrieben

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
