Hier ist deine **M2-Dokumentation**, im gleichen Stil, Aufbau und Detaillierungsgrad wie deine M1-Dokumentation.
Sachlich, knapp, technisch sauber, ohne perfekte Glättung.

---

# **M2 – Datenmodell & Persistenzschicht (Postgres)**

## **Ziel**

In M2 sollte die Datenhaltung des Verifikationssystems aufgebaut werden.
Kernanforderungen:

* ein relationales Schema, das Artikel, Summaries, Runs und Agenten-Ergebnisse konsistent speichert
* eine stabile, reproduzierbare Postgres-Instanz (Docker)
* eine API-Anbindung über SQLAlchemy
* ein Testskript, das reale Datensätze einfügt, um die Persistenzkette zu validieren

Damit bildet M2 die Grundlage für alle späteren Evaluations- und Agentenprozesse.

---

## **Umsetzung**

### **Postgres-Setup (Docker)**

* Postgres 16 als Container (`veri-db`)
* Port-Mapping nach außen (5433), um Konflikte mit lokalem Postgres zu vermeiden
* automatisches Datenbankschema über `schema.sql`:

  * Enum-Typen (`summary_source`, `run_type`, `run_status`, `verification_dimension`)
  * Tabellen:

    * `datasets`
    * `articles`
    * `summaries`
    * `runs`
    * `verification_results`
    * `errors`
    * `explanations`
* persistentes Volume `postgres-data`

Damit entsteht eine vollständig reproduzierbare lokale Datenbankumgebung.

---

### **Applikationsanbindung**

#### **Konfiguration**

* `.env` lädt die zentrale Variable:

  ```env
  DATABASE_URL=postgresql+psycopg://veri:veri@db:5432/veri_db
  ```

* `app/core/config.py` lädt diese URL über `pydantic-settings`
  → einheitliche und austauschbare DB-Konfiguration

#### **DB-Session**

`app/db/session.py`:

* SQLAlchemy-Engine auf Basis der `DATABASE_URL`
* `SessionLocal` als Factory für Transaktionen
* `get_db()`-Dependency für API-Endpunkte

Damit kann jede API-Route sicher auf Postgres zugreifen.

---

### **Schema & Datenmodell**

Die Tabellen bilden die fachliche Struktur der Verifikationspipeline ab:

* **datasets**
  Herkunft und Split-Information der Experimente

* **articles**
  Originaltexte, optional aus FRANK, SumEval, FineSumFact usw.

* **summaries**
  von LLM generierte oder referenzbasierte Zusammenfassungen
  → inkl. `source`, `llm_model`, `prompt_version`

* **runs**
  jeder Verifikationsvorgang (ein Artikel + eine Summary + ein Agentenset)

* **verification_results**
  Scores und Labels für jede Bewertungsdimension (Factuality, Coherence, usw.)

* **errors**
  Fehlermeldungen von Agenten oder Pipeline-Schritten

* **explanations**
  Freitext- oder strukturierte Erklärungen der Agenten

Das Schema ist darauf ausgelegt, später Agenten mit sehr unterschiedlichen Ausgaben speichern zu können — insbesondere JSONB-Felder für flexible Strukturen.

---

### **API-Erweiterung**

In `app/api/routes.py` wurden zwei Basisendpunkte ergänzt:

* `GET /health` – Lebenszeichen der API
* `GET /db-check` – einfache Verbindungsprüfung zur Datenbank

Diese Endpunkte dienen als Infrastrukturtest, bevor echte Agenten integriert werden.

---

### **Testskript (M2-Abnahme)**

`scripts/m2_insert_test_data.py`:

* erzeugt Dataset, Artikel, Summary
* erzeugt einen vollständigen Verification Run
* legt einen Fake-Score in `verification_results` ab
* JSON-Konfigurationen werden per `json.dumps()` gespeichert, sodass Postgres sie als `jsonb` verarbeitet

Ergebnis:
Ein vollständiger End-to-End-Durchlauf der Persistenzkette ohne Agentenlogik.

---

## **Ergebnis**

M2 stellt die komplette Datenbasis des Projekts her.

* Dockerisierte Postgres-Instanz läuft stabil
* Datenbankschema wird automatisch initialisiert
* API ist korrekt angebunden (SQLAlchemy-Session)
* Ein Testskript bestätigt realen Datenfluss:
  Dataset → Artikel → Summary → Run → Verification Result
* Die Infrastruktur ist damit bereit für M3 (Pipeline & Dummy-Agenten) und M5–M7 (echte Agenten)

Durch M2 existiert jetzt:

1. ein persistenter Speicher für alle Evaluationsdaten
2. ein reproduzierbarer Entwicklungszustand
3. die Grundlage für wissenschaftliche Experimente (Runs vergleichen, Modelle vergleichen, Agentenfehler analysieren)

Damit ist **M2 abgeschlossen**, und das Projekt besitzt erstmals eine vollständige technische Datenbasis für das spätere agentische Verifikationsmodell.
