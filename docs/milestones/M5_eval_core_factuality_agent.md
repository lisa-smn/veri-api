

# **M5 – Factuality-Agent & Verifikationslogik**

## **Ziel**

In M5 sollte der erste „echte“ Agent entstehen: ein funktionierender Factuality-Agent, der Faktfehler in Summaries identifiziert.
Der Fokus lag nicht auf ML-Modellen oder Trainingsdaten, sondern auf:

* **korrekter Integration** in die bestehende Pipeline
* **valider Datenstruktur** (Pydantic-AgentResult)
* **sicherer End-to-End-Ausführung über den `/verify`-Endpoint**
* **nachweisbarer Erkennung offensichtlicher Faktfehler**

Mit M5 wurde der Dummy-Factuality-Agent aus M3 durch ein voll funktionsfähiges Modul ersetzt.

---

## **Umsetzung**

### **LLM-Abstraktion**

* `LLMClient` Interface definiert
* Zwei Implementierungen:

  * `OpenAIClient` (Produktivbetrieb)
  * `FakeLLMClient` (Tests ohne API-Call; deterministisch)
* Pipeline entscheidet automatisch (über `TEST_MODE`) zwischen echter und Fake-LLM

Damit ist der Factuality-Agent unabhängig vom verwendeten Modell.

---

### **Factuality-Agent**

Der Agent wurde vollständig implementiert:

* Summary wird in Sätze gesplittet
* Für jeden Satz:

  * LLM-Prompt generiert
  * JSON-Antwort geparst
  * Label + Confidence + Erklärung extrahiert
* Berechnung eines agentweiten Scores
* Globale Erklärung generiert
* Fehlerliste erstellt
* Ergebnis als **valides Pydantic-`AgentResult`** zurückgegeben

Struktur u. a.:

```json
{
  "score": 0.0,
  "explanation": "...",
  "details": {
    "sentences": [...],
    "num_errors": 1
  }
}
```

Damit ist der Agent vollständig integriert und formal kompatibel zur Pipeline.

---

### **Pipeline-Integration**

* Dummy-FactualityAgent wurde durch echten Agent ersetzt
* Coherence & Readability bleiben Dummy, aber **schema-konform**
* Pipeline erzeugt ein vollständiges `PipelineResult`
* `overall_score` wird weiterhin über alle Agenten aggregiert
* TEST_MODE erlaubt Ausführung ohne DB und ohne echte LLM-Calls

---

### **Pydantic-Modell erweitert**

`AgentResult` wurde so erweitert, dass es alle Agenten konsistent beschreiben kann:

* `name`
* `score`
* `explanation`
* `errors`
* `details` (für Satz-Level-Ausgaben)

Damit funktionieren API-Serialisierung, DB-Persistenz und Tests ohne Sonderfallbehandlung.

---

### **Tests & Qualitätssicherung**

Zwei Tests bilden den gesamten M5-Scope ab:

#### 1. **Unit-Test (FactualityAgent)**

* Erkennt den Faktfehler „Berlin vs. Paris“
* Liefert Score < 1
* Generiert num_errors ≥ 1
* Test läuft vollständig LLM-frei (FakeLLM)

#### 2. **Integrationstest (`/verify`)**

* POST-Request mit Artikel + Summary
* Pipeline wird ausgeführt
* FactualityAgent erkennt Fehler
* Response enthält vollständiges Factuality-Ergebnis
* HTTP 200 statt 500 durch TEST_MODE

Beide Tests **grün** – das Abnahmekriterium ist erfüllt.

---

## **Ergebnis**

Die Factuality-Verifikation ist technisch abgeschlossen:

**Request → Pipeline → FactualityAgent → Satzprüfung → Score → API-Response**

M5 liefert damit den ersten *echten* Agenten der Verifikationskette. Offensichtliche Faktfehler werden zuverlässig erkannt und klar im Ergebnis ausgewiesen. Die Pipeline ist stabil, modular und bereit für M6 (detaillierte Modelle, Trainingsdaten und feingranulare Fehlerkategorien).

M5 ist vollständig umgesetzt und bildet die Grundlage für inhaltlich anspruchsvolle Verifikationsmethoden in M6.
