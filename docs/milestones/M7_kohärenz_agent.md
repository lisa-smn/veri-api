# M7 – Kohärenz-Agent (Coherence Agent)

## Ziel von M7

Ziel von M7 ist die Implementierung, Integration und Evaluation eines **Kohärenz-Agenten**, der die **inhaltliche und logische Stimmigkeit** einer automatisch erzeugten Zusammenfassung bewertet.

Der Agent ergänzt die bereits existierende Dimension Factuality und liefert:

* einen **normalisierten Kohärenz-Score im Bereich [0, 1]**
* strukturierte Hinweise auf **kohärenzbezogene Probleme** innerhalb der Zusammenfassung

M7 stellt sicher, dass Kohärenz:

* konsistent zur bestehenden Agentenarchitektur modelliert ist
* speicher- und auswertbar bleibt
* reproduzierbar evaluiert werden kann

---

## Definition von Kohärenz

In diesem System bezeichnet **Kohärenz** die logische und semantische Geschlossenheit einer Zusammenfassung, insbesondere:

* logische Abfolge von Aussagen
* konsistente Referenzen (Pronomen, Entitäten)
* keine inneren Widersprüche
* nachvollziehbare Übergänge zwischen Sätzen

**Nicht** Teil der Kohärenzbewertung sind:

* faktische Korrektheit (→ Factuality)
* sprachliche Eleganz oder Lesbarkeit (→ Readability)
* Stil oder Tonalität

---

## Architektur des Coherence Agents

Der Coherence Agent folgt exakt der gleichen strukturellen Architektur wie die übrigen Agenten im System.

### Zentrale Komponenten

* `CoherenceAgent`
* `LLMCoherenceEvaluator`
* `AgentResult` (gemeinsames Pydantic-Schema)

### AgentResult-Struktur

```json
{
  "name": "coherence",
  "score": 0.0,
  "explanation": "Globale Erklärung des Kohärenzurteils",
  "issue_spans": [
    {
      "start_char": 42,
      "end_char": 97,
      "message": "Logischer Bruch zwischen zwei Aussagen",
      "severity": "medium"
    }
  ],
  "details": {
    "agent_details": { }
  }
}
```

* `score`: normalisierter Kohärenz-Score
* `issue_spans`: lokalisierte Kohärenzprobleme im Summary
* `details.agent_details`: agentenspezifische Zusatzinformationen (z. B. LLM-Rohoutput)

---

## Score-Berechnung

Der Kohärenz-Score wird vom zugrunde liegenden LLM direkt im Bereich **[0, 1]** erzeugt.

Dabei gilt:

* **1.0** → vollständig kohärent
* **0.0** → stark inkohärent

Es erfolgt **keine nachträgliche Skalierung** im Agenten selbst.
Die Normalisierung externer Datensätze erfolgt ausschließlich in der Evaluationspipeline.

---

## Fehler- und Issue-Typen

Der Coherence Agent modelliert Kohärenzprobleme als `IssueSpans` mit optionaler Schweregrad-Einstufung:

* `low`: leichte Übergangs- oder Referenzprobleme
* `medium`: spürbare logische Brüche
* `high`: schwerwiegende Inkohärenz oder Widersprüche

Die Fehler werden:

* in PostgreSQL persistiert
* als Knoten im Neo4j-Graph gespeichert
* optional über die API ausgegeben

---

## Persistenz & Integration

### PostgreSQL

* Speicherung über `verification_results`
* `issue_spans` werden als strukturierte JSONB-Daten abgelegt
* Trennung zwischen `details.agent_details` und strukturierten Issues

### Neo4j

* Kohärenzprobleme werden als `Error`-Knoten modelliert
* Verknüpfung mit Summary, Run und Dimension
* Speicherung erfolgt best-effort (keine Pipeline-Blockade)

---

## Evaluation: SumEval (Coherence)

Zur Evaluation des Coherence Agents wird der **SumEval-Datensatz** verwendet.

### Ground Truth

* Menschliche Kohärenzbewertungen (typisch Skala **1–5**)

### Normalisierung

Um Vergleichbarkeit mit dem Agent-Score zu gewährleisten, wird die Ground Truth wie folgt normalisiert:

```
gt_norm = (gt_raw − 1) / (5 − 1)
```

Damit liegen:

* Agent-Score ∈ [0, 1]
* Ground Truth ∈ [0, 1]

### Metriken

Die Evaluation berechnet:

* **Pearson r** (lineare Korrelation)
* **Spearman ρ** (Rangkorrelation)
* **MAE (0–1)**

---

## Reproduzierbarkeit

Die Evaluation ist reproduzierbar durch:

* versionierten Prompt (`COHERENCE_PROMPT_VERSION`)
* explizite Angabe des LLM-Modells
* optionales Caching der Agent-Ergebnisse
* persistierte Run-Artefakte (`results/sumeval/`)

---

## Limitationen

* Kohärenz bleibt teilweise subjektiv, auch in menschlichen Bewertungen
* LLM-basierte Scores sind nicht deterministisch
* Feinere Diskursphänomene (z. B. rhetorische Struktur) werden nur begrenzt erfasst

---

## Status von M7

* Coherence Agent: ✅ implementiert
* Pipeline-Integration: ✅ abgeschlossen
* Persistenz (Postgres & Neo4j): ✅ abgeschlossen
* Tests: ✅ vorhanden
* Evaluation (SumEval): ✅ angebunden
* Dokumentation: ✅ abgeschlossen

**M7 gilt damit als abgeschlossen.**
