
---

# **M7 – Coherence-Agent**

## Ziel

Ein funktionierender Coherence-Agent, der Kohärenz von Summaries bewertet und als standardisiertes `AgentResult` in Pipeline, Postgres und Neo4j eingebunden ist.

## Aufgaben

* Kriterien für Kohärenz definieren (logische Struktur, Themenfluss, Konsistenz).
* LLM-basierte Coherence-Bewertung implementieren.
* `CoherenceAgent` mit `AgentResult`-Output fertigstellen.
* Persistenz: Speicherung in Postgres + Erweiterung Neo4j (Coherence-Knoten/Fehler).
* Integration in die Verifikations-Pipeline.
* Unit-Tests (FakeLLM) + Beispielcases.

---

