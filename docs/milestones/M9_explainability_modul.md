
---

# **M9 – Explainability-Modul**

## Ziel

Ein separates Explainability-Modul, das AgentResult → erklärbare Strukturen transformiert und Instanz-Erklärungen persistiert.

## Aufgaben

* Package `app/explainability/` anlegen.
* Pydantic-Modelle definieren (`ClaimExplanation`, `InstanceExplanation`).
* Builder-Funktionen entwickeln, die AgentResult in interpretierbare Erklärungen überführen.
* Kurzrationales / Gesamt-Erklärungen generieren.
* Persistenz in Postgres und Neo4j (Error-/Claim-/Explanation-Knoten).
* Integration in die Pipeline (nach Agenten-Ausführung).
* Tests für Builder + End-to-End-Durchläufe.

---

