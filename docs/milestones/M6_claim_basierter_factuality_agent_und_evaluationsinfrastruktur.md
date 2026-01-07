
---

# **M6 – Claim-basierter Factuality-Agent & Evaluationsinfrastruktur**

## **Ziel**

M6 erweitert den vorhandenen Factuality-Agenten zu einem **wissenschaftlich belastbaren, claim-basierten Verifikationssystem**, das Summaries nicht mehr nur grob, sondern **auf Ebene einzelner Behauptungen** prüft.

Die zentralen Ziele waren:

* Einführung eines strukturierten **Claim-Extraction → Claim-Verification** Workflows
* Erweiterung des Agenten um:

  * Claim-Level-Scores und Fehlerkategorien
  * Evidenzextraktion
  * globale und lokale Erklärungen
* Komplette Kompatibilität zur bestehenden Pipeline (`/verify`)
* Vorbereitung einer empirischen Evaluation mit **FRANK** und **FineSumFact**
* Sicherstellen, dass Pipeline, Persistenz und Tests weiterhin stabil bleiben

M6 macht den Factuality-Agenten zum ersten Mal **benchmarktauglich**.

---

## **Umsetzung**

---

## **1. Claim-Extraktion**

Ein neues Modul teilt Summaries in **atomare, überprüfbare Claims**.
Jeder Claim enthält:

* `id`
* `sentence_index`
* `text`
* optionale Zusatzinformationen

Die Extraktion erfolgt über ein LLM mit einem präzise definierten Promptdesign.
Damit bekommt das System eine **strukturierte Grundlage** für eine feingranulare Faktprüfung.

---

## **2. Claim-Verifikation**

Jeder Claim wird einzeln gegen den Artikeltext geprüft.

Der ClaimVerifier liefert:

* **Label**

  * `correct`
  * `incorrect`
  * `uncertain`

* **Confidence-Wert**

* **Fehlerkategorie**

  * ENTITY
  * NUMBER
  * DATE
  * OTHER

* **Evidence** (1–2 relevante Passagen)

* **Erklärung**

Diese Detailtiefe macht den Agenten interpretierbar und wissenschaftlich auswertbar.

---

## **3. Neuer Factuality-Agent**

Der Agent kombiniert Extraktion & Verifikation und erzeugt:

* eine geordnete Claim-Liste
* Claim-Level-Ergebnisse
* globalen Score (aggregiert über alle Claims)
* Fehlertypen
* eine konsolidierte Explanation
* Satzfallback, falls keine Claims extrahiert werden können

Die zentrale Eigenschaft: **Trotz der neuen Logik ist das API-Interface unverändert kompatibel.**

---

## **4. Erweiterte Datenstruktur im AgentResult**

Der Rückgabewert enthält jetzt:

```json
"details": {
  "sentences": [...],
  "claims": [...],
  "num_incorrect": 2
}
```

Damit sind:

* Persistenz in Postgres
* Graphmodell in Neo4j
* `/verify` Endpoint
* Tests

vollständig kompatibel, aber informativer.

---

## **5. Pipeline- & Systemintegration**

* Der bestehende FactualityAgent wurde durch die neue Claim-Version ersetzt, ohne Änderungen an der Pipeline.
* `/verify` liefert nun Claim-Level-Ergebnisse in stabiler Struktur.
* Persistenz wurde minimal erweitert, um Claim-Daten sicher abzulegen.
* TEST_MODE wurde beibehalten (FakeLLM + keine DB-Zugriffe).
* Neo4j speichert Claim-Fehler und ist für spätere Visualisierungen vorbereitet.

Alles bleibt **API-stabil** und „drop-in“ austauschbar.

---

## **6. Evaluationsinfrastruktur (FRANK & FineSumFact)**

### **FRANK**

* Dataset bereinigt (`frank_clean.jsonl`)
* Evaluationsskript implementiert (`scripts/eval_frank.py`)
* TP/FP/TN/FN-Analyse aus Claim-basierten Ergebnissen
* Ausgabe: Accuracy, Precision, Recall, F1
* Erster Testlauf erfolgreich

Damit ist der Agent **messbar**.

### **FineSumFact**

* Git-LFS korrekt eingebunden
* Originaldaten erfolgreich geladen
* Konverter erstellt (`scripts/convert_finesumfact.py` → erzeugt JSONL)
* Eval-Skripte akzeptieren FineSumFact direkt
* Claim-basierte Agentenverifikation möglich

Damit stehen zwei hochwertige Benchmarks bereit.

---

# **Tests & Qualitätssicherung**

## **Unit-Tests**

* ClaimExtractor getestet
* ClaimVerifier getestet
* Fallback-Logik verifiziert
* Struktur von Claim- und Sentence-Blöcken stabil

## **Integrationstest**

* `/verify` wurde End-to-End ausgeführt:

  * Claim-Level-Ergebnisse erscheinen korrekt
  * Score-Berechnung stabil
  * Persistenz (Postgres + Neo4j) fehlerfrei
  * FakeLLM & OpenAIClient funktionieren im Wechsel

Alle Tests laufen **grün**.

---

# **Ergebnis**

M6 liefert eine **neue Generation** des Factuality-Agenten:

* claim-basiert statt satzbasiert
* kategorisierte Fehler
* Evidenzextraktion
* robuste Erklärungen
* benchmarkfähige Architektur
* vollständig kompatibel zur bestehenden Pipeline
* bereit für quantitative wissenschaftliche Evaluation

Damit ist dein System zum ersten Mal **empirisch auswertbar**, reproduzierbar und methodisch sauber genug, um in der Bachelorarbeit überzeugend argumentiert zu werden.

M6 bildet den Kern deiner empirischen Analyse – ab jetzt kannst du echte Experimente fahren.
