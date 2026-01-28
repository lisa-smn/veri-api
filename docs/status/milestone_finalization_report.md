# Milestone-Dokumentation Finalisierung

**Datum:** 2026-01-17  
**Zweck:** Finalisierung der Milestone-Dokumente für Thesis-Referenzierung

---

## Durchgeführte Korrekturen

### 1. Code-Audit & Diskrepanzen

**M1 – Projekt-Setup:**
- ✅ `main.py` → `app/server.py` korrigiert
- ✅ Startbefehl: `uvicorn app.server:app` dokumentiert

**M2 – Datenmodell:**
- ✅ `errors` Tabelle → `run_errors` korrigiert
- ✅ Code-Referenzen zu Schema und Persistenz-Funktionen ergänzt

**M3 – Evaluations-Kern:**
- ✅ Persistenz-Funktionen korrekt referenziert (aus M2, nicht M3)
- ✅ Code-Referenzen zu `store_article_and_summary` und `store_verification_run` ergänzt

**M4 – Graph-Modell:**
- ✅ `Error` Node → `IssueSpan:Error` (Dual-Label) korrigiert
- ✅ `HAS_ERROR` → `HAS_ISSUE_SPAN` korrigiert
- ✅ Neo4j Config über `settings.*` dokumentiert (nicht direkt ENV-Vars)
- ✅ Code-Referenzen zu Graph-Persistenz ergänzt

**M5 – Factuality-Agent:**
- ✅ `errors` Feld → `issue_spans` (mit Legacy-Alias) dokumentiert
- ✅ Code-Referenzen ergänzt

**M6 – Claim-basierter Factuality-Agent:**
- ✅ Evaluationsskripte korrigiert: `eval_frank.py` → mehrere spezifische Skripte
- ✅ Code-Referenzen ergänzt

**M7 – Kohärenz-Agent:**
- ✅ Issue-Typen präzisiert: LOGICAL_INCONSISTENCY, CONTRADICTION, REDUNDANCY, ORDERING, OTHER
- ✅ Code-Referenzen zu `coherence_verifier.py:140` ergänzt

**M8 – Readability-Agent:**
- ✅ Readability v2 vollständig dokumentiert:
  - Normalisierung: `score = clamp01((score_raw_1_to_5 - 1) / 4)`
  - Constraint: `score_raw_1_to_5 <= 2` → min 1 issue
  - Raw-Score-Speicherung dokumentiert
- ✅ Issue-Typen korrigiert: LONG_SENTENCE, COMPLEX_NESTING, PUNCTUATION_OVERLOAD, HARD_TO_PARSE
- ✅ Code-Referenzen ergänzt

**M9 – Explainability-Modul:**
- ✅ `ErrorSpan` → `IssueSpan` korrigiert
- ✅ Code-Referenzen zu `explainability_service.py` und `models/pydantic.py` ergänzt

**M10 – Evaluation:**
- ✅ Run-Config-Format präzisiert: YAML statt JSON/YAML
- ✅ Code-Referenzen ergänzt

### 2. README.md Aktualisierung

- ✅ Übersichtstabelle aktualisiert mit korrekten Issue-Typen
- ✅ Code-Referenzen in Ergebnissen ergänzt
- ✅ Zitierweise-Sektion hinzugefügt mit Format und Beispielen
- ✅ Git-Tag `thesis-milestones-2026-01-17` dokumentiert

### 3. Konsistenz-Checks

- ✅ Alle Milestones haben konsistente Struktur (Ziel → Umsetzung → Ergebnis)
- ✅ Code-Referenzen (Datei:Zeilen) ergänzt, wo möglich
- ✅ Terminologie vereinheitlicht:
  - `IssueSpan` (nicht `ErrorSpan`)
  - `issue_spans` (nicht `errors`)
  - `run_errors` (nicht `errors`)
  - `IssueSpan:Error` (Dual-Label in Neo4j)
  - `HAS_ISSUE_SPAN` (nicht `HAS_ERROR`)

---

## Geänderte Dateien

1. `docs/milestones/M1_setup.md`
2. `docs/milestones/M2_datenmodell.md`
3. `docs/milestones/M3_eval_core_skelett.md`
4. `docs/milestones/M4_graph_modell.md`
5. `docs/milestones/M5_eval_core_factuality_agent.md`
6. `docs/milestones/M6_claim_basierter_factuality_agent_und_evaluationsinfrastruktur.md`
7. `docs/milestones/M7_kohärenz_agent.md`
8. `docs/milestones/M8_readability_agent.md`
9. `docs/milestones/M9_explainability_modul.md`
10. `docs/milestones/M10_evaluation_setup.md`
11. `docs/milestones/README.md`
12. `docs/status/milestone_code_audit.md` (neu)

---

## Zitierweise

**Format:** `(vgl. Anhang A, M{N}; Repo-Tag: <TAG>)`

**Zitations-Strings pro Milestone:**
- M1: `Anhang A, M1 (docs/milestones/M1_setup.md; Repo-Tag: thesis-milestones-2026-01-17)`
- M2: `Anhang A, M2 (docs/milestones/M2_datenmodell.md; Repo-Tag: thesis-milestones-2026-01-17)`
- M3: `Anhang A, M3 (docs/milestones/M3_eval_core_skelett.md; Repo-Tag: thesis-milestones-2026-01-17)`
- M4: `Anhang A, M4 (docs/milestones/M4_graph_modell.md; Repo-Tag: thesis-milestones-2026-01-17)`
- M5: `Anhang A, M5 (docs/milestones/M5_eval_core_factuality_agent.md; Repo-Tag: thesis-milestones-2026-01-17)`
- M6: `Anhang A, M6 (docs/milestones/M6_claim_basierter_factuality_agent_und_evaluationsinfrastruktur.md; Repo-Tag: thesis-milestones-2026-01-17)`
- M7: `Anhang A, M7 (docs/milestones/M7_kohärenz_agent.md; Repo-Tag: thesis-milestones-2026-01-17)`
- M8: `Anhang A, M8 (docs/milestones/M8_readability_agent.md; Repo-Tag: thesis-milestones-2026-01-17)`
- M9: `Anhang A, M9 (docs/milestones/M9_explainability_modul.md; Repo-Tag: thesis-milestones-2026-01-17)`
- M10: `Anhang A, M10 (docs/milestones/M10_evaluation_setup.md; Repo-Tag: thesis-milestones-2026-01-17)`
- M12: `Anhang A, M12 (docs/milestones/M12_streamlit_interface.md; Repo-Tag: thesis-milestones-2026-01-17)`

---

## Nächste Schritte

1. **Commit durchführen:**
   ```bash
   git add docs/milestones/ docs/status/milestone_code_audit.md docs/status/milestone_finalization_report.md
   git commit -m "docs: finalize milestone documentation for thesis referencing"
   ```

2. **Git-Tag setzen:**
   ```bash
   git tag -a thesis-milestones-2026-01-17 -m "Milestone docs for thesis (finalized and code-audited)"
   git push origin thesis-milestones-2026-01-17
   ```

3. **Commit-Hash notieren:**
   Nach dem Commit: `git rev-parse HEAD` ausführen und Hash dokumentieren.

---

## Status

✅ Alle Milestone-Dokumente finalisiert  
✅ Code-Audit durchgeführt  
✅ README.md aktualisiert  
✅ Zitierweise dokumentiert  
⏳ Commit ausstehend  
⏳ Git-Tag ausstehend

