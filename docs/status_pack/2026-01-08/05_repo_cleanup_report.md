# Repo-Cleanup-Report

**Datum:** 2026-01-08  
**Zweck:** Identifiziert überflüssige Dateien, Dead Code und Archivierungs-Kandidaten.

**Wichtig:** Dieser Report schlägt nur vor, was archiviert oder gelöscht werden könnte. **NICHT automatisch löschen!**

---

## Repo-Struktur (Top-Level)

| Ordner | Zweck | Status |
|--------|-------|--------|
| `app/` | Hauptanwendung (API, Services, Agents, Pipeline) | KEEP |
| `scripts/` | Evaluations- und Utility-Scripts | KEEP (siehe Details unten) |
| `configs/` | Konfigurationsdateien (M10 Runs) | KEEP |
| `data/` | Datensätze (FRANK, SummEval, FineSumFact) | KEEP |
| `results/` | Evaluations-Ergebnisse (Artefakte) | KEEP (nicht committen) |
| `docs/` | Dokumentation (Milestones, Pläne, Status-Pack) | KEEP |
| `tests/` | Unit- und Integration-Tests | KEEP |
| `dashboard/` | Streamlit-Dashboard (optional) | REVIEW |
| `evaluation_configs/` | Evaluations-Konfigurationen | KEEP |

---

## Dead-Code-Kandidaten

### Tabelle: Dateien/Ordner | Status | Begründung | Risiko

| Datei/Ordner | Status | Begründung | Risiko |
|--------------|--------|------------|--------|
| `scripts/archive_eval_factuality_binary_v1.py` | **ARCHIVE** | Dateiname enthält 'archive', vermutlich alte Version | **low** |
| `app/services/agents/factuality/ablation_verifier.py` | **REVIEW** | Ablation-Modul (möglicherweise experimentell), nicht in Pipeline verwendet | **medium** |
| `app/services/agents/factuality/ablation_extractor.py` | **REVIEW** | Ablation-Modul (möglicherweise experimentell), nicht in Pipeline verwendet | **medium** |
| `scripts/eval_factuality_binary_v2.py` | **KEEP** | Aktuelle Version (v2), wird möglicherweise noch verwendet | **low** |
| `scripts/eval_factuality_structured.py` | **KEEP** | Strukturierte Evaluation, möglicherweise noch verwendet | **low** |
| `scripts/eval_unified.py` | **KEEP** | Unified Evaluation, möglicherweise noch verwendet | **low** |
| `scripts/aggregate_m10_results.py` | **KEEP** | M10-Aggregation, möglicherweise noch verwendet | **low** |
| `scripts/print_*.py` | **REVIEW** | Utility-Scripts für Debugging, möglicherweise nicht mehr benötigt | **low** |
| `docs/archive/` | **ARCHIVE** | Archivierte Dokumentation, explizit als Archive markiert | **low** |
| `results/archive/` | **ARCHIVE** | Archivierte Ergebnisse, explizit als Archive markiert | **low** |

---

## Verwendungsanalyse

### Scripts mit "archive" im Namen

- **`scripts/archive_eval_factuality_binary_v1.py`**
  - **Status:** ARCHIVE
  - **Begründung:** Dateiname enthält 'archive', vermutlich alte Version von `eval_factuality_binary_v2.py`.
  - **Verwendung:** Nicht in anderen Scripts importiert (grep-Ergebnis: nur in Dokumentation erwähnt).
  - **Risiko:** Low (kann in `scripts/archive/` verschoben werden).

### Ablation-Module

- **`app/services/agents/factuality/ablation_verifier.py`**
  - **Status:** REVIEW
  - **Begründung:** Ablation-Modul (experimentell), nicht in `factuality_agent.py` verwendet.
  - **Verwendung:** Nicht in Pipeline verwendet (grep-Ergebnis: nur in `factuality_agent.py` erwähnt, aber nicht importiert).
  - **Risiko:** Medium (könnte für zukünftige Experimente nützlich sein).

- **`app/services/agents/factuality/ablation_extractor.py`**
  - **Status:** REVIEW
  - **Begründung:** Ablation-Modul (experimentell), nicht in `factuality_agent.py` verwendet.
  - **Verwendung:** Nicht in Pipeline verwendet (grep-Ergebnis: nur in `factuality_agent.py` erwähnt, aber nicht importiert).
  - **Risiko:** Medium (könnte für zukünftige Experimente nützlich sein).

### Doppelte Eval-Scripts

- **`scripts/eval_factuality_binary_v1.py` vs. `eval_factuality_binary_v2.py`**
  - **Status:** v1 = ARCHIVE (nicht gefunden, möglicherweise bereits gelöscht), v2 = KEEP
  - **Begründung:** v2 ist die aktuelle Version, v1 ist archiviert.
  - **Risiko:** Low (v1 ist bereits archiviert).

---

## Empfehlungen

### Safe to Remove (niedriges Risiko)

1. **`scripts/archive_eval_factuality_binary_v1.py`** → Verschieben nach `scripts/archive/` oder löschen (wenn v2 funktioniert).

### Review Required (mittleres Risiko)

1. **`app/services/agents/factuality/ablation_*.py`** → Prüfen, ob für zukünftige Experimente benötigt. Falls nicht: Verschieben nach `app/services/agents/factuality/archive/` oder löschen.

2. **`scripts/print_*.py`** → Prüfen, ob für Debugging noch benötigt. Falls nicht: Verschieben nach `scripts/archive/` oder löschen.

### Keep (aktuell verwendet)

- Alle `scripts/eval_*.py` (außer archive-Versionen)
- Alle `scripts/aggregate_*.py`
- Alle Agent-Module (außer Ablation, falls nicht verwendet)
- Alle Service-Module
- Alle Pipeline-Module

---

## Archivierungs-Strategie

**Empfohlene Struktur:**
```
scripts/
  archive/
    archive_eval_factuality_binary_v1.py
    (weitere archivierte Scripts)

app/services/agents/factuality/
  archive/
    ablation_verifier.py
    ablation_extractor.py
    (weitere experimentelle Module)
```

**Vorgehen:**
1. Erstelle `scripts/archive/` und `app/services/agents/factuality/archive/`.
2. Verschiebe Kandidaten dorthin (nicht löschen, falls später benötigt).
3. Dokumentiere in `docs/archive/README.md`, welche Dateien archiviert wurden und warum.

---

## Git-Status-Check

**Empfehlung:** Vor dem Archivieren/Löschen:
1. Prüfe `git log` für jede Datei (wann zuletzt geändert?).
2. Prüfe `git blame` für wichtige Dateien (wer hat sie zuletzt geändert?).
3. Prüfe, ob Dateien in `.gitignore` sind (dann nicht committen, aber lokal behalten).

---

**Nächste Schritte:**
1. Manuelle Prüfung der REVIEW-Kandidaten.
2. Verschieben der ARCHIVE-Kandidaten in entsprechende Archive-Ordner.
3. Dokumentation der Archivierungs-Entscheidungen.

