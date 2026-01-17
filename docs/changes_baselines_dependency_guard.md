# Changelog: Fail-fast Dependency Checks für Factuality Baselines

**Datum:** 2025-01-07  
**Betroffene Dateien:**
- `scripts/eval_frank_factuality_baselines.py`
- `scripts/aggregate_factuality_runs.py`
- `docs/m10_factuality_baselines.md`

---

## Zusammenfassung

Implementierung von Fail-fast Dependency-Checks für Factuality-Baseline-Evaluation, um stille 0.0-Placeholder zu verhindern und Reproduzierbarkeit zu erhöhen.

---

## Änderungen

### 1. Dependency-Checks (`eval_frank_factuality_baselines.py`)

- **Neue Funktion:** `check_dependencies(baseline_type)` prüft, ob benötigte Packages vorhanden sind
- **Fail-fast:** Script bricht standardmäßig mit `SystemExit(1)` ab, wenn Dependencies fehlen
- **Debug-Modus:** `--allow_dummy_baseline` erlaubt 0.0 als Dummy-Wert (nur für Entwicklung)
- **Metadata:** `run_metadata.json` enthält `dependencies_ok` und `missing_packages`

### 2. Safety-Markierungen

- **summary.md:** Warnbox bei invalid/dummy runs
- **Aggregator:** Erkennt invalid runs und schließt sie aus Quick Comparison aus
- **CSV:** Spalten `dependencies_ok`, `allow_dummy`, `missing_packages`, `is_invalid`

### 3. Dokumentation

- **docs/m10_factuality_baselines.md:** Neuer Abschnitt "Dependency-Schutz / Fail-fast"
- Erklärt Standardverhalten, Debug-Modus, und Beispiele

---

## Commit-Message (Vorschlag)

```
Fail-fast dependency checks for factuality baselines + dummy mode + metadata

- Add check_dependencies() to verify required packages before evaluation
- Fail-fast by default (SystemExit(1)) if dependencies missing
- Add --allow_dummy_baseline flag for debugging (not for thesis)
- Store dependencies_ok and missing_packages in run_metadata.json
- Mark invalid/dummy runs in summary.md with warning box
- Extend aggregator to detect and exclude invalid runs from Quick Comparison
- Add dependency status columns to CSV output
- Update documentation with dependency guard explanation

Prevents silent 0.0 placeholders and improves reproducibility.
```

---

## Breaking Changes

**Keine** - Bestehende Runs bleiben kompatibel. Neue Runs ohne Dependencies brechen ab (erwartetes Verhalten).

---

## Migration

**Keine Migration nötig.** Bestehende Runs funktionieren weiterhin. Neue Runs müssen Dependencies installieren oder explizit `--allow_dummy_baseline` verwenden (nur für Debugging).

