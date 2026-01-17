# Explainability-Modul: Mini-Audit

**Datum:** 2026-01-17 01:06:18
**Fixture:** minimal
**Version:** m9_v1

---

## Contract Check

**Status:** ✅ PASS

✅ Alle Contract-Checks bestanden

## Determinism Check

**Status:** ✅ PASS
**Message:** All 5 runs produced identical output

---

## Beispiel-Auszug

**Anzahl Findings:** 2
- High: 2
- Medium: 0
- Low: 0

**Coverage:** 6 Zeichen (7.41%)

### Findings pro Dimension:

- **factuality:** 2
- **coherence:** 0
- **readability:** 0

### Top 3 Spans:

1. [factuality, high] (60-64): „. Da“ (score: 9.39)
2. [factuality, high] (30-32): „ T“ (score: 7.56)

### Executive Summary:

- In der Summary wurden 2 Findings identifiziert (2 high, 0 medium, 0 low).
- Der Schwerpunkt liegt bei **factuality** (meiste Findings in dieser Dimension).
- Kritische Textstellen: „. Da“, „T“.

---

## Persistence Check

⚠️  Persistence-Checks sind optional und werden nur ausgeführt, wenn DB verfügbar ist.
Siehe `tests/explainability/test_explainability_persistence_*.py` für Details.
