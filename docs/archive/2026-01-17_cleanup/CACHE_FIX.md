# Cache-Problem behoben

## Problem

Der `run_m10_factuality.py` Script hat **KEINEN Cache verwendet**, obwohl:
- ✅ Cache-Dateien vorhanden waren (882 FRANK, 200 FineSumFact)
- ✅ `cache_enabled: true` in Config gesetzt war

**Folge:** Jeder Run hat alle LLM-Aufrufe gemacht → **sehr langsam** (Stunden statt Minuten)

## Lösung

**Cache-Integration implementiert:**
- ✅ `cache_key()` Funktion (kompatibel mit `eval_factuality_binary_v2`)
- ✅ `load_cache()` Funktion
- ✅ `append_cache()` Funktion
- ✅ Cache-Prüfung vor jedem `agent.run()`
- ✅ Cache-Speicherung nach jedem `agent.run()`
- ✅ AgentResult-Rekonstruktion aus Cache

**Code-Änderungen:**
- `scripts/run_m10_factuality.py`: Cache-Funktionen hinzugefügt
- Cache-Pfad basierend auf Dataset (FRANK vs FineSumFact)
- Cache-Statistiken werden geloggt

## Erwartete Laufzeit (mit Cache)

**Vorher (ohne Cache):**
- ~3-5 Sekunden pro Example (2 LLM-Aufrufe)
- 4,200 Examples × 3-5s = **3.5-6 Stunden**

**Nachher (mit Cache):**
- ~0.1-0.5 Sekunden pro Example (nur Decision Logic)
- 4,200 Examples × 0.3s = **~20-30 Minuten**

## Nächste Schritte

1. **Aktuellen Run abbrechen** (falls noch läuft):
   ```bash
   # Prozess finden und beenden
   pkill -f run_m10_factuality
   ```

2. **Neu starten** (mit Cache):
   ```bash
   python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml --skip-baseline
   ```

3. **Progress beobachten:**
   - Cache-Hit-Rate wird geloggt
   - Sollte ~100% sein (alle Examples gecached)

## Cache-Format

**Kompatibilität:**
- Cache-Key: `sha256:...` (mit Prefix, kompatibel mit `eval_factuality_binary_v2`)
- Cache-Value: `{"score": ..., "issue_spans": ..., "details": ...}`
- Issue-Spans werden zu `ErrorSpan` Objekten rekonstruiert

**Neue Felder:**
- `confidence`, `mapping_confidence`, `evidence_found` werden aus Cache gelesen (falls vorhanden)
- Falls nicht vorhanden: `None` (optional fields)






