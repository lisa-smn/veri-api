# Final Judge Run Instructions (Factuality)

**Status:** ⚠️ AUSSTEHEND - Muss außerhalb Sandbox ausgeführt werden

## 1. Dataset-Check

```bash
# Prüfe Label-Verteilung (beide Klassen müssen vorhanden sein)
jq -r '.has_error' data/frank/frank_clean.jsonl | sort | uniq -c
# Erwartung: pos=1436, neg=810 (beide Klassen vorhanden)
```

## 2. Final Judge Run ausführen

```bash
ENABLE_LLM_JUDGE=true JUDGE_MODE=primary JUDGE_N=3 JUDGE_TEMPERATURE=0 \
python scripts/eval_frank_factuality_llm_judge.py \
  --data data/frank/frank_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --bootstrap_n 2000 \
  --cache_mode write \
  --prompt_version v2_binary \
  --judge_n 3 \
  --judge_temperature 0.0 \
  --judge_aggregation majority
```

**Notiere den Output-Ordnerpfad:** `results/evaluation/factuality/judge_factuality_<TIMESTAMP>_gpt-4o-mini_v2_binary_seed42/`

## 3. Quality Verify ausführen

```bash
python scripts/verify_quality_factuality_coherence.py \
  --judge_run results/evaluation/factuality/judge_factuality_<TIMESTAMP>_gpt-4o-mini_v2_binary_seed42
```

**Erwartung:** Alle Checks 1-5 PASS (keine NOT RUN)

## 4. Dokumentation aktualisieren

Nach erfolgreichem Final-Run und Verify:

1. **`docs/status/factuality_status.md`**:
   - Ersetze "TBD" in der Vergleichstabelle mit Final-Run-Ergebnissen (aus `summary.json`)
   - Aktualisiere Final-Run-Pfad in "Artefakte"
   - Aktualisiere Status von "ausstehend" zu "✅ erledigt"

2. **`docs/status_pack/2026-01-08/03_evaluation_results.md`**:
   - Ersetze `<FINAL_TIMESTAMP>` mit tatsächlichem Timestamp
   - Füge Judge-Metriken in Tabelle A ein

3. **`docs/status_pack/2026-01-08/06_appendix_artifacts_index.md`**:
   - Ersetze `<FINAL_TIMESTAMP>` mit tatsächlichem Timestamp

4. **`docs/status/agents_verification_audit.md`**:
   - Setze Factuality auf ✅ **VERIFIED** (nicht mehr "nach Final-Run")
   - Aktualisiere Evidence Paths mit Final-Run-Pfad

5. **Optional: `docs/status_pack/2026-01-08/00_executive_summary.md`**:
   - 2-3 Zeilen Update: "Factuality Judge executed on frank_clean (n=200), coherence status documented."

## 5. Final Sanity Pass

```bash
# Verify nochmal ausführen
python scripts/verify_quality_factuality_coherence.py --judge_run <FINAL_RUN_PATH>

# Tests
python -m pytest -q

# Konsistenz prüfen:
# - Zahlen in factuality_status.md = summary.json
# - Pfade in status_pack = tatsächliche Run-Ordner
# - Audit zeigt VERIFIED für alle drei Agenten
```

