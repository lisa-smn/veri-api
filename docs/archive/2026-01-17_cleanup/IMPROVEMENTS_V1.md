# Factuality Agent Verbesserungen V1

**Status:** Implementiert, bereit für Evaluation

## Implementierte Verbesserungen

### 1. Gewichtete Decision Logic ✅

**Problem:** Einfaches Zählen von Issues (`num_issues >= threshold`) ist zu grob.

**Lösung:** Gewichtete Issue-Aggregation:
- `weight = severity_weight * confidence_weight * uncertainty_weight`
- `severity_weight`: low=1.0, medium=1.5, high=2.0 (konfigurierbar)
- `confidence_weight`: clamp(confidence, 0.0..1.0)
- `uncertainty_weight`: 
  - `non_error`: UNCERTAIN = 0.0
  - `weight_0.5`: UNCERTAIN = 0.5
  - `error`: UNCERTAIN = 1.0
- `prediction = sum(weights) >= decision_threshold_float`

**Neue Config-Parameter:**
- `decision_threshold_float`: Float-Threshold (z.B. 0.8, 1.0, 1.2, 1.5, 1.8)
- `severity_weights`: Dict mit low/medium/high Weights
- `confidence_min`: Optional, minimale Confidence (default: 0.0)

**Code-Änderungen:**
- `scripts/run_m10_factuality.py`: `compute_weighted_issue_score()` Funktion
- `app/services/agents/factuality/factuality_agent.py`: ErrorSpan erweitert um `confidence`, `evidence_found`
- `app/models/pydantic.py`: ErrorSpan erweitert

### 2. Verbessertes Uncertainty-Handling im ClaimVerifier ✅

**Problem:** Viele False Positives durch "incorrect" ohne belastbare Evidence.

**Lösung:** 
- Nur "incorrect", wenn:
  a) Claim extrahiert wurde
  b) Passende Evidenzspan(s) im Artikel gefunden wurden
  c) Echter Widerspruch/Fehlbehauptung vorliegt (Zahlen, Orte, Personen, Daten)
- Bei vagen/interpretativen Claims: bevorzugt "uncertain"

**Code-Änderungen:**
- `app/services/agents/factuality/claim_verifier.py`: 
  - `evidence_found` Flag wird gesetzt
  - Incorrect ohne Evidence => downgrade zu uncertain
- `app/services/agents/factuality/claim_models.py`: 
  - `evidence_found`, `evidence_spans` Felder hinzugefügt
- `app/services/agents/factuality/factuality_agent.py`:
  - ErrorSpan erhält `confidence` und `evidence_found`
  - Evidence-Penalty: incorrect ohne Evidence => downgrade zu uncertain weight

### 3. Neue Tuning-Runs hinzugefügt ✅

**8 neue Runs in `configs/m10_factuality_runs.yaml`:**

**Uncertainty Policy = non_error:**
- `factuality_frank_tune_weighted_non_error_0.8_v1` (threshold=0.8)
- `factuality_frank_tune_weighted_non_error_1.0_v1` (threshold=1.0)
- `factuality_frank_tune_weighted_non_error_1.2_v1` (threshold=1.2)
- `factuality_frank_tune_weighted_non_error_1.5_v1` (threshold=1.5)

**Uncertainty Policy = weight_0.5:**
- `factuality_frank_tune_weighted_weight05_1.0_v1` (threshold=1.0)
- `factuality_frank_tune_weighted_weight05_1.2_v1` (threshold=1.2)
- `factuality_frank_tune_weighted_weight05_1.5_v1` (threshold=1.5)
- `factuality_frank_tune_weighted_weight05_1.8_v1` (threshold=1.8)

**Alle Runs:**
- `severity_min: "low"` (nicht medium, um Recall hoch zu halten)
- `severity_weights: {low: 1.0, medium: 1.5, high: 2.0}`
- `confidence_min: 0.0` (noch nicht getuned)

### 4. Best-Run-Auswahl verbessert ✅

**Code-Änderungen:**
- `scripts/select_best_tuned_run.py`:
  - Primäres Gate: `recall >= recall_min` (default: 0.70, konfigurierbar)
  - Sekundär: max `balanced_accuracy`
  - Tie-breaker: max `precision`
  - Erkennt auch "weighted" Runs

## Nächste Schritte

### 1. Evaluation ausführen

```bash
# Alle neuen gewichteten Runs ausführen
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml --skip-baseline

# Aggregation
python3 scripts/aggregate_m10_results.py

# Best Run auswählen
python3 scripts/select_best_tuned_run.py --recall-min 0.70
```

### 2. Erwartete Verbesserungen

**Ziele:**
- ✅ Recall >= 0.70 (durch `severity_min=low`)
- ✅ Specificity deutlich besser als aktuell (durch gewichtete Aggregation + uncertainty handling)
- ✅ Balanced Accuracy > 0.524 (aktueller best tuned Stand)

**Mechanismen:**
- Gewichtete Aggregation reduziert "Phantom Errors" (uncertain zählt weniger/nicht)
- Evidence-Penalty reduziert False Positives (incorrect ohne Evidence => downgrade)
- `severity_min=low` hält Recall hoch

### 3. Weitere Verbesserungen (später)

**Noch nicht implementiert:**
- Claim Extraction: Deduplizierung, Priorisierung "hard claims"
- Span-Qualität: Robustes Mapping, mapping_confidence
- Confidence-Min Tuning: confidence_min ∈ {0.0, 0.3, 0.5}

## Akzeptanzkriterien

- ✅ Gewichtete Decision Logic implementiert
- ✅ Uncertainty-Handling verbessert
- ✅ Neue Tuning-Runs definiert
- ✅ Best-Run-Auswahl angepasst
- ⏳ **Evaluation läuft** - Ergebnisse abwarten

## Dokumentation

- Alle neuen Parameter werden in Run-Dokumentation geloggt
- `summary_matrix.csv` enthält alle Metriken
- `select_best_tuned_run.py` gibt Begründung aus






