# Readability Judge: Final Improvement Attempt

**Datum:** 2026-01-08  
**Ziel:** Einziger kontrollierter Improvement-Versuch für den Readability-Judge

## Setup

### Baseline Run
- **n:** 200 (seed=42)
- **Cache:** OFF (`--cache_mode off`)
- **Judge:** LLM-as-a-Judge, Prompt v1, Committee n=3, Temperature=0
- **Agent:** ReadabilityAgent, Prompt v1
- **Bootstrap:** n=2000

### Improved Run
- **n:** 200 (seed=42, identisch zu Baseline)
- **Cache:** OFF (`--cache_mode off`)
- **Judge:** LLM-as-a-Judge, Prompt **v2_float**, Committee n=3, Temperature=0
- **Agent:** ReadabilityAgent, Prompt v1 (unverändert)
- **Bootstrap:** n=2000

### Einzige Änderung
- **Judge-Prompt:** Von `rating: 1–5 int` (v1) auf `score: 0.00–1.00 float` (v2_float)
- **Neue Features in v2_float:**
  - Score-Skala: 0.00–1.00 (zwei Dezimalstellen)
  - Anker: 0.20 (schwer lesbar), 0.50 (mittel), 0.80 (sehr gut)
  - Explizite Anweisung: "Use full scale" (Werte <0.40 und >0.80 verwenden, wenn passend)
  - Warnung: "Wenn unsicher, verwende nicht automatisch 0.50"

## Baseline Results

### Agent vs Humans
| Metrik | Wert | 95% CI |
|---|---|---|
| Spearman ρ | TBD | [TBD, TBD] |
| Pearson r | TBD | [TBD, TBD] |
| MAE | TBD | [TBD, TBD] |
| RMSE | TBD | [TBD, TBD] |
| R² | TBD | - |
| n | 200 | - |

### Judge vs Humans
| Metrik | Wert | 95% CI |
|---|---|---|
| Spearman ρ | TBD | [TBD, TBD] |
| Pearson r | TBD | [TBD, TBD] |
| MAE | TBD | [TBD, TBD] |
| RMSE | TBD | [TBD, TBD] |
| R² | TBD | - |
| n | 200 | - |

### Agent vs Judge Agreement (n=200)
| Metrik | Wert |
|---|---|
| Spearman ρ | TBD |
| Pearson r | TBD |
| Threshold Agreement (≥0.7) | TBD% |

### Committee Reliability (n=3)
| Metrik | Wert |
|---|---|
| Mean Std | TBD |
| Median Std | TBD |
| Full Agreement | TBD% |
| Majority Agreement | TBD% |

### Prediction Distribution (Baseline)
**Agent:**
- [0.0, 0.2): TBD
- [0.2, 0.4): TBD
- [0.4, 0.6): TBD
- [0.6, 0.8): TBD
- [0.8, 1.0]: TBD

**Judge:**
- [0.0, 0.2): TBD
- [0.2, 0.4): TBD
- [0.4, 0.6): TBD
- [0.6, 0.8): TBD
- [0.8, 1.0]: TBD

### Collapse Detection
- **Status:** TBD (✅ Kein Collapse / ⚠️ Collapse erkannt)

## Improved Results

### Agent vs Humans
| Metrik | Wert | 95% CI | Delta (vs Baseline) |
|---|---|---|---|
| Spearman ρ | TBD | [TBD, TBD] | TBD |
| Pearson r | TBD | [TBD, TBD] | TBD |
| MAE | TBD | [TBD, TBD] | TBD |
| RMSE | TBD | [TBD, TBD] | TBD |
| R² | TBD | - | TBD |
| n | 200 | - | - |

### Judge vs Humans
| Metrik | Wert | 95% CI | Delta (vs Baseline) |
|---|---|---|---|
| Spearman ρ | TBD | [TBD, TBD] | TBD |
| Pearson r | TBD | [TBD, TBD] | TBD |
| MAE | TBD | [TBD, TBD] | TBD |
| RMSE | TBD | [TBD, TBD] | TBD |
| R² | TBD | - | TBD |
| n | 200 | - | - |

### Agent vs Judge Agreement (n=200)
| Metrik | Wert | Delta (vs Baseline) |
|---|---|---|
| Spearman ρ | TBD | TBD |
| Pearson r | TBD | TBD |
| Threshold Agreement (≥0.7) | TBD% | TBD% |

### Committee Reliability (n=3)
| Metrik | Wert | Delta (vs Baseline) |
|---|---|---|
| Mean Std | TBD | TBD |
| Median Std | TBD | TBD |
| Full Agreement | TBD% | TBD% |
| Majority Agreement | TBD% | TBD% |

### Prediction Distribution (Improved)
**Agent:**
- [0.0, 0.2): TBD
- [0.2, 0.4): TBD
- [0.4, 0.6): TBD
- [0.6, 0.8): TBD
- [0.8, 1.0]: TBD

**Judge:**
- [0.0, 0.2): TBD
- [0.2, 0.4): TBD
- [0.4, 0.6): TBD
- [0.6, 0.8): TBD
- [0.8, 1.0]: TBD

### Collapse Detection
- **Status:** TBD (✅ Kein Collapse / ⚠️ Collapse erkannt)

## Interpretation

### Varianz (Distribution)
- **Baseline:** TBD
- **Improved:** TBD
- **Verbesserung:** TBD

### Ranking (Spearman)
- **Baseline Judge vs Humans:** TBD
- **Improved Judge vs Humans:** TBD
- **Verbesserung:** TBD

### Committee Realismus
- **Baseline Std:** TBD (erwartet: >0 wegen Cache OFF)
- **Improved Std:** TBD (erwartet: >0 wegen Cache OFF)
- **Interpretation:** TBD

## Fazit

**Letzter Versuch abgeschlossen.** Nach diesem kontrollierten Improvement-Versuch wird kein weiteres Tuning durchgeführt.

## Reproduzierbarkeit

- **Baseline Run:** `results/evaluation/readability/TBD/`
- **Improved Run:** `results/evaluation/readability/TBD/`
- **Vergleichs-Report:** `docs/status_pack/2026-01-08/readability_last_attempt.md` (dieses Dokument)
