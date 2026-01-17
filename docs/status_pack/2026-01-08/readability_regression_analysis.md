# Readability Regression & Wiring Analysis

## 1. Wiring-Check

- **Agent-Run hat pred_agent:** True
- **Agent-Run hat pred_judge:** False
- **Judge-Run hat pred_agent:** True
- **Judge-Run hat pred_judge:** True
- **Agent vs Judge Korrelation (im Judge-Run):** 0.4359
- **Scores unterscheiden sich:** True

**Fazit Wiring:** ✅ Wiring funktioniert: Agent und Judge Scores werden getrennt gespeichert und ausgewertet.

## 2. Regression-Check: v1 alt vs v1 neu

### Config-Vergleich

| | Alt (20260114) | Neu Agent (20260115) | Neu Judge (20260115) |
|---|---|---|---|
| Git Commit | 558e1744 | 558e1744 | 558e1744 |
| Prompt Version | v1 | v1 | v1 |
| Cache | False | True | True |

### Metriken-Vergleich

| | Alt (20260114) | Neu Agent (20260115) | Neu Judge (20260115) |
|---|---|---|---|
| Spearman ρ | 0.4053 | 0.1320 | 0.0990 |
| Pearson r | 0.3865 | 0.0863 | 0.0595 |
| MAE | 0.2845 | 0.4537 | 0.5729 |

### Distribution-Vergleich

#### Alt (20260114)

- [0.4, 0.6): 41
- [0.6, 0.8): 105
- [0.8, 1.0): 47
- [1.0, 1.0]: 7

#### Neu Agent (20260115)

- [0.4, 0.6): 166
- [0.6, 0.8): 34

#### Neu Judge (20260115)

- [0.2, 0.4): 115
- [0.4, 0.6): 85

### Regression-Erklärung

**Beobachtung:**
- Alt: Spearman ρ = 0.4053
- Neu Agent: Spearman ρ = 0.1320 (Verschlechterung um 0.2733)
- Neu Judge: Spearman ρ = 0.0990

**Mögliche Ursachen:**
1. **Cache-Effekt:** Alt hatte `cache=false`, neu hat `cache=true` → möglicherweise werden alte, bessere Cached-Ergebnisse verwendet
2. **Distribution-Collapse:** Neu zeigt stark komprimierte Verteilung ([0.4, 0.6): 166 von 200) → geringe Varianz führt zu niedriger Korrelation
3. **LLM-Varianz:** Gleicher Prompt, aber LLM-Output variiert zwischen Runs

**Empfehlung:**
- Für finale Evaluation: Cache deaktivieren oder Cache leeren
- Distribution prüfen: Wenn >80% in einem Bucket → Collapse-Detector sollte warnen
- Prompt-Version prüfen: Sicherstellen, dass 'use full scale' enthalten ist