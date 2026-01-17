# Klassische Metriken als Baselines

Dieses Kapitel vergleicht die Agent- und Judge-Methoden mit klassischen Baseline-Metriken für Readability.

## System vs Humans

| System | Spearman ρ | Pearson r | MAE | RMSE | R² | n |
|---|---|---|---|---|---|---|
| **Agent** | 0.4019 [0.2680, 0.5118] | 0.3900 [0.2920, 0.4676] | 0.2825 [0.2631, 0.3021] | 0.3164 [0.3002, 0.3322] | -2.7729 | 200 |
| **Judge** | 0.2803 | 0.3432 | 0.4167 | 0.4458 | -6.4921 | 200 |
| **FLESCH** | -0.0535 [-0.1973, 0.0848] | 0.1683 [-0.0895, 0.3858] | 0.3836 [0.3622, 0.4046] | 0.4135 [0.3932, 0.4331] | -4.0614 | 200 |
| **FLESCH-KINCAID** | -0.0546 [-0.1992, 0.0927] | 0.1235 [-0.1147, 0.3375] | 0.4484 [0.4247, 0.4732] | 0.4831 [0.4606, 0.5054] | -5.9084 | 200 |
| **GUNNING-FOG** | -0.0392 [-0.1723, 0.1010] | 0.0473 [-0.1248, 0.2127] | 0.5789 [0.5478, 0.6091] | 0.6217 [0.5938, 0.6486] | -10.4426 | 200 |

## Interpretation

**ROUGE/BERTScore/BLEU/METEOR nicht berechenbar:** Der SummEval-Datensatz enthält keine Referenz-Zusammenfassungen (Gold-Standard).
Daher können Similarity-Metriken wie ROUGE, BERTScore, BLEU und METEOR nicht berechnet werden, da diese einen Vergleich zwischen System-Output und Referenz erfordern.

**Readability-Formeln:** Die klassischen Readability-Formeln (Flesch Reading Ease, Flesch-Kincaid Grade Level, Gunning Fog Index) wurden berechnet, da diese keine Referenzen benötigen.
Diese Formeln basieren auf statistischen Eigenschaften des Textes (Satzlänge, Silbenanzahl, komplexe Wörter) und messen die Lesbarkeit direkt am Text.

**Korrelationen:** Die Korrelationen (Spearman ρ, Pearson r) zeigen, ob die Baseline-Metriken menschliche Urteile überhaupt treffen können.
Niedrige Korrelationen deuten darauf hin, dass die Baseline-Metriken andere Aspekte messen als menschliche Bewerter.

**Vergleich mit Agent/Judge:** 

Die Agent-Methode zeigt eine moderate Korrelation mit menschlichen Bewertungen (Spearman ρ = 0.40, Pearson r = 0.39), was darauf hindeutet, dass der Agent zumindest teilweise menschliche Lesbarkeitsbewertungen vorhersagen kann. Die Judge-Methode zeigt eine schwächere, aber immer noch positive Korrelation (Spearman ρ = 0.28, Pearson r = 0.34).

Im Vergleich dazu zeigen die klassischen Readability-Formeln (Flesch, Flesch-Kincaid, Gunning Fog) nahezu keine Korrelation (Spearman ρ ≈ -0.05 bis -0.04), was darauf hindeutet, dass diese Formeln andere Aspekte messen als menschliche Bewerter.

**Fazit:** Die klassischen Readability-Formeln, die auf statistischen Textmerkmalen basieren (Satzlänge, Silbenanzahl), sind nicht geeignet, um menschliche Lesbarkeitsbewertungen vorherzusagen. Die Agent-Methode zeigt deutlich bessere Ergebnisse als die Judge-Methode, und beide übertreffen die klassischen Formeln deutlich.