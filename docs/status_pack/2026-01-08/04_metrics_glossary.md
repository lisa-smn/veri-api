# Metriken-Glossar

**Zweck:** Kurze, verständliche Erklärung aller verwendeten Metriken.

---

## Tabelle: Metriken-Übersicht

| Metric | Was misst das? | Skala | Höher/Lower besser | Wann sinnvoll | Stolperfallen |
|--------|----------------|------|-------------------|---------------|---------------|
| **Pearson r** | Lineare Korrelation zwischen Vorhersage und Ground Truth | [-1, 1] | Höher besser | Kontinuierliche Scores, lineare Beziehung | Sensibel auf Ausreißer, misst nur lineare Beziehung |
| **Spearman ρ** | Monotone Korrelation (Rangfolge) | [-1, 1] | Höher besser | Kontinuierliche Scores, nicht-lineare Beziehung | Misst nur Rangfolge, nicht absolute Werte |
| **MAE** | Mittlerer absoluter Fehler | [0, ∞) | Lower besser | Kontinuierliche Scores, interpretierbar in Original-Skala | Nicht robust gegen Ausreißer |
| **RMSE** | Root Mean Squared Error | [0, ∞) | Lower besser | Kontinuierliche Scores, bestraft große Fehler stärker | Sensibel auf Ausreißer |
| **R²** | Anteil erklärter Varianz | (-∞, 1] | Höher besser | Kontinuierliche Scores, Vergleich mit Baseline | Kann negativ sein (schlechter als Mittelwert-Baseline) |
| **Precision** | Anteil korrekter positiver Vorhersagen | [0, 1] | Höher besser | Binäre Klassifikation, wenn False Positives teuer sind | Ignoriert False Negatives |
| **Recall** | Anteil gefundener positiver Fälle | [0, 1] | Höher besser | Binäre Klassifikation, wenn False Negatives teuer sind | Ignoriert False Positives |
| **F1** | Harmonisches Mittel aus Precision und Recall | [0, 1] | Höher besser | Binäre Klassifikation, Balance zwischen Precision/Recall | Kann bei Imbalance irreführend sein |
| **Balanced Accuracy** | Mittelwert aus Sensitivity und Specificity | [0, 1] | Höher besser | Binäre Klassifikation bei Imbalance | Weniger bekannt als F1 |
| **MCC** | Matthews Correlation Coefficient | [-1, 1] | Höher besser | Binäre Klassifikation, alle 4 Klassen berücksichtigt | Weniger bekannt, aber robuster als F1 bei Imbalance |
| **AUROC** | Area Under ROC Curve | [0, 1] | Höher besser | Binäre Klassifikation, threshold-frei | Misst Trennfähigkeit, nicht absolute Performance |
| **Bootstrap 95% CI** | Konfidenzintervall via Resampling | - | - | Alle Metriken, Unsicherheitsschätzung | Zeigt Variabilität, nicht systematische Bias |
| **ROUGE-L** | Longest Common Subsequence (Similarity) | [0, 1] | Höher besser | Baseline, Summary vs. Referenz | Misst Ähnlichkeit, nicht Qualität |
| **BERTScore** | Semantische Similarity (BERT-Embeddings) | [0, 1] | Höher besser | Baseline, Summary vs. Referenz | Misst Ähnlichkeit, nicht Qualität |
| **LLM-as-a-Judge (Baseline)** | Automatisierte Bewertung via LLM mit striktem JSON-Output | Variabel | Höher besser (je nach Dimension) | Vergleichsbaseline für Agent, nicht Ground Truth | Moderne LLMs als Baseline, aber nicht als Ersatz für menschliche Bewertungen |

---

## Detaillierte Erklärungen

### Pearson r (Korrelationskoeffizient)

**Was misst es?** Die lineare Korrelation zwischen Vorhersage und Ground Truth.  
**Skala:** [-1, 1], wobei 1 = perfekte positive Korrelation, -1 = perfekte negative Korrelation, 0 = keine Korrelation.  
**Höher = besser?** Ja, höhere Werte bedeuten stärkere lineare Beziehung.  
**Wann sinnvoll?** Bei kontinuierlichen Scores (z.B. Coherence 0-1) und wenn eine lineare Beziehung erwartet wird.  
**Stolperfallen:** Sensibel auf Ausreißer, misst nur lineare Beziehung (nicht monotone).

**Beispiel:** r = 0.35 bedeutet, dass 35% der Varianz linear erklärt werden kann.

---

### Spearman ρ (Rangkorrelation)

**Was misst es?** Die monotone Korrelation (Rangfolge) zwischen Vorhersage und Ground Truth.  
**Skala:** [-1, 1], wobei 1 = perfekte monotone Beziehung, -1 = perfekte inverse monotone Beziehung, 0 = keine monotone Beziehung.  
**Höher = besser?** Ja, höhere Werte bedeuten stärkere monotone Beziehung.  
**Wann sinnvoll?** Bei kontinuierlichen Scores, wenn die Beziehung nicht-linear sein kann (z.B. logarithmisch). **Primär für Ranking-basierte Evaluation:** Spearman ist oft die wichtigere Metrik, da sie robust gegen Skalenfehler ist und die Frage "Ist A besser als B?" beantwortet.  
**Stolperfallen:** Misst nur Rangfolge, nicht absolute Werte (z.B. "A ist besser als B" ist wichtig, nicht "A ist 0.8").

**Beispiel:** ρ = 0.41 bedeutet, dass der Agent die Rangfolge der menschlichen Ratings teilweise erfasst.

**Spearman vs Pearson:**
- **Spearman:** Misst monotone Beziehung (Rangfolge), robust gegen nicht-lineare Beziehungen und Skalenfehler. Ideal für Ranking-basierte Evaluation.
- **Pearson:** Misst lineare Beziehung, sensibel auf Ausreißer und nicht-lineare Beziehungen. Ideal wenn absolute Werte wichtig sind.
- **Empfehlung:** Für Readability/Coherence-Evaluation ist Spearman primär, da Ranking wichtiger ist als absolute Werte.

---

### MAE (Mean Absolute Error)

**Was misst es?** Den mittleren absoluten Fehler zwischen Vorhersage und Ground Truth.  
**Skala:** [0, ∞), wobei 0 = perfekt, größere Werte = größere Fehler.  
**Höher = besser?** Nein, niedrigere Werte sind besser.  
**Wann sinnvoll?** Bei kontinuierlichen Scores, wenn Fehler in der Original-Skala interpretierbar sein sollen.  
**Stolperfallen:** Nicht robust gegen Ausreißer (ein großer Fehler kann den MAE stark erhöhen).

**Beispiel:** MAE = 0.18 bedeutet, dass der Agent im Durchschnitt um 0.18 Punkte (auf Skala 0-1) von den menschlichen Ratings abweicht.

---

### RMSE (Root Mean Squared Error)

**Was misst es?** Die Wurzel des mittleren quadratischen Fehlers zwischen Vorhersage und Ground Truth.  
**Skala:** [0, ∞), wobei 0 = perfekt, größere Werte = größere Fehler.  
**Höher = besser?** Nein, niedrigere Werte sind besser.  
**Wann sinnvoll?** Bei kontinuierlichen Scores, wenn große Fehler stärker bestraft werden sollen.  
**Stolperfallen:** Sensibel auf Ausreißer (quadratische Bestrafung).

**Beispiel:** RMSE = 0.24 bedeutet, dass der Agent im Durchschnitt um 0.24 Punkte (auf Skala 0-1) von den menschlichen Ratings abweicht, wobei große Fehler stärker gewichtet werden.

---

### R² (Determinationskoeffizient)

**Was misst es?** Den Anteil der Varianz, der durch das Modell erklärt wird.  
**Skala:** (-∞, 1], wobei 1 = perfekt, 0 = so gut wie Mittelwert-Baseline, negativ = schlechter als Mittelwert-Baseline.  
**Höher = besser?** Ja, höhere Werte bedeuten mehr erklärte Varianz.  
**Wann sinnvoll?** Bei kontinuierlichen Scores, um das Modell mit einer Mittelwert-Baseline zu vergleichen.  
**Stolperfallen:** Kann negativ sein, wenn das Modell schlechter als "immer Mittelwert vorhersagen" ist. **Nicht widersprüchlich zu brauchbarem Spearman:** R² misst absolute Werte, Spearman misst Rangfolge. Ein Modell kann gute Rangkorrelation haben (Spearman ρ > 0.4), aber schlechte absolute Kalibrierung (R² < 0).

**Beispiel:** R² = 0.04 bedeutet, dass das Modell 4% der Varianz erklärt (sehr wenig). R² = -2.77 bedeutet, dass das Modell deutlich schlechter als eine Mittelwert-Baseline ist, aber dies ist nicht widersprüchlich zu Spearman ρ = 0.40 (gute Rangkorrelation).

---

### Precision

**Was misst es?** Den Anteil korrekter positiver Vorhersagen (TP / (TP + FP)).  
**Skala:** [0, 1], wobei 1 = alle positiven Vorhersagen sind korrekt, 0 = keine positiven Vorhersagen sind korrekt.  
**Höher = besser?** Ja, höhere Werte bedeuten weniger False Positives.  
**Wann sinnvoll?** Bei binärer Klassifikation, wenn False Positives teuer sind (z.B. "falscher Alarm").  
**Stolperfallen:** Ignoriert False Negatives (kann hoch sein, auch wenn viele Fälle übersehen werden).

**Beispiel:** Precision = 0.79 bedeutet, dass 79% der als "fehlerhaft" klassifizierten Summaries tatsächlich fehlerhaft sind.

---

### Recall

**Was misst es?** Den Anteil gefundener positiver Fälle (TP / (TP + FN)).  
**Skala:** [0, 1], wobei 1 = alle positiven Fälle wurden gefunden, 0 = keine positiven Fälle wurden gefunden.  
**Höher = besser?** Ja, höhere Werte bedeuten weniger False Negatives.  
**Wann sinnvoll?** Bei binärer Klassifikation, wenn False Negatives teuer sind (z.B. "übersehener Fehler").  
**Stolperfallen:** Ignoriert False Positives (kann hoch sein, auch wenn viele falsche Alarme gegeben werden).

**Beispiel:** Recall = 0.80 bedeutet, dass 80% der tatsächlich fehlerhaften Summaries gefunden wurden.

---

### F1

**Was misst es?** Das harmonische Mittel aus Precision und Recall (2 × Precision × Recall / (Precision + Recall)).  
**Skala:** [0, 1], wobei 1 = perfekte Balance zwischen Precision und Recall, 0 = schlecht.  
**Höher = besser?** Ja, höhere Werte bedeuten bessere Balance zwischen Precision und Recall.  
**Wann sinnvoll?** Bei binärer Klassifikation, wenn eine Balance zwischen Precision und Recall wichtig ist.  
**Stolperfallen:** Kann bei Imbalance irreführend sein (z.B. wenn 90% der Fälle negativ sind, kann F1 hoch sein, auch wenn der Agent schlecht ist).

**Beispiel:** F1 = 0.79 bedeutet, dass der Agent eine gute Balance zwischen Precision (0.79) und Recall (0.80) hat.

---

### Balanced Accuracy

**Was misst es?** Den Mittelwert aus Sensitivity (Recall) und Specificity ((TN / (TN + FP))).  
**Skala:** [0, 1], wobei 1 = perfekt, 0.5 = Zufall.  
**Höher = besser?** Ja, höhere Werte bedeuten bessere Performance auf beiden Klassen.  
**Wann sinnvoll?** Bei binärer Klassifikation bei Imbalance, wenn beide Klassen wichtig sind.  
**Stolperfallen:** Weniger bekannt als F1, aber robuster bei Imbalance.

**Beispiel:** Balanced Accuracy = 0.72 bedeutet, dass der Agent auf beiden Klassen (fehlerhaft/korrekt) durchschnittlich 72% korrekt ist.

---

### MCC (Matthews Correlation Coefficient)

**Was misst es?** Die Korrelation zwischen Vorhersage und Ground Truth unter Berücksichtigung aller 4 Klassen (TP, FP, TN, FN).  
**Skala:** [-1, 1], wobei 1 = perfekt, 0 = Zufall, -1 = perfekt falsch.  
**Höher = besser?** Ja, höhere Werte bedeuten stärkere Korrelation.  
**Wann sinnvoll?** Bei binärer Klassifikation bei Imbalance, wenn alle 4 Klassen wichtig sind.  
**Stolperfallen:** Weniger bekannt als F1, aber robuster bei Imbalance.

**Beispiel:** MCC = 0.45 bedeutet, dass der Agent eine moderate Korrelation zwischen Vorhersage und Ground Truth hat.

---

### AUROC (Area Under ROC Curve)

**Was misst es?** Die Fläche unter der ROC-Kurve (Receiver Operating Characteristic), die die Trennfähigkeit des Modells misst (unabhängig vom Threshold).  
**Skala:** [0, 1], wobei 1 = perfekte Trennung, 0.5 = Zufall, 0 = perfekt falsch.  
**Höher = besser?** Ja, höhere Werte bedeuten bessere Trennfähigkeit.  
**Wann sinnvoll?** Bei binärer Klassifikation, wenn der Threshold nicht festgelegt ist oder variiert werden soll.  
**Stolperfallen:** Misst Trennfähigkeit, nicht absolute Performance (z.B. AUROC = 0.89 bedeutet, dass das Modell gut trennt, aber nicht, dass 89% korrekt klassifiziert werden).

**Beispiel:** AUROC = 0.89 bedeutet, dass der Agent zuverlässig zwischen fehlerhaften und korrekten Summaries trennt (0.5 = Zufall, 1.0 = perfekt).

---

### Bootstrap 95% CI (Konfidenzintervall)

**Was misst es?** Ein Konfidenzintervall, das die Unsicherheit einer Metrik-Schätzung abbildet (via Resampling mit Ersetzung).  
**Skala:** [lower, upper], wobei 95% der Resamples innerhalb des Intervalls liegen.  
**Höher = besser?** Nicht direkt, aber engere Intervalle bedeuten weniger Unsicherheit.  
**Wann sinnvoll?** Bei allen Metriken, um die Variabilität der Schätzung zu quantifizieren.  
**Stolperfallen:** Zeigt Variabilität, nicht systematische Bias (z.B. ein CI kann eng sein, auch wenn das Modell systematisch zu hoch vorhersagt).

**Beispiel:** Spearman ρ = 0.41 [0.27, 0.53] bedeutet, dass der wahre Wert mit 95% Wahrscheinlichkeit zwischen 0.27 und 0.53 liegt.

---

### ROUGE-L (Baseline-Metrik)

**Was misst es?** Die Ähnlichkeit zwischen Summary und Referenzsummary via Longest Common Subsequence (LCS).  
**Skala:** [0, 1], wobei 1 = identisch, 0 = keine Ähnlichkeit.  
**Höher = besser?** Ja, höhere Werte bedeuten größere Ähnlichkeit zur Referenz.  
**Wann sinnvoll?** Als Baseline, um die Ähnlichkeit zur Referenz zu messen (nicht die Qualität).  
**Stolperfallen:** Misst Ähnlichkeit, nicht Qualität (z.B. eine schlechte Summary kann hohe ROUGE-L haben, wenn sie ähnlich zur Referenz ist).

**Beispiel:** ROUGE-L = 0.20 bedeutet, dass die Summary 20% Ähnlichkeit zur Referenz hat (niedrig, daher ungeeignet als Proxy für Faktentreue).

---

### BERTScore (Baseline-Metrik)

**Was misst es?** Die semantische Ähnlichkeit zwischen Summary und Referenzsummary via BERT-Embeddings.  
**Skala:** [0, 1], wobei 1 = identisch, 0 = keine Ähnlichkeit.  
**Höher = besser?** Ja, höhere Werte bedeuten größere semantische Ähnlichkeit zur Referenz.  
**Wann sinnvoll?** Als Baseline, um die semantische Ähnlichkeit zur Referenz zu messen (nicht die Qualität).  
**Stolperfallen:** Misst Ähnlichkeit, nicht Qualität. Warnings über "Roberta pooler" sind irrelevant für die Score-Nutzung (nur für Debugging).

**Beispiel:** BERTScore = 0.10 bedeutet, dass die Summary 10% semantische Ähnlichkeit zur Referenz hat (niedrig, daher ungeeignet als Proxy für Faktentreue).

---

**Details zu Evaluation:** Siehe `03_evaluation_results.md`.

