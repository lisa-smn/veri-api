# Evaluationsmetriken – Kurztext für Thesis

**Zweck:** 12-15 Sätze für Bachelorarbeit Kapitel "Evaluationsmetriken"

---

## Text

Für die Evaluation der Factuality-Dimension verwenden wir binäre Klassifikationsmetriken, da die Ground-Truth-Labels als `has_error` (boolean) vorliegen. Aus der Confusion Matrix (TP, FP, TN, FN) berechnen wir Accuracy, Precision, Recall, F1-Score und Specificity. Zusätzlich verwenden wir Balanced Accuracy (Mittelwert aus Recall und Specificity), um der unbalancierten Klassenverteilung Rechnung zu tragen, sowie den Matthews Correlation Coefficient (MCC) als robuste Metrik für binäre Klassifikation. Für die Bewertung der Ranking-Qualität berechnen wir die Area Under ROC Curve (AUROC) basierend auf den kontinuierlichen Agent-Scores. Alle Metriken werden pro Run aggregiert, indem die Confusion Matrix über alle Beispiele summiert wird.

Für die Evaluation der Coherence- und Readability-Dimensionen verwenden wir Korrelations- und Regressionsmetriken, da die Ground-Truth-Werte kontinuierliche Ratings (Skala 1-5) sind. Wir berechnen Pearson's Korrelationskoeffizient (r) für lineare Zusammenhänge und Spearman's Rangkorrelationskoeffizient (ρ) für monotone Beziehungen. Als Fehlermetriken verwenden wir Mean Absolute Error (MAE) und Root Mean Squared Error (RMSE). Zusätzlich berechnen wir R² (Coefficient of Determination) zur Bewertung der Varianzaufklärung. Die Ground-Truth-Werte werden von der ursprünglichen Skala [1, 5] auf [0, 1] normalisiert (Formel: `gt_norm = (gt_raw - 1) / 4`), um sie mit den Agent-Scores (bereits in [0, 1]) vergleichbar zu machen.

Als Baselines verwenden wir ROUGE-L (Longest Common Subsequence F1-Score) und BERTScore (Contextual Embedding Similarity F1-Score), die beide die Ähnlichkeit zwischen Summary und Referenz-Text messen. Diese Baseline-Scores werden pro Beispiel berechnet und dann mit den Ground-Truth-Ratings korreliert (Pearson r, Spearman ρ), um ihre Eignung als Proxy-Metriken für Coherence/Readability zu bewerten. Beide Metriken liefern Scores im Bereich [0, 1], wobei höhere Werte größere Ähnlichkeit zur Referenz anzeigen.

Um die Unsicherheit der Metriken zu quantifizieren, verwenden wir Bootstrap-Resampling mit 2000 Resamples und einem Konfidenzniveau von 95%. Die Konfidenzintervalle werden mittels Percentile-Methode berechnet (2.5th und 97.5th Percentile der Resample-Verteilung). Für Regression-Metriken (Pearson, Spearman, MAE, RMSE) wird jedes Resample durch zufälliges Ziehen mit Replacement erzeugt, die Metrik-Funktion darauf angewendet und die Verteilung der Resample-Metriken analysiert. Für binäre Klassifikationsmetriken (Accuracy, Precision, Recall, F1, Balanced Accuracy) wird analog vorgegangen, wobei zunächst die Confusion Matrix pro Resample berechnet wird, bevor die Metrik daraus abgeleitet wird.

---

## Wortanzahl

**Aktuell:** ~280 Wörter (ca. 15 Sätze)

**Anpassung:** Kann auf 12-15 Sätze gekürzt werden, falls nötig.
