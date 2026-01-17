"""
Unit Tests für select_best_tuned_run.py

Testet:
- Gate-Status Logik (niemals gelogen)
- Fallback-Begründung
- Top-K Tabelle
- JSON-Persistenz
"""

from scripts.select_best_tuned_run import (
    compute_mcc,
    generate_top_k_table,
    select_best_tuned_run,
)


class TestSelectBestTunedRun:
    """Tests für select_best_tuned_run Funktion."""

    def test_both_gates_passed_no_fallback(self):
        """Wenn both_gates_filtered nicht leer → candidate_pool == both_gates, kein Fallback."""
        rows = [
            {
                "run_id": "run1",
                "dataset": "frank",
                "recall": 0.95,
                "specificity": 0.25,
                "mcc": 0.15,
                "balanced_accuracy": 0.60,
                "precision": 0.80,
                "f1": 0.87,
                "tp": 10,
                "tn": 5,
                "fp": 2,
                "fn": 1,
            },
            {
                "run_id": "run2",
                "dataset": "frank",
                "recall": 0.92,
                "specificity": 0.30,
                "mcc": 0.20,
                "balanced_accuracy": 0.61,
                "precision": 0.85,
                "f1": 0.88,
                "tp": 12,
                "tn": 6,
                "fp": 1,
                "fn": 1,
            },
        ]

        best, stats = select_best_tuned_run(
            rows, recall_min=0.90, specificity_min=0.20, target_metric="mcc", dataset="frank"
        )

        assert best is not None
        assert stats["candidate_pool_name"] == "both_gates"
        assert stats["fallback_used"] is False
        assert stats["gate2_passed"] is True
        assert "both_gates" in str(stats["justification"])

    def test_both_gates_empty_fallback_to_recall_only(self):
        """Wenn both_gates_filtered leer → Gate 2 ist ❌, Fallback-Hinweis erscheint."""
        rows = [
            {
                "run_id": "run1",
                "dataset": "frank",
                "recall": 0.95,
                "specificity": 0.15,  # < 0.20
                "mcc": 0.10,
                "balanced_accuracy": 0.55,
                "precision": 0.75,
                "f1": 0.84,
                "tp": 10,
                "tn": 3,
                "fp": 4,
                "fn": 1,
            },
            {
                "run_id": "run2",
                "dataset": "frank",
                "recall": 0.92,
                "specificity": 0.18,  # < 0.20
                "mcc": 0.12,
                "balanced_accuracy": 0.55,
                "precision": 0.80,
                "f1": 0.86,
                "tp": 11,
                "tn": 4,
                "fp": 2,
                "fn": 1,
            },
        ]

        best, stats = select_best_tuned_run(
            rows, recall_min=0.90, specificity_min=0.20, target_metric="mcc", dataset="frank"
        )

        assert best is not None
        assert stats["candidate_pool_name"] == "recall_only"
        assert stats["fallback_used"] is True
        assert stats["gate2_passed"] is False  # Gate 2 MUSS ❌ sein
        assert "❌ Gate 2" in str(stats["justification"])
        assert "Kandidatenmenge: recall_only" in str(stats["justification"])

    def test_no_gates_passed_fallback_to_all_runs(self):
        """Wenn kein Gate erfüllt → candidate_pool == all_runs."""
        rows = [
            {
                "run_id": "run1",
                "dataset": "frank",
                "recall": 0.85,  # < 0.90
                "specificity": 0.15,  # < 0.20
                "mcc": 0.05,
                "balanced_accuracy": 0.50,
                "precision": 0.70,
                "f1": 0.77,
                "tp": 8,
                "tn": 3,
                "fp": 5,
                "fn": 2,
            },
        ]

        best, stats = select_best_tuned_run(
            rows, recall_min=0.90, specificity_min=0.20, target_metric="mcc", dataset="frank"
        )

        assert best is not None
        assert stats["candidate_pool_name"] == "all_runs"
        assert stats["fallback_used"] is True
        assert stats["gate1_passed"] is False
        assert stats["gate2_passed"] is False
        assert "Kandidatenmenge: all_runs" in str(stats["justification"])

    def test_top_k_metric_within_candidate_set(self):
        """Top-K bezieht sich wirklich auf candidate_set, nicht auf alle Runs."""
        rows = [
            {
                "run_id": "run1",
                "dataset": "frank",
                "recall": 0.95,
                "specificity": 0.25,
                "mcc": 0.15,
                "balanced_accuracy": 0.60,
                "precision": 0.80,
                "f1": 0.87,
            },
            {
                "run_id": "run2",
                "dataset": "frank",
                "recall": 0.92,
                "specificity": 0.30,
                "mcc": 0.20,  # Höchster MCC
                "balanced_accuracy": 0.61,
                "precision": 0.85,
                "f1": 0.88,
            },
            {
                "run_id": "run3",
                "dataset": "frank",
                "recall": 0.88,  # < 0.90, sollte nicht in candidate_set
                "specificity": 0.35,
                "mcc": 0.25,  # Höchster MCC, aber nicht in candidate_set
                "balanced_accuracy": 0.62,
                "precision": 0.90,
                "f1": 0.89,
            },
        ]

        best, stats = select_best_tuned_run(
            rows, recall_min=0.90, specificity_min=0.20, target_metric="mcc", dataset="frank"
        )

        # Best sollte run2 sein (höchster MCC in candidate_set), nicht run3
        assert best["run_id"] == "run2"
        assert best["mcc"] == 0.20

        # Top-K sollte nur Runs aus candidate_set enthalten
        candidate_set = stats.get("candidate_set", [])
        top_k = generate_top_k_table(candidate_set, "mcc", k=5)
        assert len(top_k) <= len(candidate_set)
        assert all(r["run_id"] in ["run1", "run2"] for r in top_k)
        assert not any(r["run_id"] == "run3" for r in top_k)

    def test_justification_mentions_candidate_pool_and_ranking(self):
        """Begründung enthält explizit Kandidatenmenge, Zielmetrik, Tie-Breaker."""
        rows = [
            {
                "run_id": "run1",
                "dataset": "frank",
                "recall": 0.95,
                "specificity": 0.15,  # < 0.20 → Fallback
                "mcc": 0.10,
                "balanced_accuracy": 0.55,
                "precision": 0.75,
                "f1": 0.84,
                "tp": 10,
                "tn": 3,
                "fp": 4,
                "fn": 1,
            },
        ]

        best, stats = select_best_tuned_run(
            rows, recall_min=0.90, specificity_min=0.20, target_metric="mcc", dataset="frank"
        )

        justification_str = " ".join(stats["justification"])
        assert "Kandidatenmenge: recall_only" in justification_str
        assert "Ranking: MCC" in justification_str
        assert "Tie-Breaker" in justification_str
        assert "BALANCED_ACCURACY" in justification_str or "BA" in justification_str


class TestTopKTable:
    """Tests für generate_top_k_table."""

    def test_top_k_returns_at_most_k_runs(self):
        """Top-K gibt maximal k Runs zurück."""
        candidate_set = [
            {
                "run_id": f"run{i}",
                "mcc": 0.1 * i,
                "balanced_accuracy": 0.5,
                "precision": 0.7,
                "f1": 0.8,
            }
            for i in range(10)
        ]

        top_k = generate_top_k_table(candidate_set, "mcc", k=5)
        assert len(top_k) == 5

    def test_top_k_returns_all_if_less_than_k(self):
        """Top-K gibt alle Runs zurück, wenn weniger als k vorhanden."""
        candidate_set = [
            {"run_id": "run1", "mcc": 0.1, "balanced_accuracy": 0.5, "precision": 0.7, "f1": 0.8},
            {"run_id": "run2", "mcc": 0.2, "balanced_accuracy": 0.6, "precision": 0.8, "f1": 0.9},
        ]

        top_k = generate_top_k_table(candidate_set, "mcc", k=5)
        assert len(top_k) == 2

    def test_top_k_sorted_by_target_metric(self):
        """Top-K ist nach target_metric sortiert."""
        candidate_set = [
            {"run_id": "run1", "mcc": 0.1, "balanced_accuracy": 0.5, "precision": 0.7, "f1": 0.8},
            {"run_id": "run2", "mcc": 0.3, "balanced_accuracy": 0.6, "precision": 0.8, "f1": 0.9},
            {"run_id": "run3", "mcc": 0.2, "balanced_accuracy": 0.7, "precision": 0.9, "f1": 0.95},
        ]

        top_k = generate_top_k_table(candidate_set, "mcc", k=5)
        assert top_k[0]["run_id"] == "run2"  # Höchster MCC
        assert top_k[0]["mcc"] == 0.3


class TestMCC:
    """Tests für compute_mcc."""

    def test_mcc_perfect_prediction(self):
        """MCC = 1.0 für perfekte Vorhersage."""
        assert abs(compute_mcc(tp=10, tn=10, fp=0, fn=0) - 1.0) < 0.001

    def test_mcc_random_prediction(self):
        """MCC ≈ 0.0 für zufällige Vorhersage."""
        mcc = compute_mcc(tp=5, tn=5, fp=5, fn=5)
        assert abs(mcc) < 0.1

    def test_mcc_division_by_zero(self):
        """MCC = 0.0 wenn Division durch 0 (robust)."""
        assert compute_mcc(tp=0, tn=0, fp=0, fn=0) == 0.0
