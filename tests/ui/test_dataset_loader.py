"""Tests für Dataset Loader."""

import json
from pathlib import Path
import tempfile

from ui import dataset_loader


class TestDatasetLoader:
    """Tests für Dataset-Loader-Funktionen."""

    def test_get_article_frank(self):
        """Test FRANK Article-Extraktion."""
        row = {
            "article": "Test article text",
            "summary": "Test summary",
            "has_error": True,
        }
        article = dataset_loader.get_article_frank(row)
        assert article == "Test article text"

    def test_get_summary_frank(self):
        """Test FRANK Summary-Extraktion."""
        row = {
            "article": "Test article text",
            "summary": "Test summary",
            "has_error": True,
        }
        summary = dataset_loader.get_summary_frank(row)
        assert summary == "Test summary"

    def test_get_example_id_frank(self):
        """Test FRANK Example-ID-Extraktion."""
        row = {
            "article": "Test article",
            "summary": "Test summary",
            "meta": {
                "hash": "abc123",
                "model_name": "BERT",
            },
        }
        example_id = dataset_loader.get_example_id_frank(row, 0)
        assert example_id == "abc123_BERT"

    def test_get_article_sumeval(self):
        """Test SummEval Article-Extraktion."""
        row = {
            "article": "Test article",
            "summary": "Test summary",
        }
        article = dataset_loader.get_article_sumeval(row)
        assert article == "Test article"

    def test_get_summary_sumeval(self):
        """Test SummEval Summary-Extraktion."""
        row = {
            "article": "Test article",
            "summary": "Test summary",
        }
        summary = dataset_loader.get_summary_sumeval(row)
        assert summary == "Test summary"

    def test_load_random_example_frank_format(self):
        """Test load_random_example mit FRANK-Format."""
        # Erstelle temporäre JSONL-Datei
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            example = {
                "article": "Test article content",
                "summary": "Test summary content",
                "has_error": True,
                "meta": {
                    "hash": "test123",
                    "model_name": "TEST",
                },
            }
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            temp_path = Path(f.name)

        try:
            # Temporärer Pfad für FRANK
            original_path = dataset_loader.DATASET_PATHS.get("FRANK")
            dataset_loader.DATASET_PATHS["FRANK"] = temp_path

            article, summary, example_id, metadata = dataset_loader.load_random_example(
                "FRANK", seed=42
            )

            assert article == "Test article content"
            assert summary == "Test summary content"
            assert example_id is not None
            assert metadata is not None
            assert metadata["dataset"] == "FRANK"
            assert metadata["has_error"] is True
        finally:
            # Restore
            if original_path:
                dataset_loader.DATASET_PATHS["FRANK"] = original_path
            temp_path.unlink()

    def test_load_random_example_empty_file(self):
        """Test load_random_example mit leerer Datei."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            # Leere Datei
            temp_path = Path(f.name)

        try:
            original_path = dataset_loader.DATASET_PATHS.get("FRANK")
            dataset_loader.DATASET_PATHS["FRANK"] = temp_path

            article, summary, example_id, metadata = dataset_loader.load_random_example(
                "FRANK", seed=42
            )

            assert article is None
            assert summary is None
            assert example_id is None
            assert metadata is None
        finally:
            if original_path:
                dataset_loader.DATASET_PATHS["FRANK"] = original_path
            temp_path.unlink()

    def test_load_random_example_missing_dataset(self):
        """Test load_random_example mit nicht existierendem Dataset."""
        article, summary, example_id, metadata = dataset_loader.load_random_example(
            "NonExistentDataset", seed=42
        )

        assert article is None
        assert summary is None
        assert example_id is None
        assert metadata is None

    def test_get_dataset_path(self):
        """Test get_dataset_path."""
        # Prüfe ob FRANK-Pfad existiert (kann fehlen in Tests)
        path = dataset_loader.get_dataset_path("FRANK")
        # Kann None sein wenn Datei nicht existiert, das ist OK
        assert path is None or path.exists()

    def test_get_dataset_info(self):
        """Test get_dataset_info."""
        info = dataset_loader.get_dataset_info("FRANK")
        # Kann error enthalten wenn Datei nicht existiert
        assert isinstance(info, dict)
        assert "dataset" in info or "error" in info
