"""API Client für veri-api Backend."""

import os
from typing import Any

import requests

API_BASE_URL = os.getenv("VERI_API_BASE_URL", "http://localhost:8000")
TIMEOUT = 120  # Erhöht für Verify-Requests (können länger dauern mit Explainability/Persist)

# Fallback-Listen für Datasets und Models
DEFAULT_DATASETS = ["FRANK", "SummEval", "SummaC", "SummaCoz", "FineSumFact"]
DEFAULT_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]


def health_check() -> dict[str, Any]:
    """Prüft ob API erreichbar ist."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return {"status": "ok", "available": True}
        return {"status": "error", "available": False, "message": f"Status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "available": False, "message": str(e)}


def verify(
    article_text: str,
    summary_text: str,
    dataset: str | None = None,
    llm_model: str | None = None,
    meta: dict[str, str] | None = None,
    run_llm_judge: bool = False,
    judge_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Sendet Verify-Request an API.

    Args:
        article_text: Artikel-Text
        summary_text: Summary-Text
        dataset: Optional dataset name
        llm_model: Optional LLM model
        meta: Optional metadata dict
        run_llm_judge: Ob LLM-as-a-Judge ausgeführt werden soll
        judge_params: Optional dict mit judge_mode, judge_n, judge_temperature, judge_aggregation

    Returns:
        dict mit Response-Daten oder Error-Info
    """
    url = f"{API_BASE_URL}/verify"
    payload = {
        "article_text": article_text,
        "summary_text": summary_text,
    }
    if dataset:
        payload["dataset"] = dataset
    if llm_model:
        payload["llm_model"] = llm_model
    if meta:
        payload["meta"] = meta

    # Judge-Parameter hinzufügen
    if run_llm_judge:
        payload["run_llm_judge"] = True
        if judge_params:
            payload.update(judge_params)

    try:
        response = requests.post(url, json=payload, timeout=TIMEOUT)
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.exceptions.HTTPError:
        return {
            "success": False,
            "error": f"HTTP {response.status_code}: {response.text}",
            "status_code": response.status_code,
        }
    except requests.exceptions.RequestException as e:
        return {"success": False, "error": str(e)}


def db_check() -> dict[str, Any]:
    """Prüft ob DB über API erreichbar ist."""
    try:
        response = requests.get(f"{API_BASE_URL}/db-check", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return {"available": data.get("db_ok", False)}
        return {"available": False, "message": f"Status {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"available": False, "message": str(e)}


def get_datasets() -> list[str]:
    """
    Holt verfügbare Datasets von der API oder gibt Fallback-Liste zurück.

    Returns:
        Liste von Dataset-Namen
    """
    try:
        # Versuche /meta/capabilities oder /datasets
        for endpoint in ["/meta/capabilities", "/datasets", "/meta/datasets"]:
            try:
                response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    # Verschiedene mögliche Response-Formate
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict):
                        datasets = data.get("datasets") or data.get("data", [])
                        if isinstance(datasets, list):
                            return datasets
            except requests.exceptions.RequestException:
                continue
    except Exception:
        pass

    # Fallback: Standard-Liste
    return DEFAULT_DATASETS.copy()


def get_models() -> list[str]:
    """
    Holt verfügbare LLM Models von der API oder gibt Fallback-Liste zurück.

    Returns:
        Liste von Model-Namen
    """
    try:
        # Versuche /meta/capabilities oder /models
        for endpoint in ["/meta/capabilities", "/models", "/meta/models"]:
            try:
                response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    # Verschiedene mögliche Response-Formate
                    if isinstance(data, list):
                        return data
                    if isinstance(data, dict):
                        models = (
                            data.get("llm_models") or data.get("models") or data.get("data", [])
                        )
                        if isinstance(models, list):
                            return models
            except requests.exceptions.RequestException:
                continue
    except Exception:
        pass

    # Fallback: Standard-Liste
    return DEFAULT_MODELS.copy()
