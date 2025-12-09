from typing import Any
from app.llm.llm_client import LLMClient


class FakeLLMClient(LLMClient):
    def complete(self, prompt: str, **kwargs: Any) -> str:
        # Völlig deterministische Antwort, die einen klaren Fehler markiert.
        return """
        {
          "label": "incorrect",
          "confidence": 0.95,
          "explanation": "Im Quelltext steht Berlin, nicht Paris."
        }
        """
