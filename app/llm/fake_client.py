from typing import Any

from app.llm.llm_client import LLMClient


class FakeLLMClient(LLMClient):
    def __init__(self):
        super().__init__()
        self._response_queue: list[str] = []
        self._default_response = """
        {
          "label": "incorrect",
          "confidence": 0.95,
          "explanation": "Im Quelltext steht Berlin, nicht Paris."
        }
        """

    def set_response(self, response: str) -> None:
        """Setzt eine einzelne Response für den nächsten Call."""
        self._response_queue = [response]

    def set_responses(self, responses: list[str]) -> None:
        """Setzt eine Queue von Responses für mehrere Calls."""
        self._response_queue = list(responses)

    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Gibt die nächste Response aus der Queue zurück, oder die Default-Response."""
        if self._response_queue:
            return self._response_queue.pop(0)
        return self._default_response
