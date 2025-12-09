from abc import ABC, abstractmethod
from typing import Any

class LLMClient(ABC):
    @abstractmethod
    def complete(self, prompt: str, **kwargs: Any) -> str:
        """Sendet Prompt an ein LLM und gibt nur den Text-Output zurück."""
        raise NotImplementedError
