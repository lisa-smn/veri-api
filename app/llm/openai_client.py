from typing import Any

from openai import OpenAI
from app.llm.llm_client import LLMClient


class OpenAIClient(LLMClient):
    def __init__(self, model_name: str):
        self.model_name = model_name
        # Liest OPENAI_API_KEY automatisch aus der Umgebung
        self.client = OpenAI()

    def complete(self, prompt: str, **kwargs: Any) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
        )

        return response.choices[0].message.content or ""
