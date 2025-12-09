# tests/test_factuality_agent_unit.py

from app.services.agents.factuality_agent import FactualityAgent
from app.llm.llm_client import LLMClient


class FakeLLMClient(LLMClient):
    def complete(self, prompt: str, **kwargs) -> str:
        # Tut so, als hätte das LLM geprüft und klar falschen Satz erkannt
        return """
        {
          "label": "incorrect",
          "confidence": 0.95,
          "explanation": "Im Quelltext steht Berlin, nicht Paris."
        }
        """


def test_factuality_agent_detects_obvious_error():
    llm = FakeLLMClient()
    agent = FactualityAgent(llm)

    article = "Lisa wohnt in Berlin."
    summary = "Lisa wohnt in Paris."

    result = agent.run(article, summary, meta={})

    assert result.score < 1.0
    assert result.details["num_errors"] >= 1
