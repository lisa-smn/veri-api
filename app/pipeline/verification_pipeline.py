import os

from app.models.pydantic import PipelineResult
from app.services.agents.factuality.factuality_agent import FactualityAgent
from app.services.agents.coherence.coherence_agent import CoherenceAgent
from app.services.agents.readability.readability_agent import ReadabilityAgent
from app.llm.openai_client import OpenAIClient
from app.llm.fake_client import FakeLLMClient

TEST_MODE = os.getenv("TEST_MODE") == "1"


class VerificationPipeline:
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        # LLM-Client einmal zentral instanziieren
        if TEST_MODE:
            self.llm_client = FakeLLMClient()
        else:
            self.llm_client = OpenAIClient(model_name=model_name)

        # Factuality: echter Agent mit Claim-Logik
        self.factuality_agent = FactualityAgent(self.llm_client)

        # Coherence: jetzt ebenfalls echter Agent (mit LLMCoherenceEvaluator intern)
        self.coherence_agent = CoherenceAgent(self.llm_client)

        # Readability: weiterhin Dummy (M8)
        self.readability_agent = ReadabilityAgent()

    def run(self, article: str, summary: str, meta: dict | None = None) -> PipelineResult:
        factuality = self.factuality_agent.run(article, summary, meta)
        coherence = self.coherence_agent.run(article, summary, meta)
        readability = self.readability_agent.run(article, summary, meta)

        overall = (factuality.score + coherence.score + readability.score) / 3.0

        return PipelineResult(
            factuality=factuality,
            coherence=coherence,
            readability=readability,
            overall_score=overall,
        )
