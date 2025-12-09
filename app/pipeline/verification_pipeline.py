import os

from app.models.pydantic import PipelineResult
from app.services.agents.factuality_agent import FactualityAgent
from app.services.agents.coherence_agent import CoherenceAgent
from app.services.agents.readability_agent import ReadabilityAgent
from app.llm.openai_client import OpenAIClient
from app.llm.fake_client import FakeLLMClient

TEST_MODE = os.getenv("TEST_MODE") == "1"

class VerificationPipeline:
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        if TEST_MODE:
            self.llm_client = FakeLLMClient()
        else:
            # LLM-Client einmal instanziieren
            self.llm_client = OpenAIClient(model_name=model_name)

        # Echter Agent
        self.factuality_agent = FactualityAgent(self.llm_client)

        # Platzhalter (noch Dummy)
        self.coherence_agent = CoherenceAgent()
        self.readability_agent = ReadabilityAgent()

    def run(self, article: str, summary: str, meta: dict | None = None) -> PipelineResult:
        # Factuality → jetzt echter Agent
        factuality = self.factuality_agent.run(article, summary, meta)

        # Coherence / Readability → noch Dummy, aber korrektes Format
        coherence = self.coherence_agent.run(article, summary, meta)
        readability = self.readability_agent.run(article, summary, meta)

        # Overall Score aktuell einfach der Mittelwert
        overall = (factuality.score + coherence.score + readability.score) / 3.0

        return PipelineResult(
            factuality=factuality,
            coherence=coherence,
            readability=readability,
            overall_score=overall,
        )
