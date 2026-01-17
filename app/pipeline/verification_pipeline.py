"""
- Nimmt einen Artikel und eine dazugehörige Summary.
- Lässt drei spezialisierte Prüfer (Agenten) darüber laufen:
  1) Factuality: Stimmen die Fakten mit dem Artikel überein?
  2) Coherence: Ist die Summary logisch und widerspruchsfrei?
  3) Readability: Ist der Text gut lesbar und verständlich?
- Berechnet danach einen Gesamt-Score (Durchschnitt der drei Scores).
- Gibt alles gebündelt als PipelineResult zurück.


Man bekommt so nicht nur eine “Zahl”, sondern sieht getrennt,
ob Probleme eher Fakten, Logik oder Lesbarkeit betreffen.

"""

import os

from app.llm.fake_client import FakeLLMClient
from app.llm.openai_client import OpenAIClient
from app.models.pydantic import PipelineResult
from app.services.agents.coherence.coherence_agent import CoherenceAgent
from app.services.agents.factuality.factuality_agent import FactualityAgent
from app.services.agents.readability.readability_agent import ReadabilityAgent
from app.services.explainability.explainability_service import ExplainabilityService  # <- neu

TEST_MODE = os.getenv("TEST_MODE") == "1"


class VerificationPipeline:
    def __init__(self, model_name: str = "gpt-4o-mini") -> None:
        if TEST_MODE:
            self.llm_client = FakeLLMClient()
        else:
            self.llm_client = OpenAIClient(model_name=model_name)

        self.factuality_agent = FactualityAgent(self.llm_client)
        self.coherence_agent = CoherenceAgent(self.llm_client)
        self.readability_agent = ReadabilityAgent(self.llm_client)

        self.explainability_service = ExplainabilityService()  # <- neu

    def run(self, article: str, summary: str, meta: dict | None = None) -> PipelineResult:
        factuality = self.factuality_agent.run(article, summary, meta)
        coherence = self.coherence_agent.run(article, summary, meta)
        readability = self.readability_agent.run(article, summary, meta)

        overall = (factuality.score + coherence.score + readability.score) / 3.0

        # PipelineResult zuerst bauen...
        result = PipelineResult(
            factuality=factuality,
            coherence=coherence,
            readability=readability,
            overall_score=overall,
            explainability=None,  # <- neu (Feld muss existieren)
        )

        # ...dann Explainability deterministisch daraus generieren
        result.explainability = self.explainability_service.build(result, summary_text=summary)

        return result
