from app.models.schemas import PipelineResult
from app.services.agents.factuality_agent import FactualityAgent
from app.services.agents.coherence_agent import CoherenceAgent
from app.services.agents.readability_agent import ReadabilityAgent


class VerificationPipeline:
    def __init__(self) -> None:
        self.factuality_agent = FactualityAgent()
        self.coherence_agent = CoherenceAgent()
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
