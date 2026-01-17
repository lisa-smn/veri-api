from app.models.pydantic import AgentResult, IssueSpan
from app.services.explainability.explainability_service import ExplainabilityService


def _agent(
    name: str,
    score: float = 1.0,
    label: str = "ok",
    confidence: float = 1.0,
    explanation: str = "",
    issue_spans=None,
    details=None,
) -> AgentResult:
    return AgentResult(
        name=name,
        score=score,
        label=label,
        confidence=confidence,
        explanation=explanation,
        issue_spans=issue_spans or [],
        details=details if details is not None else {},
    )


class _FakePipelineResult:
    """
    Minimaler Stub, der sowohl Attribute als auch Dict-Zugriff abdeckt,
    weil wir nicht wissen wollen, wie dein Service intern drauf zugreift.
    """

    def __init__(self, factuality: AgentResult, coherence: AgentResult, readability: AgentResult):
        self.factuality = factuality
        self.coherence = coherence
        self.readability = readability

        # häufige Patterns in Projekten
        self.by_dimension = {
            "factuality": factuality,
            "coherence": coherence,
            "readability": readability,
        }
        self.results = self.by_dimension
        self.agent_results = self.by_dimension

    def __getitem__(self, key: str) -> AgentResult:
        return self.by_dimension[key]


def test_explainability_clusters_overlapping_spans_into_one_finding():
    svc = ExplainabilityService()

    readability = _agent(
        name="ReadabilityAgent",
        score=0.3,
        label="issues",
        confidence=0.9,
        issue_spans=[
            # sehr starke Überlappung -> sollte in praktisch jedem Clustering zu 1 Finding werden
            IssueSpan(start_char=10, end_char=50, message="verschachtelt", severity="medium"),
            IssueSpan(start_char=12, end_char=48, message="unklar", severity="medium"),
        ],
    )

    pipeline = _FakePipelineResult(
        factuality=_agent("FactualityAgent"),
        coherence=_agent("CoherenceAgent"),
        readability=readability,
    )

    report = svc.build(pipeline, summary_text="x" * 200)

    readability_findings = [f for f in report.findings if f.dimension == "readability"]
    assert len(readability_findings) == 1


def test_explainability_number_issue_type_becomes_high():
    svc = ExplainabilityService()

    factuality = _agent(
        name="FactualityAgent",
        score=0.2,
        label="issues",
        confidence=0.9,
        issue_spans=[
            IssueSpan(
                start_char=0,
                end_char=5,
                message="Zahl falsch",
                severity="low",
                issue_type="NUMBER",
            )
        ],
    )

    pipeline = _FakePipelineResult(
        factuality=factuality,
        coherence=_agent("CoherenceAgent"),
        readability=_agent("ReadabilityAgent"),
    )

    report = svc.build(pipeline, summary_text="12345 blah")

    factuality_findings = [f for f in report.findings if f.dimension == "factuality"]
    assert len(factuality_findings) >= 1
    assert factuality_findings[0].severity == "high"
