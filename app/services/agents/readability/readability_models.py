"""
Dieses Modul definiert die zentralen Datenmodelle für die Darstellung von
Lesbarkeitsproblemen (Readability-Issues) innerhalb des Readability-Agents.

Das ReadabilityIssue-Modell beschreibt einzelne, klar abgegrenzte Probleme
in der Summary, die den Lesefluss oder die Verständlichkeit beeinträchtigen,
z.B. überlange Sätze, starke Verschachtelungen oder übermäßige Interpunktion.

Die Modelle sind bewusst einfach gehalten und orientieren sich strukturell
am CoherenceIssue-Modell, um Konsistenz innerhalb der Agentenarchitektur
sicherzustellen. Sie enthalten ausschließlich textuelle Informationen
(z.B. einen kurzen Summary-Auszug und eine knappe Beschreibung des Problems)
und keine positions- oder UI-spezifischen Details.

Die Readability-Modelle dienen als internes Austauschformat zwischen
Readability-Verifier und Readability-Agent. Die Abbildung auf konkrete
IssueSpans sowie Persistenz- und Pipeline-Aspekte erfolgen außerhalb dieses
Moduls.

"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class ReadabilityIssue:
    type: Literal[
        "LONG_SENTENCE",
        "COMPLEX_NESTING",
        "PUNCTUATION_OVERLOAD",
        "HARD_TO_PARSE",
    ]
    severity: Literal["low", "medium", "high"]
    summary_span: str
    comment: str

    # Optional: Metrik-Infos aus dem Evaluator
    metric: str | None = None
    metric_value: float | None = None

    # Optional: Verbesserungsvorschlag
    hint: str | None = None
