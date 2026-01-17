from dataclasses import dataclass


@dataclass
class SentenceInfo:
    index: int
    text: str


def split_summary_into_sentences(summary_text: str) -> list[SentenceInfo]:
    """
    Sehr einfacher, naiver Satzsplitter für den Coherence-Kontext.
    Reicht für M7 völlig aus. Später ggf. durch spaCy o.Ä. ersetzen.
    """
    raw_sentences = [s.strip() for s in summary_text.split(".") if s.strip()]
    return [SentenceInfo(index=i, text=s) for i, s in enumerate(raw_sentences)]
