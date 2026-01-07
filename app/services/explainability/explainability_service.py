"""
Baut aus den Agenten-Ergebnissen einen nachvollziehbaren Explainability-Report.

- Agenten liefern Roh-Ergebnisse (Scores, issue_spans, details). Das ist für Menschen
  schwer zu überblicken und oft uneinheitlich.
- Dieser Service macht daraus einen stabilen, testbaren Report (ExplainabilityResult),
  der zeigt: Was ist das Problem? Wo im Text ist es? Wie wichtig ist es? Was tun?

1) Normalisieren:
   - Agent-Outputs werden in ein gemeinsames Finding-Format übersetzt
     (Severity vereinheitlicht, optionaler Span).
   - Primäre Quelle: AgentResult.issue_spans (kanonisch in deinem Projekt).
   - Factuality nutzt zusätzlich details (z.B. NUMBER/DATE -> high), falls vorhanden.
2) Deduplizieren & Clustern:
   - Überlappende Findings werden zusammengeführt, damit nicht 20 Einträge
     denselben Satz meinen.
3) Ranking:
   - Findings werden nach einer einfachen, erklärbaren Formel priorisiert:
     Severity * Dimension-Gewichtung * (log(Span-Länge)).
4) Top-Spans + Stats:
   - Wichtigste Textstellen (Top-K) und Basis-Statistiken (Coverage, Counts).
5) Executive Summary:
   - Kurze Zusammenfassung, regelbasiert aus Findings abgeleitet.

Ergebnis:
- Nachvollziehbare, reproduzierbare Explainability, gut für Swagger-Demo,
  Tests und spätere Speicherung (Postgres/Neo4j).
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from app.services.explainability.explainability_models import (
    Dimension,
    EvidenceItem,
    ExplainabilityResult,
    ExplainabilityStats,
    Finding,
    Severity,
    Span,
    TopSpan,
)


@dataclass(frozen=True)
class RankWeights:
    severity: Dict[Severity, float]
    dimension: Dict[Dimension, float]


DEFAULT_WEIGHTS = RankWeights(
    severity={"low": 1.0, "medium": 2.0, "high": 3.0},
    dimension={
        Dimension.factuality: 1.2,
        Dimension.coherence: 1.0,
        Dimension.readability: 0.8,
    },
)

FACTUALITY_HIGH_TYPES = {"NUMBER", "DATE"}
FACTUALITY_MEDIUM_TYPES = {"ENTITY", "NAME", "LOCATION", "ORGANIZATION"}

RECOMMENDATIONS: Dict[Tuple[Dimension, Optional[str]], str] = {
    (Dimension.factuality, "NUMBER"): "Zahlen/Einheiten mit dem Artikel abgleichen und ggf. korrigieren.",
    (Dimension.factuality, "DATE"): "Datum/Zeitraum mit dem Artikel abgleichen und ggf. korrigieren.",
    (Dimension.factuality, None): "Kernaussagen mit dem Artikel abgleichen; falsche Claims entfernen oder präzisieren.",
    (Dimension.coherence, None): "Satzfolge prüfen: klare Bezüge, keine Sprünge, keine internen Widersprüche.",
    (Dimension.readability, None): "Lange/verschachtelte Sätze teilen und unnötige Nebensätze reduzieren.",
}


# -----------------------------
# Helpers (robust + deterministic)
# -----------------------------

def _get_attr_or_key(obj: Any, key: str, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    # pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    # pydantic v1
    if hasattr(obj, "dict"):
        return obj.dict()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def _safe_get(d: Any, *keys: str, default=None):
    """Safe dict get (ignores non-dicts)."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default


def _get_issue_spans(agent_result: Any) -> List[Dict[str, Any]]:
    """
    Liest AgentResult.issue_spans (Pydantic-Objekte oder dicts) in ein einheitliches dict-Format.
    Erwartete Keys (aus deinem IssueSpan): start_char, end_char, message, severity, optional issue_type.
    """
    spans = _get_attr_or_key(agent_result, "issue_spans", None)
    if not spans:
        return []
    out: List[Dict[str, Any]] = []
    for s in spans:
        out.append(_as_dict(s))
    return out


def _span_from_issue_span(item: Dict[str, Any], summary_text: str) -> Optional[Span]:
    start = item.get("start_char")
    end = item.get("end_char")
    if start is None or end is None:
        return None
    try:
        start_i = max(0, int(start))
        end_i = max(0, int(end))
    except Exception:
        return None
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    start_i = min(start_i, len(summary_text))
    end_i = min(end_i, len(summary_text))
    text = summary_text[start_i:end_i] if end_i > start_i else ""
    return Span(start_char=start_i, end_char=end_i, text=text or None)


def _span_from_item(item: Dict[str, Any], summary_text: str) -> Optional[Span]:
    # akzeptiert mehrere Shapes:
    # - item["span"] = {"start_char":..., "end_char":...}
    # - item["span"] = {"start":..., "end":...}
    # - item["start_char"]/["end_char"]
    raw_span = _safe_get(item, "span", default=None)
    if isinstance(raw_span, dict):
        start = _safe_get(raw_span, "start_char", "start", default=None)
        end = _safe_get(raw_span, "end_char", "end", default=None)
    else:
        start = _safe_get(item, "start_char", "start", default=None)
        end = _safe_get(item, "end_char", "end", default=None)

    if start is None or end is None:
        return None

    try:
        start_i = max(0, int(start))
        end_i = max(0, int(end))
    except Exception:
        return None

    if end_i < start_i:
        start_i, end_i = end_i, start_i

    start_i = min(start_i, len(summary_text))
    end_i = min(end_i, len(summary_text))

    text = summary_text[start_i:end_i] if end_i > start_i else ""
    return Span(start_char=start_i, end_char=end_i, text=text or None)


def _severity_from_raw(dimension: Dimension, issue_type: Optional[str], raw: Any) -> Severity:
    # factuality: issue_type dominiert (NUMBER/DATE high, etc.)
    if dimension == Dimension.factuality:
        if issue_type in FACTUALITY_HIGH_TYPES:
            return "high"
        if issue_type in FACTUALITY_MEDIUM_TYPES:
            return "medium"
        # falls kein issue_type vorhanden ist, aber raw severity als String kommt, übernehmen
        if isinstance(raw, str):
            s = raw.lower().strip()
            if s in {"low", "medium", "high"}:
                return s  # type: ignore[return-value]
        return "medium"

    # coherence/readability: raw severity übernehmen wenn möglich
    if isinstance(raw, str):
        s = raw.lower().strip()
        if s in {"low", "medium", "high"}:
            return s  # type: ignore[return-value]
    if isinstance(raw, (int, float)):
        if raw >= 0.75:
            return "high"
        if raw >= 0.4:
            return "medium"
        return "low"

    return "medium"


def _recommendation(dimension: Dimension, issue_type: Optional[str]) -> str:
    return RECOMMENDATIONS.get((dimension, issue_type)) or RECOMMENDATIONS.get((dimension, None)) or "Überarbeiten."


def _stable_id(*parts: str) -> str:
    h = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()
    return f"f_{h[:12]}"


def _overlap(a: Span, b: Span) -> bool:
    return not (a.end_char <= b.start_char or b.end_char <= a.start_char)


def _evidence_hash_data(data: Dict[str, Any]) -> str:
    # deterministic-ish hashing even if values are nested/unhashable
    return hashlib.sha1(repr(sorted(data.items(), key=lambda x: x[0])).encode("utf-8")).hexdigest()


def _merge_evidence(evs: Iterable[EvidenceItem]) -> List[EvidenceItem]:
    out: List[EvidenceItem] = []
    seen = set()
    for e in evs:
        data_h = _evidence_hash_data(e.data or {})
        key = (e.kind, e.source, e.quote, data_h)
        if key in seen:
            continue
        seen.add(key)
        out.append(e)
    return out


def _model_copy(obj: Any, deep: bool = True) -> Any:
    # pydantic v2
    if hasattr(obj, "model_copy"):
        return obj.model_copy(deep=deep)
    # pydantic v1
    if hasattr(obj, "copy"):
        return obj.copy(deep=deep)
    return obj


# -----------------------------
# Explainability Service
# -----------------------------

class ExplainabilityService:
    VERSION = "m9_v1"

    def __init__(self, weights: RankWeights = DEFAULT_WEIGHTS, top_k: int = 5):
        self.weights = weights
        self.top_k = top_k

    def build(self, pipeline_result: Any, summary_text: str) -> ExplainabilityResult:
        # pipeline_result expected to contain .factuality/.coherence/.readability AgentResults (or dict keys)
        factuality = _get_attr_or_key(pipeline_result, "factuality", None)
        coherence = _get_attr_or_key(pipeline_result, "coherence", None)
        readability = _get_attr_or_key(pipeline_result, "readability", None)

        findings: List[Finding] = []
        findings += self._normalize_factuality(factuality, summary_text)
        findings += self._normalize_generic(Dimension.coherence, coherence, summary_text)
        findings += self._normalize_generic(Dimension.readability, readability, summary_text)

        findings = self._dedupe_and_cluster(findings)
        ranked = self._rank(findings)

        top_spans = self._top_spans(ranked)

        by_dim: Dict[Dimension, List[Finding]] = {d: [] for d in Dimension}
        for f in ranked:
            by_dim[f.dimension].append(f)

        stats = self._stats(ranked, summary_text)
        summary = self._executive_summary(ranked, top_spans)

        return ExplainabilityResult(
            summary=summary,
            findings=ranked,
            by_dimension=by_dim,
            top_spans=top_spans,
            stats=stats,
            version=self.VERSION,
        )

    def _normalize_factuality(self, agent_result: Any, summary_text: str) -> List[Finding]:
        if not agent_result:
            return []

        out: List[Finding] = []

        # (A) bevorzugt: issue_spans (kanonisch)
        span_items = _get_issue_spans(agent_result)
        for idx, it in enumerate(span_items):
            msg = (it.get("message") or "Faktisches Problem erkannt.").strip()
            issue_type = it.get("issue_type")  # optional, falls du es am Span mitgibst
            raw_sev = it.get("severity")
            sev = _severity_from_raw(Dimension.factuality, issue_type, raw_sev)

            span = _span_from_issue_span(it, summary_text)
            rec = _recommendation(Dimension.factuality, issue_type)

            f_id = _stable_id(
                Dimension.factuality.value,
                sev,
                issue_type or "",
                str(span.start_char if span else -1),
                str(span.end_char if span else -1),
                msg,
            )

            out.append(
                Finding(
                    id=f_id,
                    dimension=Dimension.factuality,
                    severity=sev,
                    message=msg,
                    span=span,
                    evidence=[EvidenceItem(kind="raw", source="agent:factuality", data={"issue_span": it})],
                    recommendation=rec,
                    source={
                        "agent": "factuality",
                        "source_list": "issue_spans",
                        "issue_type": issue_type,
                        "item_index": idx,
                    },
                )
            )

        # (B) zusätzlich: details-basiert (liefert oft issue_type wie NUMBER/DATE, Evidence Quotes etc.)
        details = _get_attr_or_key(agent_result, "details", None) or {}
        details = _as_dict(details)

        items = (
            _safe_get(details, "issues", default=None)
            or _safe_get(details, "incorrect_claims", default=None)
            or _safe_get(details, "claims_incorrect", default=None)
            or []
        )

        if isinstance(items, list):
            for idx, item in enumerate(items):
                if not isinstance(item, dict):
                    continue

                issue_type = _safe_get(item, "issue_type", "error_type", "type", default=None)
                sev = _severity_from_raw(Dimension.factuality, issue_type, _safe_get(item, "severity", default=None))
                span = _span_from_item(item, summary_text)

                msg = (
                    _safe_get(item, "message", default=None)
                    or _safe_get(item, "description", default=None)
                    or "Faktischer Widerspruch oder falscher Claim."
                )
                msg = str(msg).strip()

                evidence: List[EvidenceItem] = []
                quote = _safe_get(item, "evidence_quote", "evidence", default=None)
                if isinstance(quote, str) and quote.strip():
                    evidence.append(EvidenceItem(kind="quote", source="agent:factuality", quote=quote.strip()))
                claim = _safe_get(item, "claim", default=None)
                if claim:
                    evidence.append(EvidenceItem(kind="claim", source="agent:factuality", data={"claim": claim}))
                evidence.append(EvidenceItem(kind="raw", source="agent:factuality", data={"item": item}))

                rec = _recommendation(Dimension.factuality, issue_type)

                f_id = _stable_id(
                    Dimension.factuality.value,
                    sev,
                    issue_type or "",
                    str(span.start_char if span else -1),
                    str(span.end_char if span else -1),
                    msg,
                )

                out.append(
                    Finding(
                        id=f_id,
                        dimension=Dimension.factuality,
                        severity=sev,
                        message=msg,
                        span=span,
                        evidence=evidence,
                        recommendation=rec,
                        source={
                            "agent": "factuality",
                            "source_list": "details",
                            "issue_type": issue_type,
                            "item_index": idx,
                        },
                    )
                )

        return out

    def _normalize_generic(self, dimension: Dimension, agent_result: Any, summary_text: str) -> List[Finding]:
        if not agent_result:
            return []

        # 1) bevorzugt: issue_spans
        span_items = _get_issue_spans(agent_result)
        if span_items:
            out: List[Finding] = []
            for idx, it in enumerate(span_items):
                msg = (it.get("message") or f"Problem in {dimension.value} erkannt.").strip()
                raw_sev = it.get("severity", None)
                issue_type = it.get("issue_type", None)  # optional, falls du es später ergänzt
                sev = _severity_from_raw(dimension, issue_type, raw_sev)

                span = _span_from_issue_span(it, summary_text)
                rec = _recommendation(dimension, issue_type)

                f_id = _stable_id(
                    dimension.value,
                    sev,
                    issue_type or "",
                    str(span.start_char if span else -1),
                    str(span.end_char if span else -1),
                    msg,
                )

                out.append(
                    Finding(
                        id=f_id,
                        dimension=dimension,
                        severity=sev,
                        message=msg,
                        span=span,
                        evidence=[EvidenceItem(kind="raw", source=f"agent:{dimension.value}", data={"issue_span": it})],
                        recommendation=rec,
                        source={
                            "agent": dimension.value,
                            "source_list": "issue_spans",
                            "issue_type": issue_type,
                            "item_index": idx,
                        },
                    )
                )
            return out

        # 2) fallback: details["issues"]
        details = _get_attr_or_key(agent_result, "details", None) or {}
        details = _as_dict(details)
        items = _safe_get(details, "issues", default=[]) or []
        if not isinstance(items, list):
            return []

        out: List[Finding] = []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue

            issue_type = _safe_get(item, "issue_type", "type", default=None)
            sev = _severity_from_raw(dimension, issue_type, _safe_get(item, "severity", default=None))
            span = _span_from_item(item, summary_text)
            msg = (
                _safe_get(item, "message", default=None)
                or _safe_get(item, "description", default=None)
                or f"Problem in {dimension.value} erkannt."
            )
            msg = str(msg).strip()

            rec = _recommendation(dimension, issue_type)

            f_id = _stable_id(
                dimension.value,
                sev,
                issue_type or "",
                str(span.start_char if span else -1),
                str(span.end_char if span else -1),
                msg,
            )

            out.append(
                Finding(
                    id=f_id,
                    dimension=dimension,
                    severity=sev,
                    message=msg,
                    span=span,
                    evidence=[EvidenceItem(kind="raw", source=f"agent:{dimension.value}", data={"item": item})],
                    recommendation=rec,
                    source={
                        "agent": dimension.value,
                        "source_list": "details.issues",
                        "issue_type": issue_type,
                        "item_index": idx,
                    },
                )
            )
        return out

    def _dedupe_and_cluster(self, findings: List[Finding]) -> List[Finding]:
        # 1) harte Dedupe über ID
        by_id: Dict[str, Finding] = {}
        for f in findings:
            if f.id not in by_id:
                by_id[f.id] = f
            else:
                prev = by_id[f.id]
                merged = _model_copy(prev, deep=True)
                merged.evidence = _merge_evidence([*prev.evidence, *f.evidence])
                # provenance sammeln (stabil genug für Debugging)
                merged.source = {
                    **prev.source,
                    "merged_from": list({repr(prev.source.get("merged_from", [])), repr(f.source)}),
                }
                by_id[f.id] = merged

        deduped = list(by_id.values())

        # 2) Cluster innerhalb jeder Dimension nach überlappenden Spans
        out: List[Finding] = []
        for dim in Dimension:
            dim_items = [f for f in deduped if f.dimension == dim]
            with_span = [f for f in dim_items if f.span is not None]
            no_span = [f for f in dim_items if f.span is None]

            with_span.sort(key=lambda x: (x.span.start_char, x.span.end_char))  # type: ignore[union-attr]

            clusters: List[List[Finding]] = []
            for f in with_span:
                if not clusters:
                    clusters.append([f])
                    continue
                last = clusters[-1][-1]
                if last.span and f.span and _overlap(last.span, f.span):
                    clusters[-1].append(f)
                else:
                    clusters.append([f])

            for cluster in clusters:
                if len(cluster) == 1:
                    out.append(cluster[0])
                    continue

                # primary = höchste Severity (stabil)
                cluster_sorted = sorted(
                    cluster,
                    key=lambda x: (-self.weights.severity[x.severity], x.id),
                )
                primary = _model_copy(cluster_sorted[0], deep=True)

                # union-span
                starts = [c.span.start_char for c in cluster if c.span]  # type: ignore[union-attr]
                ends = [c.span.end_char for c in cluster if c.span]      # type: ignore[union-attr]
                primary.span.start_char = min(starts)  # type: ignore[union-attr]
                primary.span.end_char = max(ends)      # type: ignore[union-attr]

                # evidence + provenance
                primary.evidence = _merge_evidence([e for c in cluster for e in c.evidence])
                primary.source = {
                    **primary.source,
                    "cluster_size": len(cluster),
                    "cluster_members": [c.id for c in cluster],
                }
                out.append(primary)

            out.extend(no_span)

        # stabil sort (damit Snapshot-Tests ruhig schlafen können)
        out.sort(key=lambda f: (f.dimension.value, f.span.start_char if f.span else 10**12, f.id))
        return out

    def _rank(self, findings: List[Finding]) -> List[Finding]:
        def score(f: Finding) -> float:
            sev_w = self.weights.severity[f.severity]
            dim_w = self.weights.dimension[f.dimension]
            span_len = 1
            if f.span:
                span_len = max(1, f.span.end_char - f.span.start_char)
            span_w = 1.0 + math.log(1.0 + span_len)
            return sev_w * dim_w * span_w

        scored: List[Tuple[Finding, float]] = [(f, score(f)) for f in findings]
        scored.sort(key=lambda t: (-t[1], t[0].id))
        return [f for f, _ in scored]

    def _top_spans(self, ranked: List[Finding]) -> List[TopSpan]:
        tops: List[TopSpan] = []
        seen_ranges = set()

        def score_for(f: Finding) -> float:
            sev_w = self.weights.severity[f.severity]
            dim_w = self.weights.dimension[f.dimension]
            span_len = 1
            if f.span:
                span_len = max(1, f.span.end_char - f.span.start_char)
            return sev_w * dim_w * (1.0 + math.log(1.0 + span_len))

        for f in ranked:
            if not f.span:
                continue
            key = (f.span.start_char, f.span.end_char, f.dimension.value)
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            tops.append(
                TopSpan(
                    span=f.span,
                    dimension=f.dimension,
                    severity=f.severity,
                    finding_id=f.id,
                    rank_score=score_for(f),
                )
            )
            if len(tops) >= self.top_k:
                break
        return tops

    def _stats(self, findings: List[Finding], summary_text: str) -> ExplainabilityStats:
        counts = {"high": 0, "medium": 0, "low": 0}
        spans: List[Tuple[int, int]] = []

        for f in findings:
            counts[f.severity] += 1
            if f.span and f.span.end_char > f.span.start_char:
                spans.append((f.span.start_char, f.span.end_char))

        coverage = self._interval_union_len(spans)
        total = max(1, len(summary_text))

        return ExplainabilityStats(
            num_findings=len(findings),
            num_high_severity=counts["high"],
            num_medium_severity=counts["medium"],
            num_low_severity=counts["low"],
            coverage_chars=coverage,
            coverage_ratio=coverage / total,
        )

    @staticmethod
    def _interval_union_len(intervals: List[Tuple[int, int]]) -> int:
        if not intervals:
            return 0
        intervals.sort()
        cur_s, cur_e = intervals[0]
        total = 0
        for s, e in intervals[1:]:
            if s <= cur_e:
                cur_e = max(cur_e, e)
            else:
                total += max(0, cur_e - cur_s)
                cur_s, cur_e = s, e
        total += max(0, cur_e - cur_s)
        return total

    def _executive_summary(self, ranked: List[Finding], top_spans: List[TopSpan]) -> List[str]:
        if not ranked:
            return [
                "Es wurden keine Findings erzeugt (keine erkannten Probleme oder keine Issues geliefert).",
                "Das ist kein Beweis für Korrektheit, nur ein Hinweis auf fehlende/geringe Detektion.",
                "Für mehr Sicherheit: Factuality besonders bei Zahlen/Daten zusätzlich prüfen.",
            ]

        total = len(ranked)
        hi = sum(1 for f in ranked if f.severity == "high")
        med = sum(1 for f in ranked if f.severity == "medium")
        low = sum(1 for f in ranked if f.severity == "low")

        by_dim = {d: 0 for d in Dimension}
        for f in ranked:
            by_dim[f.dimension] += 1
        top_dim = sorted(by_dim.items(), key=lambda x: (-x[1], x[0].value))[0][0]

        sentences = [
            f"In der Summary wurden {total} Findings identifiziert ({hi} high, {med} medium, {low} low).",
            f"Der Schwerpunkt liegt bei **{top_dim.value}** (meiste Findings in dieser Dimension).",
        ]

        if top_spans:
            snippets = []
            for ts in top_spans[:3]:
                txt = (ts.span.text or "").replace("\n", " ").strip()
                if len(txt) > 70:
                    txt = txt[:67] + "..."
                if txt:
                    snippets.append(f"„{txt}“")
            if snippets:
                sentences.append("Kritische Textstellen: " + ", ".join(snippets) + ".")

        sentences.append("Priorität: erst factuality-high fixen (Zahlen/Daten), dann Kohärenz, dann Lesbarkeit glätten.")
        return sentences[:6]
