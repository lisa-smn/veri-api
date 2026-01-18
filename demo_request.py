#!/usr/bin/env python3
"""Demo-Request für Kolloquium-Screenshots"""

import json
import sys

import requests

BASE_URL = "http://localhost:8000"

payload = {
    "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner und ist seit 1990 wieder vereint.",
    "summary_text": "Paris ist die Hauptstadt von Deutschland.",
    "dataset": "demo",
    "llm_model": "gpt-4o-mini",
}

print("=" * 70)
print("INPUT:")
print("=" * 70)
print(json.dumps(payload, indent=2, ensure_ascii=False))
print()

try:
    response = requests.post(f"{BASE_URL}/verify", json=payload, timeout=60)
    response.raise_for_status()
    result = response.json()
except requests.exceptions.ConnectionError:
    print(f"❌ Server nicht erreichbar unter {BASE_URL}")
    print("   Bitte starten Sie den Server mit: docker-compose up")
    sys.exit(1)
except Exception as e:
    print(f"❌ Fehler: {e}")
    sys.exit(1)

print("=" * 70)
print("OUTPUT: SCORES (3 Dimensionen)")
print("=" * 70)
print(f"  Overall Score:     {result['overall_score']:.3f}")
print(f"  Factuality Score:  {result['factuality']['score']:.3f}")
print(f"  Coherence Score:   {result['coherence']['score']:.3f}")
print(f"  Readability Score: {result['readability']['score']:.3f}")
print()

print("=" * 70)
print("OUTPUT: ISSUE SPANS (markierte Textstellen)")
print("=" * 70)
fact_spans = result["factuality"].get("issue_spans", [])
if fact_spans:
    for i, span in enumerate(fact_spans[:2], 1):
        print(f"  Span {i}:")
        print(f"    Position: Zeichen {span.get('start_char')} bis {span.get('end_char')}")
        print(f"    Verdict: {span.get('verdict', 'N/A')}")
        print(f"    Evidence Found: {span.get('evidence_found', 'N/A')}")
        print(f"    Severity: {span.get('severity', 'N/A')}")
        print(f"    Message: {span.get('message', '')[:80]}...")
        print()
else:
    print("  Keine Issue Spans gefunden")
print()

print("=" * 70)
print("OUTPUT: ERKLÄRUNGEN (Explanations)")
print("=" * 70)

# Zeige factuality.explanation (immer vorhanden)
if result.get("factuality", {}).get("explanation"):
    print("  Globale Erklärung (Factuality):")
    print(f"    {result['factuality']['explanation'][:100]}...")
    print()

# Zeige Claims mit Erklärungen (immer vorhanden)
if result.get("factuality", {}).get("details", {}).get("claims"):
    print("  Claim-spezifische Erklärungen:")
    for i, claim in enumerate(result["factuality"]["details"]["claims"][:2], 1):
        print(f"    Claim {i}:")
        print(f"      Label: {claim.get('label', 'N/A')}")
        print(f"      Erklärung: {claim.get('explanation', 'N/A')[:80]}...")
        if claim.get("evidence_quote"):
            print(f"      Evidence Quote: {claim['evidence_quote'][:80]}...")
        print(f"      Evidence Found: {claim.get('evidence_found', 'N/A')}")
        print()

# Zeige Explainability falls vorhanden (optional)
if result.get("explainability"):
    print("  Explainability Report (optional):")
    exp = result["explainability"]
    if exp.get("summary"):
        print("    Executive Summary:")
        for line in exp["summary"][:2]:
            print(f"      • {line}")
    if exp.get("stats"):
        print(f"    Findings: {exp['stats'].get('num_findings', 0)}")
else:
    print("  (Explainability nicht verfügbar - verwende factuality.details.claims)")

print("=" * 70)
print("✅ Demo abgeschlossen")
print("=" * 70)
