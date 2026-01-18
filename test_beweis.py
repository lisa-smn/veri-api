#!/usr/bin/env python3
"""Beweis: Evidence-Gate im Factuality-Agent - 3 Testfälle"""

import json

import requests

BASE_URL = "http://localhost:8000"


def make_request(payload, test_name):
    """Hilfsfunktion für API-Requests mit Fehlerbehandlung"""
    try:
        response = requests.post(f"{BASE_URL}/verify", json=payload, timeout=60)

        if response.status_code != 200:
            print(f"❌ HTTP {response.status_code}: {response.text[:200]}")
            return None

        result = response.json()

        # Prüfe auf Fehler in Response
        if "detail" in result:
            print(f"❌ API Fehler: {result['detail']}")
            return None

        if "factuality" not in result:
            print(f"❌ Unerwartete Response-Struktur. Keys: {list(result.keys())}")
            print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)[:500]}")
            return None

        return result
    except requests.exceptions.ConnectionError:
        print(f"❌ Fehler: Server nicht erreichbar unter {BASE_URL}")
        print("   Bitte starten Sie den Server mit:")
        print("   docker-compose up")
        print("   ODER")
        print("   uvicorn app.server:app --host 0.0.0.0 --port 8000 --reload")
        return None
    except Exception as e:
        print(f"❌ Fehler bei {test_name}: {e}")
        return None


# Beispiel 1: Falsch + Beleg
print("\n" + "=" * 60)
print("Beispiel 1: Falsch + Beleg vorhanden")
print("=" * 60)
result1 = make_request(
    {
        "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner.",
        "summary_text": "Paris ist die Hauptstadt von Deutschland.",
        "dataset": "demo",
        "llm_model": "gpt-4o-mini",
    },
    "Beispiel 1",
)

if result1:
    fact1 = result1["factuality"]
    if fact1.get("issue_spans"):
        span1 = fact1["issue_spans"][0]
        print(f"✅ Verdict: {span1.get('verdict')}")
        print(f"✅ Evidence Found: {span1.get('evidence_found')}")
        print(f"✅ Message: {span1.get('message', '')[:80]}...")
        if fact1.get("details", {}).get("claims"):
            claim1 = fact1["details"]["claims"][0]
            if claim1.get("evidence_quote"):
                print(f"✅ Evidence Quote: {claim1['evidence_quote'][:80]}...")
    else:
        print("⚠️  Keine issue_spans gefunden")

# Beispiel 2: Ohne Beleg
print("\n" + "=" * 60)
print("Beispiel 2: Behauptung ohne Beleg")
print("=" * 60)
result2 = make_request(
    {
        "article_text": "Berlin ist eine große Stadt in Deutschland. Die Stadt wurde 1990 wieder vereint.",
        "summary_text": "Berlin hat genau 4 Millionen Einwohner.",
        "dataset": "demo",
        "llm_model": "gpt-4o-mini",
    },
    "Beispiel 2",
)

if result2:
    fact2 = result2["factuality"]
    if fact2.get("issue_spans"):
        span2 = fact2["issue_spans"][0]
        print(f"✅ Verdict: {span2.get('verdict')}")
        print(f"✅ Evidence Found: {span2.get('evidence_found')}")
        print(f"✅ Message: {span2.get('message', '')[:80]}...")
    else:
        print("⚠️  Keine issue_spans gefunden")

# Beispiel 3: Markierungen
print("\n" + "=" * 60)
print("Beispiel 3: Markierungen (issue_spans)")
print("=" * 60)
result3 = make_request(
    {
        "article_text": "Berlin ist die Hauptstadt von Deutschland. Die Stadt hat über 3,7 Millionen Einwohner.",
        "summary_text": "Paris ist die Hauptstadt von Deutschland. Die Stadt hat 4 Millionen Einwohner.",
        "dataset": "demo",
        "llm_model": "gpt-4o-mini",
    },
    "Beispiel 3",
)

if result3:
    fact3 = result3["factuality"]
    print(f"✅ Anzahl Issue Spans: {len(fact3.get('issue_spans', []))}")
    for i, span in enumerate(fact3.get("issue_spans", [])[:2]):
        print(f"\n   Span {i + 1}:")
        print(f"   - Start: {span.get('start_char')}, End: {span.get('end_char')}")
        print(f"   - Verdict: {span.get('verdict')}")
        print(f"   - Message: {span.get('message', '')[:60]}...")

print("\n" + "=" * 60)
if result1 and result2 and result3:
    print("✅ Beweis abgeschlossen!")
else:
    print("⚠️  Einige Tests konnten nicht ausgeführt werden")
print("=" * 60)
