"""
Debug-Script: Prüft, was das LLM zurückgibt und warum keine Evidence gefunden wird.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.llm.openai_client import OpenAIClient
from app.services.agents.factuality.claim_models import Claim
from app.services.agents.factuality.claim_verifier import LLMClaimVerifier


def main():
    # Load one example from FRANK
    dataset_path = ROOT / "data" / "frank" / "frank_clean.jsonl"
    with dataset_path.open("r", encoding="utf-8") as f:
        line = f.readline()
        ex = json.loads(line)

    article = ex["article"]
    summary = ex["summary"]
    has_error = ex.get("has_error", False)

    print("=" * 80)
    print("Debug: Evidence Retrieval")
    print("=" * 80)
    print(f"\nHas Error (GT): {has_error}")
    print("\nSummary (first 200 chars):")
    print(summary[:200] + "...")
    print("\nArticle (first 300 chars):")
    print(article[:300] + "...")
    print()

    # Setup verifier
    llm_client = OpenAIClient(model_name="gpt-4o-mini")
    verifier = LLMClaimVerifier(
        llm_client,
        use_evidence_retriever=True,
        evidence_retriever_top_k=5,
        strict_mode=False,  # Debug-Script: Standard False
    )

    # Create a test claim from first sentence
    sentences = summary.split(". ")
    if sentences:
        first_sentence = sentences[0].strip() + "."
        test_claim = Claim(
            id="test_1",
            sentence_index=0,
            sentence=first_sentence,
            text=first_sentence,
        )

        print("=" * 80)
        print(f"Testing Claim: {test_claim.text}")
        print("=" * 80)
        print()

        # Get context
        context = verifier._select_context(
            article,
            test_claim.text,
            top_k=15,
            neighbor_window=1,
            max_chars=8000,
        )

        print("Context (first 500 chars):")
        print(context[:500] + "...")
        print()

        # Verify claim
        verified = verifier.verify(article, test_claim)

        print("=" * 80)
        print("Verification Result:")
        print("=" * 80)
        print(f"Label: {verified.label}")
        print(f"Confidence: {verified.confidence}")
        print(f"Evidence Found: {verified.evidence_found}")
        print(f"Evidence (raw): {verified.evidence}")
        print(f"Evidence Spans (structured): {len(verified.evidence_spans_structured)}")
        print(f"Explanation: {verified.explanation}")
        print()

        # Check what LLM returned
        prompt = verifier._build_prompt(context, test_claim.text)
        raw_output = llm_client.complete(prompt)

        print("=" * 80)
        print("LLM Raw Output:")
        print("=" * 80)
        print(raw_output)
        print()

        # Parse output
        parsed = verifier._parse_output(raw_output)
        print("=" * 80)
        print("Parsed Output:")
        print("=" * 80)
        print(json.dumps(parsed, indent=2, ensure_ascii=False))
        print()

        # Check evidence matching
        if parsed.get("evidence"):
            print("=" * 80)
            print("Evidence Matching:")
            print("=" * 80)
            for ev in parsed.get("evidence", []):
                ev_str = str(ev).strip()
                print(f"\nEvidence from LLM: {ev_str[:100]}...")
                exact_match = ev_str in context
                print(f"  Exact match: {exact_match}")

                if not exact_match:
                    # Try fuzzy match
                    fuzzy_match = verifier._fuzzy_match_evidence(ev_str, context)
                    if fuzzy_match:
                        print(f"  Fuzzy match found: {fuzzy_match[:100]}...")
                    else:
                        print("  No fuzzy match found")
        else:
            print("⚠️  No evidence in LLM output!")


if __name__ == "__main__":
    main()
