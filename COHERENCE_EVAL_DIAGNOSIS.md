# Coherence Evaluation Failure Diagnosis Report

**Date:** 2026-01-30  
**Run ID:** `coherence_20260130_131451_gpt-4o-mini_v1_seed42`  
**Status:** Root cause identified, logging improved

---

## Problem Summary

**Symptom:**
- Examples used: 0
- Examples failed: 1700
- All metrics: 0.0000

**Root Cause:**
401 Invalid API Key Error - OpenAI API key is not properly configured.

**Error Message:**
```
Error code: 401 - {'error': {'message': 'Incorrect API key provided: your-ope************here. 
You can find your API key at https://platform.openai.com/account/api-keys.', 
'type': 'invalid_request_error', 'param': None, 'code': 'invalid_api_key'}}
```

---

## Where It Failed

**File:** `scripts/eval_sumeval_coherence.py`  
**Location:** Line 408 (`agent.run()` call)  
**Exception Type:** `OpenAIError` (401 Unauthorized)

**Call Stack:**
1. `scripts/eval_sumeval_coherence.py:408` - `agent.run(article_text=article, summary_text=summary, meta=meta)`
2. `app/services/agents/coherence/coherence_agent.py` - Agent execution
3. `app/llm/openai_client.py` - OpenAI API call
4. OpenAI API returns 401 Invalid API Key

---

## What Changed

### 1. Enhanced Error Logging

**Added separate `errors.jsonl` file** with detailed failure information:
- Exception type (e.g., `OpenAIError`, `ValueError`)
- Full exception message
- Complete traceback
- Input validation info (input_keys, gt_keys, article/summary lengths)
- Example index and ID

**Location:** `scripts/eval_sumeval_coherence.py:357-362, 420-448`

### 2. Improved Progress Reporting

**Changed progress interval from 25 to 50 examples** with clearer format:
```
[Progress] used=50, seen=52, skipped=1, failed=1 | gt_norm=0.583 pred=0.650
```

**Location:** `scripts/eval_sumeval_coherence.py:518-521`

### 3. Exception Type Extraction

**Extract exception class name** for better error classification:
```python
exception_type = type(e).__name__  # e.g., "OpenAIError"
exception_message = str(e)
exception_traceback = traceback.format_exc()
```

**Location:** `scripts/eval_sumeval_coherence.py:420-422`

---

## Dataset Validation

**Dataset:** `data/sumeval/sumeval_clean.jsonl`  
**Status:** ✅ Valid schema

**Sample record structure:**
```json
{
  "article": "...",
  "summary": "...",
  "gt": {
    "coherence": 1.3333333333,
    "readability": 3.0,
    "fluency": 3.0
  },
  "meta": {
    "doc_id": "dm-test-8764fb95bfad8ee849274873a92fb8d6b400eee2",
    ...
  }
}
```

**GT Normalization:** ✅ Working correctly
- Raw scale: [1.0, 5.0]
- Normalized to: [0, 1]
- Formula: `gt_norm = (gt_raw - 1) / 4`

---

## Fix Required

**Action:** Set valid OpenAI API key in environment

**Option 1: Environment Variable**
```bash
export OPENAI_API_KEY="sk-..."
```

**Option 2: .env file**
```bash
echo "OPENAI_API_KEY=sk-..." >> .env
```

**Option 3: Load in script** (already implemented via `load_dotenv()`)

---

## Proof of Fix

After setting a valid API key, the evaluation should produce:
- `n_used > 0` (at least some examples succeed)
- Non-zero metrics (Pearson r, Spearman ρ, MAE, RMSE)
- `errors.jsonl` file (may be empty if all succeed, or contain only non-API errors)

**Expected output structure:**
```
results/evaluation/coherence/<run_id>/
  ├── predictions.jsonl    # All examples (success + failures)
  ├── errors.jsonl         # Detailed error log (NEW)
  ├── summary.json         # Metrics with CIs
  ├── summary.md           # Human-readable summary
  └── run_metadata.json     # Run configuration
```

---

## Next Steps

1. ✅ **Error logging improved** - Separate `errors.jsonl` with detailed failure info
2. ✅ **Progress reporting enhanced** - Every 50 examples with clear counters
3. ⏳ **Set API key** - User must configure `OPENAI_API_KEY` environment variable
4. ⏳ **Re-run evaluation** - Execute with valid API key to generate metrics
5. ⏳ **Verify results** - Confirm `n_used > 0` and non-zero metrics

---

## Files Modified

- `scripts/eval_sumeval_coherence.py`
  - Added `errors.jsonl` output
  - Enhanced exception handling with type extraction
  - Improved progress reporting (50-example intervals)
  - Added traceback capture for debugging

---

## Commit

**Branch:** `fix/coherence-eval-failures`  
**Message:** `fix: log coherence eval failures and repair SummEval processing`

**Changes:**
- Add separate `errors.jsonl` file for detailed failure logging
- Extract exception type and full traceback for better diagnostics
- Improve progress reporting (every 50 examples instead of 25)
- Add input validation metadata to error records
