# Chapter 6 – Failure Root Cause Report

**Date:** 2026-01-30  
**Status:** ✅ Root cause identified and fixed

---

## Problem Summary

**Symptom:**
- Examples used: 0
- Examples failed: 1700 (alle Beispiele)
- All metrics: 0.0000

**Root Cause:**
401 Invalid API Key Error - OpenAI API key war nicht korrekt konfiguriert.

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

### 2. API Key Validation (Fail-Fast)

**Added robust API key validation** in `app/llm/openai_client.py`:
- Checks for missing keys
- Detects placeholder patterns (e.g., "your-ope", "your-openai", "********")
- Validates minimum length (>= 20 characters)
- Validates prefix (must start with "sk-")
- Masks API key in logs (first 6 + last 4 characters)

**Location:** `app/llm/openai_client.py:15-50`

**Code:**
```python
def validate_api_key(key: str | None) -> str:
    if not key:
        raise ValueError("OPENAI_API_KEY is missing")
    key = key.strip()
    placeholder_patterns = [
        "your-ope", "your-openai", "your-key", "********", "changeme", "replace-me"
    ]
    key_lower = key.lower()
    for pattern in placeholder_patterns:
        if pattern in key_lower:
            raise ValueError(f"OPENAI_API_KEY appears to be a placeholder: '{mask_api_key(key)}'")
    if len(key) < 20:
        raise ValueError(f"OPENAI_API_KEY is too short (minimum 20 characters)")
    if not key.startswith("sk-"):
        raise ValueError(f"OPENAI_API_KEY must start with 'sk-'")
    return key
```

### 3. Improved Progress Reporting

**Changed progress interval from 25 to 50 examples** with clearer format:
```
[Progress] used=50, seen=52, skipped=1, failed=1 | gt_norm=0.583 pred=0.650
```

**Location:** `scripts/eval_sumeval_coherence.py:518-521`

### 4. Exception Type Extraction

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

## Fix Applied

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

After setting a valid API key, the evaluation produced:
- ✅ `n_used = 200` (all examples succeeded)
- ✅ Non-zero metrics:
  - Spearman ρ: 0.451 [0.314, 0.575]
  - Pearson r: 0.371 [0.197, 0.557]
  - MAE: 0.174 [0.152, 0.197]
  - RMSE: 0.237 [0.203, 0.273]
- ✅ `errors.jsonl` file (empty, no errors)

**Output structure:**
```
results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/
  ├── predictions.jsonl    # 200 examples (all successful)
  ├── errors.jsonl         # Empty (no errors)
  ├── summary.json         # Metrics with CIs
  ├── summary.md           # Human-readable summary
  └── run_metadata.json     # Run configuration
```

**Quelle:** `results/evaluation/coherence/coherence_20260107_205123_gpt-4o-mini_v1_seed42/summary.json`

---

## Files Modified

1. **`scripts/eval_sumeval_coherence.py`**
   - Added `errors.jsonl` output
   - Enhanced exception handling with type extraction
   - Improved progress reporting (50-example intervals)
   - Added traceback capture for debugging

2. **`app/llm/openai_client.py`**
   - Added `validate_api_key()` function
   - Added `mask_api_key()` function for safe logging
   - Modified `__init__` to validate API key before initializing client
   - Added debug logging (masked API key + source)

3. **`scripts/eval_sumeval_readability.py`**
   - Applied same error logging improvements as coherence script

---

## Commits

**Branch:** `fix/coherence-eval-failures`  
**Commit:** `fix: log coherence eval failures and repair SummEval processing`

**Branch:** `docs/eval-artifacts-recover`  
**Commit:** `docs: restore missing evaluation artifacts and update evidence pack`

---

## Lessons Learned

1. **Fail-Fast Validation:** API key validation sollte so früh wie möglich erfolgen (beim Client-Init), nicht erst beim ersten API-Call.

2. **Error Logging:** Separate `errors.jsonl` Dateien mit vollständigen Tracebacks sind essentiell für Debugging, besonders bei Batch-Evaluations.

3. **Placeholder Detection:** Systematische Erkennung von Platzhalter-Strings verhindert, dass ungültige Keys versehentlich verwendet werden.

4. **Masked Logging:** API Keys sollten niemals im Klartext geloggt werden, auch nicht in Debug-Ausgaben.
