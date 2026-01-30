# Chapter 6 – Reproducibility Instructions

**Repo-Stand:** `f0dc39532bad0e760bb8aedc8db16550efb025bd`  
**Date:** 2026-01-30

---

## Prerequisites

### Environment Variables

**Required:**
- `OPENAI_API_KEY`: Valid OpenAI API key (starts with `sk-`)

**Set via:**
```bash
export OPENAI_API_KEY="sk-..."
```

Or create `.env` file:
```bash
echo "OPENAI_API_KEY=sk-..." >> .env
```

### Python Environment

**Virtual Environment:**
```bash
source venv/bin/activate  # or: python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## M10 Factuality Evaluation (FRANK, n=50)

### Run Single Evaluation

```bash
python3 scripts/run_m10_factuality.py \
  configs/m10_factuality_runs.yaml \
  --run-id evidence_gate_test_count_as_error
```

**Expected Outputs:**
- `results/evaluation/runs/results/evidence_gate_test_count_as_error.json`
- `results/evaluation/runs/results/evidence_gate_test_count_as_error_examples.jsonl`
- `results/evaluation/runs/docs/evidence_gate_test_count_as_error.md`

### Run All Evaluations

```bash
python3 scripts/run_m10_factuality.py configs/m10_factuality_runs.yaml
```

### Aggregate Results

```bash
python3 scripts/aggregate_m10_results.py
```

**Expected Outputs:**
- `results/evaluation/summary_matrix.csv`
- `results/evaluation/summary.md`

---

## SummEval Coherence Evaluation (n=200)

### Run Evaluation

```bash
python3 scripts/eval_sumeval_coherence.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --model gpt-4o-mini \
  --prompt_version v1 \
  --seed 42 \
  --max_examples 200 \
  --bootstrap_n 2000
```

**Expected Outputs:**
- `results/evaluation/coherence/coherence_<timestamp>_gpt-4o-mini_v1_seed42/`
  - `predictions.jsonl` (200 lines)
  - `summary.json`
  - `summary.md`
  - `run_metadata.json`
  - `errors.jsonl` (may be empty)

**Verify Success:**
- Check `summary.json`: `n_used` should be 200, `n_failed` should be 0
- Check `errors.jsonl`: Should be empty or contain only non-API errors

---

## SummEval Readability Evaluation (n=200)

### Run Evaluation

```bash
python3 scripts/eval_sumeval_readability.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --model gpt-4o-mini \
  --prompt_version v1 \
  --seed 42 \
  --max_examples 200 \
  --bootstrap_n 2000
```

**Expected Outputs:**
- `results/evaluation/readability/readability_<timestamp>_gpt-4o-mini_v1_seed42/`
  - `predictions.jsonl` (200 lines)
  - `summary.json`
  - `summary.md`
  - `run_metadata.json`
  - `errors.jsonl` (may be empty)

**Verify Success:**
- Check `summary.json`: `n_used` should be 200, `n_failed` should be 0
- Check `errors.jsonl`: Should be empty or contain only non-API errors

---

## FRANK Manifest Factuality Evaluation (n=200)

### Prerequisites

**Build Manifest:**
```bash
python3 scripts/build_frank_subset_manifest.py \
  --benchmark data/frank/benchmark.json \
  --annotations data/frank/annotations.json \
  --output data/frank/frank_subset_manifest.jsonl \
  --max_examples 200 \
  --seed 42
```

### Run Evaluation

```bash
python3 scripts/eval_frank_factuality_agent_on_manifest.py \
  --manifest data/frank/frank_subset_manifest.jsonl \
  --model gpt-4o-mini \
  --seed 42
```

**Expected Outputs:**
- `results/evaluation/factuality/factuality_agent_manifest_<timestamp>_gpt-4o-mini/`
  - `predictions.jsonl` (200 lines)
  - `summary.json`
  - `summary.md`
  - `run_metadata.json`

---

## Troubleshooting

### API Key Issues

**Symptom:** `used=0, failed=1700` or `401 Invalid API Key`

**Fix:**
1. Verify `OPENAI_API_KEY` is set: `echo $OPENAI_API_KEY`
2. Check API key format: Should start with `sk-` and be >= 20 characters
3. Check for placeholder patterns: Should not contain "your-ope", "your-openai", etc.
4. Re-run with valid key

**Validation:**
```bash
python3 -c "from app.llm.openai_client import validate_api_key; import os; validate_api_key(os.getenv('OPENAI_API_KEY'))"
```

### Dataset Issues

**Symptom:** `n_used=0` or schema errors

**Fix:**
1. Verify dataset exists: `ls data/sumeval/sumeval_clean.jsonl`
2. Check first line: `head -1 data/sumeval/sumeval_clean.jsonl | python3 -m json.tool`
3. Verify required fields: `article`, `summary`, `gt.coherence` (or `gt.readability`)

### Error Logging

**Check errors.jsonl:**
```bash
cat results/evaluation/coherence/<run_id>/errors.jsonl | python3 -m json.tool | head -50
```

**Common error types:**
- `OpenAIError` (401): Invalid API key
- `OpenAIError` (429): Rate limit exceeded (add retry/backoff)
- `ValueError`: Dataset schema mismatch
- `KeyError`: Missing required field in dataset

---

## Expected Run Times

- **M10 Factuality (n=50):** ~5-10 minutes
- **Coherence (n=200):** ~30-60 minutes
- **Readability (n=200):** ~30-60 minutes
- **FRANK Manifest (n=200):** ~60-90 minutes

**Note:** Times depend on API rate limits and model response times.

---

## Verification Checklist

After running evaluations, verify:

- [ ] `n_used` matches expected count (50 or 200)
- [ ] `n_failed` is 0 or very low (< 5% of n_used)
- [ ] `summary.json` contains non-zero metrics
- [ ] `predictions.jsonl` has correct number of lines
- [ ] `errors.jsonl` is empty or contains only non-critical errors
- [ ] `run_metadata.json` contains correct config (model, seed, prompt_version)

---

## Docker (Optional)

If using Docker:

```bash
docker-compose up -d
docker-compose exec app python3 scripts/eval_sumeval_coherence.py ...
```

**Note:** Ensure `OPENAI_API_KEY` is set in Docker environment or `.env` file.
