# Run Recipe: coherence_20260107_205123_gpt-4o-mini_v1_seed42

**Source:** `docs/status/coherence_status.md:78-86`

## Exact Command

```bash
python scripts/eval_sumeval_coherence.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --bootstrap_n 2000 \
  --cache
```

**Note:** The original recipe uses `--cache_mode write`, but the script uses `--cache` flag.

## Parameters

- **Script:** `scripts/eval_sumeval_coherence.py`
- **Dataset:** `data/sumeval/sumeval_clean.jsonl`
- **Model:** `gpt-4o-mini` (default)
- **Prompt:** `v1` (default from ENV or script)
- **Seed:** `42`
- **Max Examples:** `200`
- **Bootstrap Resamples:** `2000`
- **Cache:** Enabled (`--cache`)
- **GT Normalization:** `[1.0, 5.0] -> [0, 1]` (default)

## Expected Output

- **Run-ID:** `coherence_20260107_205123_gpt-4o-mini_v1_seed42` (must be renamed after generation)
- **Output Directory:** `results/evaluation/coherence/coherence_<timestamp>_gpt-4o-mini_v1_seed42/`
- **Files:**
  - `predictions.jsonl`
  - `summary.json`
  - `summary.md`
  - `run_metadata.json`
  - `cache.jsonl` (if cache enabled)

## Expected Metrics

- **n_used:** 200
- **n_failed:** 0 (or low)
- **Spearman ρ:** ~0.41 [0.27, 0.53]
- **Pearson r:** ~0.35 [0.17, 0.53]
- **MAE:** ~0.18 [0.16, 0.20]
- **RMSE:** ~0.24 [0.21, 0.28]

## Reproduction Strategy

Since the script generates a timestamp-based Run-ID, we need to:
1. Run the evaluation with the exact parameters above
2. Rename the generated output directory to match the historical Run-ID
3. Verify metrics match expected values (within reasonable tolerance)
