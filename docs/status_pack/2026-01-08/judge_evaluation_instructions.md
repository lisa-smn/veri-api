# LLM-as-a-Judge Evaluation: Anleitung

## Voraussetzungen

Bevor das Agreement-Script (`compare_agent_vs_judge.py`) verwendet werden kann, müssen **zwei separate Runs** durchgeführt werden:

1. **Agent-Run:** Standard-Evaluation ohne Judge (oder mit Judge im `secondary`-Mode, aber `--score_source agent`)
2. **Judge-Run:** Evaluation mit Judge aktiviert und `--score_source judge`

## Schritt 1: Agent-Run (Readability)

```bash
python3 scripts/eval_sumeval_readability.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --model gpt-4o-mini \
  --prompt_version v1 \
  --bootstrap_n 2000 \
  --score_source agent
```

**Output:** `results/evaluation/readability/readability_YYYYMMDD_HHMMSS_gpt-4o-mini_v1_seed42/`

## Schritt 2: Judge-Run (Readability)

```bash
ENABLE_LLM_JUDGE=true \
JUDGE_MODE=secondary \
JUDGE_MODEL=gpt-4o-mini \
JUDGE_PROMPT_VERSION=v1 \
JUDGE_N=3 \
JUDGE_TEMPERATURE=0 \
JUDGE_AGGREGATION=mean \
python3 scripts/eval_sumeval_readability.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --model gpt-4o-mini \
  --prompt_version v1 \
  --bootstrap_n 2000 \
  --score_source judge
```

**Output:** `results/evaluation/readability/readability_YYYYMMDD_HHMMSS_gpt-4o-mini_v1_seed42/`

**Wichtig:** Beide Runs müssen **identische Subsets** verwenden (gleicher `--seed 42` und `--max_examples 200`).

## Schritt 3: Agreement-Analyse

Nach beiden Runs:

```bash
python3 scripts/compare_agent_vs_judge.py \
  --run_dir_agent results/evaluation/readability/readability_YYYYMMDD_HHMMSS_gpt-4o-mini_v1_seed42 \
  --run_dir_judge results/evaluation/readability/readability_YYYYMMDD_HHMMSS_gpt-4o-mini_v1_seed42 \
  --out docs/status_pack/2026-01-08/judge_agreement_readability.md \
  --threshold 0.7
```

**Hinweis:** Ersetze `YYYYMMDD_HHMMSS` mit den tatsächlichen Timestamps deiner Runs.

## Schritt 4: Coherence (analog)

```bash
# Agent-Run
python3 scripts/eval_sumeval_coherence.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --model gpt-4o-mini \
  --bootstrap_n 2000 \
  --score_source agent

# Judge-Run
ENABLE_LLM_JUDGE=true JUDGE_MODE=secondary JUDGE_N=3 \
python3 scripts/eval_sumeval_coherence.py \
  --data data/sumeval/sumeval_clean.jsonl \
  --max_examples 200 \
  --seed 42 \
  --model gpt-4o-mini \
  --bootstrap_n 2000 \
  --score_source judge

# Agreement-Analyse
python3 scripts/compare_agent_vs_judge.py \
  --run_dir_agent results/evaluation/coherence/coherence_YYYYMMDD_HHMMSS_gpt-4o-mini_v1_seed42 \
  --run_dir_judge results/evaluation/coherence/coherence_YYYYMMDD_HHMMSS_gpt-4o-mini_v1_seed42 \
  --out docs/status_pack/2026-01-08/judge_agreement_coherence.md \
  --threshold 0.7
```

## Prüfen, ob ein Run Judge-Daten enthält

```bash
# Prüfe, ob pred_judge oder judge-Daten vorhanden sind
head -1 <run_dir>/predictions.jsonl | python3 -m json.tool | grep -E "(pred_judge|judge)"
```

Falls `pred_judge` oder `judge_committee` vorhanden sind, enthält der Run Judge-Daten.

## Troubleshooting

### "Keine übereinstimmenden Beispiele gefunden"
- Prüfe, ob beide Runs den **gleichen Seed** und **gleiche max_examples** haben
- Prüfe, ob `example_id` in beiden `predictions.jsonl` identisch sind

### "Judge-Score nicht verfügbar"
- Stelle sicher, dass `ENABLE_LLM_JUDGE=true` beim Judge-Run gesetzt war
- Prüfe `run_metadata.json` auf `"judge"` in `config`

### "predictions.jsonl nicht gefunden"
- Prüfe, ob der Run-Verzeichnispfad korrekt ist
- Prüfe, ob `predictions.jsonl` im Run-Verzeichnis existiert

