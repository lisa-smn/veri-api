"""
Stress-Tests für Coherence-Agent: Prüft, ob der Agent "merkt", wenn Text kaputt gemacht wird.

Modi:
1. shuffle: Permutiert Satzreihenfolge
2. inject: Injiziert klar inkohärenten Satz

Output:
- results/evaluation/coherence_stress/<run_id>/
  - stress_results.jsonl (pro Beispiel original/perturbed scores + delta)
  - summary.json + summary.md (success_rate, mean_delta, median_delta, CIs optional)
"""

import argparse
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from app.llm.openai_client import OpenAIClient
from app.services.agents.coherence.coherence_agent import CoherenceAgent

load_dotenv()


# ---------------------------
# IO helpers
# ---------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Ungültiges JSONL in {path} @ Zeile {line_no}: {e}") from e
    return rows


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ---------------------------
# Sentence splitting
# ---------------------------

def split_sentences(text: str) -> List[str]:
    """
    Einfacher Satz-Splitter (für Demo-Zwecke ausreichend).
    Nutzt Punkt + Leerzeichen als Trenner.
    """
    # Einfache Regex: Punkt gefolgt von Leerzeichen oder Ende
    sentences = re.split(r'\.\s+', text)
    # Entferne leere Sätze und füge Punkt wieder hinzu (außer letzter)
    cleaned = []
    for i, sent in enumerate(sentences):
        sent = sent.strip()
        if sent:
            if i < len(sentences) - 1:
                sent += "."
            cleaned.append(sent)
    return cleaned if cleaned else [text]  # Fallback: ganzer Text als ein Satz


# ---------------------------
# Perturbation functions
# ---------------------------

def shuffle_sentences(summary: str, seed: Optional[int] = None) -> str:
    """Permutiert Satzreihenfolge."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    sentences = split_sentences(summary)
    if len(sentences) <= 1:
        return summary  # Keine Permutation möglich
    rng.shuffle(sentences)
    return " ".join(sentences)


def inject_incoherent_sentence(summary: str, seed: Optional[int] = None) -> str:
    """Injiziert klar inkohärenten Satz an zufälliger Position."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    sentences = split_sentences(summary)
    
    # Inkohärenter Satz
    incoherent_sentences = [
        "Übrigens, das hat nichts damit zu tun: Gestern war es sehr sonnig.",
        "Nebenbei bemerkt, dies ist völlig unzusammenhängend mit dem Rest.",
        "Apropos, hier ist eine zufällige Information: Die Hauptstadt von Madagaskar ist Antananarivo.",
        "Zwischenbemerkung: Dies passt überhaupt nicht zum Kontext.",
    ]
    injected = rng.choice(incoherent_sentences)
    
    # Position: zufällig zwischen Sätzen
    if len(sentences) == 0:
        return injected
    pos = rng.randint(0, len(sentences))
    sentences.insert(pos, injected)
    return " ".join(sentences)


# ---------------------------
# Core stress test
# ---------------------------

def run_stress_test(
    rows: List[Dict[str, Any]],
    mode: str,  # "shuffle" oder "inject"
    llm_model: str,
    max_examples: Optional[int],
    results_path: Path,
    retries: int,
    sleep_s: float,
    seed: Optional[int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Führt Stress-Test durch.

    Returns:
        (summary_dict, results_list)
    """
    if seed is not None:
        random.seed(seed)

    llm = OpenAIClient(model_name=llm_model)
    agent = CoherenceAgent(llm)

    results_list: List[Dict[str, Any]] = []
    n_seen = 0
    n_used = 0
    n_failed = 0
    n_skipped = 0

    if results_path.exists():
        results_path.unlink()

    deltas: List[float] = []
    successes: List[bool] = []

    for row in rows:
        if max_examples is not None and n_used >= max_examples:
            break

        n_seen += 1

        article = row.get("article")
        summary_original = row.get("summary")
        meta = row.get("meta", {})
        example_id = meta.get("doc_id") or meta.get("id") or f"example_{n_seen}"

        if not isinstance(article, str) or not article.strip() or not isinstance(summary_original, str) or not summary_original.strip():
            n_skipped += 1
            continue

        # Perturbation
        if mode == "shuffle":
            summary_perturbed = shuffle_sentences(summary_original, seed=seed + n_seen if seed is not None else None)
        elif mode == "inject":
            summary_perturbed = inject_incoherent_sentence(summary_original, seed=seed + n_seen if seed is not None else None)
        else:
            raise ValueError(f"Unbekannter Modus: {mode}")

        # Original score
        score_original: Optional[float] = None
        for attempt in range(retries + 1):
            try:
                res_orig = agent.run(article_text=article, summary_text=summary_original, meta=meta)
                score_original = float(res_orig.score)
                break
            except Exception as e:
                if attempt < retries:
                    time.sleep(sleep_s)
                else:
                    n_failed += 1
                    print(f"Fehler bei Original für {example_id}: {e}")
                    continue

        if score_original is None:
            continue

        # Perturbed score
        score_perturbed: Optional[float] = None
        for attempt in range(retries + 1):
            try:
                res_pert = agent.run(article_text=article, summary_text=summary_perturbed, meta=meta)
                score_perturbed = float(res_pert.score)
                break
            except Exception as e:
                if attempt < retries:
                    time.sleep(sleep_s)
                else:
                    n_failed += 1
                    print(f"Fehler bei Perturbed für {example_id}: {e}")
                    continue

        if score_perturbed is None:
            continue

        # Delta: original sollte besser sein (höherer Score = besser)
        delta = score_original - score_perturbed
        success = delta > 0  # Original besser als perturbed

        deltas.append(delta)
        successes.append(success)
        n_used += 1

        rec = {
            "example_id": example_id,
            "mode": mode,
            "score_original": score_original,
            "score_perturbed": score_perturbed,
            "delta": delta,
            "success": success,
        }
        results_list.append(rec)

        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if n_used % 25 == 0:
            print(f"[{n_used}] delta={delta:.3f} (orig={score_original:.3f}, pert={score_perturbed:.3f}, success={success})")

    # Summary statistics
    if not deltas:
        return {
            "n_seen": n_seen,
            "n_used": 0,
            "n_skipped": n_skipped,
            "n_failed": n_failed,
            "mode": mode,
            "success_rate": 0.0,
            "mean_delta": 0.0,
            "median_delta": 0.0,
        }, results_list

    success_rate = sum(successes) / len(successes) if successes else 0.0
    mean_delta = sum(deltas) / len(deltas)
    sorted_deltas = sorted(deltas)
    median_delta = sorted_deltas[len(sorted_deltas) // 2]

    summary = {
        "n_seen": n_seen,
        "n_used": n_used,
        "n_skipped": n_skipped,
        "n_failed": n_failed,
        "mode": mode,
        "success_rate": success_rate,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "min_delta": min(deltas),
        "max_delta": max(deltas),
    }
    return summary, results_list


# ---------------------------
# Output generation
# ---------------------------

def write_summary_json(summary: Dict[str, Any], out_path: Path) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def write_summary_md(summary: Dict[str, Any], out_path: Path) -> None:
    mode_name = summary["mode"].upper()
    lines = [
        f"# Coherence Stress Test Summary ({mode_name})",
        "",
        f"**Mode:** {mode_name}",
        f"**Examples used:** {summary['n_used']}",
        f"**Examples skipped:** {summary['n_skipped']}",
        f"**Examples failed:** {summary['n_failed']}",
        "",
        "## Results",
        "",
        f"- **Success rate:** {summary['success_rate']:.2%} (Anteil, wo original > perturbed)",
        f"- **Mean Δ:** {summary['mean_delta']:.4f} (original - perturbed)",
        f"- **Median Δ:** {summary['median_delta']:.4f}",
        f"- **Min Δ:** {summary['min_delta']:.4f}",
        f"- **Max Δ:** {summary['max_delta']:.4f}",
        "",
        "**Interpretation:**",
        f"- Positive Δ bedeutet: Original besser als perturbed (erwartet)",
        f"- Success rate > 0.5 bedeutet: Agent erkennt Perturbationen in der Mehrzahl",
    ]

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_run_metadata(
    run_id: str,
    timestamp: str,
    data_path: Path,
    mode: str,
    llm_model: str,
    seed: Optional[int],
    n_total: int,
    n_used: int,
    n_failed: int,
    config_params: Dict[str, Any],
    out_path: Path,
) -> None:
    metadata = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "python_version": sys.version,
        "seed": seed,
        "dataset_path": str(data_path),
        "mode": mode,
        "llm_model": llm_model,
        "n_total": n_total,
        "n_used": n_used,
        "n_failed": n_failed,
        "config": config_params,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stress-Tests für Coherence-Agent")
    ap.add_argument("--data", type=str, required=True, help="Pfad zur JSONL-Datei")
    ap.add_argument("--mode", type=str, choices=["shuffle", "inject"], required=True, help="Stress-Test-Modus")
    ap.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM-Modell")
    ap.add_argument("--max_examples", type=int, help="Maximale Anzahl Beispiele")
    ap.add_argument("--max", type=int, help="Alias für --max_examples")
    ap.add_argument("--seed", type=int, help="Random seed für Reproduzierbarkeit")
    ap.add_argument("--out_dir", type=str, help="Output-Verzeichnis (default: results/evaluation/coherence_stress)")
    ap.add_argument("--retries", type=int, default=1, help="Anzahl Retries bei Fehlern")
    ap.add_argument("--sleep-s", type=float, default=1.0, help="Sleep zwischen Retries (Sekunden)")

    args = ap.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        ap.error(f"Datei nicht gefunden: {data_path}")

    max_examples = args.max_examples or args.max
    seed = args.seed

    rows = load_jsonl(data_path)

    # Run ID
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"coherence_stress_{args.mode}_{ts}_{args.model}"
    if seed is not None:
        run_id += f"_seed{seed}"

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir) / run_id
    else:
        out_dir = Path("results") / "evaluation" / "coherence_stress" / run_id
    ensure_dir(out_dir)

    results_path = out_dir / "stress_results.jsonl"
    summary_json_path = out_dir / "summary.json"
    summary_md_path = out_dir / "summary.md"
    metadata_path = out_dir / "run_metadata.json"

    print(f"Data: {data_path} (rows={len(rows)})")
    print(f"Mode: {args.mode}")
    print(f"Model: {args.model}")
    print(f"Seed: {seed}")
    print(f"Output: {out_dir}")

    summary, results_list = run_stress_test(
        rows=rows,
        mode=args.mode,
        llm_model=args.model,
        max_examples=max_examples,
        results_path=results_path,
        retries=args.retries,
        sleep_s=args.sleep_s,
        seed=seed,
    )

    # Write outputs
    write_summary_json(summary, summary_json_path)
    write_summary_md(summary, summary_md_path)
    write_run_metadata(
        run_id=run_id,
        timestamp=ts,
        data_path=data_path,
        mode=args.mode,
        llm_model=args.model,
        seed=seed,
        n_total=len(rows),
        n_used=summary["n_used"],
        n_failed=summary["n_failed"],
        config_params={
            "max_examples": max_examples,
            "retries": args.retries,
            "sleep_s": args.sleep_s,
        },
        out_path=metadata_path,
    )

    print("\n" + "=" * 60)
    print("Stress-Test abgeschlossen!")
    print("=" * 60)
    print(f"\nErgebnisse ({args.mode}):")
    print(f"  Success rate: {summary['success_rate']:.2%}")
    print(f"  Mean Δ:       {summary['mean_delta']:.4f}")
    print(f"  Median Δ:     {summary['median_delta']:.4f}")
    print(f"\nArtefakte gespeichert in: {out_dir}")


if __name__ == "__main__":
    main()

