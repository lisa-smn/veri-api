import json
from pathlib import Path

paths = sorted(Path("results").rglob("run_*_gpt-4o-mini_*.json"))
for p in paths:
    d = json.loads(p.read_text(encoding="utf-8"))
    m = d["metrics"]
    args = d["args"]
    print(
        p,
        "| dataset:",
        d["dataset"],
        "| pv:",
        d["prompt_version"],
        "| mode:",
        args["decision_mode"],
        "| thr:",
        args["issue_threshold"],
        "| cutoff:",
        args["score_cutoff"],
        "| P/R/F1:",
        round(m["precision"], 3),
        round(m["recall"], 3),
        round(m["f1"], 3),
        "| bal_acc:",
        round(m["balanced_accuracy"], 3),
    )
