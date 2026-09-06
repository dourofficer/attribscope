"""Per-subset corpus statistics for the appendix's dataset table.

Reads the corpus JSON directly (`data/<ds>/<subset>/*.json`), so the numbers describe
the data every method sees, not one method's view of it. Per subset: trajectory
count, turns per trajectory (mean / median / max), scoreable steps (turns minus the
never-scored human question turn of hand-crafted trajectories), distinct agents per
trajectory, the gold decisive step's mean position (0-indexed) and its mean relative
position gold / (turns - 1), and how many trajectories place the decisive error on
the first turn. CE is listed per subset with a macro-average row, as the manuscript
reports it.

    python scripts/tables/dataset_stats.py            # writes tables/appendix_datasets.tsv
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tables" / "appendix_datasets.tsv"

# (column, dataset, subset, agent system)
SUBSETS = [("WW-AG", "ww", "algorithm-generated", "CaptainAgent"),
           ("WW-HC", "ww", "hand-crafted", "Magentic-One"),
           ("CE-arc", "correct-error", "arc", "CORRECT"),
           ("CE-gaia", "correct-error", "gaia", "CORRECT"),
           ("CE-hotpot", "correct-error", "hotpot", "CORRECT"),
           ("CE-math500", "correct-error", "math500", "CORRECT"),
           ("CE-mmlu_pro", "correct-error", "mmlu_pro", "CORRECT"),
           ("CE-musique", "correct-error", "musique", "CORRECT"),
           ("CE-wikimqa", "correct-error", "wikimqa", "CORRECT"),
           ("TE-Cap", "traceelephant", "captain", "CaptainAgent"),
           ("TE-Mag", "traceelephant", "magentic", "Magentic-One")]


def subset_stats(ds: str, subset: str) -> dict:
    files = sorted((REPO / "data" / ds / subset).glob("*.json"),
                   key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    files = [f for f in files if f.stem.isdigit()]
    turns, steps, agents, gold, rel, gold0 = [], [], [], [], [], 0
    for f in files:
        j = json.loads(f.read_text())
        h = j["history"]
        turns.append(len(h))
        # hand-crafted trajectories open with a human question turn that is never scored
        unscored = 1 if (ds == "ww" and subset == "hand-crafted") else 0
        steps.append(len(h) - unscored)
        agents.append(len({x.get("role") or x.get("name") or "" for x in h}))
        try:
            g = int(j["mistake_step"])
        except (TypeError, ValueError, KeyError):
            continue
        gold.append(g)
        rel.append(g / max(1, len(h) - 1))
        gold0 += int(g == 0)
    return {"n_traj": len(files),
            "turns_mean": st.mean(turns), "turns_median": st.median(turns),
            "turns_max": max(turns), "steps_mean": st.mean(steps),
            "agents_mean": st.mean(agents), "gold_mean": st.mean(gold),
            "gold_rel_mean": st.mean(rel), "gold_first_turn": gold0,
            "n_labeled": len(gold)}


def main() -> int:
    rows = []
    for column, ds, subset, system in SUBSETS:
        rows.append({"column": column, "dataset": ds, "subset": subset, "system": system,
                     **subset_stats(ds, subset)})
    df = pd.DataFrame(rows)
    ce = df[df.dataset == "correct-error"]
    ce_row = {"column": "CE (macro)", "dataset": "correct-error", "subset": "all",
              "system": "CORRECT", "n_traj": int(ce.n_traj.sum()),
              "n_labeled": int(ce.n_labeled.sum()), "gold_first_turn": int(ce.gold_first_turn.sum())}
    for c in ("turns_mean", "turns_median", "turns_max", "steps_mean", "agents_mean",
              "gold_mean", "gold_rel_mean"):
        ce_row[c] = float(ce[c].mean()) if c != "turns_max" else int(ce[c].max())
    df = pd.concat([df, pd.DataFrame([ce_row])], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, sep="\t", index=False, float_format="%.4f")
    print(df.round(2).to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
