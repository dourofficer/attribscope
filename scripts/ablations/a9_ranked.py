"""A9 / appendix "accuracy at k" — step@k, agent@k and MRR for SOAP, OAT and StepFinder.

All three methods score every step of a trajectory, so a ranked-set comparison is
fair; the prompting judges emit one verdict and are left out. Two halves:

  * SOAP and its base score — re-scored at the Table-1 anchor of every cell (both
    backbones, all eleven subsets, both GT trees) with the sweep's exact primitives,
    then ranked at k in {1, 3, 5} plus the gold step's mean reciprocal rank. The k=1
    numbers must reproduce `results-{nogt,gt}/<ds>/select/selection.tsv`.
  * OAT and StepFinder — their prediction JSONs under
    `../attrib-prompting/outputs-rb-{nogt,gt}/` store per-step `scores` and
    `score_step_indices`, so the ranked metrics are read off the stored scores with
    no re-run. Ranking is descending with the earliest tied step first, as in
    `main.metrics`; a gold step the method never scored (OAT skips human and
    environment turns) counts as a miss at every k and contributes 0 to MRR; a
    missing prediction file counts as a miss. Mean over the frozen triple, then over
    the five training seeds, exactly as B1 does. The k=1 numbers must reproduce
    `results-ablations/b1_rb_baselines/by_cell.tsv`.

Output: results-ablations/a9_ranked.tsv — one row per (with_gt, dataset, subset,
backbone, method) with step@1/3/5, agent@1/3/5 and mrr on the test split.

    python scripts/ablations/a9_ranked.py --device cuda
    python scripts/ablations/a9_ranked.py --only baselines
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO, RESULTS_DIR, anchor_rows, assert_close, base_scores,  # noqa: E402
                    cell_paths, load_selection, position_load_names)
from main import config as C                                          # noqa: E402
from main.metrics import KeeperContext, compute_metrics_batch, standardize_role  # noqa: E402
from main.rescore import aggregate_attn, apply_strategy, build_W, coerce_w  # noqa: E402
from main.stores import load_representations, split_files             # noqa: E402

spec = importlib.util.spec_from_file_location("ev", REPO / "scripts/prompting/evaluate.py")
ev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ev)

KS = (1, 3, 5)
METRIC_COLS = [f"step@{k}" for k in KS] + [f"agent@{k}" for k in KS] + ["mrr"]
CONFIGS = ["configs-main/ww.yaml", "configs-main/ww-gt.yaml",
           "configs-main/traceelephant.yaml", "configs-main/traceelephant-gt.yaml",
           "configs-main/correct-error.yaml", "configs-main/correct-error-gt.yaml"]
BASELINE_JUDGES = ["qwen3.5-9b", "deepseek-8b"]
TRAIN_SEEDS = range(42, 47)
OUT = RESULTS_DIR / "a9_ranked.tsv"
B1_CELL = RESULTS_DIR / "b1_rb_baselines" / "by_cell.tsv"


# ── SOAP and base at the anchor ─────────────────────────────────────────────
def soap_cell(cfg, model, subset, device) -> list[dict]:
    seeds = C.seeds_for(cfg, subset)
    svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
    position = svd_row["position"]
    cb, ce = int(svd_row["c_begin"]), int(svd_row["c_end"])
    gamma, w = float(bp_row["gamma"]), coerce_w(bp_row["w"])
    rep_dir, data_dir, files = cell_paths(cfg, model, subset)
    members, names = position_load_names(rep_dir, files, position)

    r_idx, weightings = None, None
    if gamma > 0:
        weightings, bounds = aggregate_attn(C.attn_root(cfg), model, subset,
                                            n_ranges=cfg["n_ranges"], device=device)
        labels = [f"{lo}-{hi}" for lo, hi in bounds]
        r_idx = labels.index(str(bp_row["layer_range"]))

    acc = {m: {c: 0.0 for c in METRIC_COLS} for m in ("base", "soap")}
    for seed in seeds:
        parts = split_files(files, cfg["splits"], seed)
        train, test = (load_representations(rep_dir, data_dir, poolings=["mean"],
                                            weight_names=names, files=parts[sp],
                                            device=device)
                       for sp in ("train", "test"))
        s = base_scores(cfg, position, cb, ce, train, test, members)
        if gamma > 0:
            mats = {"backprop": build_W(test.keeper, weightings[r_idx], w, device)}
            s_tilde = apply_strategy(s, test.keeper, mats, "backprop", [gamma])[:, 0]
        else:
            s_tilde = s
        m = compute_metrics_batch(torch.stack([s, s_tilde]), None, KS,
                                  ctx=KeeperContext(test.keeper))
        for i, name in enumerate(("base", "soap")):
            for c in METRIC_COLS:
                acc[name][c] += float(m[c][i]) / len(seeds)
        del train, test
        if device == "cuda":
            torch.cuda.empty_cache()

    assert_close(acc["base"]["step@1"], float(svd_row["step_acc_test"]),
                 f"{model}/{subset} base step@1 vs selection")
    assert_close(acc["soap"]["step@1"], float(bp_row["step_acc_test"]),
                 f"{model}/{subset} soap step@1 vs selection")
    common = {"with_gt": bool(cfg["gt"]), "dataset": cfg["dataset"], "subset": subset,
              "backbone": model, "seeds": ",".join(map(str, seeds)), "n_runs": 1}
    return [{**common, "method": name, **acc[name]} for name in ("base", "soap")]


# ── OAT and StepFinder from their stored per-step scores ────────────────────
def ranked_hits(row: dict, roles: list[str]) -> dict[str, float]:
    """step@k / agent@k / rr for one trajectory from its stored per-step scores."""
    out = {c: 0.0 for c in METRIC_COLS}
    scores, idx = row.get("scores"), row.get("score_step_indices")
    if not scores or not idx:
        return out
    try:
        gold = int(row["gold_step"])
    except (TypeError, ValueError):
        return out
    gold_role = str(row.get("gold_agent") or "").strip().lower()
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))   # earliest tie first
    ranked = [idx[i] for i in order]
    ranked_roles = [standardize_role(roles[t]).strip().lower() if t < len(roles) else ""
                    for t in ranked]
    if gold in ranked:
        rank = ranked.index(gold) + 1
        out["mrr"] = 1.0 / rank
        for k in KS:
            out[f"step@{k}"] = float(rank <= k)
    if gold_role:
        for k in KS:
            out[f"agent@{k}"] = float(gold_role in ranked_roles[:k])
    return out


def trajectory_roles(data_dir: Path) -> dict[str, list[str]]:
    roles = {}
    for p in data_dir.glob("*.json"):
        if p.stem.isdigit():
            roles[p.stem] = [h.get("role", "") for h in json.loads(p.read_text())["history"]]
    return roles


def baseline_cells() -> list[dict]:
    b1 = pd.read_csv(B1_CELL, sep="\t")
    rows = []
    for root, with_gt, tree, suffix in ev.SETTINGS:
        root = root.replace("outputs", "outputs-rb")
        for ds in ev.DATASETS:
            cfg = C.load_config(REPO / "configs-main" / f"{ds}{suffix}.yaml")
            for subset in cfg["subsets"]:
                seeds = C.seeds_for(cfg, subset)
                ids_by_seed = {s: ev.test_ids(ds, subset, tree, cfg["splits"], s)
                               for s in seeds}
                roles = trajectory_roles(C.data_root(cfg) / subset)
                for judge in BASELINE_JUDGES:
                    for family in ("oat", "stepfinder"):
                        acc = {c: 0.0 for c in METRIC_COLS}
                        n_runs = 0
                        for ts in TRAIN_SEEDS:
                            method = f"{family}.s{ts}"
                            preds = ev.read_cell(root, ds, subset, judge, method)
                            if not preds:
                                print(f"  [miss] {root}/{ds}/{subset}/{judge}/{method}")
                                continue
                            n_runs += 1
                            run = {c: 0.0 for c in METRIC_COLS}
                            for seed in seeds:
                                ids = ids_by_seed[seed]
                                for i in ids:
                                    p = preds.get(str(i))
                                    if p is None:
                                        continue
                                    h = ranked_hits(p, roles.get(str(i), []))
                                    for c in METRIC_COLS:
                                        run[c] += h[c] / len(ids) / len(seeds)
                            ref = b1[(b1.with_gt == with_gt) & (b1.judge == judge)
                                     & (b1.method == method) & (b1.dataset == ds)
                                     & (b1.subset == subset)]
                            assert len(ref) == 1, (with_gt, judge, method, ds, subset)
                            assert_close(run["step@1"], float(ref.step_acc.iloc[0]),
                                         f"{judge}/{method}/{ds}/{subset} step@1 vs B1")
                            assert_close(run["agent@1"], float(ref.agent_acc.iloc[0]),
                                         f"{judge}/{method}/{ds}/{subset} agent@1 vs B1")
                            for c in METRIC_COLS:
                                acc[c] += run[c]
                        if n_runs == 0:
                            continue
                        rows.append({"with_gt": with_gt, "dataset": ds, "subset": subset,
                                     "backbone": judge, "seeds": ",".join(map(str, seeds)),
                                     "n_runs": n_runs, "method": family,
                                     **{c: acc[c] / n_runs for c in METRIC_COLS}})
    return rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cuda")
    p.add_argument("--only", choices=["soap", "baselines"], default=None)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    rows: list[dict] = []
    if args.only != "baselines":
        for cfg_path in CONFIGS:
            cfg = C.load_config(REPO / cfg_path)
            for model in cfg["models"]:
                if model not in BASELINE_JUDGES:
                    continue
                for subset in cfg["subsets"]:
                    cell = soap_cell(cfg, model, subset, args.device)
                    print(f"[{cfg['dataset']}{'-gt' if cfg['gt'] else ''}] {model}/{subset}: "
                          + "  ".join(f"{r['method']} step@1/3/5="
                                      f"{r['step@1']:.4f}/{r['step@3']:.4f}/{r['step@5']:.4f}"
                                      for r in cell))
                    rows.extend(cell)
    if args.only != "soap":
        rows.extend(baseline_cells())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}  ({len(df)} rows)")
    for (gt, bb), g in df.groupby(["with_gt", "backbone"]):
        g = g.copy()
        g["cell"] = g["dataset"] + "/" + g["subset"]
        for c in ("step@1", "step@3", "step@5"):
            piv = g.pivot_table(index="method", columns="cell", values=c) * 100
            print(f"\n=== with_gt={gt} backbone={bb} {c} (%) ===\n{piv.round(2).to_string()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
