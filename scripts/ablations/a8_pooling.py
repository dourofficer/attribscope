"""A8 / appendix pooling ablation — last-token vs mean pooling of the step representation.

`main/` freezes pooling to `mean`, but the extractor wrote both poolings, so the
last-token variant is a CPU-only re-selection. For every without-GT cell (both
backbones x {WW-AG, WW-HC, TE-Cap, TE-Mag}) the full configuration is re-selected on
the `last` stores exactly as Table 1 selected it on the `mean` stores: dense base
grid (position x band), then the backprop rescore grid on the winning base config,
test-selected over the frozen triple by the standard rule. Each row is therefore
"the best that pooling can do", matching the optimistic protocol everywhere else.
The `mean` rows are copied from `results-nogt/<ds>/select/selection.tsv` (Table 1).

The grid machinery is A7's (`a7_datasize.py`) at fraction 1, with its module-level
pooling switched to `last`; A7 already proved that path reproduces Table 1 on `mean`.

Output: results-ablations/a8_pooling.tsv — base and soap rows per (backbone, subset,
pooling) with the selected configuration and val/test metrics.

    python scripts/ablations/a8_pooling.py --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a7_datasize as a7                                              # noqa: E402
from common import (CONFIGS_NOGT, RESULTS_DIR, anchor_rows, cell_paths,  # noqa: E402
                    iter_cells, load_selection, select_config)
from main import config as C                                          # noqa: E402
from main.rescore import aggregate_attn                               # noqa: E402

OUT = RESULTS_DIR / "a8_pooling.tsv"
CONFIG_AXES = ("position", "c_begin", "c_end", "layer_range", "gamma", "w")


def selection_rows(cfg, model, subset, seeds) -> list[dict]:
    """Table 1's mean-pooling rows, straight from the selection table."""
    svd_row, bp_row = anchor_rows(load_selection(cfg), model, subset)
    common = {"dataset": cfg["dataset"], "model": model, "subset": subset,
              "seeds": ",".join(map(str, seeds)), "pooling": "mean"}
    rows = []
    for name, r in (("base", svd_row), ("soap", bp_row)):
        rows.append({**common, "row": name,
                     **{ax: ("" if pd.isna(r[ax]) else r[ax]) for ax in CONFIG_AXES},
                     "step_acc_test": float(r["step_acc_test"]),
                     "agent_acc_test": float(r["agent_acc_test"]),
                     "step_acc_val": float(r["step_acc_val"]),
                     "agent_acc_val": float(r["agent_acc_val"])})
    return rows


def last_rows(cfg, model, subset, seeds, device) -> list[dict]:
    rep_dir, data_dir, files = cell_paths(cfg, model, subset)
    weightings, bounds = aggregate_attn(C.attn_root(cfg), model, subset,
                                        n_ranges=cfg["n_ranges"], device=device)
    labels = [f"{lo}-{hi}" for lo, hi in bounds]
    base_rows, _ = a7.base_grid_rows(cfg, model, subset, seeds, rep_dir, data_dir,
                                     files, 1.0, device)
    base = select_config(pd.DataFrame(base_rows), ["position", "c_begin", "c_end"],
                         seeds, "step_acc_test@1", "agent_acc_test@1")
    assert base is not None
    resc_rows = a7.rescore_grid_rows(cfg, model, subset, seeds, base["config"], rep_dir,
                                     data_dir, files, 1.0, weightings, labels, device)
    soap = select_config(pd.DataFrame(resc_rows), ["layer_range", "gamma", "w"], seeds,
                         "step_acc_test@1", "agent_acc_test@1")
    assert soap is not None
    print(f"[{cfg['dataset']}] {model}/{subset} pooling=last: base={base['config']} "
          f"{base['step']:.4f} soap={soap['config']} {soap['step']:.4f}")
    common = {"dataset": cfg["dataset"], "model": model, "subset": subset,
              "seeds": ",".join(map(str, seeds)), "pooling": "last",
              **{k: base["config"][k] for k in ("position", "c_begin", "c_end")}}
    return [{**common, "row": "base", "layer_range": "", "gamma": 0.0, "w": "",
             "step_acc_test": base["step"], "agent_acc_test": base["agent"],
             "step_acc_val": base["step_val"], "agent_acc_val": base["agent_val"]},
            {**common, "row": "soap", **soap["config"],
             "step_acc_test": soap["step"], "agent_acc_test": soap["agent"],
             "step_acc_val": soap["step_val"], "agent_acc_val": soap["agent_val"]}]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--configs", nargs="+", default=CONFIGS_NOGT)
    p.add_argument("--models", nargs="+", default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args()

    a7.POOLING = "last"          # every a7 helper reads this module global at call time
    rows: list[dict] = []
    for cfg, model, subset in iter_cells(args.configs):
        if args.models and model not in args.models:
            continue
        seeds = C.seeds_for(cfg, subset)
        rows.extend(selection_rows(cfg, model, subset, seeds))
        rows.extend(last_rows(cfg, model, subset, seeds, args.device))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, sep="\t", index=False)
    print(f"wrote {out}  ({len(df)} rows)")
    df["cell"] = df["model"] + " " + df["subset"]
    piv = df.pivot_table(index=["pooling", "row"], columns="cell", values="step_acc_test") * 100
    print(f"\n=== step acc (%) ===\n{piv.round(2).to_string()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
