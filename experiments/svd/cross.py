"""
Cross-dataset ceiling sweep.

Outer loop: model × rep × loss × mode × (val_subset, test_subset)
Inner loop: handled by fit_all / score_all (components, centered, direction, k)
"""

from itertools import product as iproduct
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from attribscope.svd.compute_all import (
    fit_all, load_representations, resolve_model_tag, score_all,
)

# ── Config ────────────────────────────────────────────────────────────────────

OUTPUTS_ROOT = Path("/data/hoang/attrib")
DATA_ROOT    = Path("data/ww")
DEVICE       = torch.device("cuda")
OUT_BASE     = Path("/data/hoang/attrib/cross")

N_COMPONENTS_FIT   = 10
N_COMPONENTS_SCORE = list(range(1, N_COMPONENTS_FIT + 1))
KS                 = [1, 3, 5, 10]

MODELS = [
    # "llama-3.1-8b", 
    "qwen3-8b"
]

REPS = [
    ("grads",  "grad"),
    ("hidden", "mean"),
    ("hidden", "last"),
]

LOSSES = [
    ("ntp",        None),
    ("kl_uniform", None),
    ("kl_temp",    1.2),
    ("kl_temp",    1.4),
    ("kl_temp",    1.6),
    ("kl_temp",    1.8),
    ("kl_temp",    1.8),
    ("kl_temp",    2.0),
    ("kl_temp",    2.2),
    ("kl_temp",    2.4),
    ("kl_temp",    2.8),
    ("kl_temp",    3.0),
]

MODES = [
    "A", 
    # "B",
]

CROSS_CONFIGS = [
    ("hand-crafted",        "algorithm-generated"),
    ("algorithm-generated", "hand-crafted"),
    ("tau-retail",          "algorithm-generated"),
    ("tau-retail",          "hand-crafted"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_output_path(
        model, rep, pooling, loss, 
        temperature, mode, val_subset, test_subset
) -> Path:
    tag = loss if temperature is None else f"{loss}_{temperature}"
    if rep == "hidden": tag = pooling
    path = OUT_BASE / model / rep / tag / mode
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{val_subset}_{test_subset}.tsv"


def load_all(subset, base_dir, rep_dir, pooling):
    files = sorted((base_dir / rep_dir).glob("*.safetensors"), key=lambda x: int(x.stem))
    return load_representations(
        base_dir=base_dir, rep_dir=rep_dir, subset=subset,
        pooling=pooling, weight_names="all", data_dir=DATA_ROOT / subset,
        device=DEVICE, files=files,
    )


# ── Sweep ─────────────────────────────────────────────────────────────────────

merge_on = ["weight", "pooling", "method", "c", "centered", "direction", "k"]

combos = list(iproduct(MODELS, REPS, LOSSES, MODES, CROSS_CONFIGS))

for model, (rep, pooling), (loss, temperature), mode, (val_subset, test_subset) in tqdm(combos):

    if rep == "hidden" and loss != "ntp":
        continue

    out_path = get_output_path(
        model, rep, pooling, loss, 
        temperature, mode, val_subset, test_subset
    )
    if out_path.exists():
        print(f"[skip] {out_path}")
        continue

    print(f"[run]  model={model} rep={rep} loss={loss} temp={temperature} "
          f"mode={mode} val={val_subset} test={test_subset}")

    model_tag = resolve_model_tag(model, loss, temperature or 0)
    base_dir  = OUTPUTS_ROOT / rep

    val_dir  = Path(model_tag) / "reps" / val_subset
    test_dir = Path(model_tag) / "reps" / test_subset

    for d in (base_dir / val_dir, base_dir / test_dir):
        if not d.exists():
            print(f"[warn] not found: {d}")
            break
    else:
        val_reps  = load_all(val_subset,  base_dir, val_dir,  pooling)
        test_reps = load_all(test_subset, base_dir, test_dir, pooling)

        svd     = fit_all((val_reps if mode == "A" else test_reps).stores, N_COMPONENTS_FIT)
        val_df  = score_all(val_reps.stores,  svd, val_reps.keeper,  N_COMPONENTS_SCORE, KS, DEVICE)
        test_df = score_all(test_reps.stores, svd, test_reps.keeper, N_COMPONENTS_SCORE, KS, DEVICE)

        merged = pd.merge(val_df, test_df, on=merge_on, suffixes=("|val", "|test"))
        merged.to_csv(out_path, sep="\t", index=False)
        print(f"[saved] {out_path}  ({len(merged)} rows)")