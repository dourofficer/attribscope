"""
Ceiling performance sweep for failure attribution via SVD-based OOD detection.

Usage:
    python experiments/svd/pipeline.py
    python experiments/svd/pipeline.py --config experiments/svd/configs/pipeline.yaml

Outer loop: model × subset × rep × loss × mode × split_ratio × split_seed
Inner loop: handled by fit_all / score_all (components, centered, direction, k)
"""

from __future__ import annotations

import argparse
import random
from itertools import product as iproduct
from pathlib import Path

import pandas as pd
import torch
import yaml
from tqdm import tqdm

from attribscope.svd.compute_all import (
    fit_all,
    load_representations,
    resolve_model_tag,
    score_all,
)

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "pipeline.yaml"


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg["svd"]["n_components_score"] is None:
        n = cfg["svd"]["n_components_fit"]
        cfg["svd"]["n_components_score"] = list(range(1, n + 1))
    return cfg


def split_data(data: list, ratio: float, seed: int):
    data = data.copy()
    random.seed(seed)
    random.shuffle(data)
    i = int(len(data) * ratio)
    return data[:i], data[i:]


def get_output_path(
        base: Path, 
        model, 
        subset, 
        rep, 
        pooling, 
        loss, 
        temperature, 
        mode, 
        ratio, 
        seed
    ) -> Path:
    tag = loss if temperature is None else f"{loss}_{temperature}"
    if rep == "hidden": tag = pooling
    path = base / model / subset / rep / tag / mode
    path.mkdir(parents=True, exist_ok=True)
    return path / f"{ratio}_{seed}.tsv"


def run_sweep(cfg: dict):
    device   = torch.device(cfg["device"])
    out_base = Path(cfg["paths"]["out_base"])
    svd_cfg  = cfg["svd"]

    total = count_iterations(cfg)
    print(cfg["reps"])
    combos = iproduct(
        cfg["models"],
        cfg["subsets"],
        [(r["name"], r["pooling"]) for r in cfg["reps"]],        # (rep, pooling)
        cfg["losses"],              # [loss, temperature]
        cfg["modes"],
        cfg["split_ratios"],
        cfg["split_seeds"],
    )

    for model, subset, (rep, pooling), (loss, temperature), mode, ratio, seed \
        in tqdm(combos, total=total):

        # hidden states are loss-agnostic; skip redundant loss variants
        if rep == "hidden" and loss != "ntp":
            continue

        out_path = get_output_path(
            out_base, 
            model, 
            subset, 
            rep, 
            pooling, 
            loss, 
            temperature, 
            mode, 
            ratio, 
            seed
        )
        if out_path.exists():
            print(f"[skip] {out_path}")
            continue

        print(f"[run]  model={model} subset={subset} rep={rep} loss={loss} "
              f"temp={temperature} mode={mode} split=({ratio},{seed})")

        # --- paths ---
        model_tag = resolve_model_tag(model, loss, temperature or 0)
        rep_dir   = Path(model_tag) / "reps" / subset
        base_dir  = Path(cfg["paths"]["outputs_root"]) / rep
        data_dir  = Path(cfg["paths"]["data_root"]) / subset

        safetensors_dir = base_dir / rep_dir
        if not safetensors_dir.exists():
            print(f"[warn] not found: {safetensors_dir}")
            continue

        files = sorted(safetensors_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
        if not files:
            print(f"[warn] no files in {safetensors_dir}")
            continue

        # --- split + load ---
        val_files, test_files = split_data(files, ratio, seed)

        load_kwargs = dict(rep_dir=rep_dir, subset=subset, pooling=pooling,
                           weight_names="all", data_dir=data_dir, device=device)
        val_reps  = load_representations(base_dir=base_dir, files=val_files,  **load_kwargs)
        test_reps = load_representations(base_dir=base_dir, files=test_files, **load_kwargs)

        # --- fit (mode A: val, mode B: test) ---
        svd = fit_all((val_reps if mode == "A" else test_reps).stores, svd_cfg["n_components_fit"])

        # --- score ---
        score_kwargs = dict(svd=svd, n_components_score=svd_cfg["n_components_score"],
                            ks=svd_cfg["ks"], device=device)
        val_df  = score_all(val_reps.stores,  keeper=val_reps.keeper,  **score_kwargs)
        test_df = score_all(test_reps.stores, keeper=test_reps.keeper, **score_kwargs)

        # --- merge + save ---
        merge_on = ["weight", "pooling", "method", "c", "centered", "direction", "k"]
        merged = pd.merge(val_df, test_df, on=merge_on, suffixes=("|val", "|test"))
        merged.to_csv(out_path, sep="\t", index=False)
        print(f"[saved] {out_path}  ({len(merged)} rows)")


def load_all_results(base: Path) -> pd.DataFrame:
    """Concatenate all result TSVs with config columns parsed from the path."""
    frames = []
    for tsv in sorted(base.rglob("*.tsv")):
        model, subset, rep, loss_tag, mode, fname = tsv.relative_to(base).parts
        ratio, seed = fname.replace(".tsv", "").split("_")
        df = pd.read_csv(tsv, sep="\t").assign(
            model=model, subset=subset, rep=rep, loss_tag=loss_tag,
            mode=mode, split_ratio=float(ratio), split_seed=int(seed),
        )
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def count_iterations(cfg: dict) -> int:
    reps = [(r["name"], r["pooling"]) for r in cfg["reps"]]
    return sum(
        1 for _, _, (rep, _), (loss, _), *_
        in iproduct(cfg["models"], cfg["subsets"], reps,
                    cfg["losses"], cfg["modes"], cfg["split_ratios"], cfg["split_seeds"])
        if not (rep == "hidden" and loss != "ntp")
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = load_config(args.config)
    print(f"Total iterations: {count_iterations(config)}!")
    if True: run_sweep(config)