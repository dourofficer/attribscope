"""
python -m attribscope.svd.compute_scores \
    --models qwen3-8b \
    --score-subsets hand-crafted \
    --fit-subset hand-crafted \
    --base-dir outputs/grads \
    --data-dir data/ww \
    --out-dir outputs/grads \
    --n-components-fit 10 \
    --n-components-score 5 \
    --centered \
    --ks 1 3 5 10

"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file, load_file
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable
from itertools import product as iproduct

from concurrent.futures import ThreadPoolExecutor
from .utils import RepresentationStore, RepresentationStores, StoreKeeper
from .utils import load_and_stack, save_results, compute_metrics, evaluate_weights
from .core import (
    projection_svd, reconstruction_svd, 
    make_projection_scoring, make_reconstruction_scoring
)
 
# ─────────────────────────────────────────────────────────────────────────────
# Scoring-function registries
# Each dict maps a canonical name → zero-argument callable → scoring fn.
# ─────────────────────────────────────────────────────────────────────────────
 
def build_svd_functions() -> dict:
    """SVD-based (projection, reconstruction)
    × c ∈ {1..5} × centered ∈ {True, False}."""
    fns = {}
 
    for c, centered in iproduct(range(1, 10), [True, False]):
        tag = "cen" if centered else "raw"
        fns[f"proj_c{c}_{tag}"]  = make_projection_scoring(c=c, centered=centered)
        fns[f"recon_c{c}_{tag}"] = make_reconstruction_scoring(c=c, centered=centered)
 
    return fns


def load_singular_vectors(
    base_dir: Path,
    model: str,
    subset: str,
    pooling: str,
    n_components_fit: int,
    centered: bool,
):
    """Loads the V matrices from a fitted SVD."""
    centered_str =  f"{'centered' if centered else 'raw'}"
    svd_dir = base_dir / f"{model}/svd/{subset}"
    V_file = f"{pooling}_c{n_components_fit}_{centered_str}/V.safetensors"
    fp = svd_dir / V_file
    assert fp.exists(), f"Missing SVD file: {fp}"
    with safe_open(fp, framework="pt") as f:
        data = {k: f.get_tensor(k) for k in f.keys()}
    return data

 
# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────

def get_evaluation_metadata(keeper: StoreKeeper):
    # --- Precompute trajectory metadata ---
    mistake_indices: list[int | None] = []
    mistake_roles:   list[str | None] = []

    for start, end in keeper.traj_ranges:
        traj_index  = keeper.index[start:end]
        mistake_entry = next((e for e in traj_index if e.is_mistake), None)
        mistake_role = keeper.traj_meta[mistake_entry.traj_idx]['mistake_agent']
        mistake_idx = mistake_entry.step_idx

        mistake_roles.append(mistake_role)
        mistake_indices.append(mistake_idx)

    return mistake_indices, mistake_roles


def sweep(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    print("\nBuilding scoring functions …")
    scoring_functions = build_svd_functions()
    print(f"  Total: {len(scoring_functions)} scoring configs\n")
 
    for model, subset, pooling in \
        iproduct(args.models, args.subsets, args.poolings):

        print(f"\n{'━'*60}")
        print(f"  Model : {model}")
        print(f"  Subset: {subset}")
        print(f"  Pooling: {pooling}")
        print(f"{'━'*60}")
 
        representations = load_and_stack(
            model=model,
            subset=subset,
            pooling=pooling,
            weight_names="all",
            data_dir=args.data_dir / subset,
            base_dir=args.base_dir,
            device=device,
        )

        stores = representations.stores
        keeper = representations.keeper
        mistake_indices, mistake_roles = get_evaluation_metadata(keeper)

        out_dir = args.out_dir / model / "metrics" / subset

        for centered in [False, True]:
            singulars = load_singular_vectors(
                base_dir=args.base_dir,
                model=model,
                subset=subset,
                pooling=pooling,
                n_components_fit=args.n_components_fit,
                centered=centered,
            )
            for name, store in stores.items():
                V = singulars[name]
                R = store.R
                scores = projection_svd(
                    R, V, 
                    c=args.n_components_score, 
                    centered=centered
                )
                df = compute_metrics(
                    scores=scores.cpu().numpy(),
                    keeper=keeper,
                    mistake_indices=mistake_indices,
                    mistake_roles=mistake_roles,
                    ks=args.ks,
                    direction="asc"
                )

 
 
# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
 
KNOWN_MODELS  = ["llama-3.1-8b", "qwen3-8b"]
KNOWN_SUBSETS = ["hand-crafted", "algorithm-generated"]
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return parser.parse_args()
 
 
if __name__ == "__main__":
    sweep(parse_args())