from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Callable
from itertools import product as iproduct

from concurrent.futures import ThreadPoolExecutor
from .utils import GradientStore, load_and_stack, save_results

from .scoring import (
    make_mean_distance_scoring,
    make_coordinate_median_scoring,
    make_geometric_median_scoring,
    make_projection_scoring,
    make_reconstruction_scoring,
    make_knn_scoring,
)

def standardize_role(role: str) -> str:
    if "orchestrator" in role.lower(): return "Orchestrator"
    else: return role

def compute_metrics(
    scores: np.ndarray,
    store: GradientStore,
    mistake_indices: list[int | None],  # absolute step_idx in history
    mistake_roles: list[str | None],
    ks: list[int],
    direction: str,
) -> dict:
    ascending    = (direction == "asc")
    total_trajs  = len(store.traj_ranges)
    step_hits    = {k: 0 for k in ks}
    agent_hits   = {k: 0 for k in ks}

    for (start, end), mistake_step, mistake_role in zip(
        store.traj_ranges, mistake_indices, mistake_roles
    ):
        if mistake_step is None:
            continue

        # Pair each entry with its score, then rank by score
        traj_entries = store.index[start:end]
        traj_scores  = scores[start:end]
        step_scores  = [(entry.step_idx, entry.role, score) 
                        for entry, score in zip(traj_entries, traj_scores)]
        step_scores.sort(key=lambda x: x[2], reverse=not ascending)

        ranked_steps  = [step_idx for step_idx, _, _ in step_scores]
        ranked_roles  = [standardize_role(role) for _, role, _ in step_scores]
        mistake_rank  = ranked_steps.index(mistake_step) + 1  # 1-based ranking.

        for k in ks:
            if mistake_rank <= k:
                step_hits[k] += 1
            if mistake_role in ranked_roles[:k]:
                agent_hits[k] += 1

    return {
        **{f"step@{k}_{direction}":  step_hits[k]  / total_trajs for k in ks},
        **{f"agent@{k}_{direction}": agent_hits[k] / total_trajs for k in ks},
    }

def evaluate_weights(
    store: GradientStore,
    scoring_fn: Callable[[torch.Tensor], torch.Tensor],
    ks: list[int] = [1, 3, 5],
) -> pd.DataFrame:

    # --- Phase 1: Compute scores for all weights ---
    all_scores: dict[str, np.ndarray] = {}
    for weight_name, G in tqdm(store.Gs.items(), desc="Scoring"):
        all_scores[weight_name] = scoring_fn(G).cpu().numpy()

    # --- Precompute trajectory metadata ---
    mistake_indices: list[int | None] = []
    mistake_roles:   list[str | None] = []

    for start, end in store.traj_ranges:
        traj_index  = store.index[start:end]
        mistake_entry = next((e for e in traj_index if e.is_mistake), None)
        mistake_role = store.traj_meta[mistake_entry.traj_idx]['mistake_agent']
        mistake_idx = mistake_entry.step_idx

        mistake_roles.append(mistake_role)
        mistake_indices.append(mistake_idx)
        # mistake_roles.append(mistake_entry.role if mistake_entry else None)

    # --- Phase 2: Evaluate predictions (parallelized over weights) ---
    results = []
    for weight_name, scores in tqdm(all_scores.items(), desc="Predicting"):
        row = {"weight": weight_name}
        for direction in ["asc", "desc"]:
            row |= compute_metrics(
                scores, 
                store, 
                mistake_indices, 
                mistake_roles, 
                ks, 
                direction
            )
        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values("step@1_asc", ascending=False).reset_index(drop=True)
    return df
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Scoring-function registries
# Each dict maps a canonical name → zero-argument callable → scoring fn.
# ─────────────────────────────────────────────────────────────────────────────
 
METRICS = ["l1", "l2", "cosine"]

def build_central_functions() -> dict:
    """Family 1: central-tendency (mean, coord-median, geometric-median)
    × distance metric (l1, l2, cosine)."""
    fns = {}
 
    for metric in METRICS:
        fns[f"mean_dist_{metric}"]    = make_mean_distance_scoring(metric=metric)
        fns[f"coord_median_{metric}"] = make_coordinate_median_scoring(metric=metric)
        fns[f"geom_median_{metric}"]  = make_geometric_median_scoring(metric=metric)
 
    return fns
 
 
def build_svd_functions() -> dict:
    """Family 2: SVD-based (projection, reconstruction)
    × c ∈ {1..5} × centered ∈ {True, False}."""
    fns = {}
 
    for c, centered in iproduct(range(1, 10), [True, False]):
        tag = "cen" if centered else "raw"
        fns[f"proj_c{c}_{tag}"]  = make_projection_scoring(c=c, centered=centered)
        fns[f"recon_c{c}_{tag}"] = make_reconstruction_scoring(c=c, centered=centered)
 
    return fns
 
 
def build_gradnorm_functions() -> dict:
    """GradNorm: raw L1 / L2 norm of gradient vectors (no reference)."""
    return {
        "gradnorm_l1": (lambda G: G.float().norm(p=1, dim=1)),
        "gradnorm_l2": (lambda G: G.float().norm(p=2, dim=1)),
    }
 
 
def build_knn_functions() -> dict:
    """Family 3: kNN distance × k ∈ {1, 3, 5, 10, 20} × normalize ∈ {True, False}."""
    fns = {}
 
    for k, normalize in iproduct([1, 3, 5, 10, 20], [True, False]):
        tag = "norm" if normalize else "raw"
        fns[f"knn_k{k}_{tag}"] = make_knn_scoring(k=k, normalize=normalize)
 
    return fns
 
 
FAMILY_BUILDERS = {
    "central":  build_central_functions,
    "svd":      build_svd_functions,
    "gradnorm": build_gradnorm_functions,
    "knn":      build_knn_functions,
}
 
 
def build_scoring_functions(skip: list[str]) -> dict:
    fns = {}
    for family, builder in FAMILY_BUILDERS.items():
        if family in skip:
            print(f"  [skip] {family}")
            continue
        family_fns = builder()
        print(f"  [{family}] {len(family_fns)} configs")
        fns.update(family_fns)
    return fns
 
 
# ─────────────────────────────────────────────────────────────────────────────
# Sweep
# ─────────────────────────────────────────────────────────────────────────────
 
def sweep(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
 
    print("\nBuilding scoring functions …")
    scoring_functions = build_scoring_functions(skip=args.skip_families)
    print(f"  Total: {len(scoring_functions)} scoring configs\n")
 
    for model, subset in iproduct(args.models, args.subsets):
        print(f"\n{'━'*60}")
        print(f"  Model : {model}")
        print(f"  Subset: {subset}")
        print(f"{'━'*60}")
 
        store = load_and_stack(
            model=model,
            subset=subset,
            weight_names="all",
            data_dir=args.data_dir / subset,
            device=device,
            grad_dir=args.grad_dir,
        )
        out_dir = args.out_dir / model / "metrics"
 
        for name, fn in tqdm(scoring_functions.items(), desc="Scoring functions"):
            df = evaluate_weights(store, scoring_fn=fn, ks=args.ks)
            save_results(df, out_dir=out_dir / name, subset=subset, ks=args.ks)
 
        del store
        if device.type == "cuda":
            torch.cuda.empty_cache()
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
 
KNOWN_MODELS  = ["llama-3.1-8b", "qwen3-8b"]
KNOWN_SUBSETS = ["hand-crafted", "algorithm-generated"]
 
 
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep OOD scoring functions over gradient stores.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    # --- Target selection ---
    parser.add_argument(
        "--models", nargs="+", default=KNOWN_MODELS,
        metavar="MODEL",
        help="Model name(s) to evaluate.",
    )
    parser.add_argument(
        "--subsets", nargs="+", default=KNOWN_SUBSETS,
        metavar="SUBSET",
        help="Dataset subset(s) to evaluate.",
    )
 
    # --- Paths ---
    parser.add_argument(
        "--data-dir", type=Path, default=Path("ww"),
        metavar="DIR",
        help="Root directory of raw trajectory JSON files.",
    )
    parser.add_argument(
        "--grad-dir", type=Path, default=Path("outputs/grads"),
        metavar="DIR",
        help="Root directory of extracted gradient .safetensors files.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/grads"),
        metavar="DIR",
        help="Root directory for metric TSV output files.",
    )
 
    # --- Evaluation ---
    parser.add_argument(
        "--ks", nargs="+", type=int, default=[1, 3, 5, 10],
        metavar="K",
        help="Top-k values for step@k and agent@k metrics.",
    )
 
    # --- Family toggles ---
    parser.add_argument(
        "--skip-families", nargs="*", default=[],
        choices=list(FAMILY_BUILDERS.keys()),
        metavar="FAMILY",
        help=f"Scoring families to skip. Choices: {list(FAMILY_BUILDERS.keys())}",
    )
 
    return parser.parse_args()
 
 
if __name__ == "__main__":
    sweep(parse_args())