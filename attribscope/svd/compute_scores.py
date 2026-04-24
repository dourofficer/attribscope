"""
attribscope.svd.compute_scores
==============================

Score each step in a trajectory using SVD-based anomaly scores (projection or
reconstruction residual), then evaluate how well those scores rank the gold
mistake step.

Example
-------
    python -m attribscope.svd.compute_scores \
        --models qwen3-8b \
        --score-subsets hand-crafted \
        --fit-subset hand-crafted \
        --base-dir outputs/grads \
        --data-dir data/ww \
        --out-dir outputs/grads \
        --poolings grad \
        --n-components-fit 10 \
        --n-components-score all \
        --ks 1 3 5 10

CUDA_VISIBLE_DEVICES=1 python -m attribscope.svd.compute_scores \
        --models qwen3-8b \
        --score-subsets hand-crafted \
        --fit-subset hand-crafted \
        --base-dir outputs/hidden \
        --data-dir data/ww \
        --out-dir outputs/hidden \
        --poolings mean last \
        --n-components-fit 10 \
        --n-components-score all \
        --ks 1 3 5 10

Output layout
-------------
    {out-dir}/{model}/svd/metrics/{score_subset}/from_{fit_subset}/
        proj_mean_c5_centered/
            k1_asc.tsv
            k1_desc.tsv
            k3_asc.tsv
            ...
        recon_mean_c5_centered/
            ...
        proj_mean_c5_raw/
            ...

Each TSV has columns: weight, step_acc, agent_acc
"""
from __future__ import annotations

import argparse
from itertools import product as iproduct
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from safetensors import safe_open

from .core import projection_svd, reconstruction_svd
from .utils import (
    RepresentationStores,
    StoreKeeper,
    compute_metrics,
    load_and_stack,
)

SCORING_FNS: dict[str, Callable] = {
    "proj":  projection_svd,
    "recon": reconstruction_svd,
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_singular_vectors(
    base_dir:         Path,
    model:            str,
    fit_subset:       str,
    pooling:          str,
    n_components_fit: int,
    centered:         bool,
    device:           torch.device,
) -> dict[str, torch.Tensor]:
    """Load V from a previously fitted SVD.

    Returns a dict mapping weight_name -> (d, n_components_fit) tensor.
    """
    tag = "centered" if centered else "raw"
    artifact_dir  = (
        base_dir / model / "svd" / fit_subset
        / f"{pooling}_c{n_components_fit}_{tag}"
    )
    V_file = artifact_dir / "V.safetensors"
    assert V_file.exists(), f"Missing SVD file: {V_file}"
    with safe_open(V_file, framework="pt") as f:
        V =  {k: f.get_tensor(k).to(device) for k in f.keys()}
    
    result = {"V": V, "ref": None}
    if not centered: return result
    ref_file = artifact_dir / "ref.safetensors"
    assert ref_file.exists(), f"Missing reference file: {ref_file}"
    with safe_open(artifact_dir / "ref.safetensors", framework="pt") as f:
        refs = {k: f.get_tensor(k).to(device) for k in f.keys()}
        result["ref"] = refs
    return result

def get_evaluation_metadata(
    keeper: StoreKeeper,
) -> tuple[list[int | None], list[str | None]]:
    """Extract per-trajectory mistake step index and agent role from keeper."""
    mistake_indices: list[int | None] = []
    mistake_roles:   list[str | None] = []

    for start, end in keeper.traj_ranges:
        traj_index    = keeper.index[start:end]
        mistake_entry = next((e for e in traj_index if e.is_mistake), None)
        if mistake_entry is None:
            mistake_indices.append(None)
            mistake_roles.append(None)
        else:
            meta = keeper.traj_meta[mistake_entry.traj_idx]
            mistake_indices.append(mistake_entry.step_idx)
            mistake_roles.append(meta.get("mistake_agent"))

    return mistake_indices, mistake_roles


def save_results(df: pd.DataFrame, out_dir: Path, ks: list[int]) -> None:
    """Write one TSV per (k, direction) into ``out_dir``.

    Each file is named ``k{k}_{direction}.tsv`` and contains columns:
        weight, step_acc, agent_acc
    sorted by step_acc descending.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in ks:
        for direction in ("asc", "desc"):
            out              = df[["weight"]].copy()
            out["step_acc"]  = df[f"step@{k}_{direction}"]
            out["agent_acc"] = df[f"agent@{k}_{direction}"]
            out              = out.sort_values("step_acc", ascending=False).reset_index(drop=True)
            path             = out_dir / f"k{k}_{direction}.tsv"
            out.to_csv(path, sep="\t", index=False)
            print(f"    Saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# Modular sweep components
# ─────────────────────────────────────────────────────────────────────────────

def score_one_weight(
    weight_name:     str,
    R:               torch.Tensor,          # (T, d)
    V:               torch.Tensor,          # (d, n_components_fit)
    ref:             torch.Tensor | None,  # (d,) or None
    score_fn:        Callable,
    c:               int,
    centered:        bool,
    keeper:          StoreKeeper,
    mistake_indices: list[int | None],
    mistake_roles:   list[str | None],
    ks:              list[int],
) -> dict:
    """Compute scores for one weight and return a metrics row dict."""
    if centered: assert ref is not None, "Centered config requires reference vector"
    if not centered: assert ref is None, "Raw config shouldn't have reference vector"

    scores = score_fn(R, V, c=c, ref=ref)
    row: dict = {"weight": weight_name}
    for direction in ("asc", "desc"):
        row.update(compute_metrics(
            scores          = scores.cpu().numpy(),
            keeper          = keeper,
            mistake_indices = mistake_indices,
            mistake_roles   = mistake_roles,
            ks              = ks,
            direction       = direction,
        ))
    return row


def score_one_config(
    stores:          dict,                  # store_key -> RepresentationStore
    singulars:       dict[str, torch.Tensor],
    method:          str,
    score_fn:        Callable,
    c:               int,
    centered:        bool,
    pooling:         str,
    keeper:          StoreKeeper,
    mistake_indices: list[int | None],
    mistake_roles:   list[str | None],
    ks:              list[int],
    out_dir:         Path,
) -> None:
    """Score all weights for one (method, c, centered) config and save results."""
    tag         = "centered" if centered else "raw"
    scoring_tag = f"{method}_{pooling}_c{c}_{tag}"
    config_dir  = out_dir / scoring_tag
    print(f"\n  [{scoring_tag}]")

    rows: list[dict] = []
    for store in stores.values():
        if store.name not in singulars["V"]:
            print(f"    [skip] {store.name} not in singular vectors")
            continue
        row = score_one_weight(
            weight_name     = store.name,
            R               = store.R,
            V               = singulars["V"][store.name],
            ref             = singulars["ref"][store.name] \
                              if singulars["ref"] else None,
            score_fn        = score_fn,
            c               = c,
            centered        = centered,
            keeper          = keeper,
            mistake_indices = mistake_indices,
            mistake_roles   = mistake_roles,
            ks              = ks,
        )
        rows.append(row)

    if not rows:
        print("    No weights scored — skipping save.")
        return

    save_results(pd.DataFrame(rows), config_dir, ks)


def score_one_subset(
    model:        str,
    score_subset: str,
    pooling:      str,
    args:         argparse.Namespace,
    device:       torch.device,
) -> None:
    """Load representations for one (model, subset, pooling) and run all configs."""
    print(f"\n{'━' * 60}")
    print(f"  Model        : {model}")
    print(f"  Score subset : {score_subset}")
    print(f"  Fit subset   : {args.fit_subset}")
    print(f"  Pooling      : {pooling}")
    print(f"{'━' * 60}")

    representations: RepresentationStores = load_and_stack(
        model        = model,
        subset       = score_subset,
        pooling      = pooling,
        weight_names = "all",
        data_dir     = args.data_dir / score_subset,
        base_dir     = args.base_dir,
        device       = device,
    )
    stores          = representations.stores
    keeper          = representations.keeper
    mistake_indices, mistake_roles = get_evaluation_metadata(keeper)

    base_out = (
        args.out_dir / model / "svd" / "metrics"
        / score_subset / f"from_{args.fit_subset}"
    )

    for centered in (True, False):
        # try:
        singulars = load_singular_vectors(
            base_dir         = args.base_dir,
            model            = model,
            fit_subset       = args.fit_subset,
            pooling          = pooling,
            n_components_fit = args.n_components_fit,
            centered         = centered,
            device           = device,
        )
        # except AssertionError as exc:
        #     print(f"  [skip] {exc}")
        #     continue

        for method, score_fn in SCORING_FNS.items():
            for c in args.n_components_score:
                score_one_config(
                    stores          = stores,
                    singulars       = singulars,
                    method          = method,
                    score_fn        = score_fn,
                    c               = c,
                    centered        = centered,
                    pooling         = pooling,
                    keeper          = keeper,
                    mistake_indices = mistake_indices,
                    mistake_roles   = mistake_roles,
                    ks              = args.ks,
                    out_dir         = base_out,
                )

    del representations
    if device.type == "cuda":
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Sweep entry point
# ─────────────────────────────────────────────────────────────────────────────

def sweep(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    print(f"Device: {device}")
    for model, score_subset, pooling in iproduct(args.models, args.score_subsets, args.poolings):
        score_one_subset(model, score_subset, pooling, args, device)
    print("\nDone.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description     = "Score trajectories with SVD-based anomaly scores.",
        formatter_class = argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="Model tags, e.g. qwen3-8b llama-3.1-8b")
    p.add_argument("--score-subsets", nargs="+", required=True,
                   help="Subsets to score, e.g. hand-crafted algorithm-generated")
    p.add_argument("--fit-subset", required=True,
                   help="Subset used to fit the SVD (must have V.safetensors on disk)")
    p.add_argument("--base-dir", type=Path, required=True,
                   help="Root for both reps/ and svd/ directories")
    p.add_argument("--data-dir", type=Path, default=Path("data/ww"),
                   help="Who&When JSON root (for step role lookup)")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output root (defaults to --base-dir)")
    p.add_argument("--poolings", nargs="+", default=["mean"],
                   help="Pooling strategies to evaluate")
    p.add_argument("--n-components-fit", type=int,  default=10,
                   help="Number of singular vectors that were used when fitting")
    p.add_argument("--n-components-score", type=str,  nargs="+", default=["5"],
                   help="Number(s) of singular vectors to use when scoring")
    p.add_argument("--ks", type=int,  nargs="+", default=[1, 3, 5, 10],
                   help="Top-k values for step@k and agent@k metrics")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.out_dir is None:
        args.out_dir = args.base_dir

    if args.n_components_score == ["all"]:
        args.n_components_score = list(range(1, args.n_components_fit + 1))
        print("Using all n_components_score values:", args.n_components_score)
    else:
        args.n_components_score = [int(c) for c in args.n_components_score]

    invalid = [c for c in args.n_components_score if c > args.n_components_fit]
    if invalid:
        raise ValueError(
            f"--n-components-score values {invalid} exceed "
            f"--n-components-fit={args.n_components_fit}"
        )

    sweep(args)