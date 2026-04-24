"""
attribscope.svd.utils
--------------------------------------------
Utility functions and data structures for 
loading, organizing, and saving representations 
and scores in the SVD-based evaluation pipeline.

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import pandas as pd
from tqdm import tqdm
from typing import Callable
import numpy as np

from safetensors import safe_open


# ─────────────────────────────────────────────────────────────────────────────
# Core data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class StepIndex:
    """Row-level metadata for one entry in a stacked representation matrix."""
    row:        int     # row index in R
    traj_idx:   int     # 1-based index into the loaded data list
    step_idx:   int     # step index within the trajectory
    role:       str     # e.g. "WebSurfer", "Orchestrator (thought)"
    is_mistake: bool    # whether this is the gold mistake step


@dataclass
class RepresentationStore:
    """A single (T, d) matrix for one (name, pooling) pair."""
    R:       torch.Tensor   # (T, d) stacked representations, one row per step
    name:    str            # weight name or layer name, e.g. "v/35"
    pooling: str            # "last" / "mean" for activations; "grad" / "l1_norm" / ... for gradients


@dataclass
class StoreKeeper:
    """Shared metadata across all stores in a RepresentationStores container.

    Every RepresentationStore in a given container indexes the same set of
    (traj_idx, step_idx) pairs in the same order, so book-keeping lives here,
    not per-store.
    """
    index:       list[StepIndex]
    lookup:      dict[tuple[int, int], int]
    traj_meta:   dict[int, dict]
    traj_ranges: list[tuple[int, int]]
    device:      torch.device


@dataclass
class RepresentationStores:
    """Container for multiple RepresentationStore, keyed by "{pooling}.{name}"."""
    stores: dict[str, RepresentationStore]
    keeper: StoreKeeper

# ─────────────────────────────────────────────────────────────────────────────
# Safetensors key helpers
#   Keys are formatted as "{step_idx}.{pooling}.{weight_name}".
#   Weight-name shorthand uses "/" (e.g. "v/35"), so splitting on "." with
#   maxsplit=2 cleanly separates the three components.
# ─────────────────────────────────────────────────────────────────────────────
def _parse_key(key: str) -> tuple[int, str, str]:
    step_str, pooling, weight_name = key.split(".", 2)
    return int(step_str), pooling, weight_name


def get_all_rep_names(fp: Path) -> list[str]:
    """Inspect the keys of a safetensors file to find all weight names."""
    with safe_open(fp, framework="pt") as f:
        names = set()
        for k in f.keys():
            step_idx, pooling, name = _parse_key(k)
            names.add(name)
    return sorted(names)

# ─────────────────────────────────────────────────────────────────────────────
# Main loader
# ─────────────────────────────────────────────────────────────────────────────
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

def load_and_stack(
    model:        str,
    subset:       str,
    pooling:      list[str] | str,        # list of poolings to load, or "all"
    weight_names: list[str] | str,        # list of shorthand names, or "all"
    data_dir:     Path,
    base_dir:     Path,
    device:       torch.device,
) -> RepresentationStores:
    """Load per-trajectory safetensors files and build one RepresentationStore
    per (pooling, weight_name) pair, sharing a single StoreKeeper."""
    input_dir = base_dir / model / "reps" / subset
    files = sorted(input_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    assert files, f"No .safetensors files in {input_dir}"

    # Resolve "all" using the first file as a reference.
    if weight_names == "all":
        weight_names = get_all_rep_names(files[0])
    assert isinstance(weight_names, list) \
        and all(isinstance(w, str) for w in weight_names)

    # Per-(pooling, weight) tensor accumulators
    collections = {w: [] for w in weight_names}

    index:       list[StepIndex]            = []
    lookup:      dict[tuple[int, int], int] = {}
    traj_meta:   dict[int, dict]            = {}
    traj_ranges: list[tuple[int, int]]      = []

    row = 0
    for fp in tqdm(files, desc=f"Loading [{model}/{subset}]"):
        traj_idx = int(fp.stem)  # /path/to/1.safetensors -> 1

        with safe_open(fp, framework="pt", device="cpu") as f:
            # --- trajectory metadata ---------------------------------------
            header       = f.metadata()
            metadata     = json.loads(header.get("payload_metadata", "{}"))
            mistake_step = int(metadata.get("mistake_step", -1))
            traj_meta[traj_idx] = metadata

            # --- step indices, deduplicated -------------------------------
            # Use the first (pooling, weight) pair as the pivot to enumerate
            # steps. Assumes every step has entries for all requested pairs.
            pivot_w = weight_names[0]
            suffix  = f".{pooling}.{pivot_w}"
            step_indices = sorted({
                int(k.split(".", 1)[0]) for k in f.keys() if k.endswith(suffix)
            })

            # --- roles from Who&When JSON ---------------------------------
            with open(data_dir / fp.with_suffix(".json").name) as jf:
                history = json.load(jf)["history"]

            # --- one row per step, one tensor per (p, w) pair -------------
            start_row = row
            for step_idx in step_indices:
                for w in weight_names:
                    key = f"{step_idx}.{pooling}.{w}"
                    collections[w].append(f.get_tensor(key))

                index.append(StepIndex(
                    row=row, traj_idx=traj_idx, step_idx=step_idx,
                    role=history[step_idx]["role"],
                    is_mistake=(step_idx == mistake_step),
                ))
                lookup[(traj_idx, step_idx)] = row
                row += 1

            traj_ranges.append((start_row, row))

    # Stack each collection and wrap in a RepresentationStore.
    stores = {
        w: RepresentationStore(
            R=torch.stack(tensors).to(device),
            name=w,
            pooling=pooling,
        )
        for w, tensors in collections.items()
    }

    keeper = StoreKeeper(
        index=index, lookup=lookup,
        traj_meta=traj_meta, traj_ranges=traj_ranges,
        device=device,
    )

    return RepresentationStores(stores=stores, keeper=keeper)


# ─────────────────────────────────────────────────────────────────────────────
# Results writer
# ─────────────────────────────────────────────────────────────────────────────
def save_results(df: pd.DataFrame, out_dir: Path, subset: str, ks: list[int]) -> None:
    """Split the wide evaluation DataFrame into per-(k, direction) TSV files.

    Output: {out_dir}/{subset}_k{k}_{direction}.tsv
    Columns: weight, step_acc, agent_acc
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in ks:
        for direction in ["asc", "desc"]:
            out = df[["weight"]].copy()
            out["step_acc"]  = df[f"step@{k}_{direction}"]
            out["agent_acc"] = df[f"agent@{k}_{direction}"]
            out = out.sort_values("step_acc", ascending=False).reset_index(drop=True)

            path = out_dir / f"{subset}_k{k}_{direction}.tsv"
            out.to_csv(path, sep="\t", index=False)
            print(f"Saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# Metric builders
# ─────────────────────────────────────────────────────────────────────────────
def standardize_role(role: str) -> str:
    if "orchestrator" in role.lower(): return "Orchestrator"
    else: return role

def compute_metrics(
    scores: np.ndarray,
    store: RepresentationStore,
    keeper: StoreKeeper,
    mistake_indices: list[int | None],  # absolute step_idx in history
    mistake_roles: list[str | None],
    ks: list[int],
    direction: str,
) -> dict:
    ascending    = (direction == "asc")
    total_trajs  = len(keeper.traj_ranges)
    step_hits    = {k: 0 for k in ks}
    agent_hits   = {k: 0 for k in ks}

    for (start, end), mistake_step, mistake_role in zip(
        keeper.traj_ranges, mistake_indices, mistake_roles
    ):
        if mistake_step is None:
            continue

        # Pair each entry with its score, then rank by score
        traj_entries = keeper.index[start:end]
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
    stores: RepresentationStores,
    keeper: StoreKeeper,
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

    for start, end in keeper.traj_ranges:
        traj_index  = keeper.index[start:end]
        mistake_entry = next((e for e in traj_index if e.is_mistake), None)
        mistake_role = keeper.traj_meta[mistake_entry.traj_idx]['mistake_agent']
        mistake_idx = mistake_entry.step_idx

        mistake_roles.append(mistake_role)
        mistake_indices.append(mistake_idx)

    # --- Phase 2: Evaluate predictions (parallelized over weights) ---
    results = []
    for weight_name, scores in tqdm(all_scores.items(), desc="Predicting"):
        row = {"weight": weight_name}
        for direction in ["asc", "desc"]:
            row |= compute_metrics(
                scores, 
                store, 
                keeper,
                mistake_indices, 
                mistake_roles, 
                ks, 
                direction
            )
        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values("step@1_asc", ascending=False).reset_index(drop=True)
    return df