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

from safetensors.torch import save_file
from safetensors import safe_open


@dataclass
class StepIndex:
    """Row-level metadata for one entry in the stacked gradient matrix."""
    row:        int     # row index in G
    traj_idx:   int     # 1-based index into the loaded data list
    step_idx:   int     # step index within the trajectory
    role:       str     # e.g. "WebSurfer", "Orchestrator (thought)"
    is_mistake: bool    # whether this is the gold mistake step


@dataclass
class GradientStore:
    """
    Stores gradients for multiple weight names across the same set of steps.
    """
    # Maps: weight_name -> (T, d) matrix
    Gs: dict[str, torch.Tensor] 
    
    # Metadata is shared across all layers because step indices are identical
    index:       list[StepIndex]
    lookup:      dict[tuple[int, int], int]
    traj_meta:   dict[dict]
    traj_ranges: list[tuple[int, int]]
    device:      torch.device

    def __getitem__(self, weight_name: str) -> torch.Tensor:
        return self.Gs[weight_name]

    @property
    def layer_names(self) -> list[str]:
        return list(self.Gs.keys())


def get_all_weight_names(fp: Path):
    with safe_open(fp, framework="pt") as f:
        # return sorted({k.split(".", 1)[1] for k in f.keys()})
        return sorted({k.split(".", 2)[-1] for k in f.keys()})
        
def load_and_stack(
    model: str,
    subset: str,
    weight_names: list[str],  # List of layers to load, e.g. ["down/31", "up/31"]
    data_dir: Path,
    device: torch.device,
    grad_dir: Path,
):
    # input_dir = grad_dir / model / subset
    input_dir = grad_dir / model / "reps" / subset
    files = sorted(input_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    
    # Initialize containers for each requested layer
    if weight_names == "all":
        weight_names = get_all_weight_names(files[0])
        
    layer_collections = {name: [] for name in weight_names}
    index: list[StepIndex] = []
    lookup: dict[tuple[int, int], int] = {}
    traj_meta: dict[dict] = {}
    traj_ranges: list[tuple[int, int]] = []

    row = 0
    for file_idx, fp in enumerate(tqdm(files, desc="Loading Multi-Layers")):
        traj_idx = int(fp.stem) # /path/to/1.safetensors -> 1
        
        with safe_open(fp, framework="pt", device="cpu") as f:
            # Load metadata
            header = f.metadata()
            metadata = json.loads(header.get("payload_metadata", "{}"))
            mistake_step = int(metadata.get("mistake_step", -1))
            traj_meta[traj_idx] = metadata
            
            # Use the first requested layer to determine step indices 
            # (assuming all layers exist for all steps)
            first_layer = weight_names[0]
            step_keys = [k for k in f.keys() if k.endswith(f".{first_layer}")]
            step_indices = sorted([int(k.split(".")[0]) for k in step_keys])
            
            # Load matching JSON for roles
            with open(data_dir / fp.with_suffix(".json").name) as jf:
                traj_data = json.load(jf)
                history = traj_data['history']
                
            start_row = row
            for step_idx in step_indices:
                # 1. Collect tensors for EVERY requested layer at this step
                for name in weight_names:
                    # key = f"{step_idx}.{name}"
                    key = f"{step_idx}.grad.{name}"
                    layer_collections[name].append(f.get_tensor(key))
                
                # 2. Record index (only once per step)
                index.append(StepIndex(
                    row=row, traj_idx=traj_idx, step_idx=step_idx,
                    role=history[step_idx]["role"],
                    is_mistake=(step_idx == mistake_step),
                ))
                lookup[(traj_idx, step_idx)] = row
                row += 1

            traj_ranges.append((start_row, row))

    # Convert lists to stacked matrices and move to device
    Gs = {
        name: torch.stack(tensors).to(device) 
        for name, tensors in layer_collections.items()
    }

    return GradientStore(
        Gs=Gs, index=index, lookup=lookup,
        traj_meta=traj_meta, traj_ranges=traj_ranges,
        device=device,
    )

def save_results(df: pd.DataFrame, out_dir: Path, subset: str, ks: list[int]) -> None:
    """
    Splits the wide evaluation DataFrame into per-(k, direction) TSV files.

    Output: {out_dir}/metrics/{subset}_k{k}_{direction}.tsv
    Columns: weight, step_acc, agent_acc
    """
    metrics_dir = out_dir
    metrics_dir.mkdir(parents=True, exist_ok=True)

    for k in ks:
        for direction in ["asc", "desc"]:
            out = df[["weight"]].copy()
            out["step_acc"]  = df[f"step@{k}_{direction}"]
            out["agent_acc"] = df[f"agent@{k}_{direction}"]
            out = out.sort_values("step_acc", ascending=False).reset_index(drop=True)

            path = metrics_dir / f"{subset}_k{k}_{direction}.tsv"
            out.to_csv(path, sep="\t", index=False)
            print(f"Saved {path}")
