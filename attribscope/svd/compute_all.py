"""fit SVD + score in one call, return DataFrame."""
from __future__ import annotations

from collections import defaultdict
from itertools import product as iproduct
from pathlib import Path
from typing import Callable

import json
import pandas as pd
import torch
from tqdm import tqdm
from safetensors import safe_open

from attribscope.svd.core import projection_svd, reconstruction_svd
from attribscope.svd.utils import compute_metrics, get_all_rep_names
from attribscope.svd.utils import (
    RepresentationStore,
    RepresentationStores,
    StepIndex,
    StoreKeeper
)


def load_representations(
    base_dir:     Path, # e,g, outputs/ or /data/username/attrib
    rep_dir:      Path, # e.g, grads/llama-3.1-8b/reps/{subset}
                        # only contains .safetensors files
    subset:       str,  # hand-crafted | algorithm-generated
    pooling:      list[str] | str,        # list of poolings to load, or "all"
    weight_names: list[str] | str,        # list of shorthand names, or "all"
    data_dir:     Path,
    device:       torch.device,
    files:        list[Path] | None = None, # load specific files within base_dir / rep_dir
) -> RepresentationStores:
    """Load per-trajectory safetensors files and build one RepresentationStore
    per (pooling, weight_name) pair, sharing a single StoreKeeper."""

    # Resolve files
    input_dir = base_dir / rep_dir
    if not files:
        files = sorted(input_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
        assert files, f"No .safetensors files in {input_dir}"
    else:
        files = [input_dir / file for file in files]
        assert all(file.exists() for file in files), "Invalid file list."
        assert all(file.name.endswith("safetensors") for file in files), \
            "Only accept .safetensors files."

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
    for fp in tqdm(files, desc=f"Loading [{subset}] at {input_dir}"):
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
                    # assert key in f.keys(), f"Missing key {key} in {fp}"
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
        ) for w, tensors in collections.items()
    }

    keeper = StoreKeeper(
        index=index, lookup=lookup,
        traj_meta=traj_meta, traj_ranges=traj_ranges,
        device=device,
    )

    return RepresentationStores(stores=stores, keeper=keeper)

# ── Tunables ──────────────────────────────────────────────────────────────────

SCORING_FNS: dict[str, Callable] = {
    "proj":  projection_svd,
    # "recon": reconstruction_svd,
}

DEFAULT_POOLING = {
    "hidden": ["mean", "last"],
    "grads":  ["grad"],
}


# ── Path helpers ──────────────────────────────────────────────────────────────

def resolve_model_tag(
    model: str, 
    loss: str,
    temperature: float
):
    if   loss == "ntp": return f"{model}"
    elif loss == "kl_temp":
        return f"{model}-kl/temp_{temperature}"
    elif loss == "kl_uniform":
        return f"{model}-kl/uniform"
    else:
        raise ValueError(f"Not support {loss}")


# ── SVD fit (in-memory) ───────────────────────────────────────────────────────

def fit_svd(G: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k right singular vectors of G.  Shape: (d, k)."""
    torch.manual_seed(100)
    _, _, V = torch.svd_lowrank(G.float(), q=k, niter=10)
    return V.contiguous()


def fit_all(stores, n_components: int) -> dict:
    """
    Fit raw + centered SVD for every (pooling, weight).

    Returns
    -------
    {
      pooling: {
        weight_name: {
          "V_raw":      Tensor(d, n_components),
          "V_centered": Tensor(d, n_components),
          "ref":        Tensor(d,),          # per-weight mean (=SAL reference)
        }
      }
    }
    """
    by_pooling: dict[str, dict] = defaultdict(dict)
    for s in stores.values():
        by_pooling[s.pooling][s.name] = s.R

    out = {}
    for pooling, tensors in by_pooling.items():
        out[pooling] = {}
        for name in tqdm(sorted(tensors), desc=f"SVD [{pooling}]"):
            G    = tensors[name].float()
            mean = G.mean(dim=0)
            out[pooling][name] = {
                "V_raw":      fit_svd(G,        n_components).cpu(),
                "V_centered": fit_svd(G - mean, n_components).cpu(),
                "ref":        mean.cpu(),
            }
    return out


# ── Scoring ───────────────────────────────────────────────────────────────────

def get_mistake_meta(keeper: StoreKeeper):
    indices, roles = [], []
    for start, end in keeper.traj_ranges:
        entry = next((e for e in keeper.index[start:end] if e.is_mistake), None)
        indices.append(entry.step_idx if entry else None)
        roles.append(keeper.traj_meta[entry.traj_idx].get("mistake_agent") if entry else None)
    return indices, roles


def score_all(stores, svd, keeper, n_components_score, ks, device) -> list[dict]:
    """Score every (weight × method × c × centered × direction × k) combination."""
    mistake_indices, mistake_roles = get_mistake_meta(keeper)
    rows = []

    for s in stores.values():
        if s.pooling not in svd or s.name not in svd[s.pooling]:
            continue
        W = svd[s.pooling][s.name]
        R = s.R  # (T, d), already on device

        for centered, (method, fn), c in iproduct(
            (True, False), SCORING_FNS.items(), n_components_score
        ):
            V   = W["V_centered" if centered else "V_raw"].to(device)
            ref = W["ref"].to(device) if centered else None
            scores = fn(R, V, c=c, ref=ref).cpu().numpy()

            for direction in ("asc", "desc"):
                m = compute_metrics(scores, keeper, mistake_indices, mistake_roles,
                                    ks, direction)
                for k in ks:
                    rows.append({
                        "weight":    s.name,
                        "pooling":   s.pooling,
                        "method":    method,
                        "c":         c,
                        "centered":  centered,
                        "k":         k,
                        "direction": direction,
                        "step_acc":  m[f"step@{k}_{direction}"],
                        "agent_acc": m[f"agent@{k}_{direction}"],
                    })
    return pd.DataFrame(rows)


# ── Main entry point ──────────────────────────────────────────────────────────

def run_pipeline(
    model:  str,
    subset: str,
    loss:   str,   # "ntp" | "kl_uniform" | "kl_temp"
    rep:    str,   # "hidden" | "grads"
    *,
    n_components:       int              = 10,
    n_components_score: list[int] | str  = "all",
    ks:                 list[int]        = (1, 3, 5, 10),
    pooling:            list[str] | None = None,
    temperature:        float | None     = None,
    outputs_root:       Path             = Path("outputs"),
    data_dir:           Path             = Path("data/ww"),
    device:             str | None       = None,
) -> pd.DataFrame:
    """
    File structure for .
    .
    ├── grads
    │   ├── llama-3.1-8b
    │   ├── llama-3.1-8b-kl
    │   │   ├── temp_1.x
    │   │   └── uniform
    │   ├── qwen3-8b
    │   └── qwen3-8b-kl
    │       ├── temp_1.x
    │       └── uniform
    └── hidden
        ├── llama-3.1-8b
        └── qwen3-8b
    """
    dev      = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ks       = list(ks)
    poolings = pooling or DEFAULT_POOLING[rep]
    bdir     = outputs_root / rep

    if n_components_score == "all":
        n_components_score = list(range(1, n_components + 1))

    all_rows = []
    for pool in poolings:
        """
        For load_and_stack(), some artifacts:
        input_dir = "grads/llama-3.1-8b/reps/subset
                           ------------
                           model_tag, or                   
        input_dir = "grads/llama-3.1-8b-kl/temp_1.2/reps/subset
                           ------------------------
                           this as well for kl
        """
        model_tag = resolve_model_tag(model, loss, temperature)
        rep_dir = Path(model_tag) / "reps"
        reps = load_representations(
            base_dir     = bdir,
            rep_dir      = rep_dir,
            subset       = subset,
            pooling      = pool,
            weight_names = "all",
            data_dir     = data_dir / subset,
            
            device       = dev,
        )
        svd  = fit_all(reps.stores, n_components)
        rows = score_all(reps.stores, svd, reps.keeper,
                         n_components_score, ks, dev)
        all_rows.extend(rows)

        del reps, svd
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_rows)
    df.insert(0, "model",  model)
    df.insert(1, "subset", subset)
    df.insert(2, "loss",   loss)
    df.insert(3, "temperature", temperature)
    df.insert(3, "rep",    rep)
    return df