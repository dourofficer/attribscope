"""fit SVD + score in one call, return DataFrame.

attribscope.svd.computation
"""
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

from attribscope.svd2.core import _run_svd, projection_svd
from attribscope.svd2.utils import (
    compute_metrics, 
    get_mistake_meta, 
    run_metrics
)
from attribscope.svd2.utils import (
    load_representations,
    RepresentationStore,
    RepresentationStores,
    StoreKeeper,
)
# Sweep helpers — assumed to live alongside the other utils. Move/import
# from wherever they actually reside in your project.

# ── Tunables ──────────────────────────────────────────────────────────────────

SCORING_FNS: dict[str, Callable] = {
    "proj": projection_svd,
}

DEFAULT_POOLING = {
    "hidden": ["mean", "last"],
    "grads":  ["grad"],
}


def fit_one(store: RepresentationStore, n_components: int) -> dict:
    """Fit raw + centered SVD for a single RepresentationStore.

    Returns
    -------
    {
      "V_raw":      Tensor(d, n_components),   # CPU
      "V_centered": Tensor(d, n_components),   # CPU
      "ref":        Tensor(d,),                # per-weight mean, CPU
    }
    """
    G    = store.R.float()
    mean = G.mean(dim=0)
    return {
        "V_raw":      _run_svd(G,        n_components).cpu(),
        "V_centered": _run_svd(G - mean, n_components).cpu(),
        "ref":        mean.cpu(),
    }


def fit_all(stores: dict, n_components: int) -> dict:
    """Fit raw + centered SVD for every store.

    Returns
    -------
    {pooling: {weight_name: {"V_raw", "V_centered", "ref"}}}
    """
    out: dict[str, dict] = defaultdict(dict)
    for s in tqdm(
        sorted(stores.values(), key=lambda s: (s.pooling, s.name)),
        desc="SVD fit",
    ):
        out[s.pooling][s.name] = fit_one(s, n_components)
    return out


# ── Scoring ───────────────────────────────────────────────────────────────────

def score_one(
    store:              RepresentationStore,
    svd_entry:          dict,       # {"V_raw", "V_centered", "ref"}
    n_components_score: list[int],
    device:             torch.device,
) -> list[dict]:
    """Compute anomaly scores for one store across all (centered × method × c) combos.

    Returns
    -------
    List of score records, one per combo:
        {"weight", "pooling", "method", "c", "centered", "scores": np.ndarray (T,)}
    """
    # TODO: add gradnorm computation in here.
    R    = store.R.to(device)
    rows = []
    for centered, (method, fn), c in iproduct(
        (True, False), SCORING_FNS.items(), n_components_score,
    ):
        V      = svd_entry["V_centered" if centered else "V_raw"].to(device)
        ref    = svd_entry["ref"].to(device) if centered else None
        scores = fn(R, V, c=c, ref=ref).cpu().numpy()
        rows.append({
            "weight":   store.name,
            "pooling":  store.pooling,
            "method":   method,
            "c":        c,
            "centered": centered,
            "scores":   scores,      # (T,) np.ndarray — no keeper needed here
        })
    return rows


def score_all(
    stores:             dict,
    svd:                dict,
    n_components_score: list[int],
    device:             torch.device,
) -> list[dict]:
    """Compute anomaly scores for every store.

    Returns
    -------
    Flat list of score records (see score_one).
    """
    records = []
    for s in stores.values():
        if s.pooling not in svd or s.name not in svd[s.pooling]:
            continue
        records.extend(score_one(
            store              = s,
            svd_entry          = svd[s.pooling][s.name],
            n_components_score = n_components_score,
            device             = device,
        ))
    return records

# ── Main entry point ──────────────────────────────────────────────────────────

def run_pipeline(
    val_reps:           RepresentationStores,
    test_reps:          RepresentationStores,
    mode:               str,        # "A" → fit on val, "B" → fit on test
    n_components_fit:   int,
    n_components_score: list[int],
    ks:                 list[int],
    device:             torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit SVD on one split and score both splits.

    Parameters
    ----------
    val_reps / test_reps : pre-loaded representation stores for each split.
    mode                 : "A" fits on val, "B" fits on test. The split that
                           was *also* used for fitting is the transductive
                           output; the other is inductive.
    n_components_fit     : number of singular vectors to retain when fitting.
    n_components_score   : list of c values to evaluate at scoring time.
    ks                   : top-k cutoffs for the accuracy metrics.
    device               : torch device the store tensors live on.

    Returns
    -------
    (val_df, test_df) — one row per (weight × method × c × centered × direction × k).
    """
    if mode not in ("A", "B"):
        raise ValueError(f"mode must be 'A' or 'B', got {mode!r}")

    fit_reps = val_reps if mode == "A" else test_reps
    svd      = fit_all(fit_reps.stores, n_components_fit)

    score_kwargs = dict(svd=svd, n_components_score=n_components_score, device=device)
    val_scores  = score_all(val_reps.stores,  **score_kwargs)
    test_scores = score_all(test_reps.stores, **score_kwargs)

    val_df  = run_metrics(val_scores,  keeper=val_reps.keeper,  ks=ks)
    test_df = run_metrics(test_scores, keeper=test_reps.keeper, ks=ks)

    merged_df = pd.merge(
        val_df, test_df, suffixes=('|val', '|test'),
        on=['weight', 'pooling', 'method', 'c', 'centered', 'direction', 'k'],
    )
    return merged_df