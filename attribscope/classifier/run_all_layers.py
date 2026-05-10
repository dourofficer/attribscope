"""
sweep_classifier.py

Sweeps over MODEL x SUBSET x SEED x RATIO combinations.
For each combination, trains a classifier per layer and saves results to a TSV.
"""

import itertools
import re
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn

from attribscope.classifier.classifier import (
    MLPClassifier, 
    infer, 
    quick_eval, 
    seed_everything, 
    train
)

from attribscope.svd2.utils import (
    _resolve_dir,
    compute_metrics,
    get_mistake_meta,
    load_representations,
    split_data,
)

def downsample(X: torch.Tensor, y: torch.Tensor, weight: int = 1):
    """
    Undersample label 0 so that n(label 0) / n(label 1) = weight.
    All label 1 samples are kept.
    """
    idx_pos = (y == 1).nonzero(as_tuple=True)[0]
    idx_neg = (y == 0).nonzero(as_tuple=True)[0]

    n_neg_keep = min(len(idx_pos) * weight, len(idx_neg))
    idx_neg_sampled = idx_neg[torch.randperm(len(idx_neg))[:n_neg_keep]]

    idx_all = torch.cat([idx_pos, idx_neg_sampled])
    idx_all = idx_all[torch.randperm(len(idx_all))]  # shuffle

    return X[idx_all], y[idx_all]


def upsample(X: torch.Tensor, y: torch.Tensor, weight: int = 1):
    """
    Upsample label 1 (with replacement) so that n(label 0) / n(label 1) = weight.
    All label 0 samples are kept. If the current ratio is already <= weight,
    positives are left untouched (no downsampling of label 1).
    """
    idx_pos = (y == 1).nonzero(as_tuple=True)[0]
    idx_neg = (y == 0).nonzero(as_tuple=True)[0]

    n_pos_target = max(len(idx_neg) // weight, len(idx_pos))
    idx_pos_sampled = idx_pos[torch.randint(0, len(idx_pos), (n_pos_target,))]

    idx_all = torch.cat([idx_pos_sampled, idx_neg])
    idx_all = idx_all[torch.randperm(len(idx_all))]  # shuffle

    return X[idx_all], y[idx_all]

# ---------------------------------------------------------------------------
# Fixed config
# ---------------------------------------------------------------------------

REPS_ROOT: Path = Path("/data/hoang/attrib/outputs")
DATA_ROOT: Path = Path("data/ww")
OUT_ROOT:  Path = Path("outputs/temps/classifier2")

REP_TYPE:     str        = "hidden"
POOLINGS:     list[str]  = ["last", "mean"]
WEIGHT_NAMES: str        = "all"
LOSS:         str        = "ntp"
TEMPERATURE:  float|None = None
DEVICE: torch.device     = torch.device("cuda")

# Classifier hyperparameters
HIDDEN_DIM:    int   = 1024
EPOCHS:        int   = 300
BATCH_SIZE:    int   = 512
LEARNING_RATE: float = 0.05
WEIGHT_DECAY:  float = 3e-4
MOMENTUM:      float = 0.9

# ---------------------------------------------------------------------------
# Sweep config
# ---------------------------------------------------------------------------

MODELS:  list[str]   = ["llama-3.1-8b", "qwen3-8b"]
SUBSETS: list[str]   = ["algorithm-generated", "hand-crafted"]
SEEDS:   list[int]   = [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sort_key(s: str):
    if s == "embed": return (-1, 0, "")
    match = re.search(r"(\d+)", s)
    return (0, int(match.group(1)), s)


def run_one(model: str, subset: str, seed: int, pooling: str):
    print(f"\n{'='*60}")
    print(f"model={model}  subset={subset}  seed={seed}  pooling={pooling}")
    print(f"{'='*60}")

    out_path = OUT_ROOT / model / subset / f"{pooling}_{seed}.tsv"
    if out_path.exists():
        print(f"Already exists, skipping: {out_path}")
        return

    # -- Resolve dirs -------------------------------------------------------
    rep_dir  = _resolve_dir(
        root_dir=REPS_ROOT,
        model=model,
        subset=subset,
        rep_type=REP_TYPE,
        loss=LOSS,
        temperature=TEMPERATURE,
        dir_type="representations",
    )
    data_dir = DATA_ROOT / subset
    print(f"Representation dir: {rep_dir}")
    print(f"Data dir:           {data_dir}")

    # -- Split files --------------------------------------------------------
    files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    assert files, (f"No .safetensors files in {rep_dir}")
    train_files, test_files = split_data(files, 0.5, seed)
    train_files, val_files  = split_data(train_files, 0.8, seed)

    print(f"Total train trajectories: {len(train_files)}")
    print(f"Total val trajectories:   {len(val_files)}")
    print(f"Total test trajectories:  {len(test_files)}")

    # -- Load representations -----------------------------------------------
    rep_kwargs = dict(
        rep_dir=rep_dir,
        data_dir=data_dir,
        pooling=pooling,
        weight_names=WEIGHT_NAMES,
        device=DEVICE,
    )

    train_reps = load_representations(**rep_kwargs, files=train_files)
    val_reps   = load_representations(**rep_kwargs, files=val_files)
    test_reps  = load_representations(**rep_kwargs, files=test_files)

    # -- Build labels -------------------------------------------------------
    y_train = torch.tensor([idx.is_mistake for idx in train_reps.keeper.index], 
                           dtype=torch.float32, device=DEVICE)
    y_val   = torch.tensor([idx.is_mistake for idx in val_reps.keeper.index], 
                           dtype=torch.float32, device=DEVICE)
    y_test  = torch.tensor([idx.is_mistake for idx in test_reps.keeper.index], 
                           dtype=torch.float32, device=DEVICE)

    # -- Precompute mistake meta (same for all layers) ----------------------
    train_mistake_indices, train_mistake_roles = get_mistake_meta(train_reps.keeper)
    val_mistake_indices, val_mistake_roles     = get_mistake_meta(val_reps.keeper)
    test_mistake_indices, test_mistake_roles   = get_mistake_meta(test_reps.keeper)

    # -- Layer sweep --------------------------------------------------------
    layer_idxs = sorted(train_reps.stores.keys(), key=sort_key)
    rows = []

    for layer_idx in layer_idxs:
        X_train = train_reps.stores[layer_idx].R
        X_val   = val_reps.stores[layer_idx].R
        X_test  = test_reps.stores[layer_idx].R

        train_loader = DataLoader(TensorDataset(X_train, y_train), 
                                  batch_size=BATCH_SIZE, shuffle=True)
        val_loader   = DataLoader(TensorDataset(X_val, y_val),     
                                  batch_size=BATCH_SIZE, shuffle=False)

        seed_everything(seed)
        model = MLPClassifier(input_dim=X_train.shape[1], hidden_dim=HIDDEN_DIM)
        clf, _ = train(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            momentum=MOMENTUM,
            logging_steps=int(1e9),
            val_metric="f1",
            device=DEVICE,
        )

        # Train metrics
        train_scores = infer(clf, X_train, return_logits=False, device=DEVICE)
        train_metrics = compute_metrics(
            scores=train_scores, keeper=train_reps.keeper,
            mistake_indices=train_mistake_indices, mistake_roles=train_mistake_roles,
            ks=[1], direction="desc",
        )
        train_step_acc, train_agent_acc = list(train_metrics.values())

        # Validation metrics
        val_scores = infer(clf, X_val, return_logits=False, device=DEVICE)
        test_metrics = compute_metrics(
            scores=val_scores, keeper=val_reps.keeper,
            mistake_indices=val_mistake_indices, mistake_roles=val_mistake_roles,
            ks=[1], direction="desc",
        )
        val_step_acc, val_agent_acc = list(test_metrics.values())

        # Test metrics
        test_scores = infer(clf, X_test, return_logits=False, device=DEVICE)
        test_metrics = compute_metrics(
            scores=test_scores, keeper=test_reps.keeper,
            mistake_indices=test_mistake_indices, mistake_roles=test_mistake_roles,
            ks=[1], direction="desc",
        )
        test_step_acc, test_agent_acc = list(test_metrics.values())

        print(
            f"  Layer {layer_idx:>10} | "
            f"Train Step@1: {train_step_acc:.4f}  Agent@1: {train_agent_acc:.4f} | "
            f"Validation Step@1: {val_step_acc:.4f}  Agent@1: {val_agent_acc:.4f} | "
            f"Test  Step@1: {test_step_acc:.4f}  Agent@1: {test_agent_acc:.4f}"
        )

        rows.append({
            "layer_idx":       layer_idx,
            "train_step_acc":  train_step_acc,
            "train_agent_acc": train_agent_acc,
            "val_step_acc":    val_step_acc,
            "val_agent_acc":   val_agent_acc,
            "test_step_acc":   test_step_acc,
            "test_agent_acc":  test_agent_acc,
        })

    # -- Save ---------------------------------------------------------------
    df = pd.DataFrame(rows).sort_values("test_step_acc", ascending=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for model, subset, seed, pooling in itertools.product(
        MODELS, SUBSETS, SEEDS, POOLINGS
    ):
        run_one(model, subset, seed, pooling)