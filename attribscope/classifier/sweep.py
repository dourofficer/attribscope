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

from attribscope.classifier.classifier import MLPClassifier, infer, quick_eval, seed_everything, train
from attribscope.svd2.utils import (
    _resolve_dir,
    compute_metrics,
    get_mistake_meta,
    load_representations,
    split_data,
)

# ---------------------------------------------------------------------------
# Fixed config
# ---------------------------------------------------------------------------

REPS_ROOT: Path = Path("/data/hoang/attrib/outputs")
DATA_ROOT: Path = Path("data/ww")
OUT_ROOT:  Path = Path("outputs/temps/classifier")

REP_TYPE:     str        = "hidden"
POOLINGS:     list[str]  = ["last", "mean"]
WEIGHT_NAMES: str        = "all"
LOSS:         str        = "ntp"
TEMPERATURE:  float|None = None
DEVICE: torch.device     = torch.device("cuda")

# Classifier hyperparameters
HIDDEN_DIM:    int   = 512
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
RATIOS:  list[float] = [0.3, 0.5]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sort_key(s: str):
    if s == "embed":
        return (-1, 0, "")
    match = re.search(r"(\d+)", s)
    return (0, int(match.group(1)), s)


def run_one(model: str, subset: str, seed: int, ratio: float, pooling: str):
    print(f"\n{'='*60}")
    print(f"model={model}  subset={subset}  seed={seed}  ratio={ratio}  pooling={pooling}")
    print(f"{'='*60}")

    out_path = OUT_ROOT / model / subset / f"{pooling}_{ratio}_{seed}.tsv"
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

    # -- Split files --------------------------------------------------------
    files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
    assert files, f"No .safetensors files in {rep_dir}"
    train_files, test_files = split_data(files, ratio, seed)
    print(f"Train trajectories: {len(train_files)}  |  Test trajectories: {len(test_files)}")

    # -- Load representations -----------------------------------------------
    train_reps = load_representations(
        rep_dir=rep_dir, data_dir=data_dir, pooling=pooling,
        weight_names=WEIGHT_NAMES, device=DEVICE, files=train_files,
    )
    test_reps = load_representations(
        rep_dir=rep_dir, data_dir=data_dir, pooling=pooling,
        weight_names=WEIGHT_NAMES, device=DEVICE, files=test_files,
    )

    # -- Build labels -------------------------------------------------------
    y_train = torch.tensor(
        [idx.is_mistake for idx in train_reps.keeper.index], dtype=torch.float32, device=DEVICE
    )
    y_test = torch.tensor(
        [idx.is_mistake for idx in test_reps.keeper.index], dtype=torch.float32, device=DEVICE
    )

    # -- Precompute mistake meta (same for all layers) ----------------------
    train_mistake_indices, train_mistake_roles = get_mistake_meta(train_reps.keeper)
    test_mistake_indices,  test_mistake_roles  = get_mistake_meta(test_reps.keeper)

    # -- Layer sweep --------------------------------------------------------
    layer_idxs = sorted(train_reps.stores.keys(), key=sort_key)
    rows = []

    for layer_idx in layer_idxs:
        X_train = train_reps.stores[layer_idx].R
        X_test  = test_reps.stores[layer_idx].R

        seed_everything()
        clf, _ = train(
            X=X_train,
            y=y_train,
            input_dim=X_train.shape[1],
            hidden_dim=HIDDEN_DIM,
            seed=42,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            momentum=MOMENTUM,
            logging_steps=int(1e9),
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
            f"Test  Step@1: {test_step_acc:.4f}  Agent@1: {test_agent_acc:.4f}"
        )

        rows.append({
            "layer_idx":       layer_idx,
            "train_step_acc":  train_step_acc,
            "train_agent_acc": train_agent_acc,
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
    for model, subset, seed, ratio, pooling in itertools.product(
        MODELS, SUBSETS, SEEDS, RATIOS, POOLINGS
    ):
        run_one(model, subset, seed, ratio, pooling)