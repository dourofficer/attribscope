
"""
For each position (either hidden or gradient)
- Report results from training classifier with SVD-based pseudo data
- Report results from training classifier with oracle data.

CUDA_VISIBLE_DEVICES=1 python -m attribscope.classifier.run_all_positions \
    --reps-root /data/hoang/attrib/outputs \
    --data-root data/ww \
    --outputs-root /data/hoang/attrib/results_svd \
    --models llama-3.1-8b qwen3-8b \
    --subsets algorithm-generated hand-crafted \
    --poolings grad mean last \
    --losses ntp kl_uniform kl_temp \
    --temperatures 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0 \
    --weight-names all \
    --thresholds 0.0 0.01 0.02 0.025 0.05 0.075 0.1 0.125 0.15 0.175 0.2 \
    --seeds 1 \
    --device cuda
"""

import argparse
import re
import pandas as pd
import numpy as np
import itertools
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm

from attribscope.classifier.classifier import (
    MLPClassifier,
    train, infer, quick_eval,
    seed_everything
)
from attribscope.classifier.classifier import (
    upsample, downsample, key_grads, key_hidden
)
from attribscope.svd2.utils import (
    RepresentationStores,
    load_representations,
    _resolve_dir,
    split_data,
    compute_metrics,
    get_mistake_meta
)
from attribscope.svd2.computation import (
    fit_one, score_one, fit_all, score_all
)
from attribscope.svd2.utils import run_metrics

def build_rep_configs(poolings, losses, temperatures):
    """Should return
    [
        ("hidden", "last", None, None),
        ("hidden", "mean", None, None),
        ("grads", "grad", "ntp", None),
        ("grads", "grad", "kl_uniform", None),
        ("grads", "grad", "kl_temp", 1.2),
        ...
        ("grads", "grad", "kl_temp", 3.0),
    ]
    """
    rep_map = dict(last="hidden", mean="hidden", grad="grads")
    rep_configs = []
    seen = set()
    for pooling, loss, temperature in itertools.product(
        poolings, losses, temperatures
    ):
        rep_type = rep_map[pooling]
        if rep_type == "hidden":
            config = (rep_type, pooling, None, None)
        elif loss == "kl_temp":
            config = (rep_type, pooling, loss, temperature)
        else:  # ntp or kl_uniform — no temperature
            config = (rep_type, pooling, loss, None)
        if config not in seen:
            seen.add(config)
            rep_configs.append(config)

    return rep_configs


def precompute_svd(
    train_reps: RepresentationStores,
    val_reps:   RepresentationStores,
    test_reps:  RepresentationStores,
    n_components: int = 10,
    device: torch.device = torch.device("cuda")
):
    svd_components = fit_all(train_reps.stores, n_components=n_components)

    n_components_score = list(range(1, n_components + 1))
    score_kwargs = dict(
        svd=svd_components, 
        n_components_score=n_components_score, 
        device=device
    )
    train_scores = score_all(train_reps.stores, **score_kwargs)
    val_scores   = score_all(val_reps.stores,   **score_kwargs)
    test_scores  = score_all(test_reps.stores,  **score_kwargs)

    val_df  = run_metrics(val_scores,  keeper=val_reps.keeper,  ks=[1])
    test_df = run_metrics(test_scores, keeper=test_reps.keeper, ks=[1])
    merged_df = pd.merge(
        val_df, test_df, suffixes=('_val', '_test'),
        on=['weight', 'pooling', 'method', 'c', 'centered', 'direction', 'k'],
    )

    return dict(
        svd_components = svd_components,
        svd_accuracy   = merged_df,
        train_scores   = train_scores,
        val_scores     = val_scores,
        test_scores    = test_scores
    )

def get_pseudo_labels(
    train_scores,
    val_scores,
    val_reps,
    layer_idx,
    threshold,
    device
):
    val_metrics = run_metrics(val_scores, keeper=val_reps.keeper, ks=[1])
    config = val_metrics.query(
        f"weight == '{layer_idx}' and direction == 'asc'"
    ).sort_values(["step_acc"], ascending=False).iloc[0].to_dict()
    QUERY = (
        f"weight == '{layer_idx}' "
        f"and pooling == '{config["pooling"]}' "
        f"and method  == '{config["method"]}' "
        f"and c == {config["c"]} "
        f"and centered == {config["centered"]}"
    )
    print("---" * 20)
    print(f"Using best validation config:\n{QUERY}")
    print(f"The best validation results with direct projection on SVD components")
    print(f"Step@1: {config['step_acc']:.4f} Agent@1: {config['agent_acc']:.4f}\n")

    pseudo_scores = pd.DataFrame(train_scores).query(QUERY).iloc[0].scores
    wild_threshold = np.sort(pseudo_scores)[int(len(pseudo_scores) * threshold)]

    y_train_pseudo = torch.Tensor( # <- use this instead of y_train
        (pseudo_scores < wild_threshold) # <- using < because asc is better.
    ).to(device=device)
    return y_train_pseudo, config

def prepare_data(
    train_reps:   RepresentationStores,
    val_reps:     RepresentationStores,
    test_reps:    RepresentationStores,

    train_scores: list[dict], # [dict(weight, pooling, method, c, centered, scores)]
    val_scores:   list[dict], # [dict(weight, pooling, method, c, centered, scores)]

    layer_idx:    str,
    threshold:    float,
    mode: str, # oracle | pseudo
    device:       torch.device = torch.device("cuda"),
):
    
    # --- Load training data and create pseudo labels
    X_train = train_reps.stores[layer_idx].R
    X_train = X_train.float().to(device)
    y_train = torch.Tensor(
        [idx.is_mistake for idx in  train_reps.keeper.index],
    ).to(device=X_train.device)

    y_train_pseudo, config = get_pseudo_labels(
        train_scores, val_scores, val_reps,
        layer_idx, threshold,
        device=X_train.device
    )

    # --- Load validation data
    X_val = val_reps.stores[layer_idx].R
    X_val = X_val.float().to(device)
    y_val = torch.Tensor(
        [idx.is_mistake for idx in  val_reps.keeper.index],
    ).to(device=X_val.device)

    # --- Load test data
    X_test = test_reps.stores[layer_idx].R
    y_test = torch.Tensor(
        [idx.is_mistake for idx in  test_reps.keeper.index],
    ).to(device=X_test.device)

    if mode == "oracle":
        print("Using labeled training data as ceiling reference.")
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, shuffle=True)
    elif mode == "pseudo":
        train_loader = DataLoader(TensorDataset(X_train, y_train_pseudo), batch_size=512, shuffle=True)
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    val_loader   = DataLoader(TensorDataset(X_val, y_val), batch_size=512, shuffle=False)
    return dict(
        train_loader = train_loader,
        val_loader   = val_loader,
        train        = (X_train, y_train),
        validation   = (X_val, y_val), 
        test         = (X_test, y_test),
        best_config  = config,
    )


def get_metrics(
    clf: MLPClassifier,
    val_reps: RepresentationStores,
    test_reps: RepresentationStores,
    layer_idx: str,
    threshold: float,
    device: torch.device = torch.device("cuda")
):

    # --- Precompute mistake meta (same for all layers)
    val_mistake_indices, val_mistake_roles     = get_mistake_meta(val_reps.keeper)
    test_mistake_indices, test_mistake_roles   = get_mistake_meta(test_reps.keeper)

    # --- Validation metrics
    X_val = val_reps.stores[layer_idx].R
    X_val = X_val.float().to(device)
    val_scores = infer(clf, X_val, return_logits=False, device=device)
    val_metrics = compute_metrics(
        scores=val_scores, keeper=val_reps.keeper,
        mistake_indices=val_mistake_indices, mistake_roles=val_mistake_roles,
        ks=[1], direction="desc",
    )
    val_step_acc, val_agent_acc = list(val_metrics.values())

    # --- Test metrics
    X_test = test_reps.stores[layer_idx].R
    X_test = X_test.float().to(device)
    test_scores = infer(clf, X_test, return_logits=False, device=device)
    test_metrics = compute_metrics(
        scores=test_scores, keeper=test_reps.keeper,
        mistake_indices=test_mistake_indices, mistake_roles=test_mistake_roles,
        ks=[1], direction="desc",
    )
    test_step_acc, test_agent_acc = list(test_metrics.values())

    print(
        f"  Layer {layer_idx:>10} | "
        f"Validation Step@1: {val_step_acc:.4f}  Agent@1: {val_agent_acc:.4f} | "
        f"Test  Step@1: {test_step_acc:.4f}  Agent@1: {test_agent_acc:.4f}"
    )

    final_metrics = dict(
        labels         = "oracle" if threshold == 0.0 else "pseudo",
        position       = layer_idx,
        threshold      = threshold,
        val_step_acc   = val_step_acc,
        val_agent_acc  = val_agent_acc,
        test_step_acc  = test_step_acc,
        test_agent_acc = test_agent_acc
    )
    return final_metrics


def run_one(
    clf:          MLPClassifier,

    # data configs
    train_loader: DataLoader,
    val_loader:   DataLoader,
    val_reps:     RepresentationStores,
    test_reps:    RepresentationStores,
    
    # hyperparameters data selection
    layer_idx:    str,
    threshold:    float,

    # training configs
    epochs: int = 500,
    learning_rate: float = 0.05,
    weight_decay: float = 3e-4,
    momentum: float = 0.9,
    pos_weight: float = None,
    logging_steps: int = 100,
    val_metric: str = "f1",

    device: torch.device = torch.device("cuda"),
):

    clf, metrics = train(
        clf,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        momentum=momentum,
        pos_weight=pos_weight,
        logging_steps=logging_steps,
        val_metric=val_metric,
        device=next(iter(train_loader))[0].device,    
    )

    final_metrics = get_metrics(
        clf,
        val_reps,
        test_reps,
        layer_idx,
        threshold,
        device
    )
    return clf, final_metrics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps-root",    type=Path, default=Path("/data/hoang/attrib/outputs"))
    parser.add_argument("--data-root",    type=Path, default=Path("data/ww"))
    parser.add_argument("--outputs-root", type=Path, default=Path("/data/hoang/attrib/results_pseudo"))
    parser.add_argument("--models",       nargs="+", default=["llama-3.1-8b", "qwen3-8b"])
    parser.add_argument("--subsets",      nargs="+", default=["algorithm-generated", "hand-crafted"])
    parser.add_argument("--poolings",     nargs="+", default=["grad", "mean", "last"])
    parser.add_argument("--losses",       nargs="+", default=["ntp", "kl_uniform", "kl_temp"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    parser.add_argument("--weight-names", nargs="+", default=["all"])
    parser.add_argument("--thresholds",   nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2])
    parser.add_argument("--seeds",        nargs="+", type=int,   default=[1, 2, 3])
    parser.add_argument("--device",       type=str,  default="cuda")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    rep_configs = build_rep_configs(
        poolings=args.poolings,
        losses=args.losses,
        temperatures=args.temperatures 
    )
    print(f"All representation configs:\n{rep_configs}")
    print(f"Sweeping models: {args.models}")
    REPS_ROOT    = args.reps_root
    OUTPUTS_ROOT = args.outputs_root
    DATA_ROOT    = args.data_root
    WEIGHT_NAMES = args.weight_names
    if WEIGHT_NAMES == ["all"]: WEIGHT_NAMES = "all"
    DEVICE = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    assert isinstance(DEVICE, str)
    DEVICE = torch.device(DEVICE)

    combos = list(itertools.product(args.models, args.subsets, args.seeds, rep_configs))

    for model, subset, seed, (rep_type, pooling, loss, temperature) in tqdm(combos):
        print(f"\n{'='*60}")
        print(f"Model: {model} | Subset: {subset} | Seed: {seed}")
        print(f"Rep: {rep_type}/{pooling} | Loss: {loss} | Temp: {temperature}")
        print(f"{'='*60}")

        rep_dir = _resolve_dir(
            root_dir=REPS_ROOT,
            model=model,
            subset=subset,
            rep_type=rep_type,
            loss=loss,
            temperature=temperature,
            dir_type="representations"
        )
        output_dir = _resolve_dir(
            root_dir=OUTPUTS_ROOT, model=model, subset=subset,
            rep_type=rep_type, loss=loss, temperature=temperature,
            dir_type="metrics"
        )
        output_path = output_dir / f"pooling-{pooling}_seed-{seed}.tsv"
        if output_path.exists():
            print("[skipped] Skipped.")
            continue

        data_dir = DATA_ROOT / subset
        print(f"Representation dir: {rep_dir}")
        print(f"Data dir:           {data_dir}")

        files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
        assert files, (f"No .safetensors files in {rep_dir}")

        train_files, test_files = split_data(files, 0.5, seed)
        train_files, val_files  = split_data(train_files, 0.8, seed)

        print(f"Total train trajectories: {len(train_files)}")
        print(f"Total val trajectories:   {len(val_files)}")
        print(f"Total test trajectories:  {len(test_files)}")

        rep_kwargs = dict(
            rep_dir=rep_dir,
            data_dir=data_dir,
            pooling=pooling,
            weight_names=WEIGHT_NAMES,
            device=args.device,
        )

        train_reps = load_representations(**rep_kwargs, files=train_files)
        val_reps   = load_representations(**rep_kwargs, files=val_files)
        test_reps  = load_representations(**rep_kwargs, files=test_files)

        sort_key = dict(grads=key_grads, hidden=key_hidden)[rep_type]
        LAYER_IDXS = list(train_reps.stores.keys())
        LAYER_IDXS = sorted(LAYER_IDXS, key=sort_key)[:]
        print(f"All positions: {LAYER_IDXS}\n")

        # ---------------------------------------------------------------
        # SVD direct projection
        # ---------------------------------------------------------------
        precomputed_svd = precompute_svd(train_reps, val_reps, test_reps, 
                                         n_components=20, device=DEVICE)
        
        train_scores = precomputed_svd["train_scores"]
        val_scores   = precomputed_svd["val_scores"]
        svd_accuracy = precomputed_svd["svd_accuracy"]

        svd_accuracy = svd_accuracy[svd_accuracy["direction"] == "asc"]
        svd_accuracy = svd_accuracy.sort_values("step_acc_val", ascending=False)
        svd_outpath = output_dir / f"svd_pooling-{pooling}_seed-{seed}.tsv"
        if svd_outpath.exists():
            print(" [skipped] SVD computation, file exists.")
        else:
            svd_accuracy.to_csv(svd_outpath, sep="\t", index=False)

        # ---------------------------------------------------------------
        # Truncate list of positions to run
        # ---------------------------------------------------------------
        truncate_threshold = 40
        best_accuracy  = (svd_accuracy.groupby("weight")["step_acc_val"]
                           .max().sort_values(ascending=False))
        best_positions = best_accuracy.keys().to_list()[:truncate_threshold]
        print(f"Selected positions: {best_positions}\n")

        metric_rows = []
        inner_combos = list(itertools.product(best_positions, args.thresholds))
        for icb, (layer_idx, threshold) in tqdm(enumerate(inner_combos)):
            print("---" * 20)
            print(f"COMBO: [{icb + 1}/{len(inner_combos)}] | POSITION: {layer_idx} | Wild THRESHOLD: {threshold}")
            if threshold == 0.0: mode = "oracle" # --> for oracle training
            else:                mode = "pseudo" # --> for pseudo label training

            prepared_data = prepare_data(
                train_reps, val_reps, test_reps, train_scores, val_scores,
                layer_idx=layer_idx, threshold=threshold, mode=mode, device=DEVICE
            )

            train_loader = prepared_data["train_loader"]
            val_loader   = prepared_data["val_loader"]
            X_train, y_train = prepared_data["train"]
            X_val, y_val     = prepared_data["validation"]
            X_test, y_test   = prepared_data["test"]
            best_val_config  = prepared_data["best_config"]

            assert float(best_accuracy[layer_idx]) == best_val_config['step_acc'], \
            "Two SVD computations seem to diverge, check prepare_data vs. precompute_svd."

            seed_everything(seed)
            clf = MLPClassifier(
                input_dim=X_train.shape[1], 
                hidden_dim=1024
            )
            clf, metrics = run_one(
                clf, train_loader, val_loader,
                val_reps, test_reps,
                layer_idx, threshold,
                epochs = 500,
                learning_rate = 0.02,
                device = DEVICE
            )
            metric_rows.append({**best_val_config, **metrics})
        
        metric_df = pd.DataFrame(metric_rows).sort_values("test_step_acc", ascending=False)
        output_path = output_dir / f"classifer_pooling-{pooling}_seed-{seed}.tsv"
        metric_df.to_csv(output_path, sep="\t", index=False)
        