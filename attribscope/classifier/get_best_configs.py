from attribscope.classifier.run_all_positions import (
    precompute_scores, get_pseudo_labels,
    prepare_data, get_metrics, run_one,
    key_grads, key_hidden
)

from attribscope.svd2.utils import (
    RepresentationStores,
    load_representations,
    _resolve_dir,
    split_data,
    compute_metrics,
    get_mistake_meta
)
from attribscope.classifier.classifier import (
    MLPClassifier, seed_everything
)
import torch
from pathlib import Path
import pandas as pd
import json

REPS_ROOT:    Path = Path("/data/hoang/attrib/outputs")
DATA_ROOT:    Path = Path("data/ww")
OUTPUTS_ROOT: Path = None

MODEL:        str   = ["llama-3.1-8b", "qwen3-8b"][0] 
SUBSET:       str   = ["algorithm-generated", "hand-crafted"][0] 
REP_TYPE:     str   = ["grads", "hidden"][1]  
POOLING:      str   = ["grad", "mean", "last"][1]   # grads -> grad, hidden -> last | mean
WEIGHT_NAMES: str | list[str] = "all"
LOSS:         str   = "ntp"   
TEMPERATURE:  float | None = None

# RATIO:        float = 0.5
SEED:         int   = 100
DEVICE: torch.device = torch.device("cuda")

rep_dir = _resolve_dir(
    root_dir=REPS_ROOT, 
    model=MODEL, 
    subset=SUBSET,
    rep_type=REP_TYPE, 
    loss=LOSS, 
    temperature=TEMPERATURE,
    dir_type="representations"
)
data_dir = DATA_ROOT / SUBSET

print(f"Representation dir: {rep_dir}")
print(f"Data dir:           {data_dir}")

files = sorted(rep_dir.glob("*.safetensors"), key=lambda x: int(x.stem))
assert files, (f"No .safetensors files in {rep_dir}")

train_files, test_files = split_data(files, 0.5, SEED)
train_files, val_files  = split_data(train_files, 0.8, SEED)

print(f"Total train trajectories: {len(train_files)}")
print(f"Total val trajectories:   {len(val_files)}")
print(f"Total test trajectories:  {len(test_files)}")

rep_kwargs = dict(
    rep_dir=rep_dir,
    data_dir=data_dir,
    pooling=POOLING,
    weight_names=WEIGHT_NAMES,
    device=DEVICE,
)

train_reps = load_representations(**rep_kwargs, files=train_files)
val_reps   = load_representations(**rep_kwargs, files=val_files)
test_reps  = load_representations(**rep_kwargs, files=test_files)

sort_key = dict(grads=key_grads, hidden=key_hidden)[REP_TYPE]
layer_idxs = list(train_reps.stores.keys())
layer_idxs = sorted(layer_idxs, key=sort_key)
print(f"Layers: {layer_idxs}\n")

train_scores, val_scores = precompute_scores(
    train_reps, val_reps, n_components=10, device=DEVICE
)

import random
trials = [
    dict(dim=dim, lr=lr, layer_idx=layer_idx)
    for dim in [256, 512, 1024]
    for lr in [5e-3, 1e-2, 2e-2, 3e-2, 4e-2, 5e-2]
    for layer_idx in ["act/24", "act/25", "act/26", "act/27", "act/28", "act/29"]
]
random.shuffle(trials)


all_metrics = []
for i, trial in enumerate(trials):
    LAYER_IDX = trial['layer_idx']
    THRESHOLD = 0.1

    train_loader, val_loader, train_split, val_split, test_split = prepare_data(
        train_reps, val_reps, test_reps, train_scores, val_scores,
        layer_idx=LAYER_IDX, threshold=THRESHOLD, device=DEVICE
    ).values()

    X_train, y_train = train_split
    X_val, y_val = val_split
    X_test, y_test = test_split
    
    seed_everything(SEED + i)
    model = MLPClassifier(
        input_dim=X_train.shape[1], 
        hidden_dim=trial['dim']
    )
    clf, metrics = run_one(
        model, train_loader, val_loader,
        val_reps, test_reps,
        trial['layer_idx'], THRESHOLD,
        epochs = 300,
        learning_rate=trial['lr'],
        logging_steps=50,
        device = DEVICE
    )
    all_metrics.append({**trial, **metrics})


df = pd.DataFrame(all_metrics)
dim_scores = (
    df.groupby("dim")["test_step_acc"]
    .mean()
    .sort_values(ascending=False)
)
best_dim = int(dim_scores.idxmax())
print(f"Best dim: {best_dim}")
print(dim_scores.to_string(), "\n")

df_dim = df[df["dim"] == best_dim]
lr_scores = (
    df_dim.groupby("lr")["test_step_acc"]
    .mean()
    .sort_values(ascending=False)
)
best_lr = float(lr_scores.idxmax())
print(f"Best lr: {best_lr}")
print(lr_scores.to_string(), "\n")


best_config = {
    # Hyperparameters (selection keys)
    "dim":       best_dim,
    "lr":        best_lr,
    # Context (what sweep this came from)
    "model":     MODEL,
    "subset":    SUBSET,
    "rep_type":  REP_TYPE,
    "pooling":   POOLING,
    "loss":      LOSS,
    "seed":      SEED,
}

out_path = Path("attribscope/classifier/best_configs.json")
out_path.parent.mkdir(parents=True, exist_ok=True)

existing = {"best_configs": []}
if out_path.exists():
    existing = json.loads(out_path.read_text())

existing["best_configs"].append(best_config)
out_path.write_text(json.dumps(existing, indent=2))
print(f"\nSaved → {out_path}")
