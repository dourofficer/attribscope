# `experiments/` — SVD runner

YAML-driven wrapper around `attribscope.svd.compute_svd` and
`attribscope.svd.compute_scores`. Shells out unchanged — the runner is pure
argv glue.

```
experiments/
├── run.py
├── configs/default.yaml
└── README.md
```

## Usage

```bash
# Full sweep:
python -m experiments.svd.run fit   --config experiments/configs/default.yaml
python -m experiments.svd.run score --config experiments/configs/default.yaml

# Dry-run (prints the exact subprocess commands, executes nothing):
python -m experiments.svd.run fit --config experiments/configs/default.yaml --dry-run

# One-shot overrides (values parsed as YAML, so lists/ints/bools work):
python -m experiments.svd.run score \
    --config experiments/configs/default.yaml \
    --set shared.models=[qwen3-8b] \
    --set shared.rep_kinds=[grads] \
    --set score.fit_subsets=[hand-crafted] \
    --set score.score_subsets=[algorithm-generated]
```

## Config

See `experiments/configs/default.yaml`. Three top-level blocks:

- `shared:` — models, rep_kinds, poolings, paths, device.
- `fit:` — only read by `run fit`.
- `score:` — only read by `run score`.

Pooling is per-rep-kind (`mean last` for `hidden`, `grad` for `grads`). Set
`shared.poolings` to a list to force a single choice across all rep kinds.

To persist a sweep variant, copy `default.yaml` to a new file and pass
`--config experiments/configs/<name>.yaml`.

## What it runs

Per phase, the runner loops over the outer axes and lets the underlying
module handle the rest:

```
fit:    rep_kinds × poolings(rep_kind)
          → compute_svd handles: models × subsets
score:  rep_kinds × fit_subsets
          → compute_scores handles: models × score_subsets × poolings
                                    × {proj, recon} × {raw, centered}
                                    × n_components_score
```

Keep `fit.n_components` equal to `score.n_components_fit` — the scoring module
reads `V.safetensors` from the directory named `<pooling>_c<N>_*`.