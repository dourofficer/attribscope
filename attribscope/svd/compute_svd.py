"""
attribscope.svd.compute_svd
===========================

Fit top-c right singular vectors of the per-weight gradient matrices on a
given (model, subset) pair. Produces two variants per run:

    {pooling}_c{N}_raw/        V.safetensors, meta.json
    {pooling}_c{N}_centered/   V.safetensors, ref.safetensors, meta.json

The centered variant subtracts the mean gradient (the SAL reference) before
SVD; the raw variant fits directly on the unmodified gradient matrix.

Example
-------
    python -m attribscope.svd.compute_svd \
        --models qwen3-8b \
        --subsets hand-crafted \
        --rep-dir outputs/hidden \
        --pooling mean \
        --n-components 10

    python -m attribscope.svd.compute_svd \
        --models qwen3-8b \
        --subsets hand-crafted \
        --rep-dir outputs/grads \
        --pooling grad \
        --n-components 10

Output layout
-------------
    {rep-dir}/{model}/svd/{subset}/
        mean_c10_raw/            # {pooling}_c{k}_{centered/raw}/
            V.safetensors        # keys: weight_name -> (d, n_components)
            meta.json
        mean_c10_centered/
            V.safetensors        # keys: weight_name -> (d, n_components)
            ref.safetensors      # keys: weight_name -> (d,) mean gradient
            meta.json

`meta.json` schema:
    {
        "fit_subset":       str,
        "model":            str,
        "pooling":          str,
        "centered":         bool,
        "n_components_fit": int,
        "weight_names":     list[str],
        "n_rows":           int,
        "fit_rep_hash":     str,     # SHA1 of first 1 MiB of stacked reps
        "created_at":       str      # ISO-8601 UTC
    }
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import torch
from safetensors.torch import save_file
from tqdm import tqdm

# ── External dependencies (adjust imports to your module layout) ──────────────
from .utils import RepresentationStores, load_and_stack


# ─────────────────────────────────────────────────────────────────────────────
# SVD
# ─────────────────────────────────────────────────────────────────────────────
def _fit_svd(G: torch.Tensor, n_components: int) -> torch.Tensor:
    """Return the top-``n_components`` right singular vectors of ``G``.

    Shape: (d, n_components).
    """
    _, _, V = torch.svd_lowrank(G.float(), q=n_components, niter=10)
    return V.contiguous()


def _hash_first_mib(G: torch.Tensor, n_bytes: int = 1 << 20) -> str:
    """SHA1 over the first ``n_bytes`` of the tensor's byte representation.

    Cheap way to detect stale SVD artifacts if the fit representations are
    silently regenerated. Uses CPU-side bytes so the hash is deterministic
    across devices.
    """
    buf = G.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha1(buf[:n_bytes]).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Per-model, per-subset driver
# ─────────────────────────────────────────────────────────────────────────────
def fit_and_save(
    stores:      RepresentationStores,
    model:       str,
    subset:      str,
    svd_dir:     Path,
    n_components: int,
) -> None:
    """Fit raw + centered SVD for every (pooling, weight) in ``stores`` and
    write artifacts.

    Stores are grouped by pooling so that each pooling gets its own output
    directory pair:
        {svd_dir}/{pooling}_c{n_components}_raw/
        {svd_dir}/{pooling}_c{n_components}_centered/
    """
    # ── Group RepresentationStore objects by their pooling tag ────────────────
    # Keys in stores.stores are "{pooling}.{weight_name}".
    by_pooling: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
    for store_key, rep_store in stores.stores.items():
        by_pooling[rep_store.pooling][rep_store.name] = rep_store.R

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for pooling, weight_tensors in by_pooling.items():
        raw_dir      = svd_dir / f"{pooling}_c{n_components}_raw"
        centered_dir = svd_dir / f"{pooling}_c{n_components}_centered"
        raw_dir.mkdir(parents=True, exist_ok=True)
        centered_dir.mkdir(parents=True, exist_ok=True)

        V_raw:      dict[str, torch.Tensor] = {}
        V_centered: dict[str, torch.Tensor] = {}
        ref_dict:   dict[str, torch.Tensor] = {}
        rep_hash:   dict[str, str]          = {}

        weight_names = sorted(weight_tensors.keys())
        n_rows       = next(iter(weight_tensors.values())).shape[0]

        for name in tqdm(weight_names, desc=f"Fitting SVD [{model}/{subset}/{pooling}]"):
            G = weight_tensors[name]

            rep_hash[name] = _hash_first_mib(G)

            # --- raw SVD --------------------------------------------------
            V_raw[name] = _fit_svd(G, n_components).cpu()

            # --- centered SVD ---------------------------------------------
            mean = G.float().mean(dim=0)
            ref_dict[name]   = mean.cpu()
            V_centered[name] = _fit_svd(G.float() - mean, n_components).cpu()

        base_meta = dict(
            fit_subset       = subset,
            model            = model,
            pooling          = pooling,
            n_components_fit = n_components,
            weight_names     = weight_names,
            n_rows           = int(n_rows),
            fit_rep_hash     = rep_hash,
            created_at       = now,
        )

        # ── Write raw ─────────────────────────────────────────────────────
        save_file(V_raw, str(raw_dir / "V.safetensors"))
        (raw_dir / "meta.json").write_text(json.dumps(
            {**base_meta, "centered": False}, indent=2))
        print(f"  Saved {raw_dir}")

        # ── Write centered ────────────────────────────────────────────────
        save_file(V_centered, str(centered_dir / "V.safetensors"))
        save_file(ref_dict,   str(centered_dir / "ref.safetensors"))
        (centered_dir / "meta.json").write_text(json.dumps(
            {**base_meta, "centered": True}, indent=2))
        print(f"  Saved {centered_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    p.add_argument("--models",       nargs="+", required=True,
                   help="Model tags, e.g. qwen3-8b llama-3.1-8b")
    p.add_argument("--subsets",      nargs="+", required=True,
                   help="Fit subsets, e.g. hand-crafted algorithm-generated")
    p.add_argument("--rep-dir",      type=Path,  required=True,
                   help="Root directory containing {model}/reps/{subset}/*.safetensors")
    p.add_argument("--data-dir",     type=Path,  default=Path("data/ww"),
                   help="Who&When JSON root (for step role lookup)")
    p.add_argument("--pooling",      default="mean",
                   help="Pooling strategy stored in the safetensors files (default: mean)")
    p.add_argument("--n-components", type=int,   default=10,
                   help="Maximum number of singular vectors to fit")
    p.add_argument("--weights",      nargs="+",  default=["all"],
                   help="Weight names to include (default: all)")
    p.add_argument("--device",       default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main() -> None:
    args   = parse_args()
    device = torch.device(args.device)

    for model in args.models:
        for subset in args.subsets:
            print(f"\n=== {model} / {subset} ===")

            stores = load_and_stack(
                model        = model,
                subset       = subset,
                pooling      = args.pooling,
                weight_names = args.weights if args.weights != ["all"] else "all",
                data_dir     = args.data_dir / subset,
                device       = device,
                base_dir     = args.rep_dir,
            )

            svd_dir = args.rep_dir / model / "svd" / subset
            fit_and_save(stores, model, subset, svd_dir, args.n_components)

            del stores
            if device.type == "cuda":
                torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()