import torch
from typing import Callable, Literal
from pathlib import Path
import argparse

from .core import _run_svd

"""
python -m attribscope.svd.compute_svd \
    --models qwen3-8b \
    --subsets hand-crafted \
    --rep-dir outputs/grads \
    --output outputs/svd \
    --n-components 5 \
    --centered

>>> read representations from outputs/grads/qwen3-8b/hand-crafted
>>> compute singular vectors with c=5 components, centered
>>> save singular vectors and scores to outputs/svd/qwen3-8b/hand-crafted
"""

KNOWN_MODELS  = ["llama-3.1-8b", "qwen3-8b"]
KNOWN_SUBSETS = ["hand-crafted", "algorithm-generated"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    # --- Target selection ---
    parser.add_argument(
        "--models", nargs="+", default=KNOWN_MODELS,
        metavar="MODEL",
        help="Model name(s) to evaluate.",
    )
    parser.add_argument(
        "--subsets", nargs="+", default=KNOWN_SUBSETS,
        metavar="SUBSET",
        help="Dataset subset(s) to evaluate.",
    )
 
    # --- Paths ---
    parser.add_argument(
        "--data-dir", type=Path, default=Path("ww"),
        metavar="DIR",
        help="Root directory of raw trajectory JSON files.",
    )
    parser.add_argument(
        "--grad-dir", type=Path, default=Path("outputs/grads"),
        metavar="DIR",
        help="Root directory of extracted gradient .safetensors files.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("outputs/grads"),
        metavar="DIR",
        help="Root directory for metric TSV output files.",
    )
 
    return parser.parse_args()