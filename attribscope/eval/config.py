"""
Core scoring and evaluation functions for gradient-norm ablation studies.

This module owns the math: scoring steps, scoring trajectories, and evaluating
predictions against ground-truth annotations.  It knows nothing about file paths,
strategy construction, or plotting.

Typical usage
-------------
>>> from ablation.core import build_strategies, load_trajectories, \
...     get_param_names_and_sizes, discover_n_layers
>>> from ablation.core import CompiledConfigs, score_step, evaluate_trajectories
>>>
>>> trajs = load_trajectories(results_dir)
>>> param_names, param_sizes = get_param_names_and_sizes(trajs)
>>> strategies = build_strategies(discover_n_layers(param_names))
>>>
>>> cc = CompiledConfigs.compile(strategies["layer"], param_names, param_sizes)
>>> df = evaluate_trajectories(trajs, cc, "l1_norm", k=1)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ── constants ─────────────────────────────────────────────────────────────────

STRATEGIES = ["layer", "mlp", "attn", "mlp_weights", "attn_weights"]
NORM_TYPES = ["l1_norm", "l2_norm"]


# ── strategy definitions ──────────────────────────────────────────────────────

def build_strategies(L: int) -> dict[str, dict[str, str]]:
    def per_layer(prefix, suffix=""):
        return {f"{prefix}/{i}": rf"model\.layers\.{i}\.{suffix}" for i in range(L)}

    return {
        "layer": {
            **per_layer("layer"),
            "lm_head":      r"lm_head\.",
            "embed_tokens":  r"model\.embed_tokens\.",
        },
        "mlp":         per_layer("mlp",  r"mlp\."),
        "mlp_weights": {
            **per_layer("gate", r"mlp\.gate_proj\."),
            **per_layer("up",   r"mlp\.up_proj\."),
            **per_layer("down", r"mlp\.down_proj\."),
        },
        "attn":        per_layer("attn", r"self_attn\."),
        "attn_weights": {
            **per_layer("q", r"self_attn\.q_proj\."),
            **per_layer("k", r"self_attn\.k_proj\."),
            **per_layer("v", r"self_attn\.v_proj\."),
            **per_layer("o", r"self_attn\.o_proj\."),
        },
    }


# ── data loading ──────────────────────────────────────────────────────────────

def load_trajectories(results_dir: Path) -> list[dict]:
    return [json.loads(f.read_text()) for f in sorted(results_dir.glob("*.json"))]


def get_param_names_and_sizes(trajectories: list[dict]) -> tuple[list[str], np.ndarray]:
    sample_stats = next(
        log["statistics"]
        for data in trajectories
        for log in data["logs"]
        if log.get("statistics")
    )
    param_names = list(sample_stats.keys())
    param_sizes = np.array(
        [sample_stats[p]["n_params"] for p in param_names], dtype=np.float64,
    )
    return param_names, param_sizes


def discover_n_layers(param_names: list[str]) -> int:
    """Infer number of layers from the highest layer index in param names."""
    indices = [
        int(m.group(1))
        for p in param_names
        if (m := re.search(r"model\.layers\.(\d+)\.", p))
    ]
    if not indices:
        raise ValueError("Could not discover n_layers: no 'model.layers.N.' params found.")
    return max(indices) + 1


# ── compiled config representation ────────────────────────────────────────────

@dataclass(frozen=True)
class CompiledConfigs:
    """Pre-compiled parameter-group masks ready for vectorised scoring.

    Attributes
    ----------
    names : list[str]
        Human-readable config names (e.g. ``["layer/0", "layer/1", …]``).
    param_names : list[str]
        Ordered parameter names matching the ``statistics`` dicts in log
        entries.  Length *P*.
    mask : np.ndarray
        Boolean matrix of shape ``(C, P)`` where ``mask[i, j]`` is True if
        parameter *j* belongs to config *i*.  Stored as ``float64`` so it
        can be used directly in matrix multiplications.
    n_params : np.ndarray
        Total parameter count per config, shape ``(C,)``.  Derived as
        ``mask @ param_sizes``.
    """

    names:       list[str]
    param_names: list[str]
    mask:        np.ndarray   # (C, P)  float64
    n_params:    np.ndarray   # (C,)    float64

    @classmethod
    def compile(
        cls,
        configs:     dict[str, str],
        param_names: list[str],
        param_sizes: np.ndarray,
    ) -> CompiledConfigs:
        """Build a ``CompiledConfigs`` from a raw strategy config dict.

        Parameters
        ----------
        configs : dict[str, str]
            Mapping from config name to a regex pattern that matches the
            parameter names belonging to that group (e.g.
            ``{"layer/0": r"model\\.layers\\.0\\.", …}``).
        param_names : list[str]
            Ordered list of all parameter names in the model.
        param_sizes : np.ndarray
            Number of scalar parameters per entry in *param_names*, shape
            ``(P,)``.
        """
        names    = list(configs.keys())
        patterns = list(configs.values())

        mask = np.array(
            [[bool(re.search(pat, p)) for p in param_names] for pat in patterns],
            dtype=np.float64,
        )
        n_params = mask @ param_sizes

        return cls(
            names=names,
            param_names=param_names,
            mask=mask,
            n_params=n_params,
        )