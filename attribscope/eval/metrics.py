from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CompiledConfigs

# ── constants ─────────────────────────────────────────────────────────────────

STRATEGIES = ["layer", "mlp", "attn", "mlp_weights", "attn_weights"]
NORM_TYPES = ["l1_norm", "l2_norm"]


# ── step-level scoring ────────────────────────────────────────────────────────

def score_step(
    step_log:  dict,
    cc:        CompiledConfigs,
    norm_type: str,
) -> np.ndarray:
    """Score a single log entry for every config in *cc*.

    Parameters
    ----------
    step_log : dict
        A single log entry whose ``"statistics"`` dict is keyed by parameter
        name.  Each value must contain ``"l1_norm"`` and ``"l2_norm_sq"``
        fields.
    cc : CompiledConfigs
        Pre-compiled config masks.
    norm_type : ``"l1_norm"`` | ``"l2_norm"``
        Which norm to aggregate.

    Returns
    -------
    np.ndarray
        Scores of shape ``(C,)`` — one per config.  Configs with zero
        parameters produce ``NaN``.
    """
    stats       = step_log["statistics"]
    safe_counts = np.where(cc.n_params > 0, cc.n_params, np.nan)

    with np.errstate(invalid="ignore"):
        if norm_type == "l2_norm":
            vals = np.array(
                [stats[p]["l2_norm_sq"] for p in cc.param_names],
                dtype=np.float64,
            )
            return np.sqrt(cc.mask @ vals) / safe_counts
        else:
            vals = np.array(
                [stats[p]["l1_norm"] for p in cc.param_names],
                dtype=np.float64,
            )
            return (cc.mask @ vals) / safe_counts


# ── trajectory-level scoring ──────────────────────────────────────────────────

def score_trajectory(
    traj:      dict,
    cc:        CompiledConfigs,
    norm_type: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Score every valid step in a trajectory.

    Parameters
    ----------
    traj : dict
        A single trajectory dict with ``"logs"`` (list of log entries).
    cc : CompiledConfigs
        Pre-compiled config masks.
    norm_type : ``"l1_norm"`` | ``"l2_norm"``

    Returns
    -------
    score_matrix : np.ndarray
        Shape ``(S, C)`` where *S* is the number of valid log entries and
        *C* is the number of configs.
    step_indices : np.ndarray
        Shape ``(S,)`` — the ``step_idx`` of each valid log, aligned with
        the rows of *score_matrix*.

    Raises
    ------
    ValueError
        If the trajectory contains no valid log entries (none with
        ``"statistics"``).
    """
    valid_logs = [log for log in traj["logs"] if log.get("statistics")]
    if not valid_logs:
        raise ValueError("Trajectory contains no valid log entries.")

    score_matrix = np.stack([
        score_step(log, cc, norm_type) for log in valid_logs
    ])  # (S, C)
    step_indices = np.array(
        [int(log["step_idx"]) for log in valid_logs],
        dtype=np.int64,
    )
    return score_matrix, step_indices


# ── per-trajectory evaluation ─────────────────────────────────────────────────

def _resolve_agent(role: str) -> str:
    """Normalise a step role string for agent-level matching.

    The Who&When hand-crafted subset uses ``"Orchestrator (thought)"``,
    ``"Orchestrator (-> WebSurfer)"``, etc.  We collapse all Orchestrator
    variants to ``"Orchestrator"`` so they match the ground-truth label.
    """
    if "orchestrator" in role.lower():
        return "Orchestrator"
    return role


def evaluate_trajectory(
    traj:      dict,
    cc:        CompiledConfigs,
    norm_type: str,
    k:         int,
    ascending: bool = True,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Evaluate top-*k* predictions for one trajectory.

    The predicted mistake step is the step with the **lowest** score
    (``argmin``).  When *k* > 1 the top-*k* lowest-scoring steps are
    considered; a prediction is correct if the ground-truth step or agent
    appears among them.

    Parameters
    ----------
    traj : dict
        Trajectory dict with ``"logs"``, ``"metadata"`` (containing
        ``"mistake_step"`` and ``"mistake_agent"``), and ``"steps"``
        (each with ``"step_idx"`` and ``"role"``).
    cc : CompiledConfigs
        Pre-compiled config masks.
    norm_type : ``"l1_norm"`` | ``"l2_norm"``
    k : int
        Number of top predictions to consider.

    Returns
    -------
    ``(step_correct, agent_correct)`` — each of shape ``(C,)`` with
    values 0.0 or 1.0 — or ``None`` if the trajectory has no valid logs.
    """
    valid_logs = [log for log in traj["logs"] if log.get("statistics")]
    if not valid_logs:
        return None

    # Ground truth
    meta          = traj["metadata"]
    mistake_step  = int(meta["mistake_step"])
    mistake_agent = meta["mistake_agent"]
    step_roles    = {s["step_idx"]: s["role"] for s in traj["steps"]}

    # Score and rank
    score_matrix = np.stack([
        score_step(log, cc, norm_type) for log in valid_logs
    ])  # (S, C)
    step_indices = np.array(
        [int(log["step_idx"]) for log in valid_logs],
        dtype=np.int64,
    )

    # Top-k scoring (ascending/descending) step indices per config: shape (k, C)
     # pred_step_matrix = step_indices[np.argsort(score_matrix, axis=0)[:k]]
    ranked = np.argsort(score_matrix, axis=0) # sort ascending as default
    if ascending: pred_step_matrix = step_indices[ranked[:k]]
    else:         pred_step_matrix = step_indices[ranked[::-1][:k]]
   
    # Step-level accuracy: is the ground-truth step among the top-k?
    step_correct = np.any(
        pred_step_matrix == mistake_step, axis=0,
    ).astype(np.float64)

    # Agent-level accuracy: is the ground-truth agent among the predicted
    # steps' agents?
    n_configs = cc.mask.shape[0]
    agent_correct = np.empty(n_configs, dtype=np.float64)
    for c in range(n_configs):
        predicted_agents = [
            _resolve_agent(step_roles.get(int(idx), "unknown"))
            for idx in pred_step_matrix[:, c]
        ]
        agent_correct[c] = float(mistake_agent in predicted_agents)

    return step_correct, agent_correct


# ── multi-trajectory evaluation ───────────────────────────────────────────────

def evaluate_trajectories(
    trajectories: list[dict],
    cc:           CompiledConfigs,
    norm_type:    str,
    k:            int,
    ascending:    bool = True,
) -> pd.DataFrame:
    """Evaluate all trajectories and return per-config accuracy.

    Parameters
    ----------
    trajectories : list[dict]
        List of trajectory dicts.
    cc : CompiledConfigs
        Pre-compiled config masks.
    norm_type : ``"l1_norm"`` | ``"l2_norm"``
    k : int
        Number of top predictions per trajectory.

    Returns
    -------
    pd.DataFrame
        Columns: ``config``, ``step_acc``, ``agent_acc``.  One row per
        config in *cc*.
    """
    n_configs         = len(cc.names)
    step_correct_sum  = np.zeros(n_configs, dtype=np.float64)
    agent_correct_sum = np.zeros(n_configs, dtype=np.float64)
    n_total           = 0

    for traj in trajectories:
        result = evaluate_trajectory(traj, cc, norm_type, k, ascending)
        if result is None:
            continue
        step_correct, agent_correct = result
        step_correct_sum  += step_correct
        agent_correct_sum += agent_correct
        n_total           += 1

    denom = max(n_total, 1)
    return pd.DataFrame({
        "config":    cc.names,
        "step_acc":  step_correct_sum  / denom,
        "agent_acc": agent_correct_sum / denom,
    })


# ── helpers for *_dist.py scripts ─────────────────────────────────────────────

def load_top_configs(
    agg_dir:  Path,
    subset:   str,
    k_sweep:  int,
    norm_type: str,
    k_top:    int,
) -> dict[str, list[str]]:
    """Load aggregated TSV and pick the top-*k_top* configs per strategy.

    Parameters
    ----------
    agg_dir : Path
        Directory containing aggregated result TSVs.
    subset : str
        Subset name (e.g. ``"hand-crafted"``).
    k_sweep : int
        The *k* value used during the sweep that produced the TSV.
    norm_type : str
        ``"l1_norm"`` or ``"l2_norm"``.
    k_top : int
        Number of top configs to select per strategy.

    Returns
    -------
    dict[str, list[str]]
        ``{strategy_name: [config_name, …]}`` with *k_top* best configs
        ranked by ``step_acc``.
    """
    tsv_path = agg_dir / f"{subset}_k{k_sweep}_{norm_type}.tsv"
    df = pd.read_csv(tsv_path, sep="\t")
    return {
        strat: (
            df[df["strategy"] == strat]
            .sort_values("step_acc", ascending=False)
            .head(k_top)["config"]
            .tolist()
        )
        for strat in STRATEGIES
    }


def compile_top_configs(
    top_config_names: dict[str, list[str]],
    all_strategies:   dict[str, dict[str, str]],
    param_names:      list[str],
    param_sizes:      np.ndarray,
) -> dict[str, CompiledConfigs]:
    """Compile a :class:`CompiledConfigs` for each strategy's selected configs.

    Parameters
    ----------
    top_config_names : dict[str, list[str]]
        ``{strategy_name: [config_name, …]}`` — output of
        :func:`load_top_configs`.
    all_strategies : dict[str, dict[str, str]]
        Full strategy dict from :func:`build_strategies`.
    param_names : list[str]
        Ordered parameter names.
    param_sizes : np.ndarray
        Parameter counts, shape ``(P,)``.

    Returns
    -------
    dict[str, CompiledConfigs]
        ``{strategy_name: CompiledConfigs}`` ready for scoring.
    """
    compiled: dict[str, CompiledConfigs] = {}
    for strat, cfg_names in top_config_names.items():
        sub_configs = {name: all_strategies[strat][name] for name in cfg_names}
        compiled[strat] = CompiledConfigs.compile(sub_configs, param_names, param_sizes)
    return compiled