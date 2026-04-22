"""
attribscope/reps/hidden.py — Hidden-state extraction along the residual stream.

Public API
----------
layer_to_shorthand(layer_idx)         int → "hidden/{i}"
pool_last(h, ctx_len)                 last output token → (hidden_dim,)
pool_mean(h, ctx_len)                 mean over output tokens → (hidden_dim,)
extract_hidden(...)                   forward pass; returns pooled residual states
"""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor
from transformers import PreTrainedModel

# ─────────────────────────────────────────────────────────────────────────────
# Shorthand helpers
# ─────────────────────────────────────────────────────────────────────────────

def layer_to_shorthand(layer_idx: int) -> str:
    """
    >>> layer_to_shorthand(9)
    'hidden/9'
    """
    return f"hidden/{layer_idx}"


def shorthand_to_layer(shorthand: str) -> int:
    """
    >>> shorthand_to_layer('hidden/35')
    35
    """
    prefix, idx = shorthand.split("/", 1)
    if prefix != "hidden":
        raise ValueError(f"Expected 'hidden/{{i}}' shorthand, got '{shorthand}'.")
    return int(idx)


# ─────────────────────────────────────────────────────────────────────────────
# Pooling strategies
# ─────────────────────────────────────────────────────────────────────────────

def pool_last(h: Tensor, ctx_len: int) -> Tensor:
    """Return the hidden state of the last output token.

    Parameters
    ----------
    h       : float tensor of shape (seq_len, hidden_dim).
    ctx_len : number of context (input) tokens at the front of the sequence.
              The output tokens occupy positions [ctx_len:].
              Trailing padding is never present — we extract single samples
              without padding.

    Returns
    -------
    Tensor of shape (hidden_dim,).
    """
    seq_len = h.shape[0]
    assert seq_len > ctx_len, f"seq_len={seq_len} and ctx_len={ctx_len}."

    # h[-1] is always the last real token since we never pad single samples.
    return h[-1].half().cpu()


def pool_mean(h: Tensor, ctx_len: int) -> Tensor:
    """Return the mean hidden state over all output tokens.

    Parameters
    ----------
    h       : float tensor of shape (seq_len, hidden_dim).
    ctx_len : number of context tokens at the front.

    Returns
    -------
    Tensor of shape (hidden_dim,).

    """
    seq_len = h.shape[0]
    assert seq_len > ctx_len, f"seq_len={seq_len} and ctx_len={ctx_len}."

    return h[ctx_len:].float().mean(dim=0).half().cpu()


def _apply_pool(
    h:       Tensor,
    ctx_len: int,
    pool:    Literal["last", "mean", "all"],
) -> dict[str, Tensor]:
    """Dispatch to pooling functions; return a dict of stat_name → Tensor.

    Parameters
    ----------
    h       : float tensor of shape (seq_len, hidden_dim).
    ctx_len : number of context tokens.
    pool    : "last" | "mean" | "all"

    Returns
    -------
    dict with keys from {"last", "mean"} depending on `pool`.
    """
    if pool == "last":
        return {"last": pool_last(h, ctx_len)}
    if pool == "mean":
        return {"mean": pool_mean(h, ctx_len)}
    if pool == "all":
        return {
            "last": pool_last(h, ctx_len),
            "mean": pool_mean(h, ctx_len),
        }
    raise ValueError(f"Unknown pool strategy '{pool}'. Choose from: last, mean, all.")


# ─────────────────────────────────────────────────────────────────────────────
# Main extraction
# ─────────────────────────────────────────────────────────────────────────────

def _get_final_norm(model: PreTrainedModel):
    """Retrieve the final residual-stream norm from a HuggingFace causal LM.

    For Qwen / Llama / Mistral style models the layout is:
        model.model.norm   ← RMSNorm applied to the residual stream
                              before the LM head.

    Raises RuntimeError if the expected attribute is absent so the caller
    gets a clear message rather than a silent wrong result.
    """
    inner = getattr(model, "model", None)
    if inner is None:
        raise RuntimeError(
            "model.model not found. This extractor assumes a HuggingFace "
            "CausalLM where the transformer body is at model.model "
            "(Qwen/Llama/Mistral layout)."
        )
    norm = getattr(inner, "norm", None)
    if norm is None:
        raise RuntimeError(
            "model.model.norm not found. Cannot apply the final RMSNorm. "
            "Check the model architecture."
        )
    return norm


def extract_hidden(
    model:          PreTrainedModel,
    input_ids:      Tensor,
    attention_mask: Tensor | None,
    ctx_len:        int,
    layers:         list[int] | Literal["all"],
    pool:           Literal["last", "mean", "all"],
) -> dict[str, Tensor]:
    """Run one forward pass and return pooled hidden states along the residual stream.

    HuggingFace returns hidden_states as a tuple of (num_layers + 1) tensors:
        hidden_states[0]   : embedding output  (before any transformer layer)
        hidden_states[i+1] : residual stream after transformer layer i

    All states are raw residuals (pre final-norm). For the last transformer
    layer only, an additional normed version is stored under
    "hidden/{last}_normed".

    Parameters
    ----------
    model          : HuggingFace CausalLM (eval mode recommended).
    input_ids      : (1, seq_len) token ids.
    attention_mask : (1, seq_len) or None.
    ctx_len        : number of context tokens (output tokens start here).
    layers         : list of layer indices to extract, or "all".
                     Layer 0 = embedding, layer i = after transformer block i.
    pool           : pooling strategy applied to the output-token positions.

    Returns
    -------
    Flat dict mapping "{stat}.{shorthand}" → half-precision Tensor on CPU.

    Examples of keys produced
    -------------------------
        "last.hidden/0"           embedding layer, last-token pool
        "mean.hidden/35"          layer 35, mean pool
        "last.hidden/35_normed"   layer 35 (final) after model.model.norm, last-token pool
    """
    was_training = model.training
    model.eval()

    # ── Determine which layer indices to extract ──────────────────────────────
    # hidden_states tuple has length num_hidden_layers + 1:
    #   index 0        → embedding
    #   index 1..N     → after transformer blocks 0..(N-1)
    # We expose this as layer indices 0..N, matching the tuple indices directly.
    n_layers = model.config.num_hidden_layers
    all_layer_indices = list(range(n_layers + 1))   # 0 = embed, 1..N = transformer

    if layers == "all":
        wanted: set[int] = set(all_layer_indices)
    else:
        out_of_range = [i for i in layers if i not in all_layer_indices]
        if out_of_range:
            raise ValueError(
                f"Layer indices {out_of_range} are out of range. "
                f"Valid range: 0..{n_layers} (0=embedding, 1..{n_layers}=transformer)."
            )
        wanted = set(layers)

    final_layer_idx = n_layers  # the last transformer block's tuple index

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        outputs = model(
            input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )

    # hidden_states: tuple of (batch=1, seq_len, hidden_dim) tensors
    hidden_states = outputs.hidden_states  # length: n_layers + 1

    # ── Pool and collect ──────────────────────────────────────────────────────
    result: dict[str, Tensor] = {}

    for layer_idx in sorted(wanted):
        # Shape: (seq_len, hidden_dim) — squeeze the batch dim
        h = hidden_states[layer_idx][0].float()  # cast to float32 before pooling

        shorthand = layer_to_shorthand(layer_idx)
        for stat_name, vec in _apply_pool(h, ctx_len, pool).items():
            result[f"{stat_name}.{shorthand}"] = vec

        # For the final transformer layer, also store the normed version
        if layer_idx == final_layer_idx:
            final_norm = _get_final_norm(model)
            with torch.no_grad():
                h_normed = final_norm(hidden_states[layer_idx][0].unsqueeze(0))[0].float()
            normed_shorthand = f"hidden/{layer_idx}_normed"
            for stat_name, vec in _apply_pool(h_normed, ctx_len, pool).items():
                result[f"{stat_name}.{normed_shorthand}"] = vec

    if was_training:
        model.train()

    return result