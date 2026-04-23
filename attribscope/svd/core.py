import torch
from typing import Callable, Literal

DistMetric = Literal["l1", "l2", "cosine"]



# ═══════════════════════════════════════════════════════════════════
# Family 2: Spectral / subspace-based
#   Score = projection onto (or residual from) an SVD subspace.
# ═══════════════════════════════════════════════════════════════════

def _run_svd(G: torch.Tensor, c: int) -> torch.Tensor:
    """Top-c right singular vectors of G, shape (d, c)."""
    _, _, V = torch.svd_lowrank(G, q=c, niter=10)
    return V


def projection_svd(
    G: torch.Tensor,
    c: int = 1,
    centered: bool = True,
) -> torch.Tensor:
    """Mean squared projection of each row onto the top-c right singular vectors.

        τᵢ = (1/c) Σⱼ ⟨g̃ᵢ, vⱼ⟩²

    Reference : top-c singular subspace of G (optionally mean-centered).
    SAL interpretation (Du et al. 2024): the leading singular direction
    aligns with the outlier direction, so a large projection flags OOD rows.
    Higher score = more anomalous.
    """
    G_f = G.float()
    if centered:
        G_f = G_f - G_f.mean(dim=0)
    V = _run_svd(G_f, c)
    return (G_f @ V).square().mean(dim=1).to(G.dtype)


def reconstruction_svd(
    G: torch.Tensor,
    c: int = 5,
    centered: bool = True,
) -> torch.Tensor:
    """Residual L2 norm after projecting each row onto the top-c SVD subspace.

    Reference : top-c singular subspace of G (optionally mean-centered).
    Rows well-explained by the dominant subspace have small residuals;
    rows off this subspace (structural outliers) have large residuals.
    Higher score = more anomalous.
    """
    G_f = G.float()
    G_c = G_f - G_f.mean(dim=0) if centered else G_f
    V     = _run_svd(G_c, c)
    G_rec = (G_c @ V) @ V.T
    return torch.norm(G_c - G_rec, dim=1).to(G.dtype)


# ═══════════════════════════════════════════════════════════════════
# Role-based orchestration
# ═══════════════════════════════════════════════════════════════════

def group_role(role: str) -> str:
    if role.startswith("Orchestrator (->"):
        return "Orchestrator (-> Agent)"
    return role


def split_by_role(G: torch.Tensor, index: list) -> dict:
    ROLES = set(group_role(idx.role) for idx in index)
    role_Gs = {}
    for role in ROLES:
        role_idxs = [idx.row for idx in index if group_role(idx.role) == role]
        role_mask = torch.tensor(
            [group_role(idx.role) == role for idx in index],
            device=G.device,
        )
        role_Gs[role] = {"idx": role_idxs, "G": G[role_mask]}
    return role_Gs


def compute_split_scores(G: torch.Tensor, index: list, scoring: Callable) -> torch.Tensor:
    role_Gs = split_by_role(G, index)
    scores  = torch.zeros(G.shape[0], device=G.device, dtype=G.dtype)
    for role_data in role_Gs.values():
        scores[role_data["idx"]] = scoring(role_data["G"])
    return scores


def compute_scores(G: torch.Tensor, index: list, scoring: Callable) -> torch.Tensor:
    return scoring(G)


def make_scoring_fn(index, scoring, name: str = "scoring_by_role"):
    def scoring_fn(G: torch.Tensor) -> torch.Tensor:
        return compute_scores(G, index, scoring)
    scoring_fn.__name__ = name
    return scoring_fn


# ═══════════════════════════════════════════════════════════════════
# Factories
# ═══════════════════════════════════════════════════════════════════

def make_projection_scoring(c: int = 1, centered: bool = True):
    def scoring(G): return projection_svd(G, c=c, centered=centered)
    return scoring

def make_reconstruction_scoring(c: int = 5, centered: bool = True):
    def scoring(G): return reconstruction_svd(G, c=c, centered=centered)
    return scoring