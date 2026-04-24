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
    R: torch.Tensor, # (T, d) matrix of row reps to score
    V: torch.Tensor, # (d, c) matrix of top-c right singular vectors from G
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
    assert c <= V.shape[1], f"Requested c={c} exceeds n_components={V.shape[1]}"
    R_f = R.float()
    if centered: R_f = R_f - R_f.mean(dim=0)
    scores =  (R_f @ V[:, :c]).square().mean(dim=1).to(R.dtype)
    return scores


def reconstruction_svd(
    R: torch.Tensor,
    V: torch.Tensor,
    c: int = 5,
    centered: bool = True,
) -> torch.Tensor:
    """Residual L2 norm after projecting each row onto the top-c SVD subspace.

    Reference : top-c singular subspace of G (optionally mean-centered).
    Rows well-explained by the dominant subspace have small residuals;
    rows off this subspace (structural outliers) have large residuals.
    Higher score = more anomalous.
    """
    assert c <= V.shape[1], f"Requested c={c} exceeds n_components={V.shape[1]}"
    R_f = R.float()
    if centered: R_f = R_f - R_f.mean(dim=0)
    G_rec = (R_f @ V[:, :c]) @ V[:, :c].T
    return torch.norm(R_f - G_rec, dim=1).to(R.dtype)



# ═══════════════════════════════════════════════════════════════════
# Factories
# ═══════════════════════════════════════════════════════════════════

def make_projection_scoring(c: int = 1, centered: bool = True):
    def scoring(G): return projection_svd(G, c=c, centered=centered)
    return scoring

def make_reconstruction_scoring(c: int = 5, centered: bool = True):
    def scoring(G): return reconstruction_svd(G, c=c, centered=centered)
    return scoring