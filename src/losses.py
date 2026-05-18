"""Loss functions for source separation training.

Implements SI-SDR (Scale-Invariant Signal-to-Distortion Ratio) loss
following Luo & Mesgarani (2019) and Le Roux et al. (2019).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


EPS = 1e-8


def si_sdr(estimate: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute Scale-Invariant SDR between estimate and target.

    Args:
        estimate: ``[..., time]``
        target:   ``[..., time]`` — same shape as estimate.

    Returns:
        Per-element SI-SDR in dB, shape ``[...]``.
    """
    # Zero-mean
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    # Projection of estimate onto target
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + EPS
    target_proj = (dot / target_energy) * target

    # Noise component
    noise = estimate - target_proj

    # SI-SDR
    proj_energy = (target_proj ** 2).sum(dim=-1) + EPS
    noise_energy = (noise ** 2).sum(dim=-1) + EPS
    return 10 * torch.log10(proj_energy / noise_energy)


def si_sdr_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
    permutation_invariant: bool = False,
) -> torch.Tensor:
    """Negative mean SI-SDR loss.

    Args:
        estimate: ``[batch, num_sources, time]``
        target:   ``[batch, num_sources, time]``
        permutation_invariant: If True, use best permutation via PIT.

    Returns:
        Scalar loss (negative SI-SDR averaged over batch and sources).
    """
    if permutation_invariant:
        return pit_si_sdr_loss(estimate, target)

    scores = si_sdr(estimate, target)  # [batch, num_sources]
    return -scores.mean()


def pit_si_sdr_loss(
    estimate: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Permutation-Invariant Training loss using SI-SDR.

    Finds the permutation of estimated sources that maximises
    the total SI-SDR with the reference sources, then returns
    the negative mean SI-SDR under the best permutation.

    Args:
        estimate: ``[batch, num_sources, time]``
        target:   ``[batch, num_sources, time]``

    Returns:
        Scalar PIT-SI-SDR loss.
    """
    import itertools

    batch, num_sources, _ = estimate.shape
    perms = list(itertools.permutations(range(num_sources)))

    best_loss = None
    for perm in perms:
        perm_target = target[:, list(perm), :]  # [batch, num_sources, time]
        scores = si_sdr(estimate, perm_target)  # [batch, num_sources]
        loss = -scores.mean(dim=-1)  # [batch]
        if best_loss is None:
            best_loss = loss
        else:
            best_loss = torch.minimum(best_loss, loss)

    assert best_loss is not None
    return best_loss.mean()
