"""Metrics helpers for the MUSDB18 ablation study.

Wraps museval BSS_eval-based metrics (SDR/SIR/SAR/ISR) and also
computes SI-SDR in PyTorch.

Terminology used in museval v0.4:
  - track_scores() → per-frame SDR, SIR, SAR, ISR arrays
  - aggregate_scores() → median over frames & tracks
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import torch

try:
    import museval
    _MUSEVAL_AVAILABLE = True
except ImportError:
    _MUSEVAL_AVAILABLE = False


EPS = 1e-8


# ---------------------------------------------------------------------------
# PyTorch SI-SDR (used during training/validation)
# ---------------------------------------------------------------------------

def compute_si_sdr(
    estimate: torch.Tensor,
    target: torch.Tensor,
) -> float:
    """Compute mean SI-SDR over a batch of sources.

    Args:
        estimate: ``[batch, num_sources, time]``
        target:   ``[batch, num_sources, time]``

    Returns:
        Mean SI-SDR in dB (float).
    """
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)

    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = (target ** 2).sum(dim=-1, keepdim=True) + EPS
    proj = (dot / target_energy) * target
    noise = estimate - proj

    proj_e = (proj ** 2).sum(dim=-1) + EPS
    noise_e = (noise ** 2).sum(dim=-1) + EPS
    scores = 10 * torch.log10(proj_e / noise_e)  # [batch, num_sources]
    return float(scores.mean().item())


# ---------------------------------------------------------------------------
# museval-based BSS metrics
# ---------------------------------------------------------------------------

def _to_numpy_track(audio: np.ndarray) -> np.ndarray:
    """Ensure shape ``[num_frames, num_channels]`` as float32."""
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]
    elif audio.ndim == 2 and audio.shape[0] < audio.shape[1]:
        # Check if it looks like (channels, time); transpose to (time, channels)
        audio = audio.T
    return audio.astype(np.float32)


def evaluate_track(
    estimates: np.ndarray,
    references: np.ndarray,
    sample_rate: int = 44100,
    win: float = 1.0,
    hop: float = 1.0,
) -> dict[str, list[float]]:
    """Compute frame-wise BSS metrics for a single track.

    Args:
        estimates:   ``[num_sources, num_channels, num_frames]``  numpy array.
        references:  ``[num_sources, num_channels, num_frames]``  numpy array.
        sample_rate: Sample rate used for window/hop sizes.
        win:         Window size in seconds (museval default 1 s).
        hop:         Hop size in seconds.

    Returns:
        Dict with keys ``sdr``, ``sir``, ``sar``, ``isr`` — each a list of
        per-source median values (floats, NaN removed).
    """
    if not _MUSEVAL_AVAILABLE:
        raise RuntimeError("museval is not installed. Run: pip install museval")

    num_sources = estimates.shape[0]
    win_samples = int(win * sample_rate)
    hop_samples = int(hop * sample_rate)

    sdrs, sirs, sars, isrs = [], [], [], []
    for s in range(num_sources):
        est_s = _to_numpy_track(estimates[s])   # [time, channels]
        ref_s = _to_numpy_track(references[s])  # [time, channels]

        ref_s = ref_s[np.newaxis, :, :]  # [1, time, channels]
        est_s = est_s[np.newaxis, :, :]  # [1, time, channels]
        assert ref_s.ndim == 3 and ref_s.shape[0] == 1, \
            f"Expected [1, time, channels], got {ref_s.shape}"
        sdr_arr, isr_arr, sir_arr, sar_arr, _ = museval.metrics.bss_eval(
            ref_s,
            est_s,
            compute_permutation=False,
            window=win_samples,
            hop=hop_samples,
            framewise_filters=False,
        )
        # Drop NaN frames and take median
        sdrs.append(float(np.nanmedian(sdr_arr)))
        sirs.append(float(np.nanmedian(sir_arr)))
        sars.append(float(np.nanmedian(sar_arr)))
        isrs.append(float(np.nanmedian(isr_arr)))

    return {"sdr": sdrs, "sir": sirs, "sar": sars, "isr": isrs}


def aggregate_track_scores(
    per_track: Iterable[Mapping[str, list[float]]],
) -> dict[str, float]:
    """Aggregate per-source, per-track metric dicts into a single summary.

    Args:
        per_track: Iterable of dicts as returned by ``evaluate_track``.

    Returns:
        Dict with mean (over tracks and sources) for each metric key.
    """
    all_sdr: list[float] = []
    all_sir: list[float] = []
    all_sar: list[float] = []
    all_isr: list[float] = []

    for track in per_track:
        all_sdr.extend(v for v in track.get("sdr", []) if not np.isnan(v))
        all_sir.extend(v for v in track.get("sir", []) if not np.isnan(v))
        all_sar.extend(v for v in track.get("sar", []) if not np.isnan(v))
        all_isr.extend(v for v in track.get("isr", []) if not np.isnan(v))

    def _safe_mean(vals: list[float]) -> float:
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "sdr": _safe_mean(all_sdr),
        "sir": _safe_mean(all_sir),
        "sar": _safe_mean(all_sar),
        "isr": _safe_mean(all_isr),
    }
