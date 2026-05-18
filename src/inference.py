"""Chunked overlap-add inference for long audio tracks.

Separating a full MUSDB18 track (typically 3-6 minutes) in one pass
would exceed GPU memory. This module splits the mixture into overlapping
chunks, runs the model on each chunk, and re-assembles the output via
overlap-add.
"""

from __future__ import annotations

import time
from typing import Optional

import torch


def separate_track(
    model: torch.nn.Module,
    mixture: torch.Tensor,
    chunk_size: int = 65536,
    hop_size: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, float]:
    """Run source separation on a full audio track using overlap-add.

    Args:
        model:      Separation model; expects ``[1, audio_channels, time]`` or
                    ``[1, time]`` input.
        mixture:    Audio tensor with shape ``[audio_channels, time]`` or
                    ``[time,]``.
        chunk_size: Number of samples per chunk.
        hop_size:   Number of samples between chunk starts. Defaults to
                    ``chunk_size // 2`` (50 % overlap).
        device:     Device to run inference on. Defaults to the device of the
                    model's first parameter.

    Returns:
        Tuple of:
        - Separated sources: ``[num_sources, audio_channels, time]`` or
          ``[num_sources, time]`` depending on the model output shape.
        - Elapsed wall-clock time in milliseconds.
    """
    if hop_size is None:
        hop_size = chunk_size // 2

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Normalise input shape to [audio_channels, time]
    original_dim = mixture.dim()
    if original_dim == 1:
        mixture = mixture.unsqueeze(0)  # [1, time]

    audio_channels, total_length = mixture.shape

    # Padding so the last chunk is full
    pad_length = 0
    if total_length < chunk_size:
        pad_length = chunk_size - total_length
        mixture = torch.nn.functional.pad(mixture, (0, pad_length))
        total_length = mixture.shape[-1]

    model.eval()
    start_time = time.perf_counter()

    output_sum: Optional[torch.Tensor] = None
    weight_sum: Optional[torch.Tensor] = None

    with torch.inference_mode():
        starts = list(range(0, total_length - chunk_size + 1, hop_size))
        if not starts or starts[-1] + chunk_size < total_length:
            starts.append(total_length - chunk_size)

        for start in starts:
            end = start + chunk_size
            chunk = mixture[:, start:end].to(device)  # [C, chunk_size]
            # Model expects [batch, C, T] — use batch=1
            chunk_input = chunk.unsqueeze(0)  # [1, C, T]
            sep_chunk = model(chunk_input)  # [1, S, C, T] or [1, S, T]
            sep_chunk = sep_chunk.squeeze(0)  # [S, C, T] or [S, T]
            sep_chunk = sep_chunk.cpu()

            if output_sum is None:
                # Initialise accumulator tensors
                num_sources = sep_chunk.shape[0]
                extra_dims = sep_chunk.shape[1:-1]  # e.g. () or (C,)
                output_sum = torch.zeros(num_sources, *extra_dims, total_length)
                weight_sum = torch.zeros(total_length)

            # Hanning window for smooth overlap-add
            win = torch.hann_window(chunk_size, periodic=False)
            output_sum[..., start:end] += sep_chunk * win
            weight_sum[start:end] += win

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0

    # Normalise by accumulated window weights (avoid division by zero)
    weight_sum = weight_sum.clamp(min=1e-8)
    output = output_sum / weight_sum

    # Remove padding
    if pad_length > 0:
        output = output[..., :-pad_length]

    return output, elapsed_ms


def separate_batch(
    model: torch.nn.Module,
    mixture: torch.Tensor,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, float]:
    """Separate a short segment in one forward pass (training/validation).

    Args:
        model:      Separation model.
        mixture:    ``[batch, audio_channels, time]`` or ``[batch, time]``.
        device:     Inference device.

    Returns:
        Tuple of:
        - ``[batch, num_sources, ...]`` separated tensor.
        - Elapsed time in milliseconds.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    mixture = mixture.to(device)
    model.eval()
    start_time = time.perf_counter()
    with torch.inference_mode():
        output = model(mixture)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return output, elapsed_ms
