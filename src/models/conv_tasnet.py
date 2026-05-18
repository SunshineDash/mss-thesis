"""Baseline Conv-TasNet model for 4-source music separation.

Architecture following Luo & Mesgarani (2019) — "Conv-TasNet: Surpassing Ideal
Time-Frequency Magnitude Masking for Speech Separation" — adapted for stereo
multi-source separation on MUSDB18-HQ.

Key differences from the original speech-separation model:
  - ``audio_channels`` = 2 for stereo (default) or 1 for mono.
  - Encoder/decoder operate per-channel; the separator processes the average of
    encoded left/right features so the bottleneck is channel-agnostic.
  - Output has shape ``[batch, num_sources, audio_channels, time]``.
"""

from __future__ import annotations

import torch
from torch import nn

try:
    from .blocks import ChannelwiseLayerNorm, TCNSeparator
except ImportError:  # Allow running files directly from within src/
    from blocks import ChannelwiseLayerNorm, TCNSeparator  # type: ignore[no-redef]


class ConvTasNet(nn.Module):
    """Conv-TasNet for multi-source music separation.

    Args:
        num_sources:         Number of output sources (4 for MUSDB18).
        audio_channels:      Number of input/output audio channels
                             (1 = mono, 2 = stereo).
        encoder_channels:    Number of encoder filters (N in the paper).
        bottleneck_channels: Bottleneck channel count (B in the paper).
        hidden_channels:     Temporal block hidden channel count (H).
        skip_channels:       Skip-connection channel count (Sc).
        num_blocks:          Temporal blocks per repeat cycle (X).
        num_repeats:         Number of repeat cycles (R).
        kernel_size:         Kernel size for the temporal d-conv (P).
        encoder_kernel_size: Encoder / decoder kernel size (L).
        encoder_stride:      Encoder / decoder stride (L/2).
        causal:              Use causal (streaming) padding in TCN.
        dsc_layers:          Number of TCN blocks replaced with DSC (0 = none).
    """

    def __init__(
        self,
        num_sources: int = 4,
        audio_channels: int = 2,
        encoder_channels: int = 512,
        bottleneck_channels: int = 128,
        hidden_channels: int = 512,
        skip_channels: int = 128,
        num_blocks: int = 8,
        num_repeats: int = 3,
        kernel_size: int = 3,
        encoder_kernel_size: int = 16,
        encoder_stride: int = 8,
        causal: bool = False,
        dsc_layers: int = 0,
        **_: object,  # Ignore unknown config keys gracefully
    ) -> None:
        super().__init__()
        self.num_sources = num_sources
        self.audio_channels = audio_channels
        self.encoder_channels = encoder_channels

        # ------------------------------------------------------------------ #
        # Encoder  (shared weights across audio_channels via grouped conv)    #
        # ------------------------------------------------------------------ #
        # We encode each audio channel independently using the same filters.
        self.encoder = nn.Conv1d(
            audio_channels,
            encoder_channels * audio_channels,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride,
            groups=audio_channels,
            bias=False,
        )

        # ------------------------------------------------------------------ #
        # Bottleneck  (reduce from encoder_channels to bottleneck_channels)   #
        # ------------------------------------------------------------------ #
        # Applied to the mean of per-channel encoder features.
        self.layer_norm = ChannelwiseLayerNorm(encoder_channels)
        self.bottleneck_conv = nn.Conv1d(encoder_channels, bottleneck_channels, kernel_size=1)

        # ------------------------------------------------------------------ #
        # Separator (TCN)                                                      #
        # ------------------------------------------------------------------ #
        self.separator = TCNSeparator(
            bottleneck_channels=bottleneck_channels,
            hidden_channels=hidden_channels,
            skip_channels=skip_channels,
            num_blocks=num_blocks,
            num_repeats=num_repeats,
            kernel_size=kernel_size,
            causal=causal,
            dsc_layers=dsc_layers,
        )

        # ------------------------------------------------------------------ #
        # Mask head  → one mask per source × audio_channel                    #
        # ------------------------------------------------------------------ #
        self.mask_head = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(skip_channels, num_sources * encoder_channels * audio_channels, kernel_size=1),
        )

        # ------------------------------------------------------------------ #
        # Decoder                                                              #
        # ------------------------------------------------------------------ #
        self.decoder = nn.ConvTranspose1d(
            encoder_channels * audio_channels,
            audio_channels,
            kernel_size=encoder_kernel_size,
            stride=encoder_stride,
            groups=audio_channels,
            bias=False,
        )

        self._encoder_stride = encoder_stride

    # ---------------------------------------------------------------------- #
    # Forward                                                                  #
    # ---------------------------------------------------------------------- #

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        """Separate sources from a stereo or mono mixture.

        Args:
            mixture: ``[batch, audio_channels, time]``
                     Also accepts ``[batch, time]`` for mono convenience.

        Returns:
            ``[batch, num_sources, audio_channels, time]``
        """
        # -- Normalise input shape
        if mixture.dim() == 2:
            mixture = mixture.unsqueeze(1)  # [B, 1, T]

        if mixture.dim() != 3:
            raise ValueError(f"Expected 2-D or 3-D input, got {mixture.dim()}-D.")

        batch, in_channels, original_length = mixture.shape
        if in_channels != self.audio_channels:
            raise ValueError(
                f"Model has audio_channels={self.audio_channels}, "
                f"but input has {in_channels} channels."
            )

        # -- Encoder:  [B, C*N, frames]
        encoded = self.encoder(mixture)        # [B, C*N, frames]
        frames = encoded.shape[-1]

        # Reshape to [B, C, N, frames] then mean-pool over channels → [B, N, frames]
        enc_reshaped = encoded.view(batch, self.audio_channels, self.encoder_channels, frames)
        enc_mean = enc_reshaped.mean(dim=1)    # [B, N, frames]

        # -- Bottleneck
        features = self.bottleneck_conv(self.layer_norm(enc_mean))  # [B, B_ch, frames]

        # -- Separator
        skip_sum = self.separator(features)    # [B, Sc, frames]

        # -- Masks: [B, S*C*N, frames] → sigmoid → [B, S, C, N, frames]
        masks = self.mask_head(skip_sum)       # [B, S*C*N, frames]
        masks = torch.sigmoid(masks)
        masks = masks.view(batch, self.num_sources, self.audio_channels, self.encoder_channels, frames)

        # -- Apply masks to encoded features, then decode each source
        # enc_reshaped: [B, C, N, frames] → expand → [B, 1, C, N, frames]
        enc_exp = enc_reshaped.unsqueeze(1)    # [B, 1, C, N, frames]
        # masked:  [B, S, C, N, frames]
        masked = masks * enc_exp

        # Merge source × channel × encoder dims for batched decode
        # [B, S, C, N, frames] → [B*S, C*N, frames]
        masked_flat = masked.view(batch * self.num_sources, self.audio_channels * self.encoder_channels, frames)

        # -- Decoder
        decoded = self.decoder(masked_flat)    # [B*S, C, padded_time]
        if decoded.shape[-1] < original_length:
            pad_len = original_length - decoded.shape[-1]
            decoded = torch.nn.functional.pad(decoded, (0, pad_len))
        decoded = decoded[..., :original_length]  # trim to original length

        # Reshape to [B, S, C, T]
        output = decoded.view(batch, self.num_sources, self.audio_channels, original_length)
        return output
