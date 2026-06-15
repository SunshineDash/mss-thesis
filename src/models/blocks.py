"""Building blocks for Conv-TasNet and DSC-Conv-TasNet.

TCN-based temporal block architecture following Luo & Mesgarani (2019):
  "Conv-TasNet: Surpassing Ideal Time-Frequency Magnitude Masking for
   Speech Separation"

Each TemporalBlock implements a residual+skip dilated 1-D convolution.
The DepthwiseSeparableConv1d can replace regular D-Conv inside blocks
for the DSC ablation study.
"""

from __future__ import annotations

import torch
from torch import nn


class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise layer normalisation applied over the channel dimension.

    Input shape: ``[batch, channels, length]``
    """

    def __init__(self, channels: int, eps: float = 1e-8) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(1, channels, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise-separable 1-D convolution.

    Args:
        in_channels:  Input channel count.
        hidden_channels: Intermediate depthwise channel count
                         (same as in_channels for the separator blocks).
        kernel_size:  Kernel size for the depthwise conv.
        dilation:     Dilation factor.
        causal:       If True use causal (one-sided) padding.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.causal = causal
        padding = (kernel_size - 1) * dilation
        if not causal:
            padding = padding // 2

        self.depthwise = nn.Conv1d(
            hidden_channels,
            hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding if not causal else 0,
            groups=hidden_channels,
        )
        self.causal_pad = (kernel_size - 1) * dilation if causal else 0
        self.prelu = nn.PReLU()
        self.norm = ChannelwiseLayerNorm(hidden_channels)
        self.pointwise = nn.Conv1d(hidden_channels, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.causal_pad > 0:
            x = nn.functional.pad(x, (self.causal_pad, 0))
        h = self.depthwise(x)
        h = self.norm(self.prelu(h))
        return self.pointwise(h)


class TemporalBlock(nn.Module):
    """One residual temporal block (1-D TDCN block).

    Structure::

        Input -> 1x1 Conv (bottleneck -> hidden) -> PReLU -> cLN
               -> D-Conv (dilated, depthwise sep or regular) -> PReLU -> cLN
               -> 1x1 Conv (hidden -> bottleneck) [res]
               -> 1x1 Conv (hidden -> skip_channels) [skip]

    Args:
        bottleneck_channels: Channel count in the bottleneck (TCN input/output).
        hidden_channels:     Channel count inside the block.
        skip_channels:       Channel count for the skip output.
        kernel_size:         Kernel size of the (DSC) d-conv.
        dilation:            Dilation factor.
        causal:              Causal or non-causal padding.
        use_dsc:             Replace regular d-conv with depthwise-separable.
    """

    def __init__(
        self,
        bottleneck_channels: int,
        hidden_channels: int,
        skip_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        causal: bool = False,
        use_dsc: bool = False,
    ) -> None:
        super().__init__()
        self.use_dsc = use_dsc

        # 1×1 expand
        self.conv_in = nn.Conv1d(bottleneck_channels, hidden_channels, kernel_size=1)
        self.prelu_in = nn.PReLU()
        self.norm_in = ChannelwiseLayerNorm(hidden_channels)

        # depthwise conv (dilated)
        causal_padding = (kernel_size - 1) * dilation if causal else 0
        non_causal_padding = (kernel_size - 1) * dilation // 2

        if use_dsc:
            self.d_conv = DepthwiseSeparableConv1d(
                in_channels=hidden_channels,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                causal=causal,
            )
        else:
            self.d_conv = nn.Sequential(
                nn.Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=causal_padding if causal else non_causal_padding,
                    groups=1,
                ),
            )
            self._causal_pad = causal_padding
            self._causal = causal

        self.prelu_d = nn.PReLU()
        self.norm_d = ChannelwiseLayerNorm(hidden_channels)

        # 1×1 res & skip
        self.conv_res = nn.Conv1d(hidden_channels, bottleneck_channels, kernel_size=1)
        self.conv_skip = nn.Conv1d(hidden_channels, skip_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (residual_output, skip_output)."""
        h = self.norm_in(self.prelu_in(self.conv_in(x)))

        if self.use_dsc:
            h = self.d_conv(h)
        else:
            if self._causal and self._causal_pad > 0:
                h = nn.functional.pad(h, (self._causal_pad, 0))
            h = self.d_conv(h)

        h = self.norm_d(self.prelu_d(h))
        res = x + self.conv_res(h)
        skip = self.conv_skip(h)
        return res, skip


class TCNSeparator(nn.Module):
    """Stacked TCN separator module.

    ``num_blocks`` temporal blocks are repeated ``num_repeats`` times.
    The dilation doubles within each repeat cycle:
    ``d = 2^k`` for k in ``[0, num_blocks-1]``.

    Args:
        bottleneck_channels: Bottleneck channel count.
        hidden_channels:     Hidden channel count inside each temporal block.
        skip_channels:       Skip-connection channel count.
        num_blocks:          Number of temporal blocks per repeat cycle.
        num_repeats:         Number of repeat cycles.
        kernel_size:         Kernel size for the d-conv inside each block.
        causal:              Causal / non-causal.
        dsc_layers:          Number of blocks (counting from the first block of
                             the first repeat) to replace with DSC. ``0`` means
                             no DSC layers (baseline). Pass ``-1`` to replace all.
    """

    def __init__(
        self,
        bottleneck_channels: int = 128,
        hidden_channels: int = 512,
        skip_channels: int = 128,
        num_blocks: int = 8,
        num_repeats: int = 3,
        kernel_size: int = 3,
        causal: bool = False,
        dsc_layers: int = 0,
    ) -> None:
        super().__init__()
        total_blocks = num_blocks * num_repeats
        if dsc_layers < 0:
            dsc_layers = total_blocks
        if dsc_layers > total_blocks:
            raise ValueError(
                f"dsc_layers={dsc_layers} exceeds total temporal blocks "
                f"({num_blocks} blocks × {num_repeats} repeats = {total_blocks})."
            )

        self.blocks = nn.ModuleList()
        block_idx = 0
        for _ in range(num_repeats):
            for b in range(num_blocks):
                use_dsc = block_idx < dsc_layers
                dilation = 2 ** b
                self.blocks.append(
                    TemporalBlock(
                        bottleneck_channels=bottleneck_channels,
                        hidden_channels=hidden_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        causal=causal,
                        use_dsc=use_dsc,
                    )
                )
                block_idx += 1

        self.num_dsc = dsc_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the separator; returns the sum of all skip connections."""
        skip_sum: torch.Tensor | None = None
        h = x
        for block in self.blocks:
            h, skip = block(h)
            if skip_sum is None:
                skip_sum = skip
            else:
                skip_sum = skip_sum + skip
        assert skip_sum is not None
        return skip_sum
