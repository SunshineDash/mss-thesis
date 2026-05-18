"""DSC-Conv-TasNet: Conv-TasNet with depthwise-separable convolution ablation.

Instead of maintaining a separate forward() the DSC variant is simply
ConvTasNet with ``dsc_layers > 0``.  This file exists to:

  1. Provide a thin subclass that validates the ``dsc_layers`` parameter and
     documents the ablation clearly in the class name.
  2. Keep backward compatibility — old configs that use ``dsc_rate`` will
     automatically be translated to ``dsc_layers``.

Parameter mapping from old-style config to new style:
  ``dsc_rate`` × (num_blocks × num_repeats) → ``dsc_layers`` (integer).

Example::

    # configs/dsc5.yaml
    model:
      type: dsc_conv_tasnet
      num_sources: 4
      encoder_channels: 512
      bottleneck_channels: 128
      hidden_channels: 512
      skip_channels: 128
      num_blocks: 8
      num_repeats: 3
      kernel_size: 3
      dsc_layers: 5
"""

from __future__ import annotations

import math

try:
    from .conv_tasnet import ConvTasNet
except ImportError:  # Allow running files directly from within src/
    from conv_tasnet import ConvTasNet  # type: ignore[no-redef]


class DSCConvTasNet(ConvTasNet):
    """Conv-TasNet variant where the first ``dsc_layers`` TCN blocks use DSC.

    Args:
        dsc_layers: Number of temporal blocks (counting from block 0 of
                    repeat 0) to replace with Depthwise-Separable Convolution.
                    Must satisfy ``0 <= dsc_layers <= num_blocks * num_repeats``.
        dsc_rate:   **Deprecated** — fraction of total blocks to replace.
                    Ignored when ``dsc_layers`` is provided explicitly.
                    Kept for backward compatibility with old configs.
        **kwargs:   All other Conv-TasNet hyper-parameters are passed through.
    """

    def __init__(
        self,
        dsc_layers: int | None = None,
        dsc_rate: float | None = None,
        **kwargs: object,
    ) -> None:
        num_blocks = int(kwargs.get("num_blocks", 8))
        num_repeats = int(kwargs.get("num_repeats", 3))
        total_blocks = num_blocks * num_repeats

        if dsc_layers is None:
            if dsc_rate is not None:
                # Back-compat: translate rate → count
                dsc_layers = max(0, min(total_blocks, math.ceil(dsc_rate * total_blocks)))
            else:
                dsc_layers = 0

        if not (0 <= dsc_layers <= total_blocks):
            raise ValueError(
                f"dsc_layers={dsc_layers} is out of range "
                f"[0, {total_blocks}] for num_blocks={num_blocks}, "
                f"num_repeats={num_repeats}."
            )

        # Inject dsc_layers into kwargs so ConvTasNet/TCNSeparator receives it
        kwargs["dsc_layers"] = dsc_layers
        super().__init__(**kwargs)
        self.dsc_layers = dsc_layers

    def extra_repr(self) -> str:
        total = int(self.separator.blocks.__len__())
        return (
            f"dsc_layers={self.dsc_layers}/{total}, "
            f"num_sources={self.num_sources}, "
            f"audio_channels={self.audio_channels}"
        )
