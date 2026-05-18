"""Smoke test: verifies the full pipeline on synthetic data (no real dataset needed).

Run from the project root:
    python scripts/smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make sure src/ is importable when running from the project root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from models.conv_tasnet import ConvTasNet
from models.dsc_conv_tasnet import DSCConvTasNet
from losses import si_sdr_loss

# --------------------------------------------------------------------------- #
# 1 — Random mixture tensor [1, 2, 44100]  (1 sec stereo)
# --------------------------------------------------------------------------- #
print("Check 1: creating random mixture [1, 2, 44100] ...")
mixture = torch.randn(1, 2, 44100)
assert mixture.shape == (1, 2, 44100), f"Unexpected shape: {mixture.shape}"
print(f"  mixture.shape = {tuple(mixture.shape)}  ✓")

# --------------------------------------------------------------------------- #
# 2 — Forward pass: ConvTasNet (baseline)
# --------------------------------------------------------------------------- #
print("\nCheck 2a: ConvTasNet forward pass ...")
baseline = ConvTasNet(
    num_sources=4,
    audio_channels=2,
    encoder_channels=512,
    bottleneck_channels=128,
    hidden_channels=512,
    skip_channels=128,
    num_blocks=8,
    num_repeats=3,
    kernel_size=3,
    encoder_kernel_size=16,
    encoder_stride=8,
    causal=False,
)
baseline.eval()
with torch.no_grad():
    out_baseline = baseline(mixture)
print(f"  ConvTasNet output shape: {tuple(out_baseline.shape)}")

# --------------------------------------------------------------------------- #
# 2 — Forward pass: DSCConvTasNet(dsc_layers=5)
# --------------------------------------------------------------------------- #
print("\nCheck 2b: DSCConvTasNet(dsc_layers=5) forward pass ...")
dsc_model = DSCConvTasNet(
    dsc_layers=5,
    num_sources=4,
    audio_channels=2,
    encoder_channels=512,
    bottleneck_channels=128,
    hidden_channels=512,
    skip_channels=128,
    num_blocks=8,
    num_repeats=3,
    kernel_size=3,
    encoder_kernel_size=16,
    encoder_stride=8,
    causal=False,
)
dsc_model.eval()
with torch.no_grad():
    out_dsc = dsc_model(mixture)
print(f"  DSCConvTasNet output shape: {tuple(out_dsc.shape)}")

# --------------------------------------------------------------------------- #
# 3 — Check output shape == [1, 4, 2, 44100]
# --------------------------------------------------------------------------- #
print("\nCheck 3: output shape == [1, 4, 2, 44100] ...")
expected_shape = (1, 4, 2, 44100)
assert out_baseline.shape == expected_shape, (
    f"ConvTasNet shape mismatch: expected {expected_shape}, got {tuple(out_baseline.shape)}"
)
assert out_dsc.shape == expected_shape, (
    f"DSCConvTasNet shape mismatch: expected {expected_shape}, got {tuple(out_dsc.shape)}"
)
print(f"  Both models output shape {expected_shape}  ✓")

# --------------------------------------------------------------------------- #
# 4 — SI-SDR loss on random targets
# --------------------------------------------------------------------------- #
print("\nCheck 4: si_sdr_loss on random targets ...")
# Re-run baseline with grad enabled for backward check
dsc_model.train()
estimate = dsc_model(mixture)  # [1, 4, 2, 44100]
B, S, C, T = estimate.shape
est_flat = estimate.reshape(B, S * C, T)     # [1, 8, 44100]
targets = torch.randn_like(est_flat)         # [1, 8, 44100]
loss = si_sdr_loss(est_flat, targets)
print(f"  si_sdr_loss = {loss.item():.4f}  ✓")

# --------------------------------------------------------------------------- #
# 5 — Backward pass
# --------------------------------------------------------------------------- #
print("\nCheck 5: backward() ...")
loss.backward()
print("  backward() completed without errors  ✓")

# --------------------------------------------------------------------------- #
# 6 — ReduceLROnPlateau instantiation
# --------------------------------------------------------------------------- #
print("\nCheck 6: ReduceLROnPlateau ...")
optimizer = torch.optim.Adam(dsc_model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5
)
scheduler.step(loss.item())
print(f"  ReduceLROnPlateau created and stepped OK  ✓")

# --------------------------------------------------------------------------- #
# 7 — museval import
# --------------------------------------------------------------------------- #
print("\nCheck 7: import museval ...")
import museval
try:
    from importlib.metadata import version as _pkg_version
    _museval_ver = _pkg_version("museval")
except Exception:
    _museval_ver = getattr(museval, "__version__", "unknown")
print(f"  museval version: {_museval_ver}  ✓")

# --------------------------------------------------------------------------- #
# ALL CHECKS PASSED
# --------------------------------------------------------------------------- #
print("\n" + "=" * 50)
print("ALL CHECKS PASSED")
print("=" * 50)
