# Ablation Study: DSC Compression in Conv-TasNet for Music Source Separation

This repository contains the code, configs, trained checkpoints, and results for a master's thesis
ablation study on **depthwise separable convolution (DSC) compression** applied to
[Conv-TasNet](https://arxiv.org/abs/1809.07454) for 4-stem music source separation on
[MUSDB18-HQ](https://zenodo.org/record/3338373).

The core question: *how does progressively replacing standard dilated convolutions in the TCN
blocks of Conv-TasNet with depthwise separable convolutions shift the
parameters ↔ quality Pareto curve?*

---

## Background

Conv-TasNet (Luo & Mesgarani, 2019) is a lightweight temporal-domain source separation model
built around a Temporal Convolutional Network (TCN). This study systematically substitutes the
first L of the 24 TCN blocks with DSC variants and measures the effect on:

- separation quality (avg cSDR, BSS Eval v4 chunked, MUSDB18-HQ test set, 50 tracks)
- model size (parameter count)
- computational cost (MACs/s per stem)

All experiments use the same training protocol, dataset split (86/14/50), and evaluation script,
so comparisons between configurations are internally consistent.

---

## Key Results

| Configuration | L replaced | Params (M) | Compression | avg cSDR (dB) | ΔSDR (dB) | MACs/s per stem |
|---|---|---|---|---|---|---|
| Baseline      | 0  | 24.30 | —      | −4.66 | 0.00  | 16.55 GMACs |
| DSC-5         | 5  | 21.69 | −10.7% | −8.49 | −3.83 | 14.76 GMACs |
| DSC-10        | 10 | 19.09 | −21.5% | −7.22 | −2.56 | 12.96 GMACs |
| **DSC-20 ★**  | 20 | 13.88 | −42.9% | −8.12 | −3.46 |  9.38 GMACs |
| DSC-24 (full) | 24 | 11.79 | −51.5% | −6.24 | −1.58 |  7.95 GMACs |
| DSC-40†       | 40 | 19.23 | −20.9% | −6.94 | −2.28 | — |
| DSC-48†       | 48 | 22.96 |  −5.5% | −7.86 | −3.20 | — |

★ **DSC-20 is the single Pareto-optimal configuration** among all tested variants
(Wilcoxon test with Bonferroni correction, n=50, 21 pairs): it delivers a meaningful
parameter reduction (−42.9%) while maintaining non-degenerate output for all four stems.

† DSC-40 and DSC-48 use more TCN repeats (5 and 6 respectively) to compensate for capacity
loss; included to test whether quality degradation under full DSC is a capacity issue or an
architectural limitation.

**Notable findings:**

- **DSC-24 causes systematic drums degradation**: 48 of 50 test tracks collapse to avg
  drums cSDR ≈ 0.00 dB. Full DSC replacement removes cross-channel interaction in
  high-dilation layers, depriving the network of the capacity needed to model transients.
  DSC-24 is only suitable when the drums stem is not required.
- **Parameter reduction does not translate to GPU speedup**: DSC-24 shows +5.6% inference
  overhead vs baseline, because depthwise operations are memory-bandwidth-bound on GPU
  rather than compute-bound.
- **The ≤ 5 GMACs/s lightweight threshold is not reached**: the minimum achieved is
  7.95 GMACs/s at L=24 (1.59× above the threshold), because the two pointwise 1×1 layers
  inside each TCN block are unoptimizable by DSC and account for ~60% of block FLOPs.

All results are statistically verified with the non-parametric Wilcoxon signed-rank test
(Bonferroni correction).

---

## Repository Structure

```
AudioSepAblationStudy/
├── configs/              # YAML experiment configs (baseline, dsc5, …, dsc48)
├── checkpoints/          # Trained best.pt checkpoints per experiment (Git LFS)
├── data/
│   └── musdb18hq/test/   # MUSDB18-HQ test split: 50 tracks × 5 stems (WAV)
├── results/              # Per-experiment CSV metrics and aggregate summaries
├── scripts/              # Analysis, plotting, MACs measurement, audit utilities
├── src/
│   ├── models/
│   │   ├── conv_tasnet.py        # Baseline Conv-TasNet (groups=1 dilated TCN)
│   │   ├── dsc_conv_tasnet.py    # DSC subclass; takes dsc_layers=L parameter
│   │   └── blocks.py             # TemporalBlock, DepthwiseSeparableConv1d, TCNSeparator
│   ├── data/
│   │   └── musdb_dataset.py      # MUSDB18-HQ dataset loader
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # BSS-Eval v4 + SI-SDR evaluation (overlap-add)
│   ├── inference.py              # Single-track inference helper
│   ├── losses.py                 # L1 / SI-SDR loss
│   ├── metrics.py                # SDR, SIR, SAR, SI-SDR computation
│   └── utils.py                  # Config I/O, checkpoint management, CSV logging
├── notebooks/            # Exploratory Jupyter notebooks
├── run_ablation.sh       # Full train + eval pipeline for all 5 main experiments
├── run_dsc40.sh          # Variant pipeline for DSC-40 / DSC-48
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

Tested with Python 3.10, PyTorch 2.1, CUDA 11.8.  
Checkpoints are stored in Git LFS — run `git lfs pull` to fetch them.

Place the MUSDB18-HQ dataset under `data/musdb18hq/` with the standard
`train/` / `test/` split expected by `musdb18`.

---

## Training

```bash
python src/train.py --config configs/baseline.yaml
python src/train.py --config configs/dsc20.yaml
# … etc.
```

Or run the full ablation pipeline:

```bash
bash run_ablation.sh
```

Key training settings (shared across all experiments unless overridden in config):

| Parameter | Value |
|---|---|
| Optimizer | Adam, lr = 3×10⁻⁴ |
| LR schedule | Plateau reduction |
| Loss | L1 waveform |
| Segment length | 4.0 s (baseline) / 2.0 s (DSC variants) |
| Batch size | 6 segments |
| Epochs | 100–210 (config-dependent) |
| Hardware | NVIDIA A40 / RTX 4090 (24 GB VRAM) |
| Seed | 42 |
| Augmentations | pitch shift, time stretch, random gain, source remix, channel swap, sign flip |

> **Reproducibility note.** Despite seed=42, non-deterministic CUDA ops introduce
> ~0.1–0.2 dB run-to-run variance. Comparisons rely on median per-track values and
> relative differences rather than absolute scores.

---

## Evaluation

```bash
python src/evaluate.py \
    --config configs/dsc20.yaml \
    --checkpoint checkpoints/dsc20/best.pt
```

Produces per-track and aggregate CSV files in `results/`.  
Metrics: cSDR, SDR, SIR, SAR (BSS Eval v4, chunked 3 s), SI-SDR.

---

## Analysis Scripts

```bash
python scripts/summarize_results.py   # aggregate CSVs → summary tables
python scripts/analyze_results.py     # per-stem / per-track analysis + Pareto plot
python scripts/measure_macs.py        # FLOPs / MACs profiling via fvcore
python scripts/param_count.py         # parameter count table
```

---

## Dataset

**MUSDB18-HQ** — 150 full-length stereo tracks at 44.1 kHz (WAV, lossless) with isolated stems:
`vocals`, `drums`, `bass`, `other`.  
Split: 86 train / 14 validation / 50 test.  
Only the test split (50 tracks) is included in this repository for evaluation.

The full dataset is available at: <https://zenodo.org/record/3338373>

---

## DSC Modification Details

The baseline in this work uses **standard full-rank (groups=1) dilated Conv1d** in TCN blocks —
distinct from the original Luo & Mesgarani (2019) Conv-TasNet, which already uses depthwise
convolutions in `d_conv`. All ΔSDR values are relative to this groups=1 baseline.

Each DSC replacement augments the `d_conv` module with:
```
depthwise dilated Conv1d (groups=hidden_channels)
→ pointwise 1×1 Conv1d
→ PReLU
→ ChannelwiseLayerNorm
```

The substitution is parametrized by `dsc_layers=L`: only the first L of the 24 TCN blocks are
replaced; the remaining 24-L blocks retain standard dilated convolutions.

Theoretical MAC reduction for an isolated dilated layer (N=256, K=3):

```
1/R = 1/N + 1/K² ≈ 1/8.7
```

But at the full TCN-block level, the two pointwise 1×1 layers are unoptimizable,
limiting the actual MAC reduction to **≈ 2.5× per block**.

---

## References

- Luo Y., Mesgarani N. (2019). Conv-TasNet: Surpassing ideal time–frequency magnitude masking
  for speech separation. *IEEE/ACM TASLP*, 27(8), 1256–1266.
- Rafii Z. et al. (2017). MUSDB18 — a corpus for music separation. Zenodo.
- Stoter F.R. et al. (2019). MUSDB18-HQ — an uncompressed version of MUSDB18. Zenodo.
- Vincent E. et al. (2006). Performance measurement in blind audio source separation.
  *IEEE TASLP*, 14(4), 1462–1469.
