"""MUSDB18-HQ dataset loader for source separation experiments.

Expected directory layout (MUSDB18-HQ unzipped):

    <root>/
        train/
            <track name>/
                mixture.wav
                vocals.wav
                drums.wav
                bass.wav
                other.wav
        test/
            <track name>/
                mixture.wav
                vocals.wav
                ...

Usage:

    >>> from data.musdb_dataset import MUSDBDataset
    >>> ds = MUSDBDataset(
    ...     root="data/musdb18hq",
    ...     sources=["vocals", "drums", "bass", "other"],
    ...     sample_rate=44100,
    ...     segment_seconds=4.0,
    ...     split="train",
    ... )
    >>> mixture, sources = ds[0]
    >>> print(mixture.shape)   # [audio_channels, segment_samples]
    >>> print(sources.shape)   # [num_sources, audio_channels, segment_samples]
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import torch
import torchaudio
from torch.utils.data import Dataset


class MUSDBDataset(Dataset):
    """MUSDB18-HQ stereo dataset with fixed-length random segments.

    Args:
        root:             Path to MUSDB18-HQ root directory.
        sources:          List of source names matching WAV file stems.
                          Default MUSDB18 order: ``["vocals", "drums", "bass", "other"]``.
        sample_rate:      Target sample rate for resampling. Set to ``None`` to
                          load at the native rate (44 100 Hz for MUSDB18-HQ).
        segment_seconds:  Length of each training segment in seconds.
                          Use ``None`` (or the ``"eval"`` split) to load full tracks.
        split:            ``"train"`` or ``"test"`` (maps to the MUSDB18 directory name).
        full_track:       If True, always return the full track regardless of
                          ``segment_seconds``. Useful for evaluation.
        seed:             Optional fixed seed for reproducible segment sampling.
    """

    SOURCE_NAMES: tuple[str, ...] = ("vocals", "drums", "bass", "other")

    def __init__(
        self,
        root: str | Path,
        sources: Iterable[str] = SOURCE_NAMES,
        sample_rate: int | None = 44100,
        segment_seconds: float | None = 4.0,
        split: str = "train",
        full_track: bool = False,
        seed: int | None = None,
    ) -> None:
        self.root = Path(root)
        self.sources = list(sources)
        self.sample_rate = sample_rate
        self.segment_seconds = segment_seconds
        self.split = split
        self.full_track = full_track
        self.seed = seed
        self.epoch = 0

        self.index: list[Path] = self._build_index()
        if not self.index:
            raise FileNotFoundError(
                f"No track directories found in '{self.root / split}'. "
                "Check that the MUSDB18-HQ dataset is correctly placed."
            )

    # ---------------------------------------------------------------------- #
    # Index                                                                    #
    # ---------------------------------------------------------------------- #

    def _build_index(self) -> list[Path]:
        split_dir = self.root / self.split
        if not split_dir.exists():
            return []
        tracks = sorted(p for p in split_dir.iterdir() if p.is_dir())
        # Validate that mixture and all requested source WAVs exist
        valid = []
        for track_dir in tracks:
            if not (track_dir / "mixture.wav").exists():
                continue
            if all((track_dir / f"{s}.wav").exists() for s in self.sources):
                valid.append(track_dir)
        return valid

    # ---------------------------------------------------------------------- #
    # Dataset protocol                                                         #
    # ---------------------------------------------------------------------- #

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one segment (or full track) from a MUSDB18-HQ track directory.

        Returns:
            mixture: ``[audio_channels, segment_samples]``
            sources: ``[num_sources, audio_channels, segment_samples]``
        """
        track_dir = self.index[idx]

        # Load mixture to discover length and native sample rate
        mixture_path = track_dir / "mixture.wav"
        mixture_waveform, native_sr = torchaudio.load(str(mixture_path))
        # mixture_waveform: [audio_channels, time]

        # Resample if needed
        if self.sample_rate is not None and native_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=native_sr, new_freq=self.sample_rate
            )
            mixture_waveform = resampler(mixture_waveform)
            effective_sr = self.sample_rate
        else:
            effective_sr = native_sr

        total_samples = mixture_waveform.shape[-1]

        # ------------------------------------------------------------------ #
        # Determine segment boundaries                                         #
        # ------------------------------------------------------------------ #
        if self.full_track or self.segment_seconds is None:
            start = 0
            seg_len = total_samples
        else:
            seg_len = int(self.segment_seconds * effective_sr)
            if total_samples <= seg_len:
                start = 0
            else:
                max_start = total_samples - seg_len
                # Per-sample deterministic random start when seed is set
                rng = random.Random(self.seed + idx * 1000 + self.epoch if self.seed is not None else None)
                start = rng.randint(0, max_start)

        # Slice mixture
        mixture_seg = mixture_waveform[:, start: start + seg_len]
        # Zero-pad if needed (e.g., last segment)
        if mixture_seg.shape[-1] < seg_len:
            pad = seg_len - mixture_seg.shape[-1]
            mixture_seg = torch.nn.functional.pad(mixture_seg, (0, pad))

        # ------------------------------------------------------------------ #
        # Load and slice each source                                           #
        # ------------------------------------------------------------------ #
        source_tensors: list[torch.Tensor] = []
        for source_name in self.sources:
            src_path = track_dir / f"{source_name}.wav"
            src_waveform, src_sr = torchaudio.load(str(src_path))

            if self.sample_rate is not None and src_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=src_sr, new_freq=self.sample_rate
                )
                src_waveform = resampler(src_waveform)

            src_seg = src_waveform[:, start: start + seg_len]
            if src_seg.shape[-1] < seg_len:
                pad = seg_len - src_seg.shape[-1]
                src_seg = torch.nn.functional.pad(src_seg, (0, pad))

            source_tensors.append(src_seg)  # [C, T]

        sources_out = torch.stack(source_tensors, dim=0)  # [S, C, T]
        return mixture_seg, sources_out


# --------------------------------------------------------------------------- #
# DataLoader worker init — for reproducible augmentation                       #
# --------------------------------------------------------------------------- #

def worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker uniquely but reproducibly.

    Pass this to ``DataLoader(worker_init_fn=worker_init_fn)``.
    """
    import numpy as np

    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed + worker_id)
    np.random.seed((worker_seed + worker_id) % (2 ** 32))
