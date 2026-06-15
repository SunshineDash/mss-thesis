"""Trivial *mixture-as-estimate* baseline for the MUSDB18-HQ test split.

This computes the lower-reference cSDR for thesis tables 3.2 and 3.5 by
feeding the mixture itself as the estimate for *every* source. No model is
involved — this is a pure CPU museval computation.

CRITICAL — identical protocol to the trained models:
  * Data is loaded with the SAME class as ``src/evaluate.py``
    (``MUSDBDataset`` split="test", full_track=True, sources order
    [vocals, drums, bass, other], sorted track directories).
  * Metrics are computed with the SAME function ``evaluate_track`` from
    ``src/metrics.py`` with its default 1 s window / 1 s hop (museval).
    win/hop are NOT overridden here.
  * Aggregation matches ``scripts/analyze_results.py``:
        - within a track: median over 1 s frames (done inside evaluate_track)
        - per-source cSDR = mean of those medians over the 50 tracks
        - overall avg cSDR = mean over all (4 sources x 50 tracks)
        - std over tracks = std(ddof=1)
  * The per-track CSV uses the SAME columns as the other ``*_per_track.csv``
    files so the row drops into the existing analysis pipeline.

Usage:
    python scripts/eval_mixture_baseline.py
    python scripts/eval_mixture_baseline.py --root data/musdb18hq --sample-rate 44100
"""

from __future__ import annotations

import argparse
import gc
import math
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Make ``src/`` importable (same trick as src/evaluate.py)                      #
# --------------------------------------------------------------------------- #
BASE = Path(__file__).resolve().parent.parent
SRC = BASE / "src"
sys.path.insert(0, str(SRC))

import torch  # noqa: E402  (after sys.path tweak)

from data.musdb_dataset import MUSDBDataset  # noqa: E402
from metrics import evaluate_track  # noqa: E402
from utils import ensure_dir, save_metrics_csv  # noqa: E402


# Fixed protocol constants (match configs/*.yaml defaults) ------------------- #
EXP_NAME = "mixture_baseline"
SOURCES = ["vocals", "drums", "bass", "other"]   # MUSDB18 canonical order
EXPECTED_TRACKS = 50
EXPECTED_CHANNELS = 2                              # stereo


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Mixture-as-estimate cSDR baseline on MUSDB18-HQ test.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--root", default="data/musdb18hq",
                   help="MUSDB18-HQ root directory (must contain test/).")
    p.add_argument("--sample-rate", type=int, default=44100,
                   help="Sample rate used by the dataset loader (must match the "
                        "rate used to evaluate the trained models).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sample_rate = int(args.sample_rate)
    data_root = Path(args.root)

    print("=" * 70)
    print("MIXTURE-AS-ESTIMATE BASELINE  (lower reference cSDR)")
    print("=" * 70)
    print(f"  Root          : {data_root}")
    print(f"  Split         : test")
    print(f"  Sample rate   : {sample_rate} Hz")
    print(f"  Source order  : {SOURCES}")
    print(f"  evaluate_track: museval defaults win=1.0 s, hop=1.0 s (NOT overridden)")
    print(f"  estimate[s]   : the track's own mixture, for all 4 sources")
    print("=" * 70)

    # --------------------------------------------------------------------- #
    # Data presence checks — refuse to invent numbers                        #
    # --------------------------------------------------------------------- #
    test_dir = data_root / "test"
    if not test_dir.exists():
        print(f"\n[STOP] Test directory does not exist: {test_dir}")
        print("       Place the MUSDB18-HQ test split (50 track folders) there,")
        print("       then re-run this script. No results were written.")
        return 1

    try:
        test_dataset = MUSDBDataset(
            root=str(data_root),
            sources=SOURCES,
            sample_rate=sample_rate,
            segment_seconds=None,   # full tracks
            split="test",
            full_track=True,
        )
    except FileNotFoundError as exc:
        print(f"\n[STOP] {exc}")
        print("       No results were written.")
        return 1

    n_tracks = len(test_dataset)
    print(f"\nValid test tracks found: {n_tracks}")

    if n_tracks < EXPECTED_TRACKS:
        # Report exactly which directories are missing required stems.
        all_dirs = sorted(p for p in test_dir.iterdir() if p.is_dir())
        valid_names = {p.name for p in test_dataset.index}
        print(f"\n[STOP] Expected {EXPECTED_TRACKS} complete tracks, found {n_tracks}.")
        print("       Incomplete / missing track folders:")
        for d in all_dirs:
            if d.name in valid_names:
                continue
            missing = []
            if not (d / "mixture.wav").exists():
                missing.append("mixture.wav")
            for s in SOURCES:
                if not (d / f"{s}.wav").exists():
                    missing.append(f"{s}.wav")
            print(f"         - {d.name}: missing {missing or '???'}")
        if len(all_dirs) < EXPECTED_TRACKS:
            print(f"       Only {len(all_dirs)} subdirectories exist under {test_dir}.")
        print("\n       No results were written. Better empty than a wrong thesis row.")
        return 1

    # --------------------------------------------------------------------- #
    # Per-track loop (CPU, one track at a time, free memory each iteration)   #
    # --------------------------------------------------------------------- #
    per_track_rows: list[dict] = []
    # per_source_medians[source] -> list of per-track median cSDR values
    per_source_sdr: dict[str, list[float]] = {s: [] for s in SOURCES}
    length_adjusted = 0
    processed = 0

    for idx in range(n_tracks):
        track_dir = test_dataset.index[idx]
        track_name = track_dir.name

        mixture, sources_ref = test_dataset[idx]
        # mixture:     [C, T]
        # sources_ref: [S, C, T]

        # ---- sanity: channels --------------------------------------------- #
        c_mix = mixture.shape[0]
        c_ref = sources_ref.shape[1]
        if c_mix != EXPECTED_CHANNELS or c_ref != EXPECTED_CHANNELS:
            print(f"  ! {track_name}: unexpected channel count "
                  f"(mixture C={c_mix}, refs C={c_ref}); expected {EXPECTED_CHANNELS}.")

        # ---- sanity: align length / channels before evaluation ------------ #
        t_mix = mixture.shape[-1]
        t_ref = sources_ref.shape[-1]
        T = min(t_mix, t_ref)
        C = min(c_mix, c_ref)
        if t_mix != t_ref:
            length_adjusted += 1
            print(f"  ! {track_name}: length mismatch mixture={t_mix} refs={t_ref}; "
                  f"trimming both to {T}.")
        mixture = mixture[:C, :T]
        sources_ref = sources_ref[:, :C, :T]

        # ---- build estimate: mixture repeated for all 4 sources ----------- #
        # Shape [S, C, T] — exactly what evaluate_track expects.
        separated = torch.stack([mixture for _ in SOURCES], dim=0)

        sep_np = separated.cpu().numpy()      # [S, C, T]
        ref_np = sources_ref.cpu().numpy()    # [S, C, T]

        # ---- EXACT same call as src/evaluate.py (no win/hop override) ----- #
        try:
            scores = evaluate_track(sep_np, ref_np, sample_rate=sample_rate)
        except Exception as exc:  # noqa: BLE001
            print(f"  museval FAILED for '{track_name}': {exc}")
            scores = {"sdr": [float("nan")] * len(SOURCES),
                      "sir": [float("nan")] * len(SOURCES),
                      "sar": [float("nan")] * len(SOURCES),
                      "isr": [float("nan")] * len(SOURCES)}

        for s_idx, src_name in enumerate(SOURCES):
            sdr = scores["sdr"][s_idx]
            sir_val = scores["sir"][s_idx]
            # Guard inf SIR exactly like src/evaluate.py
            if sir_val is not None and not math.isnan(float(sir_val)) and math.isinf(float(sir_val)):
                sir_val = float("nan")
            if not (sdr is None or math.isnan(float(sdr))):
                per_source_sdr[src_name].append(float(sdr))
            per_track_rows.append({
                "experiment":        EXP_NAME,
                "seed":              0,
                "checkpoint":        "none (mixture-as-estimate)",
                "track":             track_name,
                "source":            src_name,
                "si_sdr":            float("nan"),   # not defined for this baseline
                "sdr":               sdr,
                "sir":               sir_val,
                "sar":               scores["sar"][s_idx],
                "isr":               scores["isr"][s_idx],
                "params":            0,
                "inference_time_ms": 0.0,
            })

        processed += 1
        print(f"  [{processed:2d}/{n_tracks}] {track_name:<45s}  "
              f"cSDR(v/d/b/o) = "
              f"{scores['sdr'][0]:6.2f} {scores['sdr'][1]:6.2f} "
              f"{scores['sdr'][2]:6.2f} {scores['sdr'][3]:6.2f}")

        # ---- free memory ------------------------------------------------- #
        del mixture, sources_ref, separated, sep_np, ref_np
        gc.collect()

    # --------------------------------------------------------------------- #
    # Save per-track CSV (same columns as other *_per_track.csv)             #
    # --------------------------------------------------------------------- #
    out_dir = ensure_dir(BASE / "results")
    per_track_csv = out_dir / f"{EXP_NAME}_per_track.csv"
    save_metrics_csv(per_track_csv, per_track_rows)
    print(f"\nPer-track metrics saved: {per_track_csv}")

    # --------------------------------------------------------------------- #
    # Aggregation (identical logic to scripts/analyze_results.py)            #
    # --------------------------------------------------------------------- #
    # per-source cSDR = mean over tracks of per-track median
    per_source_mean = {s: (float(np.mean(v)) if v else float("nan"))
                       for s, v in per_source_sdr.items()}

    # overall avg cSDR = mean over all (source x track) median values
    all_sdr = [v for vals in per_source_sdr.values() for v in vals]
    overall_avg = float(np.mean(all_sdr)) if all_sdr else float("nan")

    # std over tracks (ddof=1) over all per-(track,source) rows — matches the
    # SDR std reported in analyze_results.py §4a (df["sdr"].std(ddof=1)).
    overall_std = float(np.std(all_sdr, ddof=1)) if len(all_sdr) > 1 else float("nan")
    per_source_std = {s: (float(np.std(v, ddof=1)) if len(v) > 1 else float("nan"))
                      for s, v in per_source_sdr.items()}

    n_used = len(all_sdr)  # should be 4 * 50 = 200

    # --------------------------------------------------------------------- #
    # Sanity checks                                                          #
    # --------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    print(f"  Tracks processed                : {processed}  (expected {EXPECTED_TRACKS})")
    print(f"  evaluate_track input shape      : [S, C, T] = "
          f"[{len(SOURCES)}, {EXPECTED_CHANNELS}, T]")
    print(f"  Tracks with length adjustment   : {length_adjusted}")
    print(f"  Valid (source x track) cSDR vals: {n_used}  (expected "
          f"{len(SOURCES) * EXPECTED_TRACKS})")

    # Physically-grounded band: for N=4 sources the analytic floor when stems
    # have equal power is 10*log10(1/3) = -4.77 dB; real MUSDB targets (vocals,
    # bass) are quieter than 1/4 of the mix, pushing the museval median lower.
    # A genuine shape/order/normalisation bug would land far outside this band
    # (e.g. > 0 dB, or extreme like < -15 dB).
    plausible = (-10.0 <= overall_avg <= 0.0)
    print(f"  Overall avg cSDR plausibility   : {overall_avg:+.2f} dB  "
          f"-> {'OK (expected ~-3..-8 dB; 4-source floor ~-4.8 dB)' if plausible else 'OUT OF RANGE'}")

    if processed != EXPECTED_TRACKS:
        print(f"\n[STOP] Processed {processed} tracks, expected {EXPECTED_TRACKS}. "
              "Do NOT use these numbers.")
        return 1

    if not plausible:
        print("\n[WARNING] Overall avg cSDR is outside the expected ~0..-1 dB band "
              "for a mixture-as-estimate baseline.")
        print("          This usually signals a shape / channel / source-order / "
              "normalisation mismatch.")
        print("          Per-track CSV was written for inspection, but DO NOT paste "
              "these numbers into the thesis until the cause is resolved.")
        # Still print the numbers below for debugging, but they are flagged.

    # --------------------------------------------------------------------- #
    # Final, table-ready values                                             #
    # --------------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print("RESULTS — ready for tables 3.2 and 3.5  (cSDR = museval SDR, dB)")
    print("=" * 70)
    print(f"  avg cSDR (overall) : {overall_avg:.2f} dB")
    print(f"  vocals             : {per_source_mean['vocals']:.2f} dB")
    print(f"  drums              : {per_source_mean['drums']:.2f} dB")
    print(f"  bass               : {per_source_mean['bass']:.2f} dB")
    print(f"  other              : {per_source_mean['other']:.2f} dB")
    print(f"  std over tracks     : {overall_std:.2f} dB  (ddof=1, over all "
          f"{n_used} source x track values)")
    print("  per-source std (ddof=1):")
    for s in SOURCES:
        print(f"     {s:<7s} : {per_source_std[s]:.2f} dB")
    print(f"  tracks processed   : {processed}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
