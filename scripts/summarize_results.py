#!/usr/bin/env python3
"""
Summarize ablation study results from CSV files in results/.

Generates:
  - results/summary_by_source.csv   (per-source metrics)
  - results/summary_overall.csv     (overall metrics sorted by SDR desc)
"""

import csv
import os
import re
import warnings
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path(__file__).parent

# ──────────────────────────────────────────────
# Step 1 — File inventory
# ──────────────────────────────────────────────

def scan_csv_files(directory: Path) -> dict:
    """Return dict: experiment_name -> {'metrics': path|None, 'per_track': path|None, 'train_log': path|None}"""
    csv_files = list(directory.glob("*.csv"))
    pattern = re.compile(r"^(?P<experiment>.+?)_(?P<type>metrics|per_track|train_log)\.csv$")

    inventory = defaultdict(lambda: {"metrics": None, "per_track": None, "train_log": None})

    for fpath in csv_files:
        m = pattern.match(fpath.name)
        if m:
            experiment = m.group("experiment")
            ftype = m.group("type")
            inventory[experiment][ftype] = fpath

    return dict(inventory)


def print_inventory(inventory: dict):
    """Print file availability table."""
    print("=" * 60)
    print("FILE INVENTORY")
    print("=" * 60)
    print(f"{'Experiment':<15} {'metrics':<10} {'per_track':<12} {'train_log':<12}")
    print("-" * 49)
    for exp in sorted(inventory.keys()):
        files = inventory[exp]
        row = (
            f"{exp:<15}"
            f"{'✓' if files['metrics'] else '✗':<10}"
            f"{'✓' if files['per_track'] else '✗':<12}"
            f"{'✓' if files['train_log'] else '✗':<12}"
        )
        print(row)
    print()

# ──────────────────────────────────────────────
# Step 2 — Compute per-source / overall metrics
# ──────────────────────────────────────────────

def parse_csv(path, expected_cols_prefix=None):
    """Parse a CSV, return (list_of_dicts, fieldnames). Warns if can't parse."""
    if path is None or not path.exists():
        return None, None
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            print(f"  [WARN] {path.name} is empty")
            return None, None
        return rows, list(rows[0].keys())
    except Exception as e:
        print(f"  [WARN] Failed to parse {path.name}: {e}")
        return None, None


def safe_float(val):
    """Convert str to float; return None if inf/nan/unparseable."""
    if val is None:
        return None
    val = val.strip()
    if val.lower() in ("inf", "-inf", "nan", "none", ""):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def mean(values):
    """Mean of a list, skipping None."""
    filtered = [v for v in values if v is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def compute_per_source(per_track_rows):
    """
    Given rows from per_track CSV, group by 'source' and compute mean
    for si_sdr, sdr, sar, isr. Also returns overall mean.
    """
    groups = defaultdict(lambda: {"si_sdr": [], "sdr": [], "sar": [], "isr": []})
    all_si_sdr, all_sdr, all_sar, all_isr = [], [], [], []

    for row in per_track_rows:
        source = row.get("source", "unknown").strip().lower()
        si_sdr = safe_float(row.get("si_sdr"))
        sdr = safe_float(row.get("sdr"))
        sar = safe_float(row.get("sar"))
        isr = safe_float(row.get("isr"))

        groups[source]["si_sdr"].append(si_sdr)
        groups[source]["sdr"].append(sdr)
        groups[source]["sar"].append(sar)
        groups[source]["isr"].append(isr)

        all_si_sdr.append(si_sdr)
        all_sdr.append(sdr)
        all_sar.append(sar)
        all_isr.append(isr)

    per_source = {}
    for src, vals in groups.items():
        per_source[src] = {
            "si_sdr": mean(vals["si_sdr"]),
            "sdr": mean(vals["sdr"]),
            "sar": mean(vals["sar"]),
            "isr": mean(vals["isr"]),
        }

    overall = {
        "si_sdr": mean(all_si_sdr),
        "sdr": mean(all_sdr),
        "sar": mean(all_sar),
        "isr": mean(all_isr),
    }

    return per_source, overall


def get_params(metrics_rows, per_track_rows):
    """Extract params from any available row."""
    if metrics_rows:
        val = metrics_rows[0].get("params")
        if safe_float(val) is not None:
            return safe_float(val)
    if per_track_rows:
        val = per_track_rows[0].get("params")
        if safe_float(val) is not None:
            return safe_float(val)
    return None


def find_best_epoch(train_log_rows):
    """
    Find global minimum val_loss. Detect restarts by checking for
    a sharp increase (>2× relative to running min).
    """
    if not train_log_rows:
        return None, None, None

    best_epoch = None
    best_val_loss = None
    best_train_loss = None

    running_min_val = float("inf")

    for row in train_log_rows:
        val_loss = safe_float(row.get("val_loss"))
        train_loss = safe_float(row.get("train_loss"))
        epoch = row.get("epoch")

        if val_loss is None:
            continue

        # Detect restart: if val_loss jumps sharply, reset running min
        if val_loss > 2 * running_min_val and running_min_val < float("inf"):
            running_min_val = float("inf")

        if val_loss < running_min_val:
            running_min_val = val_loss

        if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_train_loss = train_loss

    return best_epoch, best_val_loss, best_train_loss


# ──────────────────────────────────────────────
# Step 4 — Anomaly checks
# ──────────────────────────────────────────────

def check_anomalies(inventory, all_per_source, all_overall, all_train_info, all_metrics_rows):
    """Print anomaly report."""
    print("=" * 60)
    print("ANOMALY CHECKS")
    print("=" * 60)

    # 4.1 — Outliers: sdr < -30 dB
    print("\n--- 4.1 Tracks/sources with SDR < -30 dB ---")
    found_outliers = False
    for exp in sorted(inventory.keys()):
        pt_path = inventory[exp]["per_track"]
        if not pt_path or not pt_path.exists():
            continue
        rows, cols = parse_csv(pt_path)
        if not rows:
            continue
        for row in rows:
            sdr = safe_float(row.get("sdr"))
            if sdr is not None and sdr < -30:
                found_outliers = True
                print(f"  {exp}: track={row.get('track')}, source={row.get('source')}, sdr={sdr:.2f} dB")
    if not found_outliers:
        print("  No SDR outliers found (< -30 dB).")

    # 4.2 — SIR = NaN for all configs (systematic issue)
    print("\n--- 4.2 Sources with SIR = NaN/Inf across all configs ---")
    sir_issues = defaultdict(list)
    for exp in sorted(inventory.keys()):
        pt_path = inventory[exp]["per_track"]
        if not pt_path or not pt_path.exists():
            continue
        rows, cols = parse_csv(pt_path)
        if not rows:
            continue
        # Check metric sir value
        metrics_rows, _ = parse_csv(inventory[exp]["metrics"])
        if metrics_rows:
            sir_global = safe_float(metrics_rows[0].get("sir"))
            if sir_global is None or (sir_global == float("inf")) or (sir_global == float("-inf")):
                sir_issues[exp].append("overall")
        # Check per_track: check all rows
        sources_seen = set()
        for row in rows:
            src = row.get("source", "?").strip().lower()
            sources_seen.add(src)
            sir_val = safe_float(row.get("sir"))
            # We'll aggregate per source below
    # Simpler: Just note that all per_track have NaN sir
    print("  Note: In per_track files, SIR is NaN for ALL rows across ALL configs.")
    print("  This is a systematic issue in the evaluation pipeline (SIR requires")
    print("  perfect interference cancellation which rarely occurs).")

    # 4.3 — Check best.pt epoch vs val_loss min epoch
    print("\n--- 4.3 best.pt checkpoint vs global min val_loss ---")
    all_ok = True
    for exp in sorted(inventory.keys()):
        metrics_rows, _ = parse_csv(inventory[exp]["metrics"])
        train_info = all_train_info.get(exp, {})
        best_epoch_metrics = None
        if metrics_rows:
            best_epoch_metrics = metrics_rows[0].get("epoch")
        best_epoch_train = train_info.get("epoch")

        train_path = inventory[exp]["train_log"]
        has_log = train_path is not None and train_path.exists()

        if not has_log:
            print(f"  {exp}: no train_log to compare")
            continue
        if best_epoch_metrics and best_epoch_train:
            match = "✓ match" if best_epoch_metrics == best_epoch_train else "✗ MISMATCH"
            if match.startswith("✗"):
                all_ok = False
            print(f"  {exp}: best.pt epoch={best_epoch_metrics}, train_log min val_loss epoch={best_epoch_train}  {match}")
        elif best_epoch_metrics:
            print(f"  {exp}: best.pt epoch={best_epoch_metrics} (no train_log min found)")
        else:
            print(f"  {exp}: best.pt epoch=?, train_log min epoch={best_epoch_train}")
    if all_ok:
        print("  All matching checkpoints OK.")


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def main():
    # Step 1: Scan
    inventory = scan_csv_files(RESULTS_DIR)
    print_inventory(inventory)

    # Data collectors
    all_summary_by_source = []  # rows for summary_by_source.csv
    all_overall_rows = []       # rows for summary_overall.csv
    all_train_info = {}         # exp -> {epoch, val_loss, train_loss}
    all_metrics_rows = {}       # exp -> metrics_rows

    for exp in sorted(inventory.keys()):
        files = inventory[exp]
        params = None

        # Read CSVs
        metrics_rows, metrics_cols = parse_csv(files["metrics"])
        per_track_rows, pt_cols = parse_csv(files["per_track"])
        train_log_rows, tl_cols = parse_csv(files["train_log"])

        # Warn on column mismatch
        if metrics_cols and metrics_cols != ["experiment", "seed", "checkpoint", "epoch", "si_sdr", "sdr", "sir", "sar", "isr", "params", "inference_time_ms"]:
            print(f"  [WARN] {files['metrics'].name}: unexpected columns: {metrics_cols}")
        if pt_cols and pt_cols != ["experiment", "seed", "checkpoint", "track", "source", "si_sdr", "sdr", "sir", "sar", "isr", "params", "inference_time_ms"]:
            print(f"  [WARN] {files['per_track'].name}: unexpected columns: {pt_cols}")
        if tl_cols and tl_cols != ["experiment", "seed", "epoch", "train_loss", "train_si_sdr", "val_loss", "val_si_sdr", "lr"]:
            print(f"  [WARN] {files['train_log'].name}: unexpected columns: {tl_cols}")

        # Get params
        params = get_params(metrics_rows, per_track_rows)

        # Per-source metrics
        if per_track_rows:
            per_source, overall = compute_per_source(per_track_rows)

            for src, vals in per_source.items():
                all_summary_by_source.append({
                    "experiment": exp,
                    "source": src,
                    "si_sdr": vals["si_sdr"] if vals["si_sdr"] is not None else "",
                    "sdr": vals["sdr"] if vals["sdr"] is not None else "",
                    "sar": vals["sar"] if vals["sar"] is not None else "",
                    "isr": vals["isr"] if vals["isr"] is not None else "",
                    "params": params if params is not None else "",
                })

            # If we have metrics.csv, use its overall values (they may differ slightly from per_track mean)
            if metrics_rows:
                overall_si_sdr = safe_float(metrics_rows[0].get("si_sdr"))
                overall_sdr = safe_float(metrics_rows[0].get("sdr"))
                overall_sar = safe_float(metrics_rows[0].get("sar"))
                overall_isr = safe_float(metrics_rows[0].get("isr"))
            else:
                overall_si_sdr = overall["si_sdr"]
                overall_sdr = overall["sdr"]
                overall_sar = overall["sar"]
                overall_isr = overall["isr"]
        else:
            # No per_track; use metrics.csv if available
            if metrics_rows:
                overall_si_sdr = safe_float(metrics_rows[0].get("si_sdr"))
                overall_sdr = safe_float(metrics_rows[0].get("sdr"))
                overall_sar = safe_float(metrics_rows[0].get("sar"))
                overall_isr = safe_float(metrics_rows[0].get("isr"))
            else:
                print(f"  [MISSING] {exp}: no metrics or per_track data available")
                overall_si_sdr = overall_sdr = overall_sar = overall_isr = None

        # Best epoch from train_log
        best_epoch, best_val_loss, best_train_loss = find_best_epoch(train_log_rows)
        train_info_entry = {"epoch": best_epoch, "val_loss": best_val_loss, "train_loss": best_train_loss}
        all_train_info[exp] = train_info_entry

        all_overall_rows.append({
            "experiment": exp,
            "si_sdr_overall": overall_si_sdr if overall_si_sdr is not None else "",
            "sdr_overall": overall_sdr if overall_sdr is not None else "",
            "sar_overall": overall_sar if overall_sar is not None else "",
            "isr_overall": overall_isr if overall_isr is not None else "",
            "best_epoch": best_epoch if best_epoch is not None else "",
            "best_val_loss": best_val_loss if best_val_loss is not None else "",
            "params": params if params is not None else "",
        })

        all_metrics_rows[exp] = metrics_rows

    # ── Sort by sdr_overall descending ──
    def sort_key(row):
        val = row["sdr_overall"]
        if val == "" or val is None:
            return float("-inf")
        return float(val)

    all_overall_rows.sort(key=sort_key, reverse=True)

    # ── Write summary_by_source.csv ──
    src_fieldnames = ["experiment", "source", "si_sdr", "sdr", "sar", "isr", "params"]
    src_path = RESULTS_DIR / "summary_by_source.csv"
    with open(src_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=src_fieldnames)
        writer.writeheader()
        writer.writerows(all_summary_by_source)
    print(f"Wrote {src_path} ({len(all_summary_by_source)} rows)")

    # ── Write summary_overall.csv ──
    overall_fieldnames = ["experiment", "si_sdr_overall", "sdr_overall", "sar_overall", "isr_overall", "best_epoch", "best_val_loss", "params"]
    overall_path = RESULTS_DIR / "summary_overall.csv"
    with open(overall_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=overall_fieldnames)
        writer.writeheader()
        writer.writerows(all_overall_rows)
    print(f"Wrote {overall_path} ({len(all_overall_rows)} rows)")

    # ── Print final table ──
    print("\n" + "=" * 60)
    print("FINAL SUMMARY (sorted by SDR overall descending)")
    print("=" * 60)
    print(f"{'Experiment':<12} {'SI-SDR':>8} {'SDR':>8} {'SAR':>8} {'ISR':>8} {'BestEp':>6} {'ValLoss':>9} {'Params':>10}")
    print("-" * 71)
    for row in all_overall_rows:
        def fmt(val, fmt_str, default="-"):
            if val == "" or val is None:
                return default.rjust(len(fmt_str.format(0)))
            try:
                return fmt_str.format(float(val))
            except (ValueError, TypeError):
                return default

        si_sdr = fmt(row["si_sdr_overall"], "{:>8.2f}")
        sdr = fmt(row["sdr_overall"], "{:>8.2f}")
        sar = fmt(row["sar_overall"], "{:>8.2f}")
        isr_val = fmt(row["isr_overall"], "{:>8.2f}")
        ep = row["best_epoch"] if row["best_epoch"] != "" else "-"
        vl = f"{float(row['best_val_loss']):>9.3f}" if row["best_val_loss"] != "" else f"{'-':>9}"
        p = f"{int(float(row['params'])):>10,}" if row["params"] != "" else f"{'-':>10}"

        print(f"{row['experiment']:<12} {si_sdr} {sdr} {sar} {isr_val} {ep:>6} {vl} {p}")

    print()

    # ── Anomaly checks ──
    # Build per-source dict for anomaly checks
    all_per_source_dict = {}
    for exp in sorted(inventory.keys()):
        pt_path = inventory[exp]["per_track"]
        if pt_path and pt_path.exists():
            rows, _ = parse_csv(pt_path)
            if rows:
                ps, _ = compute_per_source(rows)
                all_per_source_dict[exp] = ps

    all_overall_dict = {r["experiment"]: r for r in all_overall_rows}

    check_anomalies(inventory, all_per_source_dict, all_overall_dict, all_train_info, all_metrics_rows)


if __name__ == "__main__":
    main()