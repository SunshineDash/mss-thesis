"""
make_plots_tables.py
--------------------
Generates all figures and LaTeX tables for Chapter 3 of the MSc thesis
on Music Source Separation (Conv-TasNet with DSC blocks).

Outputs:
  results/figures/  — 5 PNG plots (300 DPI, 10×5 inches)
  results/tables/   — 2 LaTeX .tex files
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES  = os.path.join(BASE, "results")
FIG  = os.path.join(RES, "figures")
TAB  = os.path.join(RES, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────────────────────────────────────
PALETTE      = sns.color_palette("colorblind")
FIG_SIZE     = (10, 5)
DPI          = 300
TITLE_FS     = 14
LABEL_FS     = 12
TICK_FS      = 10
LEGEND_FS    = 10
GRID_ALPHA   = 0.3

plt.rcParams.update({
    "font.size":        TICK_FS,
    "axes.titlesize":   TITLE_FS,
    "axes.labelsize":   LABEL_FS,
    "xtick.labelsize":  TICK_FS,
    "ytick.labelsize":  TICK_FS,
    "legend.fontsize":  LEGEND_FS,
    "axes.grid":        True,
    "grid.alpha":       GRID_ALPHA,
})

# ─────────────────────────────────────────────────────────────────────────────
# Model ordering / display names
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ORDER   = ["baseline", "dsc5", "dsc10", "dsc20", "dsc_full"]
MODEL_LABELS  = {
    "baseline": "Baseline",
    "dsc5":     "DSC-5%",
    "dsc10":    "DSC-10%",
    "dsc20":    "DSC-20%",
    "dsc_full": "DSC-Full",
}

SOURCES = ["vocals", "drums", "bass", "other"]
# NOTE: si_sdr in per_track is a TRACK-level value (identical for all 4 sources
# of a given track), so it cannot be plotted per-source. sdr/sar/isr are source-specific.
METRICS_SOURCE = ["sdr", "sar", "isr"]       # source-specific metrics for bar charts
METRICS_TABLE  = ["sdr", "sar", "isr"]       # columns for Table 1
METRICS_WITH_SIR = ["si_sdr", "sdr", "sir", "sar"]  # for Fig 5 (SIR = inf in data)
METRIC_LABELS = {
    "si_sdr": "SI-SDR (dB)", "sdr": "SDR (dB)",
    "sir": "SIR (dB)",        "sar": "SAR (dB)",
    "isr":  "ISR (dB)",
}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load all CSVs and print structure
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING CSV FILES")
print("=" * 70)

# ── Training logs ──
train_logs = {}
for exp in ["baseline", "dsc20", "dsc_full"]:
    path = os.path.join(RES, f"{exp}_train_log.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        train_logs[exp] = df
        print(f"\n[{exp}_train_log.csv]  shape={df.shape}")
        print(f"  columns: {list(df.columns)}")
        print(df.head(3).to_string(index=False))

# ── Aggregate metrics ──
agg_metrics = {}
for exp in MODEL_ORDER:
    path = os.path.join(RES, f"{exp}_metrics.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Replace inf with NaN for calculations
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        agg_metrics[exp] = df
        print(f"\n[{exp}_metrics.csv]  shape={df.shape}")
        print(f"  columns: {list(df.columns)}")
        print(df.to_string(index=False))

# ── Per-track metrics ──
per_track = {}
for exp in MODEL_ORDER:
    path = os.path.join(RES, f"{exp}_per_track.csv")
    if os.path.exists(path):
        df = pd.read_csv(path)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        per_track[exp] = df
        print(f"\n[{exp}_per_track.csv]  shape={df.shape}")
        print(f"  columns: {list(df.columns)}")
        print(df.head(4).to_string(index=False))

print("\n" + "=" * 70)
print("ALL FILES LOADED SUCCESSFULLY")
print("=" * 70)

# ─────────────────────────────────────────────────────────────────────────────
# Helper: combine all per_track into one DataFrame
# ─────────────────────────────────────────────────────────────────────────────
all_per_track = pd.concat(per_track.values(), ignore_index=True)
all_per_track["model"] = pd.Categorical(
    all_per_track["experiment"], categories=MODEL_ORDER, ordered=True
)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 1 — Training Loss Curves
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Fig 1] Training Loss Curves …")

fig, ax = plt.subplots(figsize=FIG_SIZE)
colors = {exp: PALETTE[i] for i, exp in enumerate(train_logs.keys())}
linestyles = {"train": "-", "val": "--"}

for exp, df in train_logs.items():
    label_train = f"{MODEL_LABELS[exp]} – Train"
    label_val   = f"{MODEL_LABELS[exp]} – Val"
    c = colors[exp]
    ax.plot(df["epoch"], df["train_loss"], color=c, ls="-",  lw=1.8,
            label=label_train)
    ax.plot(df["epoch"], df["val_loss"],   color=c, ls="--", lw=1.8,
            label=label_val)

ax.set_title("Training and Validation Loss", fontsize=TITLE_FS)
ax.set_xlabel("Epoch", fontsize=LABEL_FS)
ax.set_ylabel("Loss (negative SI-SDR)", fontsize=LABEL_FS)
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.legend(loc="upper right", fontsize=LEGEND_FS)
fig.tight_layout()
out = os.path.join(FIG, "fig1_training_loss.png")
fig.savefig(out, dpi=DPI)
plt.close(fig)
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 2 — Training SI-SDR Curves
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 2] Training SI-SDR Curves …")

fig, ax = plt.subplots(figsize=FIG_SIZE)

for exp, df in train_logs.items():
    label_train = f"{MODEL_LABELS[exp]} – Train"
    label_val   = f"{MODEL_LABELS[exp]} – Val"
    c = colors[exp]
    ax.plot(df["epoch"], df["train_si_sdr"], color=c, ls="-",  lw=1.8,
            label=label_train)
    ax.plot(df["epoch"], df["val_si_sdr"],   color=c, ls="--", lw=1.8,
            label=label_val)

ax.set_title("Training and Validation SI-SDR", fontsize=TITLE_FS)
ax.set_xlabel("Epoch", fontsize=LABEL_FS)
ax.set_ylabel("SI-SDR (dB)", fontsize=LABEL_FS)
ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
ax.legend(loc="lower right", fontsize=LEGEND_FS)
fig.tight_layout()
out = os.path.join(FIG, "fig2_training_si_sdr.png")
fig.savefig(out, dpi=DPI)
plt.close(fig)
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Per-source means for bar charts
# NOTE: si_sdr is track-level (same for all 4 sources per track) — not plotted
# per source. Only sdr, sar, isr are genuinely source-specific.
# ─────────────────────────────────────────────────────────────────────────────
source_means = (
    all_per_track
    .groupby(["experiment", "source"])[METRICS_SOURCE]
    .mean()
    .reset_index()
)
source_means["model"] = pd.Categorical(
    source_means["experiment"], categories=MODEL_ORDER, ordered=True
)

n_sources = len(SOURCES)
n_models  = len(MODEL_ORDER)
bar_width = 0.15
x = np.arange(n_sources)

# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — SDR by Source  (replaces SI-SDR which is track-level, not source-level)
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 3] SDR by Source …")
print("  (Note: si_sdr is track-level; SDR is source-specific — used here)")

fig, ax = plt.subplots(figsize=FIG_SIZE)

for i, exp in enumerate(MODEL_ORDER):
    vals = []
    sub = source_means[source_means["experiment"] == exp]
    for src in SOURCES:
        row = sub[sub["source"] == src]
        vals.append(row["sdr"].values[0] if len(row) else np.nan)
    offset = (i - n_models / 2 + 0.5) * bar_width
    ax.bar(x + offset, vals, bar_width,
           label=MODEL_LABELS[exp], color=PALETTE[i], alpha=0.85)

ax.set_title("Mean SDR by Source and Model", fontsize=TITLE_FS)
ax.set_xlabel("Source", fontsize=LABEL_FS)
ax.set_ylabel("SDR (dB)", fontsize=LABEL_FS)
ax.set_xticks(x)
ax.set_xticklabels([s.capitalize() for s in SOURCES], fontsize=TICK_FS)
ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
ax.legend(loc="upper right", fontsize=LEGEND_FS)
fig.tight_layout()
out = os.path.join(FIG, "fig3_sdr_by_source.png")
fig.savefig(out, dpi=DPI)
plt.close(fig)
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — SAR by Source
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 4] SAR by Source …")

fig, ax = plt.subplots(figsize=FIG_SIZE)

for i, exp in enumerate(MODEL_ORDER):
    vals = []
    sub = source_means[source_means["experiment"] == exp]
    for src in SOURCES:
        row = sub[sub["source"] == src]
        vals.append(row["sar"].values[0] if len(row) else np.nan)
    offset = (i - n_models / 2 + 0.5) * bar_width
    ax.bar(x + offset, vals, bar_width,
           label=MODEL_LABELS[exp], color=PALETTE[i], alpha=0.85)

ax.set_title("Mean SAR by Source and Model", fontsize=TITLE_FS)
ax.set_xlabel("Source", fontsize=LABEL_FS)
ax.set_ylabel("SAR (dB)", fontsize=LABEL_FS)
ax.set_xticks(x)
ax.set_xticklabels([s.capitalize() for s in SOURCES], fontsize=TICK_FS)
ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
ax.legend(loc="upper right", fontsize=LEGEND_FS)
fig.tight_layout()
out = os.path.join(FIG, "fig4_sar_by_source.png")
fig.savefig(out, dpi=DPI)
plt.close(fig)
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Overall Metrics Comparison (mean across all sources/tracks)
# Includes SI-SDR, SDR, SIR, SAR. SIR = ∞ in BSS Eval (museval) for the
# multi-source 4-stem setup, so SIR bars are absent (NaN) — annotated.
# ─────────────────────────────────────────────────────────────────────────────
print("[Fig 5] Overall Metrics Comparison (SI-SDR, SDR, SIR, SAR) …")

# Build overall means from per_track.
# si_sdr is track-level (same per source), so mean over track rows gives correct
# per-model overall SI-SDR. sdr and sar are averaged over all source rows.
METRICS_OVERALL = ["si_sdr", "sdr", "sar"]
overall_means = (
    all_per_track
    .groupby("experiment")[METRICS_OVERALL]
    .mean()
    .reset_index()
)
# Add SIR column — all NaN because museval returns inf for multi-source
overall_means["sir"] = np.nan
overall_means["model"] = pd.Categorical(
    overall_means["experiment"], categories=MODEL_ORDER, ordered=True
)
overall_means = overall_means.sort_values("model")

fig, ax = plt.subplots(figsize=FIG_SIZE)

metric_display_all = ["SI-SDR", "SDR", "SIR", "SAR"]
n_metrics_all = len(METRICS_WITH_SIR)
bar_width2 = 0.15
x2 = np.arange(len(MODEL_ORDER))

for j, (metric, mlabel) in enumerate(zip(METRICS_WITH_SIR, metric_display_all)):
    vals = overall_means.set_index("experiment").reindex(MODEL_ORDER)[metric].values
    offset = (j - n_metrics_all / 2 + 0.5) * bar_width2
    ax.bar(x2 + offset, vals, bar_width2,
           label=mlabel, color=PALETTE[j + 4], alpha=0.85)

ax.set_title("Overall Mean Metrics by Model (mean over all sources & tracks)",
             fontsize=TITLE_FS)
ax.set_xlabel("Model", fontsize=LABEL_FS)
ax.set_ylabel("Score (dB)", fontsize=LABEL_FS)
ax.set_xticks(x2)
ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER], fontsize=TICK_FS)
ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
ax.legend(loc="upper right", fontsize=LEGEND_FS)
# Annotate that SIR = ∞ (not plotted)
ax.text(0.01, 0.02, "Note: SIR = \u221e for all models (multi-source BSS Eval) \u2014 not plotted",
        transform=ax.transAxes, fontsize=8, color="gray",
        verticalalignment="bottom")
fig.tight_layout()
out = os.path.join(FIG, "fig5_overall_metrics.png")
fig.savefig(out, dpi=DPI)
plt.close(fig)
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX Table 1 — Metrics by Source for each model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Table 1] Metrics by Source …")

def fmt(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}"

# Build wide table: rows=models, cols=source×metric (SDR, SAR, ISR — source-specific)
header_cols = []
for src in SOURCES:
    for m in METRICS_TABLE:
        header_cols.append(f"{src.capitalize()} {METRIC_LABELS[m].split(' ')[0]}")

# LaTeX header
col_spec = "l" + "r" * len(header_cols)
header_str = " & ".join(["Model"] + header_cols) + r" \\"

table1_lines = [
    r"\begin{table}[htbp]",
    r"  \centering",
    r"  \caption{Source-specific separation metrics for each model variant. "
    r"Mean values over the MUSDB18-HQ test set (50 tracks). "
    r"SDR, SAR, ISR in dB. "
    r"SI-SDR is track-level (not source-specific); SIR evaluates to $\infty$ in multi-source setup.}",
    r"  \label{tab:metrics_by_source}",
    r"  \resizebox{\textwidth}{!}{%",
    r"  \begin{tabular}{" + col_spec + "}",
    r"  \toprule",
    "  " + header_str,
    r"  \midrule",
]

for exp in MODEL_ORDER:
    sub = source_means[source_means["experiment"] == exp]
    vals = []
    for src in SOURCES:
        row = sub[sub["source"] == src]
        for m in METRICS_TABLE:
            v = row[m].values[0] if len(row) else np.nan
            vals.append(fmt(v))
    table1_lines.append("  " + MODEL_LABELS[exp] + " & " + " & ".join(vals) + r" \\")

table1_lines += [
    r"  \bottomrule",
    r"  \end{tabular}%",
    r"  }",
    r"\end{table}",
]

table1_tex = "\n".join(table1_lines)
out_t1 = os.path.join(TAB, "table1_metrics_by_source.tex")
with open(out_t1, "w", encoding="utf-8") as f:
    f.write(table1_tex)
print(f"  Saved: {out_t1}")
print(table1_tex)

# ─────────────────────────────────────────────────────────────────────────────
# LaTeX Table 2 — Summary: mean SI-SDR, SDR, SAR + params + inference time
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Table 2] Summary Table …")

summary_rows = []
for exp in MODEL_ORDER:
    sub = overall_means[overall_means["experiment"] == exp]
    agg = agg_metrics.get(exp, None)
    params = int(agg["params"].values[0]) if agg is not None else None
    inf_ms = float(agg["inference_time_ms"].values[0]) if agg is not None else None
    row = {
        "model":  MODEL_LABELS[exp],
        "si_sdr": sub["si_sdr"].values[0] if len(sub) else np.nan,
        "sdr":    sub["sdr"].values[0]    if len(sub) else np.nan,
        "sar":    sub["sar"].values[0]    if len(sub) else np.nan,
        "params": params,
        "inf_ms": inf_ms,
    }
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

# Find best (highest) per numeric column  (best = max for SI-SDR/SDR/SAR)
best_si_sdr = summary_df["si_sdr"].max()
best_sdr    = summary_df["sdr"].max()
best_sar    = summary_df["sar"].max()

def bold_if(v, best, decimals=2):
    s = fmt(v, decimals)
    if not np.isnan(v) and np.isclose(v, best, atol=1e-6):
        return r"\textbf{" + s + "}"
    return s

table2_lines = [
    r"\begin{table}[htbp]",
    r"  \centering",
    r"  \caption{Summary of mean separation performance across all sources and "
    r"MUSDB18-HQ test tracks. Best result per column is \textbf{bold}. "
    r"SI-SDR, SDR, SAR in dB; Params = number of model parameters; "
    r"Inference = wall-clock time (ms) for the full 50-track test set.}",
    r"  \label{tab:summary}",
    r"  \begin{tabular}{lrrrrrr}",
    r"  \toprule",
    r"  Model & SI-SDR & SDR & SAR & Params & Inference (ms) \\",
    r"  \midrule",
]

for _, row in summary_df.iterrows():
    params_str = f"{int(row['params']):,}" if row['params'] is not None else "—"
    inf_str    = f"{row['inf_ms']:,.0f}"   if row['inf_ms']  is not None else "—"
    line = "  {} & {} & {} & {} & {} & {} \\\\".format(
        row["model"],
        bold_if(row["si_sdr"], best_si_sdr),
        bold_if(row["sdr"],    best_sdr),
        bold_if(row["sar"],    best_sar),
        params_str,
        inf_str,
    )
    table2_lines.append(line)

table2_lines += [
    r"  \bottomrule",
    r"  \end{tabular}",
    r"\end{table}",
]

table2_tex = "\n".join(table2_lines)
out_t2 = os.path.join(TAB, "table2_summary.tex")
with open(out_t2, "w", encoding="utf-8") as f:
    f.write(table2_tex)
print(f"  Saved: {out_t2}")
print(table2_tex)

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DONE.  All outputs saved to:")
print(f"  Figures : {FIG}")
print(f"  Tables  : {TAB}")
print("=" * 70)
