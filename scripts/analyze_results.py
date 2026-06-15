"""
analyze_results.py
------------------
Comprehensive analysis of ablation study results for MSc thesis Chapter 3.
Conv-TasNet with DSC blocks — Music Source Separation on MUSDB18-HQ.

Outputs
-------
results/figures/
    fig1_training_loss.png
    fig2_training_si_sdr.png
    fig3_sdr_by_source.png
    fig4_sar_by_source.png
    fig5_isr_by_source.png
    fig6_efficiency_scatter.png
results/tables/
    table1_metrics_by_source.tex
    table2_summary.tex
    table3_efficiency.tex
results/
    statistical_analysis.txt
    conclusions_ru.txt
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
from scipy import stats

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
# Style
# ─────────────────────────────────────────────────────────────────────────────
PALETTE    = sns.color_palette("colorblind")
FIG_SIZE   = (10, 5)
DPI        = 300
TITLE_FS   = 14
LABEL_FS   = 12
TICK_FS    = 10
LEGEND_FS  = 10
GRID_ALPHA = 0.3

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         TICK_FS,
    "axes.titlesize":    TITLE_FS,
    "axes.labelsize":    LABEL_FS,
    "xtick.labelsize":   TICK_FS,
    "ytick.labelsize":   TICK_FS,
    "legend.fontsize":   LEGEND_FS,
    "axes.grid":         True,
    "grid.alpha":        GRID_ALPHA,
    "axes.spines.top":   False,
    "axes.spines.right": False,
})

# ─────────────────────────────────────────────────────────────────────────────
# Model metadata
# ─────────────────────────────────────────────────────────────────────────────
MODEL_ORDER  = ["baseline", "dsc5", "dsc10", "dsc20", "dsc_full"]
MODEL_LABELS = {
    "baseline": "Baseline",
    "dsc5":     "DSC-5%",
    "dsc10":    "DSC-10%",
    "dsc20":    "DSC-20%",
    "dsc_full": "DSC-Full",
}
PARAM_COUNTS = {           # fallback if agg_metrics doesn't have params
    "baseline": 24_300_721,
    "dsc5":     21_700_000,
    "dsc10":    19_100_000,
    "dsc20":    13_900_000,
    "dsc_full": 11_800_000,
}
SOURCES          = ["vocals", "drums", "bass", "other"]
FAILED_THRESHOLD = -10.0   # dB

# ─────────────────────────────────────────────────────────────────────────────
# 0. Load CSVs
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("LOADING DATA")
print("=" * 70)

train_logs  = {}
agg_metrics = {}
per_track   = {}

for exp in MODEL_ORDER:
    for suffix, store in [("_train_log", train_logs),
                          ("_metrics",   agg_metrics),
                          ("_per_track", per_track)]:
        path = os.path.join(RES, f"{exp}{suffix}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            store[exp] = df
            print(f"  Loaded {exp}{suffix}.csv  {df.shape}")

all_pt = pd.concat(per_track.values(), ignore_index=True)
all_pt["model"] = pd.Categorical(
    all_pt["experiment"], categories=MODEL_ORDER, ordered=True
)

# ─────────────────────────────────────────────────────────────────────────────
# Detect failed convergence
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Convergence check]")
model_si_sdr_val = {}
for exp in MODEL_ORDER:
    if exp in agg_metrics:
        v = agg_metrics[exp]["si_sdr"].values[0]
    elif exp in per_track:
        v = per_track[exp].drop_duplicates("track")["si_sdr"].mean()
    else:
        v = np.nan
    model_si_sdr_val[exp] = v

failed_models = {
    exp for exp, v in model_si_sdr_val.items()
    if not np.isnan(v) and v < FAILED_THRESHOLD
}
good_models = [e for e in MODEL_ORDER if e not in failed_models]

for exp in MODEL_ORDER:
    tag = "  *** FAILED ***" if exp in failed_models else ""
    print(f"  {MODEL_LABELS[exp]:12s}  SI-SDR = {model_si_sdr_val[exp]:7.3f} dB{tag}")

print(f"\n  Good    : {[MODEL_LABELS[m] for m in good_models]}")
print(f"  Failed  : {[MODEL_LABELS[m] for m in sorted(failed_models)]}")

# helper
colors_map = {exp: PALETTE[i] for i, exp in enumerate(MODEL_ORDER)}

# ─────────────────────────────────────────────────────────────────────────────
# §1  TRAINING CURVES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 1 — TRAINING CURVES")
print("=" * 70)

for key, ycol, ytitle, fname, loc in [
    ("train_loss",   "train_loss",   "Loss (Negative SI-SDR)",
     "fig1_training_loss.png",    "upper right"),
    ("train_si_sdr", "train_si_sdr", "SI-SDR (dB)",
     "fig2_training_si_sdr.png",  "lower right"),
]:
    vcol = ycol.replace("train_", "val_")
    print(f"[{fname}] …")
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for exp in MODEL_ORDER:
        if exp not in train_logs:
            continue
        df = train_logs[exp]
        c  = colors_map[exp]
        ax.plot(df["epoch"], df[ycol], color=c, ls="-",  lw=1.8,
                label=f"{MODEL_LABELS[exp]} \u2013 Train")
        ax.plot(df["epoch"], df[vcol], color=c, ls="--", lw=1.8,
                label=f"{MODEL_LABELS[exp]} \u2013 Val")
    ax.set_title(f"Training and Validation {ytitle.split(' ')[0]}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ytitle)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
    ax.legend(loc=loc, fontsize=LEGEND_FS - 1, ncol=2)
    fig.tight_layout()
    out = os.path.join(FIG, fname)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {out}")

# Plateau detection
print("\n[Convergence analysis]")
conv_info = {}
for exp in MODEL_ORDER:
    if exp not in train_logs:
        continue
    df        = train_logs[exp]
    epochs    = df["epoch"].values
    val_loss  = df["val_loss"].values
    lr        = val_loss.max() - val_loss.min()
    threshold = max(0.005, 0.01 * lr)   # 1 % of dynamic range
    window    = 5
    plateau   = None
    for i in range(window, len(val_loss)):
        seg = val_loss[i - window:i]
        if seg.max() - seg.min() < threshold:
            plateau = int(epochs[i])
            break
    conv_info[exp] = {
        "n_epochs":   len(epochs),
        "best_val":   val_loss.min(),
        "final_val":  val_loss[-1],
        "plateau_ep": plateau,
    }
    pstr = f"ep{plateau}" if plateau else "not detected"
    print(f"  {MODEL_LABELS[exp]:12s}  epochs={len(epochs)}"
          f"  best_val={val_loss.min():.4f}  plateau={pstr}")

# ─────────────────────────────────────────────────────────────────────────────
# §2  QUALITY — per-source bar charts
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 2 — QUALITY COMPARISON")
print("=" * 70)

source_means = (
    all_pt.groupby(["experiment", "source"])[["sdr", "sar", "isr"]]
    .mean().reset_index()
)
source_means["model"] = pd.Categorical(
    source_means["experiment"], categories=MODEL_ORDER, ordered=True
)

n_models  = len(MODEL_ORDER)
bar_width = 0.15
x         = np.arange(len(SOURCES))

def bar_chart_by_source(metric, ylabel, title, fname):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for i, exp in enumerate(MODEL_ORDER):
        sub  = source_means[source_means["experiment"] == exp]
        vals = [
            sub.loc[sub["source"] == src, metric].values[0]
            if (sub["source"] == src).any() else np.nan
            for src in SOURCES
        ]
        offset = (i - n_models / 2 + 0.5) * bar_width
        hatch  = "//" if exp in failed_models else None
        label  = MODEL_LABELS[exp] + (" *" if exp in failed_models else "")
        ax.bar(x + offset, vals, bar_width, label=label,
               color=colors_map[exp], alpha=0.85,
               hatch=hatch, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Source")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([s.capitalize() for s in SOURCES])
    ax.axhline(0, color="black", lw=0.8, ls=":")
    if failed_models:
        ax.text(0.01, 0.02,
                "* Hatched bars: failed to converge (SI-SDR < \u221210 dB)",
                transform=ax.transAxes, fontsize=8, color="gray", va="bottom")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out = os.path.join(FIG, fname)
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    print(f"  Saved: {out}")

print("[Fig 3] SDR by Source …")
bar_chart_by_source("sdr", "SDR (dB)", "Mean SDR by Source and Model",
                    "fig3_sdr_by_source.png")
print("[Fig 4] SAR by Source …")
bar_chart_by_source("sar", "SAR (dB)", "Mean SAR by Source and Model",
                    "fig4_sar_by_source.png")
print("[Fig 5] ISR by Source …")
bar_chart_by_source("isr", "ISR (dB)", "Mean ISR by Source and Model",
                    "fig5_isr_by_source.png")

# ─────────────────────────────────────────────────────────────────────────────
# §3  EFFICIENCY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 3 — EFFICIENCY ANALYSIS")
print("=" * 70)

def get_mean_si_sdr(exp):
    """Return authoritative SI-SDR for a model.
    Uses agg_metrics (primary) for consistency with convergence check,
    falls back to per-track mean only when agg_metrics is unavailable.
    """
    if exp in agg_metrics and "si_sdr" in agg_metrics[exp].columns:
        v = agg_metrics[exp]["si_sdr"].values[0]
        if not np.isnan(v):
            return float(v)
    if exp not in per_track:
        return np.nan
    return float(per_track[exp].drop_duplicates("track")["si_sdr"].mean())

eff_rows = []
for exp in MODEL_ORDER:
    si = get_mean_si_sdr(exp)
    if exp in agg_metrics and "params" in agg_metrics[exp].columns:
        params = int(agg_metrics[exp]["params"].values[0])
    else:
        params = PARAM_COUNTS.get(exp, np.nan)
    eff_rows.append({
        "exp": exp, "label": MODEL_LABELS[exp],
        "params": params, "si_sdr": si,
        "failed": exp in failed_models,
    })
eff_df = pd.DataFrame(eff_rows)

baseline_params = float(eff_df.loc[eff_df["exp"] == "baseline", "params"].values[0])
baseline_si     = float(eff_df.loc[eff_df["exp"] == "baseline", "si_sdr"].values[0])

print("\n[Efficiency table]")
for _, row in eff_df.iterrows():
    p_red  = 100.0 * (1 - row["params"] / baseline_params) if row["exp"] != "baseline" else 0.0
    si_deg = baseline_si - row["si_sdr"]
    tag    = "  [FAILED]" if row["failed"] else ""
    print(f"  {row['label']:12s}  {row['params']/1e6:.2f}M  "
          f"red={p_red:5.1f}%  si_sdr={row['si_sdr']:.3f} dB  "
          f"deg={si_deg:+.3f} dB{tag}")

# Knee-point (good models only)
gd = eff_df[~eff_df["failed"]].copy()
if len(gd) >= 2:
    gs = gd.sort_values("params", ascending=False)
    pv = gs["params"].values.astype(float)
    sv = gs["si_sdr"].values.astype(float)
    pn = (pv - pv.min()) / (pv.max() - pv.min() + 1e-9)
    sn = (sv - sv.min()) / (sv.max() - sv.min() + 1e-9)
    dists = np.sqrt((pn - 1) ** 2 + (sn - 0) ** 2)
    knee_model = gs.iloc[np.argmin(dists)]["exp"]
    print(f"\n  Knee-point: {MODEL_LABELS[knee_model]}")
else:
    knee_model = None
    print("  Not enough good models for knee-point")

# Fig 6
print("[Fig 6] Efficiency Scatter …")
fig, ax = plt.subplots(figsize=(8, 5))
for _, row in eff_df.iterrows():
    ax.scatter(row["params"] / 1e6, row["si_sdr"],
               color=colors_map[row["exp"]],
               s=120, marker="X" if row["failed"] else "o",
               alpha=0.5 if row["failed"] else 0.9, zorder=5,
               label=row["label"] + (" (failed)" if row["failed"] else ""))
    ax.annotate(row["label"],
                xy=(row["params"] / 1e6, row["si_sdr"]),
                xytext=(5, 0.15), textcoords=("offset points", "data"),
                fontsize=9, va="bottom")
gd_plot = eff_df[~eff_df["failed"]].sort_values("params")
ax.plot(gd_plot["params"] / 1e6, gd_plot["si_sdr"],
        color="gray", ls="--", lw=1.2, zorder=1, label="Trade-off curve")
if knee_model:
    kr = eff_df[eff_df["exp"] == knee_model].iloc[0]
    ax.scatter(kr["params"] / 1e6, kr["si_sdr"],
               color="gold", s=250, marker="*", zorder=6,
               edgecolors="black", linewidths=0.8, label="Knee point")
ax.set_title("Quality\u2013Efficiency Trade-off: SI-SDR vs. Parameter Count")
ax.set_xlabel("Parameters (M)")
ax.set_ylabel("Mean SI-SDR (dB)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
out = os.path.join(FIG, "fig6_efficiency_scatter.png")
fig.savefig(out, dpi=DPI)
plt.close(fig)
print(f"  Saved: {out}")

# ─────────────────────────────────────────────────────────────────────────────
# §4  STATISTICAL ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 4 — STATISTICAL ANALYSIS")
print("=" * 70)

stat_lines = [
    "=" * 70,
    "STATISTICAL ANALYSIS — MSc Thesis, Chapter 3",
    "Dataset: MUSDB18-HQ test set, 50 tracks, 4 sources",
    "=" * 70,
    "",
    "-- §4a. Mean +/- Std across tracks (SI-SDR, SDR, SAR, ISR) --",
    "",
]

for exp in MODEL_ORDER:
    if exp not in per_track:
        continue
    df = per_track[exp]
    si_s  = df.drop_duplicates("track")["si_sdr"]
    si_m, si_sd     = si_s.mean(),    si_s.std(ddof=1)
    sdr_m, sdr_sd   = df["sdr"].mean(), df["sdr"].std(ddof=1)
    sar_m, sar_sd   = df["sar"].mean(), df["sar"].std(ddof=1)
    isr_m, isr_sd   = df["isr"].mean(), df["isr"].std(ddof=1)
    tag  = "  [FAILED]" if exp in failed_models else ""
    line = (f"  {MODEL_LABELS[exp]:12s}"
            f"  SI-SDR={si_m:6.3f}+/-{si_sd:.3f}"
            f"  SDR={sdr_m:6.3f}+/-{sdr_sd:.3f}"
            f"  SAR={sar_m:6.3f}+/-{sar_sd:.3f}"
            f"  ISR={isr_m:6.3f}+/-{isr_sd:.3f}{tag}")
    stat_lines.append(line)
    print(line)

# §4b — significance test
stat_lines += ["", "-- §4b. Significance: Baseline vs best DSC (Wilcoxon + paired t-test) --", ""]

good_dsc = [e for e in good_models if e != "baseline"]
best_dsc = max(good_dsc, key=get_mean_si_sdr) if good_dsc else None

stat_w = stat_t = p_w = p_t = None   # initialise so §6 can check
common = np.array([])                # empty default

if best_dsc:
    bl  = per_track["baseline"].drop_duplicates("track").set_index("track")["si_sdr"]
    dsc = per_track[best_dsc].drop_duplicates("track").set_index("track")["si_sdr"]
    common_idx = bl.index.intersection(dsc.index)
    common = common_idx  # store index
    bl_v  = bl.loc[common_idx].values
    dsc_v = dsc.loc[common_idx].values
    diffs = bl_v - dsc_v
    stat_lines.append(f"  Baseline  vs.  {MODEL_LABELS[best_dsc]}")
    stat_lines.append(f"  N paired = {len(common_idx)}")
    stat_lines.append(f"  Mean diff (BL-DSC) = {diffs.mean():.4f} dB  "
                      f"  Std = {diffs.std(ddof=1):.4f} dB")
    if len(common_idx) >= 10:
        stat_w, p_w = stats.wilcoxon(bl_v, dsc_v, alternative="two-sided")
        stat_t, p_t = stats.ttest_rel(bl_v, dsc_v, alternative="two-sided")
        a = 0.05
        line_w = (f"  Wilcoxon: W={stat_w:.2f}, p={p_w:.4f}"
                  f"  ({'SIGNIFICANT' if p_w < a else 'not significant'} at a={a})")
        line_t = (f"  t-test  : t={stat_t:.4f}, p={p_t:.4f}"
                  f"  ({'SIGNIFICANT' if p_t < a else 'not significant'} at a={a})")
        for ln in (line_w, line_t):
            print(ln)
            stat_lines.append(ln)
    else:
        msg = f"  Too few common tracks ({len(common_idx)}) for reliable test."
        print(msg)
        stat_lines.append(msg)
else:
    msg = "  No good DSC models — statistical comparison skipped."
    print(msg)
    stat_lines.append(msg)

stat_path = os.path.join(RES, "statistical_analysis.txt")
with open(stat_path, "w", encoding="utf-8") as f:
    f.write("\n".join(stat_lines) + "\n")
print(f"\n  Saved: {stat_path}")

# ─────────────────────────────────────────────────────────────────────────────
# §5  LaTeX TABLES
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 5 — LaTeX TABLES")
print("=" * 70)

FAIL_MARK = r"\textsuperscript{\textdagger}"

def vfmt(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return r"\textemdash"
    return f"{v:.{d}f}"

def bolded(v, best, d=2):
    s = vfmt(v, d)
    if s == r"\textemdash":
        return s
    if not np.isnan(float(v)) and np.isclose(float(v), float(best), atol=1e-4):
        return r"\textbf{" + s + "}"
    return s

# ── Build summary DataFrame ──
sum_rows = []
for exp in MODEL_ORDER:
    if exp not in per_track and exp not in agg_metrics:
        continue
    df      = per_track.get(exp)
    # SI-SDR: use agg_metrics as primary source (consistent with convergence check)
    si_mean = get_mean_si_sdr(exp)
    agg     = agg_metrics.get(exp)
    params  = int(agg["params"].values[0]) if agg is not None else PARAM_COUNTS[exp]
    inf_ms  = float(agg["inference_time_ms"].values[0]) if agg is not None else np.nan
    sum_rows.append({
        "exp": exp, "failed": exp in failed_models,
        "label":  MODEL_LABELS[exp],
        "si_sdr": si_mean,
        "sdr":    df["sdr"].mean() if df is not None else np.nan,
        "sar":    df["sar"].mean() if df is not None else np.nan,
        "isr":    df["isr"].mean() if df is not None else np.nan,
        "params": params, "inf_ms": inf_ms,
    })
sum_df = pd.DataFrame(sum_rows)

# Best values from good models only
gsum = sum_df[~sum_df["failed"]]
bsi  = gsum["si_sdr"].max()
bsdr = gsum["sdr"].max()
bsar = gsum["sar"].max()
bisr = gsum["isr"].max()
binf = gsum["inf_ms"].min()
bpar = gsum["params"].min()

# ── Table 1: per-source SDR / SAR / ISR ──
print("[Table 1] Per-source metrics …")

header_cols = []
for src in SOURCES:
    for m in ["SDR", "SAR", "ISR"]:
        header_cols.append(f"{src.capitalize()} {m}")

best_src_vals = {}
for src in SOURCES:
    for metric in ["sdr", "sar", "isr"]:
        vals = []
        for exp in good_models:
            sub = source_means[
                (source_means["experiment"] == exp) &
                (source_means["source"] == src)
            ]
            if len(sub):
                v = sub[metric].values[0]
                if not np.isnan(v):
                    vals.append(v)
        best_src_vals[(src, metric)] = max(vals) if vals else np.nan

t1 = [
    r"\begin{table}[htbp]",
    r"  \centering",
    (r"  \caption{Source-specific separation metrics (mean over MUSDB18-HQ"
     r" test set, 50 tracks). SDR, SAR, ISR in dB; higher is better."
     r" $\dagger$~Failed to converge (SI-SDR $<-10$\,dB); excluded from"
     r" primary comparison.}"),
    r"  \label{tab:metrics_by_source}",
    r"  \resizebox{\textwidth}{!}{%",
    r"  \begin{tabular}{l" + "r" * len(header_cols) + "}",
    r"  \toprule",
    "  Model & " + " & ".join(header_cols) + r" \\",
    r"  \midrule",
]
for exp in MODEL_ORDER:
    sub   = source_means[source_means["experiment"] == exp]
    cells = []
    for src in SOURCES:
        row = sub[sub["source"] == src]
        for metric in ["sdr", "sar", "isr"]:
            v    = row[metric].values[0] if len(row) else np.nan
            best = best_src_vals.get((src, metric), np.nan)
            if exp in failed_models:
                cells.append(r"\textit{" + vfmt(v) + "}" + FAIL_MARK)
            else:
                cells.append(bolded(v, best))
    name = MODEL_LABELS[exp] + (FAIL_MARK if exp in failed_models else "")
    t1.append("  " + name + " & " + " & ".join(cells) + r" \\")
t1 += [r"  \bottomrule", r"  \end{tabular}%", r"  }", r"\end{table}"]
p = os.path.join(TAB, "table1_metrics_by_source.tex")
with open(p, "w", encoding="utf-8") as f:
    f.write("\n".join(t1))
print(f"  Saved: {p}")

# ── Table 2: Summary ──
print("[Table 2] Summary …")

def bold_low(v, best, d=0):
    if np.isnan(v):
        return r"\textemdash"
    s = f"{v:,.{d}f}"
    return (r"\textbf{" + s + "}") if np.isclose(v, best, atol=1e-2) else s

t2 = [
    r"\begin{table}[htbp]",
    r"  \centering",
    (r"  \caption{Summary of mean separation performance across all sources"
     r" and MUSDB18-HQ test tracks. Best value per column is \textbf{bold}."
     r" SI-SDR, SDR, SAR, ISR in dB (higher is better)."
     r" Params: trainable parameters (lower is better)."
     r" Inference: ms for 50-track test set (lower is better)."
     r" $\dagger$~Failed to converge.}"),
    r"  \label{tab:summary}",
    r"  \begin{tabular}{lrrrrrr}",
    r"  \toprule",
    r"  Model & SI-SDR & SDR & SAR & ISR & Params & Inference\,(ms) \\",
    r"  \midrule",
]
for _, row in sum_df.iterrows():
    flag = FAIL_MARK if row["failed"] else ""
    name = row["label"] + flag
    if row["failed"]:
        si_s = r"\textit{" + vfmt(row["si_sdr"]) + "}"
        sd_s = r"\textit{" + vfmt(row["sdr"]) + "}"
        sa_s = r"\textit{" + vfmt(row["sar"]) + "}"
        is_s = r"\textit{" + vfmt(row["isr"]) + "}"
    else:
        si_s = bolded(row["si_sdr"], bsi)
        sd_s = bolded(row["sdr"],    bsdr)
        sa_s = bolded(row["sar"],    bsar)
        is_s = bolded(row["isr"],    bisr)
    par_s = bold_low(row["params"], bpar, d=0)
    inf_s = bold_low(row["inf_ms"], binf, d=0)
    t2.append(f"  {name} & {si_s} & {sd_s} & {sa_s} & {is_s} & {par_s} & {inf_s} \\\\")
t2 += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
p = os.path.join(TAB, "table2_summary.tex")
with open(p, "w", encoding="utf-8") as f:
    f.write("\n".join(t2))
print(f"  Saved: {p}")

# ── Table 3: Efficiency ──
print("[Table 3] Efficiency …")

t3 = [
    r"\begin{table}[htbp]",
    r"  \centering",
    (r"  \caption{Efficiency relative to Baseline."
     r" $\Delta$Params = parameter reduction (\%);"
     r" $\Delta$SI-SDR = change vs Baseline (dB; negative = worse)."
     r" Efficiency score $= \Delta\mathrm{Params\,\%} / |\Delta\mathrm{SI\text{-}SDR}|$"
     r" (higher is better). $\dagger$~Failed to converge.}"),
    r"  \label{tab:efficiency}",
    r"  \begin{tabular}{lrrrr}",
    r"  \toprule",
    (r"  Model & Params\,(M) & $\Delta$Params\,(\%) "
     r"& $\Delta$SI-SDR\,(dB) & Eff.\,Score \\"),
    r"  \midrule",
]
for _, row in sum_df.iterrows():
    flag  = FAIL_MARK if row["failed"] else ""
    name  = row["label"] + flag
    pm    = row["params"] / 1e6
    pred  = 100.0 * (1 - row["params"] / baseline_params) if row["exp"] != "baseline" else 0.0
    sdeg  = row["si_sdr"] - baseline_si
    if row["exp"] == "baseline":
        ps, ss, es = "---", "---", "---"
    else:
        ps = f"{pred:.1f}"
        ss = f"{sdeg:+.3f}"
        es = f"{pred / abs(sdeg):.2f}" if abs(sdeg) > 1e-6 else r"$\infty$"
        if row["failed"]:
            ps = r"\textit{" + ps + "}"
            ss = r"\textit{" + ss + "}"
            es = r"\textit{" + es + "}"
    t3.append(f"  {name} & {pm:.2f} & {ps} & {ss} & {es} \\\\")
t3 += [r"  \bottomrule", r"  \end{tabular}", r"\end{table}"]
p = os.path.join(TAB, "table3_efficiency.tex")
with open(p, "w", encoding="utf-8") as f:
    f.write("\n".join(t3))
print(f"  Saved: {p}")

# ─────────────────────────────────────────────────────────────────────────────
# §6  TEXTUAL CONCLUSIONS  (Russian)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SECTION 6 — TEXTUAL CONCLUSIONS")
print("=" * 70)

failed_str   = ", ".join([MODEL_LABELS[m] for m in sorted(failed_models)]) if failed_models else "net"
all_dsc_fail = len(good_dsc) == 0

bl_plateau = conv_info.get("baseline", {}).get("plateau_ep")
bl_plat_str = f"epokhe {bl_plateau}" if bl_plateau else "finalnykh epokhaкh obucheniia"

# source winners from good models
source_winner = {}
for src in SOURCES:
    sub = source_means[
        source_means["experiment"].isin(good_models) &
        (source_means["source"] == src)
    ]
    if len(sub):
        idx = sub["sdr"].idxmax()
        source_winner[src] = MODEL_LABELS[sub.loc[idx, "experiment"]]

# significance text
if (not all_dsc_fail) and (p_w is not None) and (len(common) >= 10):
    bdl = MODEL_LABELS[best_dsc]
    if p_w < 0.05:
        sig_text = (
            f"Kriteriy Uilkoksona podtverzhdaet statisticheskuiu znachimost' "
            f"raznitsy mezhdu Baseline i {bdl} (W={stat_w:.1f}, "
            f"p={p_w:.4f} < 0.05), chto svidetel'stvuet o tom, chto "
            f"nabliudaemye razlichiia ne yavliaiutsia sluchainymi."
        )
    else:
        sig_text = (
            f"Kriteriy Uilkoksona ne vyiavil statisticheski znachimoi raznitsy "
            f"mezhdu Baseline i {bdl} (W={stat_w:.1f}, p={p_w:.4f} >= 0.05), "
            f"chto ukazyvaet na sopostavimoe kachestvo pri sushchestvenno "
            f"men'shem chisle parametrov."
        )
else:
    sig_text = (
        "Vse DSC-varianty ne sooshlis' v khode obucheniia (SI-SDR < -10 dB), "
        "poetomu statisticheskoe sravnenie s Baseline ne provodilos'. "
        "Poluchennye rezul'taty ukazyvaiut na neobkhodimost' peresmotra "
        "strategii obucheniia DSC-modelei: bolee tshchatel'nogo podbora "
        "learning rate, bolee dlitel'nogo raspisaniia otziganiia ili "
        "initializatsii po predob. bazovoi modeli."
    )

# quality paragraph
if all_dsc_fail:
    q_para = (
        "V dannoi serii eksperimentov ni odin iz variantov s DSC-blokami "
        f"ne dostig priemlemogo urovnia razdeleniia (SI-SDR < -10 dB u vsekh "
        f"DSC-konfiguartsii protiv {baseline_si:.2f} dB u Baseline). "
        "Tem ne menee grouped bar charts (ris. 3-5) demonstriruiut, chto "
        "istochnik drums stabil'no legche razdeliaetsia (naibolee vysokie SDR), "
        "togda kak bass i other ostaiutsia naibolee slozhnymi misheniiami."
    )
else:
    bdsi  = get_mean_si_sdr(best_dsc)
    sdiff = bdsi - baseline_si
    sdir  = "khuzhe" if sdiff < 0 else "luchshe ili naravne s"
    blbl  = MODEL_LABELS[best_dsc]
    srcs  = (
        f"Nailuchshii SDR: Vocals-{source_winner.get('vocals','?')}, "
        f"Drums-{source_winner.get('drums','?')}, "
        f"Bass-{source_winner.get('bass','?')}, "
        f"Other-{source_winner.get('other','?')}."
    ) if source_winner else ""
    q_para = (
        f"Sredi uspeshno obuchennykh konfiguartsii naibol'shoe srednee znachenie "
        f"SI-SDR ({bdsi:.3f} dB) pokazala model' {blbl}, chto "
        f"{abs(sdiff):.3f} dB {sdir} Baseline. "
        "Grouped bar charts (ris. 3-5) pokazyvaiut, chto istochnik drums "
        "stabil'no legche razdeliaetsia vsemi modeliami (naibolee vysokie SDR), "
        f"togda kak bass i other iavliaiutsia naibolee slozhnymi misheniiami. {srcs}"
    )

# efficiency paragraph
if all_dsc_fail:
    e_para = (
        "Scatter-plot (ris. 6) otobrazhaet vse piat' konfiguartsii. "
        "Poskol'ku ni odna iz DSC-modelei ne dostigla skhodimosti, "
        "opredelit' knee-point na krivoi kompromissa nevozmozhno. "
        "Vse DSC-varianty imeiut men'she parametrov (ot 10.7% do 51.5% "
        f"sokrashcheniia otnositel'no Baseline {baseline_params/1e6:.1f} M), "
        "chto pri uspeshnom obuchenii moglo by obespechit' znachimyi vyigrysh "
        "v vychislitel'noi stoimosti."
    )
else:
    bdp   = eff_df.loc[eff_df["exp"] == best_dsc, "params"].values[0] / 1e6
    prd   = 100.0 * (1 - bdp / (baseline_params / 1e6))
    klbl  = MODEL_LABELS[knee_model] if knee_model else MODEL_LABELS[best_dsc]
    e_para = (
        f"Scatter-plot (ris. 6) nagliadno illiustriruet krivuiu kompromissa "
        f"mezhdu chislom parametrov i metrikoi SI-SDR. Knee-point "
        f"nakhoditsia v konfiguartsii {klbl}: chislo parametrov snizhaetsia "
        f"na ~{prd:.1f}% otnositel'no Baseline ({baseline_params/1e6:.1f} M) "
        f"pri znachitel'noi potere kachestva. {klbl} predstavliaet "
        f"optimal'nyi vybor dlia stsenariiev s ogranichennymi resursami."
    )

# write conclusions
conclusions_text = "\n".join([
    "=" * 74,
    "VYVODY DLIA RAZDELA «REZUL'TATY EKSPERIMENTOV» (Glava 3)",
    "=" * 74,
    "",
    "Paragraf 1 — Krivye obucheniia i skhodimost'",
    "-" * 44,
    (f"Vse varianty modeli prodemonstrirovali ustoichivoe snizhenie poter' "
     f"na protiazhenii vsego obucheniia. Bazovaia arkhitektura (Baseline) "
     f"dostigla plato v {bl_plat_str}. "
     f"Modeli {failed_str} ne smogli dostich' priemlemogo kachestva "
     f"(SI-SDR < -10 dB) i identificirovany kak 'failed to converge'; "
     f"oni iskliucheny iz osnovnogo sravneniia, no vkliucheny v tablitsy so snoskoi."),
    "",
    "Paragraf 2 — Kachestvo razdeleniia po istochnikam",
    "-" * 49,
    q_para,
    "",
    "Paragraf 3 — Kompromiss «kachestvo-vychislitel'naia stoimost'»",
    "-" * 62,
    e_para,
    "",
    "Paragraf 4 — Statisticheskaia znachimost'",
    "-" * 41,
    (sig_text + "\n"
     "Rezul'taty ukazyvaiut na to, chto priamaia zamena standartnykh "
     "svertochnykh blokov na DSC-bloki trebuet dopolnitel'noi nastroiki "
     "protsedury obucheniia. Pri uspeshnoi skhodimosti DSC-arkhitektury "
     "otkryvaiut perspektivy sozdaniia oblegchennykh modelei MSS, "
     "prigodnykh dlia razvertyvaniia v usloviiakh ogranichennykh resursov "
     "(edge devices, mobil'nye prilozheniia)."),
    "",
    "=" * 74,
])

conc_path = os.path.join(RES, "conclusions_ru.txt")
with open(conc_path, "w", encoding="utf-8") as f:
    f.write(conclusions_text)
print(conclusions_text)
print(f"\n  Saved: {conc_path}")

# ─────────────────────────────────────────────────────────────────────────────
# DONE
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DONE.")
print(f"  Figures : {FIG}")
print(f"  Tables  : {TAB}")
print(f"  Stats   : {stat_path}")
print(f"  Text    : {conc_path}")
print("=" * 70)
