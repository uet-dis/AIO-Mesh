#!/usr/bin/env python3
"""
plot_pareto_design_space.py
────────────────────────────
Generates: "Figure – Ablation Study: HARP Architectural Design Space"

Layout: 2-row × 3-column figure.
  Row 0: NSFNET  |  Row 1: GEANT2
  Col 0: Safe Acceptance Rate (%)     ↑ higher is better
  Col 1: SLA Violation Rate (%)       ↓ lower is better
  Col 2: Average Latency (ms)         ↓ lower is better

Each subplot shows grouped bars (bursty / heavyTail / uniform)
for every ablation variant, with mean ± std error bars.

Data source: docs/AIO-mesh/results/{topo}/{topo}_master_ablation_summary_5seeds.csv
"""

from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent
DATA_DIR = BASE / "docs/AIO-mesh/results"
FIGS_DIR = BASE / "docs/AIO-mesh/Figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Publication rcParams (IEEE / ACM style) ────────────────────────────────────
plt.rcParams.update({
    # ---- Typography ----
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          8.5,
    "axes.titlesize":     9.5,
    "axes.labelsize":     8.5,
    "xtick.labelsize":    7.5,
    "ytick.labelsize":    7.5,
    "legend.fontsize":    7.5,
    "legend.title_fontsize": 8.0,
    # ---- Lines & markers ----
    "lines.linewidth":    1.0,
    "patch.linewidth":    0.6,
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "xtick.minor.width":  0.5,
    "ytick.minor.width":  0.5,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    # ---- Grid ----
    "axes.grid":          True,
    "grid.alpha":         0.35,
    "grid.linewidth":     0.5,
    "grid.linestyle":     "--",
    # ---- Spines ----
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    # ---- Figure ----
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    # ---- PDF/SVG ----
    "pdf.fonttype":       42,   # embed fonts as TrueType (required by IEEE)
    "svg.fonttype":       "none",
})


# ── Variant catalogue ─────────────────────────────────────────────────────────
# Order: HARP full first (highlighted), then ablations
VARIANTS = [
    ("HARP full",                          "HARP (Full)",         "#C0392B"),
    ("HARP w/o adaptive penalty",          "w/o Adapt. Penalty",  "#E67E22"),
    ("HARP w/o GAT",                       "w/o GAT",             "#2980B9"),
    ("HARP w/o hard masking",              "w/o Hard Mask",       "#27AE60"),
    ("HARP w/o mask + unchecked admission","w/o Mask+Unchecked",  "#8E44AD"),
    ("HARP with soft MSD penalty",         "Soft MSD Penalty",    "#616A6B"),
]
VARIANT_KEYS   = [v[0] for v in VARIANTS]
VARIANT_LABELS = [v[1] for v in VARIANTS]
VARIANT_COLORS = [v[2] for v in VARIANTS]

SCENARIOS     = ["bursty", "heavyTail", "uniform"]
SCENARIO_HATCH= ["",       "///",       "..."]   # distinct fill patterns
SCENARIO_ALPHA= [0.88,     0.72,        0.60]

TOPO_LABEL = {
    "nsfnet": "NSFNET  (14 nodes, N²=196)",
    "geant2": "GEANT2  (23 nodes, N²=529)",
}

# Metric definitions: (column_base, ylabel, better_direction, y_arrow)
METRICS = [
    ("safe_acceptance_rate", "Safe Acceptance Rate (%)", "↑ higher is better", "up"),
    ("sla_violation_rate",   "SLA Violation Rate (%)",   "↓ lower is better",  "down"),
    ("avg_latency_ms",       "Average Latency (ms)",     "↓ lower is better",  "down"),
]


# ── Helper: parse "mean ± std" strings ────────────────────────────────────────
def parse_mean_std(s: str):
    """Return (mean, std) from '31.5 ± 0.8' or just '31.5'."""
    s = str(s).strip()
    m = re.match(r"([\d.]+)\s*±\s*([\d.]+)", s)
    if m:
        return float(m.group(1)), float(m.group(2))
    try:
        return float(s), 0.0
    except ValueError:
        return np.nan, np.nan


def load_topo(topo: str) -> pd.DataFrame:
    path = DATA_DIR / topo / f"{topo}_master_ablation_summary_5seeds.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")
    df = pd.read_csv(path)
    # Parse mean ± std columns into separate float columns
    for col_base, _, _, _ in METRICS:
        if col_base in df.columns:
            parsed = df[col_base].apply(parse_mean_std)
            df[f"{col_base}_mean"] = [x[0] for x in parsed]
            df[f"{col_base}_std"]  = [x[1] for x in parsed]
    return df


# ── Drawing function ──────────────────────────────────────────────────────────
def draw_axis(ax, df, col_base: str, better: str, y_arrow: str, topo: str):
    """Draw grouped bars (variants × scenarios) on one subplot."""
    n_var  = len(VARIANT_KEYS)
    n_scen = len(SCENARIOS)
    group_w = 0.72           # total width of one variant group
    bar_w   = group_w / n_scen
    offsets = np.linspace(-(group_w - bar_w) / 2,
                          +(group_w - bar_w) / 2,
                          n_scen)
    x_pos = np.arange(n_var)

    for s_idx, (scen, hatch, alpha) in enumerate(zip(SCENARIOS, SCENARIO_HATCH, SCENARIO_ALPHA)):
        scen_df = df[df["Scenario"] == scen].copy()
        means, stds = [], []
        for vk in VARIANT_KEYS:
            row = scen_df[scen_df["variant"] == vk]
            if row.empty:
                means.append(np.nan); stds.append(0.0)
            else:
                means.append(row[f"{col_base}_mean"].values[0])
                stds.append(row[f"{col_base}_std"].values[0])

        # clip means and error bars to >=0 for rate/latency metrics
        means_arr = np.array([max(0.0, m) if not np.isnan(m) else 0.0 for m in means])
        stds_arr  = np.array(stds)
        # lower error cannot go below 0
        err_lo = np.minimum(stds_arr, means_arr)
        err_hi = stds_arr
        yerr_asym = [err_lo, err_hi]

        bars = ax.bar(
            x_pos + offsets[s_idx], means_arr,
            width=bar_w * 0.92,
            color=[VARIANT_COLORS[i] for i in range(n_var)],
            alpha=alpha,
            hatch=hatch,
            edgecolor="white",
            linewidth=0.4,
            yerr=yerr_asym,
            error_kw=dict(elinewidth=0.6, capsize=1.8, capthick=0.6,
                          ecolor="#444444", alpha=0.8),
            zorder=3,
        )

    # HARP full group: draw a subtle highlight box around it
    _x0 = x_pos[0] - group_w / 2 - 0.04
    _x1 = x_pos[0] + group_w / 2 + 0.04
    ax.axvspan(_x0, _x1, color="#C0392B", alpha=0.06, zorder=0, linewidth=0)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(VARIANT_LABELS, rotation=28, ha="right", fontsize=7.0)
    ax.set_axisbelow(True)

    # "better" direction annotation arrow
    arrow_kwargs = dict(xycoords="axes fraction", textcoords="axes fraction",
                        fontsize=6.5, color="#555555",
                        arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.7))
    if y_arrow == "up":
        ax.annotate("", xy=(1.01, 0.92), xytext=(1.01, 0.72),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#27AE60", lw=1.0))
    else:
        ax.annotate("", xy=(1.01, 0.08), xytext=(1.01, 0.28),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=1.0))


# ── Main ───────────────────────────────────────────────────────────────────────
def plot_pareto_design_space():
    topos = ["nsfnet", "geant2"]

    fig, axes = plt.subplots(
        2, 3,
        figsize=(7.16, 4.80),     # IEEE double-column width ≈ 7.16 in
        gridspec_kw={"hspace": 0.58, "wspace": 0.40},
    )

    for row_idx, topo in enumerate(topos):
        try:
            df = load_topo(topo)
        except FileNotFoundError as e:
            print(f"⚠️  {e}")
            continue

        for col_idx, (col_base, ylabel, better_txt, y_arrow) in enumerate(METRICS):
            ax = axes[row_idx, col_idx]
            draw_axis(ax, df, col_base, better_txt, y_arrow, topo)

            # Y-axis label: left column only
            if col_idx == 0:
                ax.set_ylabel(ylabel, fontsize=8.0)
            else:
                ax.set_ylabel(ylabel, fontsize=8.0)

            # Column titles: top row only
            if row_idx == 0:
                titles = [
                    "Safe Acceptance Rate\n(↑ higher is better)",
                    "SLA Violation Rate\n(↓ lower is better)",
                    "Average Latency\n(↓ lower is better)",
                ]
                ax.set_title(titles[col_idx], fontsize=9.0, fontweight="bold", pad=6)

            # Topology row label (left side of row)
            if col_idx == 0:
                ax.set_ylabel(
                    f"{TOPO_LABEL[topo]}\n\n{ylabel}",
                    fontsize=8.0, labelpad=4,
                )

            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
            ax.tick_params(axis="both", which="major", labelsize=7.0)
            # Enforce bottom=0 after all drawing is done
            ax.set_ylim(bottom=0)

    # ── Shared legend (scenarios) ─────────────────────────────────────────────
    scen_patches = []
    for scen, hatch, alpha in zip(SCENARIOS, SCENARIO_HATCH, SCENARIO_ALPHA):
        scen_patches.append(Patch(
            facecolor="#888888", hatch=hatch, alpha=alpha,
            edgecolor="white", linewidth=0.5, label=scen.capitalize(),
        ))

    # ── Shared legend (variants) ──────────────────────────────────────────────
    var_patches = []
    for label, color in zip(VARIANT_LABELS, VARIANT_COLORS):
        lw = 1.5 if "HARP★" in label else 0.5
        var_patches.append(Patch(facecolor=color, edgecolor="white",
                                 linewidth=lw, label=label))

    # Two-section legend below figure
    leg1 = fig.legend(
        handles=var_patches,
        loc="lower left",
        bbox_to_anchor=(0.01, -0.09),
        ncol=3,
        title="Ablation Variants",
        frameon=True, framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=7.2,
        title_fontsize=7.8,
    )
    leg2 = fig.legend(
        handles=scen_patches,
        loc="lower right",
        bbox_to_anchor=(0.99, -0.09),
        ncol=3,
        title="Traffic Scenarios  (bar fill patterns)",
        frameon=True, framealpha=0.95,
        edgecolor="#CCCCCC",
        fontsize=7.2,
        title_fontsize=7.8,
    )
    fig.add_artist(leg1)

    fig.suptitle(
        "Ablation Study: HARP Architectural Design Space\n"
        r"(NSFNET and GEANT2 $\cdot$ 5-seed mean $\pm$ std $\cdot$ proposed method = HARP Full)",
        fontsize=9.5, fontweight="bold", y=1.01,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    for fmt in ("pdf", "png", "svg"):
        out = FIGS_DIR / f"pareto_design_space.{fmt}"
        extra = {"dpi": 300} if fmt in ("png",) else {}
        plt.savefig(out, format=fmt, **extra)
        print(f"✓ {fmt.upper():3s} -> {out}")

    plt.close()
    print("Done.")


if __name__ == "__main__":
    plot_pareto_design_space()
