"""Figure 6 of the manuscript: held-out validation (XXZ family, coherent noise) and the A2 ladder diagnostic.

Numbers are the 30-repetition summaries printed by p6_baselines.py, p3_weighted_cp.py (kscale_auto) and the
A2 diagnostic (90% quantile of |r|/g(k) per k bin, 10 repetitions), alpha = 0.1.
  python scripts/fig_heldout.py --out results/fig_heldout.pdf
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

DATASETS = ["TFIM\n(n = 4–10)", "XXZ\n(held-out family)", "TFIM + coherent\n(0.03, 0.05 rad)", "TFIM + coherent\n(0.05, 0.08 rad)"]
COVERAGE = {  # method -> coverage per dataset
    "Split CP, k ≤ 8": [0.853, 0.918, 0.646, 0.601],
    "Gaussian ±zσ per cell": [0.895, 0.928, 0.669, 0.585],
    "CDR ±3σ": [0.994, 0.995, 0.937, 0.889],
    "k-scaled CP, vnCDR": [0.889, 0.918, 0.815, 0.775],
    "k-scaled CP, GBM": [0.938, 0.927, 0.920, 0.889],
}
COLORS = ["#F06A28", "#8C8C8C", "#C9B037", "#1F78E0", "#12A88A"]
HATCH = ["", "", "//", "", ""]

KBINS = ["0", "2", "4", "8", "16–20", "21–32", "targets"]
A2 = {
    "TFIM": [1.98, 2.02, 2.00, 2.01, 2.05, 2.14, 2.10],
    "XXZ": [2.11, 2.18, 2.20, 2.13, 2.12, 2.06, 2.01],
    "coherent (0.03, 0.05)": [2.23, 2.08, 2.03, 2.03, 2.15, 2.41, 2.72],
    "coherent (0.05, 0.08)": [2.35, 2.14, 2.11, 2.11, 2.14, 2.48, 3.06],
}
A2_COL = ["#1F78E0", "#12A88A", "#F06A28", "#B8326E"]
A2_MK = ["o", "s", "^", "D"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_heldout.pdf")
    a = ap.parse_args()
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.45, 1]})

    x = np.arange(len(DATASETS))
    w = 0.16
    for i, (name, vals) in enumerate(COVERAGE.items()):
        ax1.bar(x + (i - 2) * w, vals, w - 0.02, label=name, color=COLORS[i], hatch=HATCH[i],
                edgecolor="white", linewidth=0.6)
    ax1.axhline(0.9, color="k", lw=0.8, ls=":")
    ax1.text(3.55, 0.902, "1 − α", va="bottom", ha="right", fontsize=8)
    ax1.set_xticks(x, DATASETS, fontsize=8)
    ax1.set_ylim(0.5, 1.02)
    ax1.set_ylabel("coverage on target circuits")
    ax1.set_title("(a) coverage, α = 0.1", loc="left", fontweight="bold")
    ax1.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax1.set_axisbelow(True)
    ax1.legend(fontsize=7.5, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))

    xs = np.arange(len(KBINS))
    for (name, v), c, m in zip(A2.items(), A2_COL, A2_MK):
        ax2.plot(xs[:-1], v[:-1], marker=m, color=c, lw=1.6, ms=5, mec="white", label=name)
        ax2.plot(xs[-2:], v[-2:], color=c, lw=1.2, ls="--")
        ax2.plot(xs[-1], v[-1], marker=m, color=c, ms=7, mec="white")
    ax2.axvspan(3.5, 5.5, color="#f2f2f2", zorder=0)
    ax2.text(4.5, 3.12, "k-ladder", ha="center", fontsize=8, color="#555")
    ax2.set_xticks(xs, KBINS, fontsize=8)
    ax2.set_xlabel("non-Clifford gates k of the calibration circuits")
    ax2.set_ylabel("90% quantile of |r| / g(k)")
    ax2.set_ylim(1.8, 3.2)
    ax2.set_title("(b) A2 diagnostic on the k-ladder (vnCDR)", loc="left", fontweight="bold")
    ax2.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax2.legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight")
    fig.savefig(a.out.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
