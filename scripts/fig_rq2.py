"""Main RQ2 figures from p3_weighted_cp.py --dump outputs (and the datasets for the assumption check).

  python scripts/fig_rq2.py --rows results/p3_rows_linear.csv.gz results/p3_rows_gbm.csv.gz \
      --data data/main_n4-8.csv.gz data/kladder_n4-8.csv.gz data/main_n10.csv.gz data/kladder_n10.csv.gz

Figures (PDF + PNG in results/figs/):
  fig_rq2_coverage   coverage on target circuits vs number of non-Clifford gates, per model
  fig_rq2_width      interval width vs number of non-Clifford gates, per model
  fig_rq2_assumption 90% quantile of the normalized score |r|/g(k) vs k (calibration and targets)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from g0_split_cp import GBM, LinearCell, split_groups  # noqa: E402
from p3_weighted_cp import kscale_fit  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"
# reference categorical palette, first slots in fixed order; oracle is a neutral reference
STYLE = {
    "oracle":      dict(color="#8c8c8c", ls="--", marker="s", label="Oracle (target calibration)"),
    "split_low":   dict(color="#f47230", ls="-", marker="v", label="Split CP, k ≤ 8 (standard CDR set)"),
    "split_topk":  dict(color="#00b894", ls="-", marker="D", label="Split CP, k = 32 only"),
    "kscale_auto": dict(color="#257de0", ls="-", marker="o", label="k-scaled CP (proposed)"),
}
MODEL_TITLE = {"linear": "vnCDR (pooled linear)", "gbm": "Gradient boosting"}
K_BINS = [0, 56, 100, 150, 230]
K_LABELS = ["28–56", "57–100", "101–150", "151–228"]


def style_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def binned(rows: pd.DataFrame, value: str) -> pd.DataFrame:
    r = rows.assign(kbin=pd.cut(rows.k, K_BINS, labels=K_LABELS))
    # mean over test rows within a repetition, then mean and 2.5/97.5% across repetitions
    per_rep = r.groupby(["method", "kbin", "rep"], observed=True)[value].mean().reset_index()
    return per_rep.groupby(["method", "kbin"], observed=True)[value].agg(
        mean="mean", lo=lambda x: np.quantile(x, 0.025), hi=lambda x: np.quantile(x, 0.975)).reset_index()


def line_panel(ax, stats, ylabel, ref=None):
    x = np.arange(len(K_LABELS))
    for m, st in STYLE.items():
        s = stats[stats.method == m].set_index("kbin").reindex(K_LABELS)
        ax.fill_between(x, s.lo, s.hi, color=st["color"], alpha=0.15, lw=0)
        ax.plot(x, s["mean"], color=st["color"], ls=st["ls"], lw=2, marker=st["marker"], ms=6,
                markeredgecolor="white", markeredgewidth=1, label=st["label"])
    if ref is not None:
        ax.axhline(ref, color=INK, lw=0.8, ls=":")
        ax.text(1.01, ref, f"1 − α", transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=8, color=INK, clip_on=False)
    ax.set_xticks(x)
    ax.set_xticklabels(K_LABELS)
    ax.set_xlabel("non-Clifford gates in target circuit", color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    style_axes(ax)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def assumption_stats(d: pd.DataFrame, model_cls, reps: int, alpha: float):
    """90% quantile of |r|/g(k) for calibration rows (by k) and target rows (by k bin), per split."""
    recs = []
    for rep in range(reps):
        rng = np.random.default_rng(2000 + rep)
        sp = split_groups(d, rng)
        nc, tg = d[d.kind == "nearclifford"], d[d.kind == "target"]
        m = model_cls().fit(nc[nc.group_id.isin(sp["train"]) & (nc.n_keep <= 8)])
        cal = nc[nc.group_id.isin(sp["cal"])]
        r = cal.y_ideal.to_numpy() - m.predict(cal)
        g, _ = kscale_fit(cal, r)
        sn = np.abs(r) / g(cal)
        klev = np.where(cal.n_keep.to_numpy() >= 28, 32, cal.n_keep.to_numpy())  # 28 = k=32 capped by target
        for k, v in pd.Series(sn).groupby(klev).quantile(1 - alpha).items():
            recs.append(dict(rep=rep, set="calibration", k=float(k), q=v, raw=np.quantile(np.abs(r)[klev == k], 1 - alpha)))
        te = tg[tg.group_id.isin(sp["test"])]
        rt = te.y_ideal.to_numpy() - m.predict(te)
        st = np.abs(rt) / g(te)
        kb = pd.cut(te.n_keep, K_BINS).cat.codes.to_numpy()
        centers = [42, 78, 125, 190]
        for b in range(len(centers)):
            sel = kb == b
            if sel.any():
                recs.append(dict(rep=rep, set="target", k=float(centers[b]), q=np.quantile(st[sel], 1 - alpha),
                                 raw=np.quantile(np.abs(rt[sel]), 1 - alpha)))
    return pd.DataFrame(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", nargs="+", default=[str(ROOT / "results" / f"p3_rows_{m}.csv.gz") for m in ("linear", "gbm")])
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--reps-assumption", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "results" / "figs"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 10, "axes.titleweight": "bold"})

    rows = {}
    for p in args.rows:
        model = "gbm" if "gbm" in Path(p).name else "linear"
        r = pd.read_csv(p)
        r = r[r.method.isin(STYLE)]
        r["covered"] = (r.abs_err <= r.half_width).astype(float)
        r["width"] = 2 * r.half_width
        rows[model] = r

    # Figure 1: coverage
    fig, axes = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (model, r) in zip(axes, rows.items()):
        line_panel(ax, binned(r, "covered"), "coverage on target circuits", ref=1 - args.alpha)
        ax.set_title(MODEL_TITLE[model], color=INK, loc="left")
        ax.set_ylim(0.3, 1.02)
    axes[0].legend(frameon=False, fontsize=8, loc="lower left")
    save(fig, out, "fig_rq2_coverage")

    # Figure 2: width
    fig, axes = plt.subplots(1, len(rows), figsize=(5.2 * len(rows), 3.6))
    axes = np.atleast_1d(axes)
    for ax, (model, r) in zip(axes, rows.items()):
        rr = r[np.isfinite(r.width)]
        line_panel(ax, binned(rr, "width"), "mean interval width")
        ax.set_title(MODEL_TITLE[model], color=INK, loc="left")
        ax.set_ylim(0, None)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, out, "fig_rq2_width")

    # Figure 3: assumption check
    d = pd.concat([pd.read_csv(p) for p in args.data], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6), sharey=True)
    summary = []
    for ax, (model, cls) in zip(axes, [("linear", LinearCell), ("gbm", GBM)]):
        a = assumption_stats(d, cls, args.reps_assumption, args.alpha)
        s = a.groupby(["set", "k"]).q.agg(["mean", "min", "max"]).reset_index()
        for setname, col, mk in [("calibration", "#257de0", "o"), ("target", "#f47230", "s")]:
            ss = s[s.set == setname]
            ax.errorbar(np.log1p(ss.k), ss["mean"], yerr=[ss["mean"] - ss["min"], ss["max"] - ss["mean"]],
                        color=col, marker=mk, ms=6, lw=2, capsize=3, markeredgecolor="white",
                        label="near-Clifford calibration circuits" if setname == "calibration" else "target circuits")
        cal_mean = s[s.set == "calibration"]["mean"].mean()
        ax.axhline(cal_mean, color=INK2, lw=0.8, ls=":")
        ticks = [0, 2, 8, 32, 100, 228]
        ax.set_xticks(np.log1p(ticks)); ax.set_xticklabels(ticks)
        ax.set_xlabel("non-Clifford gates k (log scale)", color=INK2, fontsize=9)
        ax.set_ylabel(f"{int(100 * (1 - args.alpha))}% quantile of |r| / g(k)", color=INK2, fontsize=9)
        ax.set_title(MODEL_TITLE[model], color=INK, loc="left")
        style_axes(ax)
        summary.append(a.assign(model=model))
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, out, "fig_rq2_assumption")
    pd.concat(summary).groupby(["model", "set", "k"])[["q", "raw"]].mean().round(3).to_csv(out / "rq2_assumption.csv")

    # numbers behind the figures
    tab = []
    for model, r in rows.items():
        c = binned(r, "covered").assign(model=model, metric="coverage")
        w = binned(r[np.isfinite(r.width)], "width").assign(model=model, metric="width")
        tab += [c, w]
    pd.concat(tab).round(3).to_csv(out / "rq2_figure_data.csv", index=False)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
