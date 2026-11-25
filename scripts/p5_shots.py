"""Phase 5 (RQ4): how to spend a shot budget on conformal calibration of ML-QEM.

  python scripts/p5_shots.py --data data/main_n4-8.csv.gz data/kladder_n4-8.csv.gz \
      data/main_n10.csv.gz data/kladder_n10.csv.gz --reps 30

Shots are re-drawn offline from the stored exact noisy values (exact_s1/s3/s5): no new simulation.
A circuit run with M shots costs 3M shots (noise scales 1, 3, 5).

Per repetition: target groups split 40/30/30 (train / cal / test). vnCDR is trained on near-Clifford
rows (k <= 8) of train groups (stored 2000-shot values). The k-scale model g(k) is fitted ONCE on
half A of the calibration groups at 2000 shots (fixed, outside the budget). The budget is spent on the
conformalization set: n_cal near-Clifford circuits drawn from half B (k-ladder), each re-sampled
at M_cal shots. Test targets are re-sampled at M_test shots.

Experiments
  E1  fixed budget B = 3 * 2000 * 200 shots, M_cal in {125 ... 16000}, n_cal = B / (3 M_cal), M_test = 2000
  E2  M_cal = M_test = 2000, n_cal in {12, 25, 50, 100, 200, 400, 800}
  E3  M_cal = M_test = M in {125 ... 16000}, n_cal = 200, g(k) fitted at 2000 shots
  E4  as E3 but g(k) re-fitted at the same M (the k-scale model must match the shot count)
Outputs: results/p5_*.csv, results/p5_shots.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from g0_split_cp import LinearCell, conformal_q, split_groups
from p3_weighted_cp import kscale_fit

ROOT = Path(__file__).resolve().parents[1]
M_GRID = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]


def resample(df: pd.DataFrame, M: int, rng) -> pd.DataFrame:
    out = df.copy()
    for s in (1, 3, 5):
        p = np.clip((1 + df[f"exact_s{s}"].to_numpy()) / 2, 0, 1)
        out[f"noisy_s{s}"] = 2 * rng.binomial(M, p) / M - 1
    return out


def evaluate(model, g, cal_pool, test, n_cal, M_cal, M_test, alpha, rng):
    circs = cal_pool.circuit_id.unique()
    pick = rng.choice(circs, size=min(n_cal, len(circs)), replace=False)
    cal = resample(cal_pool[cal_pool.circuit_id.isin(pick)], M_cal, rng)
    q = conformal_q(np.abs(cal.y_ideal.to_numpy() - model.predict(cal)) / g(cal), alpha)
    te = resample(test, M_test, rng)
    gk = g(te)
    err = np.abs(te.y_ideal.to_numpy() - model.predict(te))
    return dict(coverage=float(np.mean(err <= q * gk)), width=float(np.median(2 * q * gk)),
                q=q, mae=float(err.mean()), n_cal_used=len(pick))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    d = pd.concat([pd.read_csv(p) for p in args.data], ignore_index=True)

    rows = []
    for rep in range(args.reps):
        rng = np.random.default_rng(5000 + rep)
        sp = split_groups(d, rng)
        nc, tg = d[d.kind == "nearclifford"], d[d.kind == "target"]
        model = LinearCell().fit(nc[nc.group_id.isin(sp["train"]) & (nc.n_keep <= 8)])
        cal_ids = sorted(sp["cal"]); rng.shuffle(cal_ids)
        half_a = set(cal_ids[: len(cal_ids) // 2])
        ca = nc[nc.group_id.isin(half_a)]
        g, _ = kscale_fit(ca, ca.y_ideal.to_numpy() - model.predict(ca))
        pool = nc[nc.group_id.isin(set(cal_ids) - half_a)]
        test = tg[tg.group_id.isin(sp["test"])]

        B = 3 * 2000 * 200
        for M in M_GRID:
            r = evaluate(model, g, pool, test, B // (3 * M), M, 2000, args.alpha, rng)
            rows.append(dict(rep=rep, exp="E1", M_cal=M, M_test=2000, n_cal=B // (3 * M), **r))
        for n_cal in [12, 25, 50, 100, 200, 400, 800]:
            r = evaluate(model, g, pool, test, n_cal, 2000, 2000, args.alpha, rng)
            rows.append(dict(rep=rep, exp="E2", M_cal=2000, M_test=2000, n_cal=n_cal, **r))
        for M in M_GRID:
            r = evaluate(model, g, pool, test, 200, M, M, args.alpha, rng)
            rows.append(dict(rep=rep, exp="E3", M_cal=M, M_test=M, n_cal=200, **r))
            # E4: as E3, but g(k) re-fitted on half A re-sampled at the same M (k-scale model matched to M)
            ca_m = resample(ca, M, rng)
            g_m, _ = kscale_fit(ca_m, ca_m.y_ideal.to_numpy() - model.predict(ca_m))
            r = evaluate(model, g_m, pool, test, 200, M, M, args.alpha, rng)
            rows.append(dict(rep=rep, exp="E4", M_cal=M, M_test=M, n_cal=200, **r))
        if (rep + 1) % 5 == 0:
            print(f"  {rep + 1}/{args.reps}", flush=True)

    res = pd.DataFrame(rows)
    agg = res.groupby(["exp", "M_cal", "M_test", "n_cal"]).agg(
        coverage=("coverage", "mean"), cov_sd=("coverage", "std"), width=("width", "mean"),
        width_sd=("width", "std"), mae=("mae", "mean"), n_used=("n_cal_used", "mean")).reset_index()
    pd.set_option("display.width", 200)
    for e, a in agg.groupby("exp"):
        print(f"\n--- {e}")
        print(a.drop(columns="exp").round(4).to_string(index=False))
    agg.to_csv(out / "p5_summary.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ink, ink2, grid = "#191919", "#666666", "#e6e6e6"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 3.7))
        # (a) E1: coverage vs calibration shots at a fixed budget; test circuits use 2000 shots
        a = agg[agg.exp == "E1"].sort_values("M_cal")
        sd = res[res.exp == "E1"].groupby("M_cal").coverage
        ax1.fill_between(a.M_cal, sd.quantile(0.025).values, sd.quantile(0.975).values, color="#257de0", alpha=0.15, lw=0)
        ax1.plot(a.M_cal, a.coverage, color="#257de0", marker="o", lw=2, markeredgecolor="white")
        ax1.axhline(1 - args.alpha, color=ink, lw=0.8, ls=":")
        ax1.axvline(2000, color="#bbbbbb", lw=0.8)
        ax1.text(2000 * 1.08, 0.62, "shots on\ntarget circuits", fontsize=8, color=ink2, va="bottom")
        for m, nc_ in zip(a.M_cal, a.n_cal):
            ax1.annotate(f"n={nc_}", (m, 1.0), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7, color=ink2)
        ax1.set_xscale("log"); ax1.set_ylim(0.6, 1.04)
        ax1.set_xlabel("shots per calibration circuit (fixed total budget)", fontsize=9, color=ink2)
        ax1.set_ylabel("coverage on target circuits", fontsize=9, color=ink2)
        ax1.set_title("(a) calibration shots must match target shots", loc="left", fontsize=10, fontweight="bold")
        # (b) E3/E4: width vs shots (calibration = test), with sqrt(floor^2 + b^2/M) fit on E4
        for e, col, mk, lab in [("E4", "#257de0", "o", "g(k) refitted at M"), ("E3", "#f47230", "s", "g(k) from 2000 shots")]:
            b_ = agg[agg.exp == e].sort_values("M_cal")
            ax2.plot(b_.M_cal, b_.width, color=col, marker=mk, lw=2, markeredgecolor="white", label=lab)
        b4 = agg[agg.exp == "E4"].sort_values("M_cal")
        Mv = b4.M_cal.values.astype(float)
        c, *_ = np.linalg.lstsq(np.c_[np.ones_like(Mv), 1 / Mv], b4.width.values ** 2, rcond=None)
        mm = np.logspace(np.log10(Mv.min()), np.log10(Mv.max()), 100)
        ax2.plot(mm, np.sqrt(c[0] + c[1] / mm), color=ink2, lw=1, ls="--", label="fit: sqrt(w0² + b²/M)")
        ax2.axhline(np.sqrt(c[0]), color=ink2, lw=0.8, ls=":")
        ax2.text(Mv.min(), np.sqrt(c[0]) * 1.05, f"floor w0 = {np.sqrt(c[0]):.2f} (model bias)", fontsize=8, color=ink2)
        ax2.set_xscale("log")
        ax2.set_xlabel("shots per circuit (calibration = target)", fontsize=9, color=ink2)
        ax2.set_ylabel("median interval width", fontsize=9, color=ink2)
        ax2.set_title("(b) more shots: diminishing returns", loc="left", fontsize=10, fontweight="bold")
        ax2.legend(frameon=False, fontsize=8)
        for ax in (ax1, ax2):
            ax.yaxis.grid(True, color=grid); ax.set_axisbelow(True)
            for s_ in ("top", "right"):
                ax.spines[s_].set_visible(False)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(out / f"p5_shots.{ext}", dpi=200)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
