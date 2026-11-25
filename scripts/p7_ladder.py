"""Remedies for A2 violation (score quantile rising with k), detected on the k-ladder.

  python scripts/p7_ladder.py --data data/coh_n4-8.csv.gz data/coh_kladder_n4-8.csv.gz --model linear

Same splits as p3/p6 (seed 2000+rep). g(k) is fitted on half A of the calibration groups; scores are taken
on half B. s = |r| / g(k).

  kscale      q_all = conformal quantile of every half-B score                      (RQ2 method)
  top         max(q_all, q_top), q_top = conformal quantile of half-B scores with k >= K_TOP
  hi          g refitted on half A rows with k >= 8 only; q from half-B scores with k >= K_TOP
  extrap      trend in the score quantile: fit log q_bin = u + c log(1+k_bin) over half-B k-bins,
              c clipped to >= 0; q(k) = max(q_all, q_top * ((1+k)/(1+k_top))^c), k_top = median k of the top bin
  hi_extrap   g as in hi; trend c fitted on the k >= 8 bins only; q(k) = q_hi * max(1, ((1+k)/(1+k_top))^c)
  diag        A2 diagnostic ratio  q_top / q_low  (q_low from scores with k <= 8), reported per rep
Outputs coverage, median width and coverage by target-k bin.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from g0_split_cp import GBM, LinearCell, conformal_q, split_groups
from p3_weighted_cp import kscale_fit

K_TOP = 16
QBINS = [-1, 0, 2, 4, 8, 20, 40]
K_BINS = [0, 56, 100, 150, 230]
K_LABELS = ["28-56", "57-100", "101-150", "151-228"]


def quantile_trend(s, k, alpha):
    b = pd.cut(k, QBINS)
    tab = pd.DataFrame({"s": s, "k": k, "b": b}).groupby("b", observed=True).agg(
        q=("s", lambda x: conformal_q(x.to_numpy(), alpha)), kk=("k", "median"), n=("s", "size"))
    tab = tab[tab.n >= 30]
    if len(tab) < 2:
        return 0.0
    c, _ = np.polyfit(np.log1p(tab.kk.to_numpy()), np.log(tab.q.to_numpy()), 1)
    return max(c, 0.0)


def one_rep(d, rep, alpha, model_cls):
    rng = np.random.default_rng(2000 + rep)
    sp = split_groups(d, rng)
    nc, tg = d[d.kind == "nearclifford"], d[d.kind == "target"]
    train = nc[nc.group_id.isin(sp["train"]) & (nc.n_keep <= 8)]
    cal_nc = nc[nc.group_id.isin(sp["cal"])]
    test = tg[tg.group_id.isin(sp["test"])].reset_index(drop=True)
    m = model_cls().fit(train)
    y, pred = test.y_ideal.to_numpy(), m.predict(test)
    r = cal_nc.y_ideal.to_numpy() - m.predict(cal_nc)

    half = set()
    for _, gg in tg[tg.group_id.isin(sp["cal"])].drop_duplicates("group_id").groupby(["n", "steps"]):
        ids = rng.permutation(gg.group_id.values)
        half |= set(ids[: len(ids) // 2])
    in_a = cal_nc.group_id.isin(half).to_numpy()
    A, B = cal_nc[in_a], cal_nc[~in_a]
    kB = B.n_keep.to_numpy()

    g, _ = kscale_fit(A, r[in_a])
    sB = np.abs(r[~in_a]) / g(B)
    gt = g(test)
    q_all = conformal_q(sB, alpha)
    q_top = conformal_q(sB[kB >= K_TOP], alpha)
    q_low = conformal_q(sB[kB <= 8], alpha)
    c = quantile_trend(sB, kB, alpha)
    k_top = np.median(kB[kB >= K_TOP])
    kt = test.n_keep.to_numpy()

    hi_a = (A.n_keep >= 8).to_numpy()
    g_hi, _ = kscale_fit(A[hi_a], r[in_a][hi_a])
    s_hi = np.abs(r[~in_a]) / g_hi(B)

    half_w = {
        "kscale": q_all * gt,
        "top": max(q_all, q_top) * gt,
        "hi": conformal_q(s_hi[kB >= K_TOP], alpha) * g_hi(test),
        "extrap": np.maximum(q_all, q_top * ((1 + kt) / (1 + k_top)) ** c) * gt,
    }
    # hi_extrap: g from k >= 8, quantile trend over the k >= 8 bins only, extrapolated to the target k
    q_hi = conformal_q(s_hi[kB >= K_TOP], alpha)
    sel = kB >= 8
    c_hi = quantile_trend(s_hi[sel], kB[sel], alpha)
    half_w["hi_extrap"] = np.maximum(q_hi, q_hi * ((1 + kt) / (1 + k_top)) ** c_hi) * g_hi(test)
    rows = []
    for name, h in half_w.items():
        rows.append(pd.DataFrame({"rep": rep, "method": name, "covered": np.abs(y - pred) <= h,
                                  "width": 2 * h, "k": kt}))
    return pd.concat(rows), {"rep": rep, "diag": q_top / q_low, "c": c, "c_hi": c_hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--model", choices=["linear", "gbm"], default="linear")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    d = pd.concat([pd.read_csv(p) for p in args.data], ignore_index=True)
    cls = LinearCell if args.model == "linear" else GBM
    res, diags = [], []
    for rep in range(args.reps):
        a, b = one_rep(d, rep, args.alpha, cls)
        res.append(a); diags.append(b)
    res, diags = pd.concat(res), pd.DataFrame(diags)
    per = res.groupby(["method", "rep"]).agg(cov=("covered", "mean"), w=("width", "median")).reset_index()
    s = per.groupby("method").agg(coverage=("cov", "mean"), cov_sd=("cov", "std"), width=("w", "mean"))
    res["kbin"] = pd.cut(res.k, K_BINS, labels=K_LABELS)
    s = s.join(res.groupby(["method", "kbin"], observed=True).covered.mean().unstack())
    print(f"model={args.model} alpha={args.alpha} reps={args.reps}")
    print(s.round(3).to_string())
    print(f"A2 diagnostic q_top/q_low: mean {diags.diag.mean():.3f} (min {diags.diag.min():.3f}, max {diags.diag.max():.3f}); "
          f"trend c: mean {diags.c.mean():.3f}; c_hi: mean {diags.c_hi.mean():.3f}")
    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        s.to_csv(Path(args.out) / "p7_summary.csv"); diags.to_csv(Path(args.out) / "p7_diag.csv", index=False)


if __name__ == "__main__":
    main()
