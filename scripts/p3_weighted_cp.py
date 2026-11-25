"""Phase 3 (RQ2): conformal prediction for target circuits when calibration uses near-Clifford circuits.

  python scripts/p3_weighted_cp.py --data data/main_n4-8.csv.gz data/kladder_n4-8.csv.gz --reps 50

Per repetition, target circuits (group_id) are split 40/30/30 into train / cal / test groups.
The base ML-QEM model is trained on near-Clifford rows with k <= --train-kmax of train groups
(standard CDR training). Methods, all evaluated on TARGET rows of test groups:

  oracle       split CP calibrated on target rows of cal groups (reference; impossible on hardware)
  split_low    split CP on near-Clifford rows with k <= train-kmax            (= G0 baseline)
  split_all    split CP on all near-Clifford rows (incl. k-ladder)
  split_topk   split CP on the near-Clifford rows with the largest k available
  mondrian     split CP per (n, steps, obs_weight) cell, all near-Clifford rows
  weighted     weighted CP for covariate shift (Tibshirani et al. 2019); density ratio from a
               logistic classifier cal-vs-test on [noisy_s1, noisy_s3, noisy_s5, obs_weight, n, steps]
  weighted_k   same, classifier also sees log(1+k)  (expect degenerate weights without a k-ladder)
  kscale_auto  picks kscale or kscale_cap per repetition by a k-holdout check on calibration data
  kscale_cap   kscale with g(k) frozen at the largest calibration k (no extrapolation in k)
  kscale       k-scaled CP: |residual| scale g(k) fitted per (n, steps, obs_weight) cell as
               log E|r| = a + b log(1+k) on one half of the cal groups; normalized scores |r|/g(k)
               conformalized on the other half; target interval = q * g(k_target).
               Valid if |r|/g(k) is exchangeable across k (to be checked: this is the RQ2 hypothesis).

Outputs: results/p3_summary.csv, results/p3_by_config.csv, results/p3_coverage.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from g0_split_cp import GBM, LinearCell, conformal_q, split_groups  # same folder

ROOT = Path(__file__).resolve().parents[1]
CELL = ["n", "steps", "obs_weight"]
DUMP_ROWS = None  # set to a list by --dump to keep per-test-row results
SELECTED: list[str] = []  # variant picked by kscale_auto in each repetition
CLF_FEATS = ["noisy_s1", "noisy_s3", "noisy_s5", "obs_weight", "n", "steps"]


def weighted_q(cal_scores, w_cal, w_test, alpha):
    """Per-test-point weighted conformal quantile (point mass at +inf with weight w_test)."""
    order = np.argsort(cal_scores)
    s, cw = cal_scores[order], np.cumsum(w_cal[order])
    W = cw[-1]
    need = (1 - alpha) * (W + w_test)  # first j with cumsum >= need
    idx = np.searchsorted(cw, need, side="left")
    q = np.full(len(w_test), np.inf)
    ok = idx < len(s)
    q[ok] = s[idx[ok]]
    return q


def density_ratio(cal, test, feats):
    X = pd.concat([cal[feats], test[feats]]).to_numpy(dtype=float)
    y = np.r_[np.zeros(len(cal)), np.ones(len(test))]
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
    def w(df):
        p = clf.predict_proba(sc.transform(df[feats].to_numpy(dtype=float)))[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return p / (1 - p)
    return w(cal), w(test)


def kscale_fit(cal_fit, resid, k_cap=None):
    """Per cell: log mean|r| = a + b log(1+k), fitted on k-binned means. Returns g(df) callable."""
    d = cal_fit.assign(ar=np.abs(resid), lk=np.log1p(cal_fit.n_keep))
    coef = {}
    for key, g in d.groupby(CELL):
        m = g.groupby("lk").ar.mean()
        m = m[m > 0]
        if len(m) >= 2:
            b, a = np.polyfit(m.index.values, np.log(m.values), 1)
        else:
            b, a = 0.0, np.log(max(g.ar.mean(), 1e-6))
        coef[key] = (a, b)
    # pooled fallback for cells absent from the fitting data
    m = d.groupby("lk").ar.mean()
    m = m[m > 0]
    b0, a0 = np.polyfit(m.index.values, np.log(m.values), 1) if len(m) >= 2 else (0.0, np.log(max(d.ar.mean(), 1e-6)))
    def g(df):
        keys = list(zip(*[df[c] for c in CELL]))
        k = df.n_keep.to_numpy()
        lk = np.log1p(np.minimum(k, k_cap) if k_cap is not None else k)
        return np.array([np.exp(coef.get(k, (a0, b0))[0] + coef.get(k, (a0, b0))[1] * x) for k, x in zip(keys, lk)])
    return g, coef


def one_rep(d, rep, alpha, model_cls, train_kmax):
    rng = np.random.default_rng(2000 + rep)
    sp = split_groups(d, rng)
    nc, tg = d[d.kind == "nearclifford"], d[d.kind == "target"]
    train = nc[nc.group_id.isin(sp["train"]) & (nc.n_keep <= train_kmax)]
    cal_nc = nc[nc.group_id.isin(sp["cal"])]
    cal_tg = tg[tg.group_id.isin(sp["cal"])]
    test = tg[tg.group_id.isin(sp["test"])]
    m = model_cls().fit(train)
    y_t = test.y_ideal.to_numpy()
    err_t = np.abs(y_t - m.predict(test))
    r_nc = cal_nc.y_ideal.to_numpy() - m.predict(cal_nc)
    s_nc = np.abs(r_nc)
    kmax = cal_nc.n_keep.max()

    q = {}
    q["oracle"] = np.full(len(test), conformal_q(np.abs(cal_tg.y_ideal - m.predict(cal_tg)), alpha))
    low = (cal_nc.n_keep <= train_kmax).to_numpy()
    q["split_low"] = np.full(len(test), conformal_q(s_nc[low], alpha))
    q["split_all"] = np.full(len(test), conformal_q(s_nc, alpha))
    top = (cal_nc.n_keep >= kmax).to_numpy()
    q["split_topk"] = np.full(len(test), conformal_q(s_nc[top], alpha))

    test_keys = list(zip(*[test[c] for c in CELL]))
    cal_cells = cal_nc[CELL].to_numpy()
    qcell = {k: conformal_q(s_nc[(cal_cells == np.array(k)).all(axis=1)], alpha) for k in set(test_keys)}
    q["mondrian"] = np.array([qcell[k] for k in test_keys])

    w_c, w_t = density_ratio(cal_nc, test, CLF_FEATS)
    q["weighted"] = weighted_q(s_nc, w_c, w_t, alpha)
    cal_k = cal_nc.assign(lk=np.log1p(cal_nc.n_keep))
    test_k = test.assign(lk=np.log1p(test.n_keep))
    w_c, w_t = density_ratio(cal_k, test_k, CLF_FEATS + ["lk"])
    q["weighted_k"] = weighted_q(s_nc, w_c, w_t, alpha)

    # k-scaled CP: split the cal groups in two halves (fit g / conformalize)
    half = set()  # stratified by (n, steps) so every cell is represented in both halves
    cal_t = tg[tg.group_id.isin(sp["cal"])].drop_duplicates("group_id")
    for _, gg in cal_t.groupby(["n", "steps"]):
        ids = rng.permutation(gg.group_id.values)
        half |= set(ids[: len(ids) // 2])
    in_a = cal_nc.group_id.isin(half).to_numpy()
    g, _ = kscale_fit(cal_nc[in_a], r_nc[in_a])
    qn = conformal_q(s_nc[~in_a] / g(cal_nc[~in_a]), alpha)
    q["kscale"] = qn * g(test)
    # capped variant: no extrapolation in k beyond the largest calibration k (g frozen at k_max)
    g_c, _ = kscale_fit(cal_nc[in_a], r_nc[in_a], k_cap=kmax)
    qn_c = conformal_q(s_nc[~in_a] / g_c(cal_nc[~in_a]), alpha)
    q["kscale_cap"] = qn_c * g_c(test)

    # kscale_auto: choose between extrapolating and capped g(k) with a k-holdout check on calibration
    # data only. Refit both on k < kmax, conformalize on half B (k < kmax), measure coverage on the
    # held-out k = kmax rows of half B; pick the variant whose coverage is closest to 1 - alpha
    # (ties -> narrower). No target labels are used.
    lo = (cal_nc.n_keep < kmax).to_numpy()
    hold = (~in_a) & ~lo
    if hold.sum() > 0 and (in_a & lo).sum() > 0:
        k_lo = cal_nc.n_keep[lo].max()
        res_sel = {}
        for name, cap in (("kscale", None), ("kscale_cap", k_lo)):
            gh, _ = kscale_fit(cal_nc[in_a & lo], r_nc[in_a & lo], k_cap=cap)
            qh = conformal_q(s_nc[(~in_a) & lo] / gh(cal_nc[(~in_a) & lo]), alpha)
            ih = qh * gh(cal_nc[hold])
            res_sel[name] = (abs((s_nc[hold] <= ih).mean() - (1 - alpha)), float(np.median(ih)))
        choice = min(res_sel, key=lambda k: res_sel[k])
    else:
        choice = "kscale_cap"
    q["kscale_auto"] = q[choice]
    SELECTED.append(choice)

    rows, cfg = [], []
    if DUMP_ROWS is not None:
        for name in ("oracle", "split_low", "split_topk", "mondrian", "weighted", "kscale", "kscale_cap", "kscale_auto"):
            DUMP_ROWS.append(pd.DataFrame({"rep": rep, "method": name, "n": test.n.values, "steps": test.steps.values,
                                           "k": test.n_keep.values, "obs_weight": test.obs_weight.values,
                                           "abs_err": err_t, "half_width": q[name]}))
    for name, qq in q.items():
        cov = err_t <= qq
        width = 2 * qq
        rows.append(dict(rep=rep, method=name, coverage=cov.mean(),
                         width_median=float(np.median(width)),
                         frac_inf=float(np.mean(~np.isfinite(qq))),
                         width_mean_finite=float(np.mean(width[np.isfinite(width)])) if np.isfinite(width).any() else np.inf))
        s = pd.DataFrame({"n": test.n.values, "steps": test.steps.values, "cov": cov, "w": np.minimum(width, 2.0)})
        for (n_, st), gg in s.groupby(["n", "steps"]):
            cfg.append(dict(rep=rep, method=name, n=n_, steps=st, coverage=gg["cov"].mean(), width=gg.w.median()))
    return rows, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", default=[str(ROOT / "data" / "main_n4-8.csv.gz")])
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--model", choices=["linear", "gbm"], default="linear")
    ap.add_argument("--train-kmax", type=int, default=8)
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--dump", action="store_true", help="save per-test-row results (for fig_rq2.py)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    global DUMP_ROWS
    if args.dump:
        DUMP_ROWS = []
    d = pd.concat([pd.read_csv(p) for p in args.data], ignore_index=True)
    ks = sorted(d[d.kind == "nearclifford"].n_keep.unique())
    print(f"{len(d)} rows; near-Clifford k values: {ks}; target non-Clifford counts: "
          f"{int(d[d.kind == 'target'].n_keep.min())}-{int(d[d.kind == 'target'].n_keep.max())}")
    model_cls = {"linear": LinearCell, "gbm": GBM}[args.model]

    rows, cfg = [], []
    for rep in range(args.reps):
        r, c = one_rep(d, rep, args.alpha, model_cls, args.train_kmax)
        rows += r; cfg += c
        if (rep + 1) % 10 == 0:
            print(f"  {rep + 1}/{args.reps}", flush=True)
    res, cfg = pd.DataFrame(rows), pd.DataFrame(cfg)
    summ = res.groupby("method").agg(coverage=("coverage", "mean"), cov_sd=("coverage", "std"),
                                     width_median=("width_median", "mean"), frac_inf=("frac_inf", "mean"))
    order = ["oracle", "split_low", "split_all", "split_topk", "mondrian", "weighted", "weighted_k", "kscale", "kscale_cap", "kscale_auto"]
    summ = summ.reindex(order)
    by_cov = cfg.groupby(["method", "n", "steps"]).coverage.mean().unstack([1, 2]).reindex(order)
    by_w = cfg.groupby(["method", "n", "steps"]).width.mean().unstack([1, 2]).reindex(order)
    pd.set_option("display.width", 200)
    print(f"\nmodel={args.model}, alpha={args.alpha} (target {1 - args.alpha:.2f}), {args.reps} reps\n")
    print(summ.round(3).to_string())
    print("kscale_auto picked:", pd.Series(SELECTED).value_counts().to_dict())
    print("\nCoverage on targets by (n, steps):")
    print(by_cov.round(2).to_string())
    print("\nMedian interval width by (n, steps) (capped at 2):")
    print(by_w.round(2).to_string())
    summ.to_csv(out / f"p3_summary_{args.model}.csv")
    if DUMP_ROWS:
        pd.concat(DUMP_ROWS).to_csv(out / f"p3_rows_{args.model}.csv.gz", index=False)
    by_cov.to_csv(out / f"p3_by_config_{args.model}.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 3.8))
        data = [res[res.method == m].coverage for m in order]
        ax.boxplot(data)
        ax.set_xticks(range(1, len(order) + 1))
        ax.set_xticklabels(order, rotation=20)
        ax.axhline(1 - args.alpha, color="k", ls="--", lw=0.8)
        ax.set_ylabel("coverage on target circuits")
        ax.set_title(f"Phase 3 methods, model = {args.model}")
        fig.tight_layout()
        fig.savefig(out / f"p3_coverage_{args.model}.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
