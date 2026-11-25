"""Supplementary analyses: existing UQ baselines, alpha sensitivity, and circuit-level conformal prediction.

  python scripts/p6_baselines.py --data data/main_n4-8.csv.gz data/kladder_n4-8.csv.gz \
      data/main_n10.csv.gz data/kladder_n10.csv.gz --reps 30 --alphas 0.05 0.1 0.2

All methods use the same vnCDR model (pooled linear per (n, steps, obs_weight) cell, trained on
near-Clifford rows with k <= 8 of train groups) and are evaluated on TARGET rows of test groups.
Nominal level 1 - alpha for every method (Gaussian methods use z_{1-alpha/2}).

  shot_prop     prediction +- z * shot std error propagated through the vnCDR coefficients
  cdr_3sigma    Czarnik et al. (2021): +- 3 sigma, sigma = sqrt(C / (L - 1)) from the training fit (per cell)
  cdr_gauss     same sigma, +- z * sigma
  bootstrap     percentile interval of predictions from 50 bootstrap refits over train groups
  bayes         Bayesian linear regression (BayesianRidge) per cell, +- z * predictive std
  cv_plus       CV+ (Barber et al. 2021), 10 folds over train groups (exchangeability with near-Clifford)
  zne           Richardson ZNE (1, 3, 5) +- z * propagated shot std error (no learning)
  split_cp      split CP on near-Clifford rows with k <= 8 of cal groups (G0)
  kscale        k-scaled CP (proposed, RQ2)
  kscale_circ   circuit-level k-scaled CP: score = max over the circuit's observables, so ALL observables
                of a target circuit are covered simultaneously with probability >= 1 - alpha
Reported: row coverage, simultaneous (all-observables) coverage per target circuit, median width.
Outputs: results/p6_baselines.csv, results/p6_by_k.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import BayesianRidge

from g0_split_cp import CELL, LinearCell, conformal_q, split_groups
from p3_weighted_cp import kscale_fit

ROOT = Path(__file__).resolve().parents[1]
X3 = ["noisy_s1", "noisy_s3", "noisy_s5"]
K_BINS = [0, 56, 100, 150, 230]
K_LABELS = ["28-56", "57-100", "101-150", "151-228"]


def cell_keys(df):
    return list(zip(*[df[c] for c in CELL]))


def design(df):
    return np.column_stack([df[X3].to_numpy(), np.ones(len(df))])


def cdr_sigma(model: LinearCell, train: pd.DataFrame) -> dict:
    sig = {}
    for key, g in train.groupby(CELL):
        r = g.y_ideal.to_numpy() - design(g) @ model.coef[key]
        sig[key] = np.sqrt(np.sum(r**2) / max(len(g) - 1, 1))
    return sig


def shot_prop_se(model: LinearCell, df: pd.DataFrame) -> np.ndarray:
    c = np.array([model.coef[k][:3] for k in cell_keys(df)])
    se = df[["se_s1", "se_s3", "se_s5"]].to_numpy()
    return np.sqrt(np.sum((c * se) ** 2, axis=1))


def bayes_intervals(train, test, z):
    lo, hi = np.empty(len(test)), np.empty(len(test))
    keys = np.array([str(k) for k in cell_keys(test)])
    for key, g in train.groupby(CELL):
        m = keys == str(key)
        if not m.any():
            continue
        br = BayesianRidge().fit(g[X3].to_numpy(), g.y_ideal.to_numpy())
        mu, sd = br.predict(test[X3].to_numpy()[m], return_std=True)
        lo[m], hi[m] = mu - z * sd, mu + z * sd
    return lo, hi


def cv_plus(train, test, alpha, rng, folds=10, per_fold=300):
    gids = rng.permutation(train.group_id.unique())
    fold_of = {g: i % folds for i, g in enumerate(gids)}
    f = train.group_id.map(fold_of).to_numpy()
    lows, highs = [], []
    for k in range(folds):
        m = LinearCell().fit(train[f != k])
        hold = train[f == k]
        if len(hold) > per_fold:
            hold = hold.sample(per_fold, random_state=int(rng.integers(1 << 31)))
        r = np.abs(hold.y_ideal.to_numpy() - m.predict(hold))
        p = m.predict(test)
        lows.append(p[:, None] - r[None, :])
        highs.append(p[:, None] + r[None, :])
    L, H = np.concatenate(lows, axis=1), np.concatenate(highs, axis=1)
    n = L.shape[1]
    lo = np.sort(L, axis=1)[:, max(int(np.floor(alpha * (n + 1))) - 1, 0)]
    hi = np.sort(H, axis=1)[:, min(int(np.ceil((1 - alpha) * (n + 1))) - 1, n - 1)]
    return lo, hi


def one_rep(d, rep, alphas, n_boot=50):
    rng = np.random.default_rng(2000 + rep)
    sp = split_groups(d, rng)
    nc, tg = d[d.kind == "nearclifford"], d[d.kind == "target"]
    train = nc[nc.group_id.isin(sp["train"]) & (nc.n_keep <= 8)]
    cal_nc = nc[nc.group_id.isin(sp["cal"])]
    test = tg[tg.group_id.isin(sp["test"])].reset_index(drop=True)
    y = test.y_ideal.to_numpy()
    model = LinearCell().fit(train)
    pred = model.predict(test)
    sig = cdr_sigma(model, train)
    sig_t = np.array([sig[k] for k in cell_keys(test)])
    se_t = shot_prop_se(model, test)
    zne = (15 * test.noisy_s1 - 10 * test.noisy_s3 + 3 * test.noisy_s5).to_numpy() / 8
    zne_se = np.sqrt((15 / 8 * test.se_s1) ** 2 + (10 / 8 * test.se_s3) ** 2 + (3 / 8 * test.se_s5) ** 2).to_numpy()

    # bootstrap refits over train groups (shared by all alphas)
    tids = train.group_id.unique()
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(tids, size=len(tids), replace=True)
        bt = pd.concat([train[train.group_id == g] for g in pick])
        boots.append(LinearCell().fit(bt).predict(test))
    boots = np.array(boots)

    # k-scaled CP: g on half A of cal groups (stratified by n, steps), conformalize on half B
    half = set()
    cal_t = tg[tg.group_id.isin(sp["cal"])].drop_duplicates("group_id")
    for _, gg in cal_t.groupby(["n", "steps"]):
        ids = rng.permutation(gg.group_id.values)
        half |= set(ids[: len(ids) // 2])
    in_a = cal_nc.group_id.isin(half).to_numpy()
    r_nc = cal_nc.y_ideal.to_numpy() - model.predict(cal_nc)
    g, _ = kscale_fit(cal_nc[in_a], r_nc[in_a])
    cal_b = cal_nc[~in_a]
    s_b = np.abs(r_nc[~in_a]) / g(cal_b)
    s_b_circ = pd.Series(s_b).groupby(cal_b.circuit_id.to_numpy()).max().to_numpy()
    g_t = g(test)
    low_k = (cal_nc.n_keep <= 8).to_numpy()
    s_low = np.abs(r_nc[low_k])

    rows = []
    for alpha in alphas:
        z = norm.ppf(1 - alpha / 2)
        iv = {
            "shot_prop": (pred - z * se_t, pred + z * se_t),
            "cdr_3sigma": (pred - 3 * sig_t, pred + 3 * sig_t),
            "cdr_gauss": (pred - z * sig_t, pred + z * sig_t),
            "bootstrap": tuple(np.quantile(boots, [alpha / 2, 1 - alpha / 2], axis=0)),
            "bayes": bayes_intervals(train, test, z),
            "cv_plus": cv_plus(train, test, alpha, rng),
            "zne": (zne - z * zne_se, zne + z * zne_se),
        }
        q_low = conformal_q(s_low, alpha)
        iv["split_cp"] = (pred - q_low, pred + q_low)
        qk = conformal_q(s_b, alpha)
        iv["kscale"] = (pred - qk * g_t, pred + qk * g_t)
        qc = conformal_q(s_b_circ, alpha)
        iv["kscale_circ"] = (pred - qc * g_t, pred + qc * g_t)
        for name, (lo, hi) in iv.items():
            cov = (y >= lo) & (y <= hi)
            simul = pd.Series(cov).groupby(test.circuit_id.to_numpy()).all().mean()
            rows.append(pd.DataFrame({"rep": rep, "alpha": alpha, "method": name, "covered": cov,
                                      "width": hi - lo, "k": test.n_keep.to_numpy(), "simul": simul,
                                      "n": test.n.to_numpy(), "steps": test.steps.to_numpy(), "w": test.obs_weight.to_numpy()}))
    return pd.concat(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    d = pd.concat([pd.read_csv(p) for p in args.data], ignore_index=True)

    res = []
    for rep in range(args.reps):
        res.append(one_rep(d, rep, args.alphas))
        if (rep + 1) % 5 == 0:
            print(f"  {rep + 1}/{args.reps}", flush=True)
    res = pd.concat(res)
    order = ["shot_prop", "zne", "cdr_3sigma", "cdr_gauss", "bootstrap", "bayes", "cv_plus",
             "split_cp", "kscale", "kscale_circ"]
    per_rep = res.groupby(["alpha", "method", "rep"]).agg(cov=("covered", "mean"), simul=("simul", "first"),
                                                          width=("width", "median")).reset_index()
    summ = per_rep.groupby(["alpha", "method"]).agg(coverage=("cov", "mean"), cov_sd=("cov", "std"),
                                                     simultaneous=("simul", "mean"), width=("width", "mean"))
    res["kbin"] = pd.cut(res.k, K_BINS, labels=K_LABELS)
    byk = res.groupby(["alpha", "method", "kbin"], observed=True).covered.mean().unstack()
    pd.set_option("display.width", 200)
    for a in args.alphas:
        print(f"\n=== alpha = {a} (target coverage {1 - a:.2f})")
        s = summ.loc[a].reindex(order).join(byk.loc[a].reindex(order))
        print(s.round(3).to_string())
    cellcov = res.groupby(["alpha", "method", "n", "steps", "w"]).covered.mean()
    worst = cellcov.groupby(["alpha", "method"]).agg(["min", lambda x: (x < 1 - x.name[0] - 0.05).mean() if False else x.quantile(0.1)])
    worst.columns = ["worst_cell", "p10_cell"]
    print("\nPer-cell (n, steps, weight) coverage: worst and 10th percentile")
    print(worst.round(3).to_string())
    worst.to_csv(out / "p6_cells.csv")
    summ.to_csv(out / "p6_baselines.csv")
    byk.to_csv(out / "p6_by_k.csv")


if __name__ == "__main__":
    main()
