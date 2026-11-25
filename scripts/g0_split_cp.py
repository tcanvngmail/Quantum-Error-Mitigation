"""Phase 2, experiment G0: does split conformal prediction cover mitigated target values?

  python scripts/g0_split_cp.py --data data/main_n4-8.csv.gz --reps 50 --alpha 0.1

Design (one repetition)
  * Target circuits (group_id) are split 40/30/30 into train / cal / test groups, stratified by (n, steps).
  * A global ML-QEM model is trained on the NEAR-CLIFFORD rows of train groups (labels are classically
    computable there, as in CDR):
        linear : y ~ x1 + x3 + x5 per (n, steps, obs_weight) cell      (vnCDR-style, pooled over targets)
        gbm    : HistGradientBoostingRegressor on x1, x3, x5, obs_weight, obs_pos, n, steps, depth, n_cx
  * Split CP with absolute-residual scores, calibrated on different sets of cal-group rows:
        oracle      target rows        (exchangeable with the test targets; not available on hardware)
        nc_all      near-Clifford rows, both modes
        nc_sample   near-Clifford rows, CDR-style "sample" mode only
        nc_mixed    near-Clifford rows, "mixed" mode only
  * Test on TARGET rows of test groups. Sanity check: nc_all-calibrated intervals on nc rows of test groups.
  * Baseline "shot-only": prediction +- z * se_s1 (pretends shot noise is the only error).

G0 go/no-go: oracle and sanity coverage ~ 1 - alpha, and nc-calibrated coverage on targets clearly below it.
Outputs: results/g0_summary.csv, results/g0_by_config.csv, results/g0_coverage.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
FEATS = ["noisy_s1", "noisy_s3", "noisy_s5", "obs_weight", "obs_pos", "n", "steps", "depth", "n_cx"]
CELL = ["n", "steps", "obs_weight"]


def split_groups(d: pd.DataFrame, rng: np.random.Generator) -> dict[str, set]:
    out = {"train": set(), "cal": set(), "test": set()}
    for _, g in d[d.kind == "target"].groupby(["n", "steps"]):
        ids = rng.permutation(g.group_id.unique())
        a, b = int(0.4 * len(ids)), int(0.7 * len(ids))
        out["train"] |= set(ids[:a]); out["cal"] |= set(ids[a:b]); out["test"] |= set(ids[b:])
    return out


class LinearCell:
    """vnCDR pooled per (n, steps, obs_weight): y = c1 x1 + c3 x3 + c5 x5 + b."""

    def fit(self, df):
        self.coef = {}
        for key, g in df.groupby(CELL):
            X = np.column_stack([g[["noisy_s1", "noisy_s3", "noisy_s5"]].to_numpy(), np.ones(len(g))])
            self.coef[key] = np.linalg.lstsq(X, g.y_ideal.to_numpy(), rcond=None)[0]
        return self

    def predict(self, df):
        out = np.empty(len(df))
        keys = list(zip(*[df[c] for c in CELL]))
        X = np.column_stack([df[["noisy_s1", "noisy_s3", "noisy_s5"]].to_numpy(), np.ones(len(df))])
        for i, k in enumerate(keys):
            out[i] = X[i] @ self.coef[k]
        return out


class GBM:
    def fit(self, df):
        self.m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=0)
        self.m.fit(df[FEATS], df.y_ideal)
        return self

    def predict(self, df):
        return self.m.predict(df[FEATS])


def conformal_q(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return np.inf if k > n else float(np.sort(scores)[k - 1])


def one_rep(d, rep, alpha, models):
    rng = np.random.default_rng(1000 + rep)
    sp = split_groups(d, rng)
    nc, tg = d[d.kind == "nearclifford"], d[d.kind == "target"]
    train = nc[nc.group_id.isin(sp["train"])]
    cal_sets = {
        "oracle": tg[tg.group_id.isin(sp["cal"])],
        "nc_all": nc[nc.group_id.isin(sp["cal"])],
        "nc_sample": nc[nc.group_id.isin(sp["cal"]) & (nc.nc_mode == "sample")],
        "nc_mixed": nc[nc.group_id.isin(sp["cal"]) & (nc.nc_mode == "mixed")],
    }
    test = tg[tg.group_id.isin(sp["test"])]
    test_nc = nc[nc.group_id.isin(sp["test"])]
    z = norm.ppf(1 - alpha / 2)
    rows, cfg_rows = [], []
    for mname, M in models.items():
        m = M().fit(train)
        pt = m.predict(test)
        err_t = np.abs(test.y_ideal.to_numpy() - pt)
        mae = float(err_t.mean())
        for cname, cal in cal_sets.items():
            q = conformal_q(np.abs(cal.y_ideal.to_numpy() - m.predict(cal)), alpha)
            cov = err_t <= q
            rows.append(dict(rep=rep, model=mname, method=f"CP[{cname}]", test="target",
                             coverage=cov.mean(), width=2 * q, mae=mae))
            cfg_rows += [dict(rep=rep, model=mname, method=f"CP[{cname}]", n=n_, steps=s_, coverage=c)
                         for (n_, s_), c in pd.Series(cov, index=pd.MultiIndex.from_frame(test[["n", "steps"]]))
                         .groupby(level=[0, 1]).mean().items()]
            if cname == "nc_all":
                cov_nc = np.abs(test_nc.y_ideal.to_numpy() - m.predict(test_nc)) <= q
                rows.append(dict(rep=rep, model=mname, method="CP[nc_all]", test="nearclifford (sanity)",
                                 coverage=cov_nc.mean(), width=2 * q, mae=np.nan))
        cov_shot = err_t <= z * test.se_s1.to_numpy()
        rows.append(dict(rep=rep, model=mname, method="shot-only", test="target",
                         coverage=cov_shot.mean(), width=float(2 * z * test.se_s1.mean()), mae=mae))
    raw_cov = np.abs(test.y_ideal - test.noisy_s1) <= z * test.se_s1
    rows.append(dict(rep=rep, model="none (raw)", method="shot-only", test="target",
                     coverage=raw_cov.mean(), width=float(2 * z * test.se_s1.mean()),
                     mae=float(np.abs(test.y_ideal - test.noisy_s1).mean())))
    return rows, cfg_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "main_n4-8.csv.gz"))
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(exist_ok=True)
    d = pd.read_csv(args.data)
    models = {"linear": LinearCell, "gbm": GBM}

    rows, cfg = [], []
    for rep in range(args.reps):
        r, c = one_rep(d, rep, args.alpha, models)
        rows += r; cfg += c
        if (rep + 1) % 10 == 0:
            print(f"  {rep + 1}/{args.reps} repetitions", flush=True)
    res, cfg = pd.DataFrame(rows), pd.DataFrame(cfg)

    summ = (res.groupby(["model", "method", "test"])
            .agg(coverage=("coverage", "mean"), cov_sd=("coverage", "std"),
                 width=("width", "mean"), mae=("mae", "mean")).reset_index())
    by_cfg = cfg.groupby(["model", "method", "n", "steps"]).coverage.mean().unstack([2, 3])
    pd.set_option("display.width", 200)
    print(f"\nalpha = {args.alpha} (target coverage {1 - args.alpha:.2f}), {args.reps} repetitions\n")
    print(summ.round(3).to_string(index=False))
    print("\nTarget coverage by (n, steps):")
    print(by_cfg.round(2).to_string())
    summ.to_csv(out / "g0_summary.csv", index=False)
    by_cfg.to_csv(out / "g0_by_config.csv")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sub = res[res.test == "target"]
        order = ["CP[oracle]", "CP[nc_all]", "CP[nc_mixed]", "CP[nc_sample]", "shot-only"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
        for ax, mname in zip(axes, models):
            s = sub[sub.model == mname]
            ax.boxplot([s[s.method == m].coverage for m in order])
            ax.set_xticks(range(1, len(order) + 1))
            ax.set_xticklabels([m.replace("CP[", "").rstrip("]") for m in order])
            ax.axhline(1 - args.alpha, color="k", ls="--", lw=0.8)
            ax.set_title(f"model = {mname}")
            ax.tick_params(axis="x", rotation=20)
        axes[0].set_ylabel("coverage on target circuits")
        fig.tight_layout()
        fig.savefig(out / "g0_coverage.png", dpi=150)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
