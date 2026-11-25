"""A2 check on the k-ladder (Fig. 4b): 90% quantile of normalized scores |r|/g(k) by k bin, and on targets.

  python scripts/a2_diagnostic.py --data data/coh_n4-8.csv.gz data/coh_kladder_n4-8.csv.gz --reps 10
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from g0_split_cp import LinearCell, split_groups  # noqa: E402
from p3_weighted_cp import kscale_fit  # noqa: E402

BINS = [-1, 0, 2, 4, 8, 20, 40]
LABELS = ["0", "2", "4", "8", "16-20", "21-32"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    d = pd.concat([pd.read_csv(p) for p in a.data], ignore_index=True)
    out = []
    for rep in range(a.reps):
        rng = np.random.default_rng(2000 + rep)
        sp = split_groups(d, rng)
        nc, tg = d[d.kind == "nearclifford"], d[d.kind == "target"]
        m = LinearCell().fit(nc[nc.group_id.isin(sp["train"]) & (nc.n_keep <= 8)])
        cal = nc[nc.group_id.isin(sp["cal"])]
        r = cal.y_ideal.to_numpy() - m.predict(cal)
        g, _ = kscale_fit(cal, r)
        s = pd.DataFrame({"s": np.abs(r) / g(cal), "k": cal.n_keep.to_numpy()})
        te = tg[tg.group_id.isin(sp["test"])]
        st = np.abs(te.y_ideal.to_numpy() - m.predict(te)) / g(te)
        q = s.groupby(pd.cut(s.k, BINS, labels=LABELS), observed=True).s.quantile(0.9)
        q["targets"] = np.quantile(st, 0.9)
        out.append(q)
    q = pd.concat(out, axis=1).mean(axis=1)
    print(q.round(2).to_string())
    print(f"ladder ratio q(21-32)/q(8) = {q['21-32'] / q['8']:.2f}")


if __name__ == "__main__":
    main()
