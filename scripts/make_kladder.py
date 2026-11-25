"""Phase 3 data: extra calibration circuits with MORE non-Clifford gates (k-ladder) for existing targets.

Re-creates every target circuit of an existing dataset from its (n, steps, J, h, dt), then generates
near-Clifford circuits keeping k in --k-list non-Clifford gates (capped at the target's count).
Rows have the same columns as make_dataset.py, the same group_id as the target, and
circuit_id '<group>_kl<k>_<j>', so the output can simply be concatenated with the base dataset.

  OMP_NUM_THREADS=1 caffeinate -i python scripts/make_kladder.py --base data/main_n4-8.csv.gz \
      --k-list 16 32 --per-k 8 --jobs 4 --out data/kladder_n4-8.csv.gz

Cost ~ the base run (16 extra circuits per target): ~20 min for n = 4-8, ~4 h for n = 10.
Circuits with k = 16-32 non-Clifford gates remain classically simulable at scale (stabilizer-rank
methods), so using them for calibration is realistic on hardware.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
sys.path.insert(0, str(HERE))

from dataclasses import replace  # noqa: E402

from cpqem.circuits import make_target  # noqa: E402
from cpqem.clifford import count_non_clifford, near_clifford  # noqa: E402
from cpqem.noise import from_fake_backend  # noqa: E402
from make_dataset import rows_for_circuit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="existing dataset whose target circuits are reused")
    ap.add_argument("--k-list", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--per-k", type=int, default=8, help="circuits per target and per k")
    ap.add_argument("--nc-modes", nargs="+", default=["sample", "mixed"])
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--backend", default="FakeSherbrooke")
    ap.add_argument("--seed", type=int, default=33)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    base = pd.read_csv(args.base)
    tg = base[base.kind == "target"].drop_duplicates("group_id")
    rng = np.random.default_rng(args.seed)
    noise_cache: dict[int, object] = {}
    tasks, mismatch = [], 0
    for _, r in tg.iterrows():
        # access columns with r["..."]: r.dt would hit pandas' .dt (datetime) accessor
        gid = r["group_id"]
        n, steps = int(r["n"]), int(r["steps"])
        J, h, dt = float(r["J"]), float(r["h"]), float(r["dt"])
        family = r["family"] if "family" in tg.columns else "tfim"
        target = make_target(family, n, steps, J, h, dt)
        n_nc = count_non_clifford(target)
        if n_nc != int(r["n_keep"]):
            mismatch += 1  # target not reproduced exactly (e.g. different Qiskit version)
        if n not in noise_cache:
            coh = {"coh_rx": float(r["dev_coh_rx"]) if "dev_coh_rx" in tg.columns else 0.0,
                   "coh_zz": float(r["dev_coh_zz"]) if "dev_coh_zz" in tg.columns else 0.0}
            noise_cache[n] = replace(from_fake_backend(args.backend, n), **coh)  # same noise as --base
        meta = {"group_id": gid, "n": n, "steps": steps, "J": J, "h": h, "dt": dt}
        if "family" in tg.columns:
            meta["family"] = family
        for k in args.k_list:
            for j in range(args.per_k):
                mode = args.nc_modes[j % len(args.nc_modes)]
                nc = near_clifford(target, k, rng, mode=mode)
                tasks.append((nc, {**meta, "circuit_id": f"{gid}_kl{k}_{j}", "kind": "nearclifford",
                                   "n_keep": min(k, n_nc), "nc_mode": mode}, noise_cache[n]))
    if mismatch:
        print(f"WARNING: {mismatch} targets have a different non-Clifford count than in --base "
              "(Qiskit version changed?). Check before using the data.", flush=True)

    print(f"{len(tg)} targets, {len(tasks)} circuits to simulate ...", flush=True)
    t0 = time.perf_counter()
    seeds = rng.integers(0, 2**31 - 1, size=len(tasks))
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(rows_for_circuit)(qc, meta, npar, args.scales, args.shots, int(sd), True)
        for (qc, meta, npar), sd in zip(tasks, seeds)
    )
    df = pd.DataFrame([row for rows in results for row in rows])[base.columns]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
