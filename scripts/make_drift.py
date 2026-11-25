"""Phase 4 data (RQ3): a time stream of probe and target circuits under drifting noise.

At each time step t = 0..T-1 and for each n, the stream contains
  * 1 PROBE: a near-Clifford circuit (k drawn from --probe-k, label classically computable) -> used
    online by ACI / conformal martingales to track coverage and detect drift;
  * --targets-per-t TARGET circuits: full TFIM Trotter circuits -> labels used only for evaluation.
The SAME circuits are used in every scenario (common random numbers): differences between
scenarios come only from the noise. Noise at t = 0 equals the main dataset's noise
(FakeSherbrooke, same qubit chain), so models trained/calibrated on main_*.csv.gz apply at t = 0.

Scenarios (T1/T2 multiplied by t_scale, gate and readout errors by err_scale):
  S0  control     constant
  S2  gradual     t_scale 1 -> 0.7 and err_scale 1 -> 1.5 linearly over the stream
  S3  jump        at T/2: t_scale 0.7, err_scale 1.6 (like a bad recalibration)
  S4  random walk log-scale random walk, t_scale in [0.6, 1.1], err_scale in [0.9, 1.8]
  S5  device swap at T/2 the noise parameters of another device (--swap-backend)

Dry run first (a few minutes):
  OMP_NUM_THREADS=1 python scripts/make_drift.py --T 4 --n-list 4 --jobs 4 --out data/drift_test.csv.gz
Full run (~1 h on 4 cores):
  OMP_NUM_THREADS=1 caffeinate -i python scripts/make_drift.py --T 300 --n-list 4 6 8 --jobs 4 \
      --out data/drift_n4-8.csv.gz
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

from cpqem.circuits import random_tfim_params, tfim_trotter  # noqa: E402
from cpqem.clifford import count_non_clifford, near_clifford  # noqa: E402
from cpqem.noise import from_fake_backend  # noqa: E402
from make_dataset import rows_for_circuit  # noqa: E402

SCENARIOS = ["S0", "S2", "S3", "S4", "S5"]


def schedules(T: int, rng: np.random.Generator) -> dict[str, list[tuple[float, float, bool]]]:
    """(t_scale, err_scale, swapped_device) for every scenario and time step."""
    out = {}
    out["S0"] = [(1.0, 1.0, False)] * T
    f = np.linspace(0, 1, T)
    out["S2"] = [(1 - 0.3 * x, 1 + 0.5 * x, False) for x in f]
    out["S3"] = [(1.0, 1.0, False) if t < T // 2 else (0.7, 1.6, False) for t in range(T)]
    lt, le = 0.0, 0.0
    walk = []
    for _ in range(T):
        walk.append((float(np.exp(lt)), float(np.exp(le)), False))
        lt = float(np.clip(lt + rng.normal(0, 0.02), np.log(0.6), np.log(1.1)))
        le = float(np.clip(le + rng.normal(0, 0.03), np.log(0.9), np.log(1.8)))
    out["S4"] = walk
    out["S5"] = [(1.0, 1.0, t >= T // 2) for t in range(T)]
    return out


def load_swap_backend(preferred: str, n: int):
    for name in [preferred, "FakeBrisbane", "FakeKyiv", "FakeOsaka", "FakeKyoto"]:
        try:
            return name, from_fake_backend(name, n)
        except Exception:  # noqa: BLE001  (backend missing in this qiskit-ibm-runtime version)
            continue
    raise RuntimeError("no alternative fake backend available for scenario S5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=300)
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--targets-per-t", type=int, default=2)
    ap.add_argument("--probe-k", type=int, nargs="+", default=[0, 2, 4, 8, 16, 32])
    ap.add_argument("--scenarios", nargs="+", default=SCENARIOS)
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--backend", default="FakeSherbrooke")
    ap.add_argument("--swap-backend", default="FakeBrisbane")
    ap.add_argument("--seed", type=int, default=44)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    sched = schedules(args.T, rng)

    # circuits: generated once per (n, t), shared by all scenarios
    circuits = {}
    for n in args.n_list:
        for t in range(args.T):
            items = []
            p = random_tfim_params(rng)
            base_target = tfim_trotter(n, args.steps, **p)
            k = int(rng.choice(args.probe_k))
            mode = ["sample", "mixed"][t % 2]
            probe = near_clifford(base_target, k, rng, mode=mode)
            items.append(("probe", probe, p, min(k, count_non_clifford(base_target)), mode))
            for j in range(args.targets_per_t):
                pj = random_tfim_params(rng)
                tj = tfim_trotter(n, args.steps, **pj)
                items.append((f"target{j}", tj, pj, count_non_clifford(tj), "none"))
            circuits[(n, t)] = items

    base_np = {n: from_fake_backend(args.backend, n) for n in args.n_list}
    swap = {n: load_swap_backend(args.swap_backend, n) for n in args.n_list}
    print("swap backend:", {n: s[0] for n, s in swap.items()},
          "| chains:", {n: base_np[n].physical_qubits for n in args.n_list}, flush=True)

    tasks = []
    for scen in args.scenarios:
        for t in range(args.T):
            ts, es, swapped = sched[scen][t]
            for n in args.n_list:
                if swapped:
                    bname, npar = swap[n]
                else:
                    bname, npar = args.backend, base_np[n]
                npar = npar.scaled(t_scale=ts, err_scale=es)
                for role, qc, p, k, mode in circuits[(n, t)]:
                    meta = {"group_id": f"drift_n{n}_t{t}", "n": n, "steps": args.steps, **p,
                            "circuit_id": f"{scen}_n{n}_t{t}_{role}",
                            "kind": "nearclifford" if role == "probe" else "target",
                            "n_keep": k, "nc_mode": mode,
                            "scenario": scen, "t": t, "role": role,
                            "t_scale": ts, "err_scale": es, "backend": bname}
                    tasks.append((qc, meta, npar))

    for scen in args.scenarios:
        ts = np.array([s[0] for s in sched[scen]]); es = np.array([s[1] for s in sched[scen]])
        print(f"{scen}: t_scale {ts.min():.2f}-{ts.max():.2f}, err_scale {es.min():.2f}-{es.max():.2f}")
    print(f"{len(tasks)} circuit simulations ...", flush=True)

    t0 = time.perf_counter()
    seeds = rng.integers(0, 2**31 - 1, size=len(tasks))
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(rows_for_circuit)(qc, meta, npar, args.scales, args.shots, int(sd), True)
        for (qc, meta, npar), sd in zip(tasks, seeds)
    )
    df = pd.DataFrame([row for rows in results for row in rows])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
