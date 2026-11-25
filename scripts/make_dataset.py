"""Generate a CP-QEM dataset: TFIM target circuits + near-Clifford circuits, ideal labels and noisy features.

One row = (circuit, observable). Columns:
  ids:      group_id (target it derives from), circuit_id, kind (target | nearclifford), n_keep
  params:   n, steps, J, h, dt
  label:    y_ideal (exact, from statevector)
  inputs:   noisy_s{scale}, se_s{scale} for each noise scale; circuit, observable and device features

Example (small pilot, ~minutes):
  OMP_NUM_THREADS=1 python scripts/make_dataset.py --n-list 4 6 --steps-list 4 8 12 \
      --targets-per-config 3 --nc-per-target 16 --shots 2000 --jobs 4 --out data/pilot_dataset.csv.gz

Near-Clifford circuits cycle over --nc-modes (default: sample = CDR-style, close to the target;
mixed = 30% uniform Clifford angles, more diverse labels). The mode is stored in column nc_mode,
so the circuit-shift experiments (RQ2) can compare calibration sets with weak vs strong overlap.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace  # noqa: E402

from cpqem.circuits import FAMILIES  # noqa: E402
from cpqem.clifford import count_non_clifford, near_clifford  # noqa: E402
from cpqem.features import circuit_features, observable_features  # noqa: E402
from cpqem.noise import build_noise_model, from_fake_backend  # noqa: E402
from cpqem.simulate import (  # noqa: E402
    fold_global,
    ideal_expectations,
    make_simulator,
    noisy_exact_expectations,
    noisy_expectations,
    sample_shots,
    z_observables,
)


def rows_for_circuit(qc, meta, noise_params, scales, shots, seed, exact=True):
    """Simulate one circuit at all noise scales; return one dict per observable.

    exact=True  (default): store exact noisy values exact_s{s} and a shot-sampled noisy_s{s}, se_s{s}
                drawn offline from them (re-sample later for other shot budgets without re-simulating).
    exact=False: old path, Aer samples the shots directly (noisy_s{s}, se_s{s} only).
    """
    n = qc.num_qubits
    obs = z_observables(n)
    nm = build_noise_model(noise_params)
    sim = make_simulator(nm, n)
    rng = np.random.default_rng(seed)

    ideal = ideal_expectations(qc, obs)
    ex, noisy = {}, {}
    for s in scales:
        folded = fold_global(qc, s)
        if exact:
            ex[s] = noisy_exact_expectations(folded, obs, sim, noise_params.readout)
            noisy[s] = {k: sample_shots(v, shots, rng) for k, v in ex[s].items()}
        else:
            noisy[s] = noisy_expectations(folded, obs, sim, shots, seed + 1000 * s)

    base = {**meta, **circuit_features(qc), **noise_params.summary(), "shots": shots}
    rows = []
    for label, qubits in obs:
        r = {**base, **observable_features(label, qubits, n), "y_ideal": ideal[label]}
        for s in scales:
            r[f"noisy_s{s}"], r[f"se_s{s}"] = noisy[s][label]
            if exact:
                r[f"exact_s{s}"] = ex[s][label]
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-list", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--steps-list", type=int, nargs="+", default=[4, 8, 12])
    ap.add_argument("--dt-range", type=float, nargs=2, default=[0.15, 0.4])
    ap.add_argument("--nc-modes", nargs="+", default=["sample", "mixed"],
                    help="near-Clifford replacement modes, cycled over the nc circuits of each target")
    ap.add_argument("--targets-per-config", type=int, default=3)
    ap.add_argument("--nc-per-target", type=int, default=16, help="near-Clifford circuits per target")
    ap.add_argument("--keep-list", type=int, nargs="+", default=[0, 2, 4, 8],
                    help="numbers of non-Clifford gates kept (cycled over the near-Clifford circuits)")
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--backend", default="FakeSherbrooke")
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--family", choices=sorted(FAMILIES), default="tfim", help="target circuit family")
    ap.add_argument("--coh-rx", type=float, default=0.0, help="coherent over-rotation (rad) after sx/x")
    ap.add_argument("--coh-zz", type=float, default=0.0, help="coherent residual ZZ (rad) after cx")
    ap.add_argument("--no-exact", action="store_true", help="let Aer sample shots directly (old path)")
    ap.add_argument("--out", default="data/pilot_dataset.csv.gz")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    tasks = []
    for n in args.n_list:
        noise_params = replace(from_fake_backend(args.backend, n), coh_rx=args.coh_rx, coh_zz=args.coh_zz)
        build, sample = FAMILIES[args.family]
        for steps in args.steps_list:
            for t in range(args.targets_per_config):
                p = sample(rng, tuple(args.dt_range)) if args.family == "tfim" else sample(rng)
                target = build(n, steps, **p)
                gid = f"{args.family}_n{n}_s{steps}_t{t}" if args.family != "tfim" else f"n{n}_s{steps}_t{t}"
                meta = {"group_id": gid, "family": args.family, "n": n, "steps": steps, **p}
                tasks.append((target, {**meta, "circuit_id": f"{gid}_target", "kind": "target",
                                       "n_keep": count_non_clifford(target), "nc_mode": "none"}, noise_params))
                for j in range(args.nc_per_target):
                    # alternate modes first, then cycle k within each mode -> balanced design
                    mode = args.nc_modes[j % len(args.nc_modes)]
                    k = args.keep_list[(j // len(args.nc_modes)) % len(args.keep_list)]
                    nc = near_clifford(target, k, rng, mode=mode)
                    tasks.append((nc, {**meta, "circuit_id": f"{gid}_nc{j}", "kind": "nearclifford",
                                       "n_keep": k, "nc_mode": mode}, noise_params))

    print(f"{len(tasks)} circuits to simulate ...", flush=True)
    t0 = time.perf_counter()
    seeds = rng.integers(0, 2**31 - 1, size=len(tasks))
    results = Parallel(n_jobs=args.jobs, verbose=5)(
        delayed(rows_for_circuit)(qc, meta, npar, args.scales, args.shots, int(sd), not args.no_exact)
        for (qc, meta, npar), sd in zip(tasks, seeds)
    )
    df = pd.DataFrame([r for rows in results for r in rows])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} rows to {out} in {time.perf_counter() - t0:.1f} s")


if __name__ == "__main__":
    main()
