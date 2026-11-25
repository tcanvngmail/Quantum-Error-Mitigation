"""Week 2, step 2: verify each pipeline block on small cases (n = 4).

  python scripts/verify_blocks.py            # all checks, ~1-3 minutes on a laptop
  python scripts/verify_blocks.py --only 1 4 # selected checks

Checks
  [1] TFIM Trotter circuit vs exact evolution exp(-iHT)|0>: error shrinks as dt -> 0 at fixed T (log-log slope 1-2)
  [2] Near-Clifford circuits: exactly k non-Clifford gates kept; k = 0 is a true Clifford circuit;
      labels y_ideal are diverse (not all ~0)
  [3] Noise parameters from the fake backend fall in physically plausible ranges
  [4] Global folding: ideal values unchanged, cx count multiplied by exactly 1/3/5
  [5] Noisy estimates: (a) noiseless sampling unbiased, (b) reported std error matches the
      empirical spread over seeds, (c) noise damps <O> linearly (CDR premise) and more at larger scale

Outputs: console PASS/FAIL lines, results/verify_blocks.txt, and plots if matplotlib is installed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qiskit.quantum_info import SparsePauliOp, Statevector  # noqa: E402
from scipy.linalg import expm  # noqa: E402

from cpqem.circuits import tfim_trotter  # noqa: E402
from cpqem.clifford import count_non_clifford, near_clifford  # noqa: E402
from cpqem.noise import build_noise_model, from_fake_backend  # noqa: E402
from cpqem.simulate import (  # noqa: E402
    fold_global,
    ideal_expectations,
    make_simulator,
    noisy_expectations,
    z_observables,
)

N = 4
J, H_FIELD = 1.0, 0.8
RESULTS = Path(__file__).resolve().parents[1] / "results"
LOG: list[str] = []


def report(name: str, ok: bool, detail: str) -> bool:
    line = f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}"
    print(line, flush=True)
    LOG.append(line)
    return ok


def try_plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


# ---------------------------------------------------------------- [1] Trotter vs exact
def check_trotter(plt):
    obs = z_observables(N)
    terms = [("ZZ", [i, i + 1], -J) for i in range(N - 1)] + [("X", [i], -H_FIELD) for i in range(N)]
    H = SparsePauliOp.from_sparse_list(terms, num_qubits=N).to_matrix()
    T = 1.0
    psi0 = np.zeros(2**N, dtype=complex)
    psi0[0] = 1.0
    exact_sv = Statevector(expm(-1j * H * T) @ psi0)
    exact = {}
    for label, qubits in obs:
        op = SparsePauliOp.from_sparse_list([("Z" * len(qubits), list(qubits), 1.0)], num_qubits=N)
        exact[label] = float(np.real(exact_sv.expectation_value(op)))

    steps_list = [5, 10, 20, 40]
    dts, errs = [], []
    for steps in steps_list:
        dt = T / steps
        trot = ideal_expectations(tfim_trotter(N, steps, J, H_FIELD, dt), obs)
        dts.append(dt)
        errs.append(max(abs(trot[k] - exact[k]) for k in exact))
    slope = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    ok = all(a > b for a, b in zip(errs, errs[1:])) and slope > 0.7 and errs[-1] < 0.05
    report("1 Trotter vs exact", ok,
           f"max error {['%.2e' % e for e in errs]} for dt={dts}; log-log slope={slope:.2f} "
           "(expect 1-2; ~2.0 for these parameters, checked with a numpy-only reference)")
    if plt:
        plt.figure(figsize=(4, 3))
        plt.loglog(dts, errs, "o-", label="max |Trotter - exact|")
        plt.loglog(dts, (np.array(dts) / dts[0]) ** 2 * errs[0], "--", label="slope 2")
        plt.xlabel("dt (T = 1 fixed)")
        plt.ylabel("error")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / "verify_trotter.png", dpi=150)
        plt.close()


# ---------------------------------------------------------------- [2] near-Clifford
def check_near_clifford():
    rng = np.random.default_rng(0)
    target = tfim_trotter(N, 4, J, H_FIELD, 0.15)
    n_nc = count_non_clifford(target)
    ok_counts = True
    for k in (0, 1, 2, 4):
        for _ in range(5):
            nc = near_clifford(target, k, rng)
            ok_counts &= count_non_clifford(nc) == min(k, n_nc)
            ok_counts &= len(nc.data) == len(target.data)  # same gate skeleton, only angles change
    report("2a near-Clifford gate counts", ok_counts, f"target has {n_nc} non-Clifford rz; kept k in {{0,1,2,4}}")

    nc0 = near_clifford(target, 0, rng)
    try:
        from qiskit.quantum_info import Clifford
        Clifford(nc0)
        report("2b k=0 circuit is Clifford", True, "qiskit.quantum_info.Clifford accepted it")
    except Exception as e:  # noqa: BLE001
        report("2b k=0 circuit is Clifford", False, f"Clifford() raised {type(e).__name__}: {e}")

    # Label diversity in the dataset regime (steps=12, dt=0.3); see diagnose_cdr.py Part B.
    obs = z_observables(N)
    target12 = tfim_trotter(N, 12, J, H_FIELD, 0.3)
    stats = {}
    for mode in ("sample", "mixed"):
        ys = []
        for j in range(30):
            ys += list(ideal_expectations(near_clifford(target12, [0, 2, 4][j % 3], rng, mode=mode), obs).values())
        ys = np.array(ys)
        stats[mode] = (float(np.std(ys)), float(np.mean(np.abs(ys) > 0.9)))
    std_m, near1_m = stats["mixed"]
    report("2c label diversity", std_m > 0.3 and near1_m < 0.5,
           ", ".join(f"mode={m}: std={a:.2f}, frac|y|>0.9={b:.2f}" for m, (a, b) in stats.items())
           + " (criterion on mode=mixed: std > 0.3 and frac|y|>0.9 < 0.5; 'sample' is expected to cluster near +-1)")


def check_noise_params(backend):
    lines, ok = [], True
    for n in (4, 6, 8, 10, 12):
        p = from_fake_backend(backend, n)
        s = p.summary()
        ok &= 10 < s["dev_t1_mean_us"] < 2000 and 1e-5 < s["dev_err2q_mean"] < 0.2
        ok &= 1e-4 < s["dev_readout_mean"] < 0.3 and np.all(p.t2 <= 2 * p.t1 + 1e-12)
        lines.append(f"n={n:2d} chain={p.physical_qubits} T1={s['dev_t1_mean_us']:.0f}us "
                     f"T2={s['dev_t2_mean_us']:.0f}us e1q={s['dev_err1q_mean']:.1e} "
                     f"e2q={s['dev_err2q_mean']:.1e} ro={s['dev_readout_mean']:.1e}")
        build_noise_model(p)
    report("3 noise parameters plausible", ok, "\n      " + "\n      ".join(lines))


# ---------------------------------------------------------------- [4] folding
def check_folding():
    qc = tfim_trotter(N, 4, J, H_FIELD, 0.15)
    obs = z_observables(N)
    base = ideal_expectations(qc, obs)
    cx1 = qc.count_ops().get("cx", 0)
    ok, details = True, []
    for s in (3, 5):
        f = fold_global(qc, s)
        diff = max(abs(ideal_expectations(f, obs)[k] - base[k]) for k in base)
        ratio = f.count_ops().get("cx", 0) / cx1
        ok &= diff < 1e-8 and abs(ratio - s) < 1e-9
        details.append(f"scale {s}: max ideal diff {diff:.1e}, cx ratio {ratio:.2f}")
    report("4 global folding", ok, "; ".join(details))


# ---------------------------------------------------------------- [5] noisy estimates
def check_noisy(backend, plt):
    rng = np.random.default_rng(1)
    target = tfim_trotter(N, 4, J, H_FIELD, 0.15)
    obs = z_observables(N)

    # (a) unbiased without noise
    ideal = ideal_expectations(target, obs)
    est = noisy_expectations(target, obs, make_simulator(None, N), shots=20000, seed=3)
    z = np.array([(est[k][0] - ideal[k]) / max(est[k][1], 1e-4) for k in ideal])
    report("5a noiseless sampling unbiased", np.mean(np.abs(z) < 3) >= 0.9,
           f"|z|-scores {np.round(np.abs(z), 2).tolist()} (expect mostly < 2)")

    # (b) standard error formula vs empirical spread
    sim0 = make_simulator(None, N)
    means, ses = [], []
    for sd in range(40):
        m, se = noisy_expectations(target, obs, sim0, shots=1000, seed=100 + sd)["Z0"]
        means.append(m)
        ses.append(se)
    ratio = np.std(means, ddof=1) / np.mean(ses)
    report("5b std error calibrated", 0.7 < ratio < 1.3, f"empirical std / reported se = {ratio:.2f} (expect ~1)")

    # (c) damping under realistic noise, CDR premise, in the dataset regime (steps=12, dt=0.3, mode=mixed)
    sim = make_simulator(build_noise_model(from_fake_backend(backend, N)), N)
    target12 = tfim_trotter(N, 12, J, H_FIELD, 0.3)
    weights = np.array([len(q) for _, q in obs])
    xs = {1: [], 3: [], 5: []}
    ys, ws = [], []
    for j in range(30):
        nc = near_clifford(target12, [0, 2, 4][j % 3], rng, mode="mixed")
        idl = ideal_expectations(nc, obs)
        for s in xs:
            nz = noisy_expectations(fold_global(nc, s), obs, sim, shots=4000, seed=200 + j * 10 + s)
            xs[s] += [nz[k][0] for k in idl]
        ys += list(idl.values())
        ws += list(weights)
    ys, ws = np.array(ys), np.array(ws)
    ok, parts = True, []
    for w in (1, 2):
        m = ws == w
        a, r2 = {}, {}
        for s, v in xs.items():
            v = np.array(v)[m]
            a[s], b = np.polyfit(ys[m], v, 1)
            r2[s] = 1 - np.sum((v - (a[s] * ys[m] + b)) ** 2) / np.sum((v - v.mean()) ** 2)
        ok &= 1 > a[1] > a[3] > a[5] > 0 and r2[1] > 0.85
        parts.append(f"weight {w}: " + ", ".join(f"s{s} a={a[s]:.2f} R2={r2[s]:.2f}" for s in a))
    report("5c noise damping (CDR premise)", ok,
           "; ".join(parts) + " (expect 1 > a1 > a3 > a5 > 0 and R2 at scale 1 > 0.85, per observable weight)")
    if plt:
        plt.figure(figsize=(4, 4))
        for s, v in xs.items():
            plt.scatter(ys, v, s=8, label=f"scale {s}")
        plt.plot([-1, 1], [-1, 1], "k--", lw=0.8)
        plt.xlabel("ideal <O>")
        plt.ylabel("noisy <O>")
        plt.legend()
        plt.tight_layout()
        plt.savefig(RESULTS / "verify_damping.png", dpi=150)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--backend", default="FakeSherbrooke")
    args = ap.parse_args()
    RESULTS.mkdir(exist_ok=True)
    plt = try_plot()

    if 1 in args.only:
        check_trotter(plt)
    if 2 in args.only:
        check_near_clifford()
    if 3 in args.only:
        check_noise_params(args.backend)
    if 4 in args.only:
        check_folding()
    if 5 in args.only:
        check_noisy(args.backend, plt)

    (RESULTS / "verify_blocks.txt").write_text("\n".join(LOG) + "\n")
    n_fail = sum(line.startswith("[FAIL]") for line in LOG)
    print(f"\n{len(LOG) - n_fail} passed, {n_fail} failed -> results/verify_blocks.txt")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
