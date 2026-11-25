"""Sanity checks to run first (pytest -q). Each test is small (n <= 4) and takes seconds."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpqem.circuits import BASIS, tfim_trotter  # noqa: E402
from cpqem.clifford import count_non_clifford, near_clifford  # noqa: E402
from cpqem.noise import NoiseParams, build_noise_model, from_fake_backend  # noqa: E402
from cpqem.simulate import (  # noqa: E402
    noisy_exact_expectations,
    fold_global,
    ideal_expectations,
    make_simulator,
    noisy_expectations,
    z_observables,
)

N = 4


def _target():
    return tfim_trotter(N, steps=3, J=1.0, h=0.8, dt=0.15)


def test_basis_and_nonclifford():
    qc = _target()
    assert set(qc.count_ops()) <= set(BASIS)
    assert count_non_clifford(qc) > 0


def test_near_clifford_counts():
    qc, rng = _target(), np.random.default_rng(0)
    assert count_non_clifford(near_clifford(qc, 0, rng)) == 0
    assert count_non_clifford(near_clifford(qc, 2, rng)) == 2


def test_folding_preserves_ideal_values():
    qc = _target()
    obs = z_observables(N)
    a, b = ideal_expectations(qc, obs), ideal_expectations(fold_global(qc, 3), obs)
    assert all(abs(a[k] - b[k]) < 1e-8 for k in a)


def test_noiseless_sampling_matches_ideal():
    qc = _target()
    obs = z_observables(N)
    ideal = ideal_expectations(qc, obs)
    est = noisy_expectations(qc, obs, make_simulator(None, N), shots=20000, seed=1)
    for k, (m, se) in est.items():
        assert abs(m - ideal[k]) < 5 * se + 1e-3, k


def test_noise_damps_and_scales():
    """Noise should pull expectations toward 0, more at scale 5 than at scale 1 (on average)."""
    qc = _target()
    obs = z_observables(N)
    ideal = ideal_expectations(qc, obs)
    npar = NoiseParams(t1=np.full(N, 100e-6), t2=np.full(N, 80e-6), err_1q=np.full(N, 5e-4),
                       err_2q=np.full(N - 1, 1e-2), readout=np.full(N, 1e-2))
    sim = make_simulator(build_noise_model(npar), N)
    err = {}
    for s in (1, 5):
        est = noisy_expectations(fold_global(qc, s), obs, sim, shots=20000, seed=2)
        err[s] = np.mean([abs(est[k][0] - ideal[k]) for k in ideal])
    assert err[5] > err[1]


def test_fake_backend_params():
    npar = from_fake_backend("FakeSherbrooke", N)
    assert npar.n == N and len(npar.err_2q) == N - 1
    assert np.all(npar.t2 <= 2 * npar.t1 + 1e-12)
    build_noise_model(npar)  # must not raise


def test_exact_matches_aer_sampling_with_readout():
    """Exact path (probabilities + analytic readout) must agree with Aer shot sampling incl. readout error."""
    qc = fold_global(_target(), 3)
    obs = z_observables(N)
    npar = NoiseParams(t1=np.full(N, 100e-6), t2=np.full(N, 80e-6), err_1q=np.full(N, 5e-4),
                       err_2q=np.full(N - 1, 1e-2), readout=np.array([0.01, 0.03, 0.05, 0.02]))
    sim = make_simulator(build_noise_model(npar), N)
    ex = noisy_exact_expectations(qc, obs, sim, npar.readout)
    est = noisy_expectations(qc, obs, sim, shots=50000, seed=4)
    for k in ex:
        assert abs(est[k][0] - ex[k]) < 4 * est[k][1] + 1e-3, (k, ex[k], est[k])


def test_xxz_trotter_vs_exact():
    from scipy.linalg import expm
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from cpqem.circuits import xxz_trotter
    J, D, T = 1.0, 0.7, 0.6
    terms = []
    for i in range(N - 1):
        terms += [("XX", [i, i + 1], J), ("YY", [i, i + 1], J), ("ZZ", [i, i + 1], D * J)]
    H = SparsePauliOp.from_sparse_list(terms, num_qubits=N).to_matrix()
    psi0 = np.zeros(2**N, dtype=complex)
    psi0[int("".join("1" if q % 2 else "0" for q in reversed(range(N))), 2)] = 1.0  # Neel |0101>
    obs = z_observables(N)
    exact = Statevector(expm(-1j * H * T) @ psi0)
    errs = []
    for steps in (6, 24):
        trot = ideal_expectations(xxz_trotter(N, steps, J, D, T / steps), obs)
        ex = {}
        for label, qubits in obs:
            op = SparsePauliOp.from_sparse_list([("Z" * len(qubits), list(qubits), 1.0)], num_qubits=N)
            ex[label] = float(np.real(exact.expectation_value(op)))
        errs.append(max(abs(trot[k] - ex[k]) for k in ex))
    # first-order Trotter: error ~ 1/steps, so 6 -> 24 steps should cut it ~4x.
    # Convergence at this rate also confirms the Neel state and the qubit ordering.
    assert errs[1] < 0.35 * errs[0] and errs[1] < 0.04, errs


def test_coherent_noise_changes_results():
    from dataclasses import replace
    qc = _target()
    obs = z_observables(N)
    npar = NoiseParams(t1=np.full(N, 100e-6), t2=np.full(N, 80e-6), err_1q=np.full(N, 5e-4),
                       err_2q=np.full(N - 1, 1e-2), readout=np.full(N, 1e-2))
    a = noisy_exact_expectations(qc, obs, make_simulator(build_noise_model(npar), N), npar.readout)
    coh = replace(npar, coh_rx=0.05, coh_zz=0.08)
    b = noisy_exact_expectations(qc, obs, make_simulator(build_noise_model(coh), N), coh.readout)
    assert max(abs(a[k] - b[k]) for k in a) > 1e-3
    assert "dev_coh_rx" in coh.summary() and "dev_coh_rx" not in npar.summary()
