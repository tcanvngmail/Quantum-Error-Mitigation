"""Ideal and noisy expectation values, with global unitary folding for noise scaling."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel

from .circuits import BASIS


def z_observables(n: int) -> list[tuple[str, tuple[int, ...]]]:
    """Z_i on every qubit and Z_i Z_{i+1} on every bond (all measurable in the Z basis)."""
    obs = [(f"Z{i}", (i,)) for i in range(n)]
    obs += [(f"Z{i}Z{i+1}", (i, i + 1)) for i in range(n - 1)]
    return obs


def ideal_expectations(qc: QuantumCircuit, obs) -> dict[str, float]:
    sv = Statevector(qc)
    n = qc.num_qubits
    out = {}
    for label, qubits in obs:
        op = SparsePauliOp.from_sparse_list([("Z" * len(qubits), list(qubits), 1.0)], num_qubits=n)
        out[label] = float(np.real(sv.expectation_value(op)))
    return out


def fold_global(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    """U -> U (U^dag U)^k with scale = 2k+1, translated to BASIS without optimization."""
    if scale < 1 or scale % 2 == 0:
        raise ValueError("scale must be an odd integer >= 1")
    if scale == 1:
        return qc
    out = qc.copy()
    inv = qc.inverse()
    for _ in range((scale - 1) // 2):
        out.barrier()
        out.compose(inv, inplace=True)
        out.barrier()
        out.compose(qc, inplace=True)
    # optimization_level=0: translate sxdg etc. to BASIS but never cancel the folds.
    return transpile(out, basis_gates=BASIS, optimization_level=0)


def make_simulator(noise_model: NoiseModel | None, n: int) -> AerSimulator:
    method = "density_matrix" if n <= 12 else "automatic"
    return AerSimulator(method=method, noise_model=noise_model)


def noisy_expectations(
    qc: QuantumCircuit,
    obs,
    sim: AerSimulator,
    shots: int,
    seed: int,
) -> dict[str, tuple[float, float]]:
    """Estimate each Z-type observable from one shot-sampled run. Returns {label: (mean, stderr)}."""
    meas = qc.copy()
    meas.measure_all()
    counts = sim.run(meas, shots=shots, seed_simulator=seed).result().get_counts()
    n = qc.num_qubits
    keys = list(counts.keys())
    freqs = np.array([counts[k] for k in keys], dtype=float) / shots
    # Qiskit bitstrings: qubit 0 is the rightmost character.
    bits = np.array([[int(k[n - 1 - q]) for q in range(n)] for k in keys])
    out = {}
    for label, qubits in obs:
        parity = bits[:, list(qubits)].sum(axis=1) % 2
        m = float(np.dot(freqs, 1 - 2 * parity))
        se = float(np.sqrt(max(1 - m * m, 0.0) / shots))
        out[label] = (m, se)
    return out


# ----------------------------------------------------------------------------------------------
# Exact noisy expectations + offline shot sampling (phase 2).
# One density-matrix run per (circuit, scale) gives exact <O> including gate noise; readout error
# is applied analytically (independent symmetric flips multiply each Z factor by 1 - 2 e_q).
# Shots are then drawn offline, so shot grids and re-sampling need no new simulation.
# ----------------------------------------------------------------------------------------------
def noisy_exact_expectations(qc: QuantumCircuit, obs, sim: AerSimulator, readout: np.ndarray) -> dict[str, float]:
    circ = qc.copy()
    circ.save_probabilities()  # method added to QuantumCircuit by importing qiskit_aer
    probs = np.asarray(sim.run(circ).result().data(0)["probabilities"], dtype=float)
    n = qc.num_qubits
    idx = np.arange(2**n)
    out = {}
    for label, qubits in obs:
        parity = np.zeros(2**n, dtype=int)
        for q in qubits:
            parity ^= (idx >> q) & 1  # Aer probabilities are little-endian: qubit 0 = least significant bit
        m = float(np.dot(probs, 1 - 2 * parity))
        out[label] = m * float(np.prod([1 - 2 * readout[q] for q in qubits]))
    return out


def sample_shots(exact: float, shots: int, rng: np.random.Generator) -> tuple[float, float]:
    """Draw a shot-noise estimate of a +-1-valued observable with exact mean `exact`."""
    p = min(max((1 + exact) / 2, 0.0), 1.0)
    m = 2 * rng.binomial(shots, p) / shots - 1
    return float(m), float(np.sqrt(max(1 - m * m, 0.0) / shots))
