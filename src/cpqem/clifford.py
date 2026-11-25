"""Near-Clifford training/calibration circuits in the style of Clifford Data Regression (CDR).

In BASIS = [rz, sx, x, cx], sx, x and cx are Clifford; rz(theta) is Clifford
iff theta is a multiple of pi/2. A near-Clifford circuit keeps `n_keep`
non-Clifford rz gates and replaces the others by Clifford rz gates.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit

HALF_PI = np.pi / 2


def is_clifford_angle(theta: float, tol: float = 1e-8) -> bool:
    k = theta / HALF_PI
    return abs(k - round(k)) < tol


def non_clifford_indices(qc: QuantumCircuit) -> list[int]:
    idx = []
    for i, inst in enumerate(qc.data):
        if inst.operation.name == "rz" and not is_clifford_angle(float(inst.operation.params[0])):
            idx.append(i)
    return idx


def count_non_clifford(qc: QuantumCircuit) -> int:
    return len(non_clifford_indices(qc))


def _replacement_angle(theta: float, rng: np.random.Generator, mode: str, sigma: float) -> float:
    if mode == "nearest":
        return round(theta / HALF_PI) * HALF_PI
    if mode == "uniform":
        # Any of the four Clifford angles: much more diverse labels, less similar to the target.
        return float(rng.integers(0, 4)) * HALF_PI
    if mode.startswith("mixed"):
        # "mixed" = uniform with prob 0.3, else CDR-style sampling; "mixed0.5" sets the prob explicitly.
        p = float(mode[5:]) if len(mode) > 5 else 0.3
        sub = "uniform" if rng.random() < p else "sample"
        return _replacement_angle(theta, rng, sub, sigma)
    if mode == "sample":
        # Sample a Clifford angle with probability ~ exp(-(distance/sigma)^2), as in CDR.
        base = np.floor(theta / HALF_PI)
        cands = (base + np.array([-1, 0, 1, 2])) * HALF_PI
        w = np.exp(-((theta - cands) / sigma) ** 2)
        return float(rng.choice(cands, p=w / w.sum()))
    raise ValueError(f"unknown mode {mode!r}")


def near_clifford(
    qc: QuantumCircuit,
    n_keep: int,
    rng: np.random.Generator,
    mode: str = "sample",
    sigma: float = 0.5,
) -> QuantumCircuit:
    """Return a copy of qc where all but `n_keep` (random) non-Clifford rz gates are Cliffordized."""
    idx = non_clifford_indices(qc)
    n_keep = min(n_keep, len(idx))
    keep = set(rng.choice(idx, size=n_keep, replace=False).tolist()) if n_keep > 0 else set()
    replace = set(idx) - keep

    out = qc.copy_empty_like()
    out.name = f"{qc.name}_nc{n_keep}"
    for i, inst in enumerate(qc.data):
        if i in replace:
            theta = float(inst.operation.params[0])
            out.rz(_replacement_angle(theta, rng, mode, sigma), inst.qubits[0])
        else:
            out.append(inst.operation, inst.qubits, inst.clbits)
    return out
