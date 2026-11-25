"""Target circuit families for the CP-QEM benchmark.

All circuits are returned already translated to the simulation basis
BASIS = [rz, sx, x, cx] on a linear chain of qubits 0..n-1, so that
non-Clifford content is carried only by rz angles.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

BASIS = ["rz", "sx", "x", "cx"]


def _to_basis(qc: QuantumCircuit, seed: int = 0) -> QuantumCircuit:
    """Translate to BASIS on a linear chain without adding SWAPs (circuits are nearest-neighbour)."""
    n = qc.num_qubits
    cmap = CouplingMap.from_line(n, bidirectional=True) if n > 1 else None
    return transpile(
        qc,
        basis_gates=BASIS,
        coupling_map=cmap,
        initial_layout=list(range(n)),
        layout_method="trivial",
        routing_method="none" if n > 1 else None,
        optimization_level=1,
        seed_transpiler=seed,
    )


def tfim_trotter(n: int, steps: int, J: float, h: float, dt: float) -> QuantumCircuit:
    """First-order Trotter evolution of H = -J sum Z_i Z_{i+1} - h sum X_i from |0...0>.

    rzz(theta) = exp(-i theta/2 ZZ), so exp(+i J dt ZZ) -> theta = -2 J dt.
    rx(theta)  = exp(-i theta/2 X),  so exp(+i h dt X)  -> theta = -2 h dt.
    """
    qc = QuantumCircuit(n, name=f"tfim_n{n}_s{steps}")
    for _ in range(steps):
        for start in (0, 1):  # even bonds, then odd bonds (brickwork)
            for q in range(start, n - 1, 2):
                qc.rzz(-2.0 * J * dt, q, q + 1)
        for q in range(n):
            qc.rx(-2.0 * h * dt, q)
    return _to_basis(qc)


def hardware_efficient(n: int, layers: int, rng: np.random.Generator) -> QuantumCircuit:
    """Random-parameter hardware-efficient ansatz: RY-RZ layer + CX ladder."""
    qc = QuantumCircuit(n, name=f"hea_n{n}_l{layers}")
    for _ in range(layers):
        for q in range(n):
            qc.ry(rng.uniform(0, 2 * np.pi), q)
            qc.rz(rng.uniform(0, 2 * np.pi), q)
        for q in range(n - 1):
            qc.cx(q, q + 1)
    for q in range(n):
        qc.ry(rng.uniform(0, 2 * np.pi), q)
    return _to_basis(qc)


def random_tfim_params(rng: np.random.Generator, dt_range: tuple[float, float] = (0.15, 0.4)) -> dict:
    """Sample Hamiltonian/time-step parameters for one TFIM target circuit.

    dt_range default (0.15, 0.4): with 4-12 Trotter steps this keeps target labels away from
    the trivial |0...0> value (diagnose_cdr.py, Part B).
    """
    return {
        "J": float(rng.uniform(0.5, 1.5)),
        "h": float(rng.uniform(0.5, 1.5)),
        "dt": float(rng.uniform(*dt_range)),
    }


def xxz_trotter(n: int, steps: int, J: float, h: float, dt: float) -> QuantumCircuit:
    """First-order Trotter evolution of the XXZ chain H = J sum (X X + Y Y + Delta Z Z) from the Neel state.

    The anisotropy Delta is passed as `h` so the dataset columns (J, h, dt) stay the same for every family.
    rxx/ryy/rzz(theta) = exp(-i theta/2 PP), so exp(-i J dt PP) -> theta = 2 J dt.
    """
    qc = QuantumCircuit(n, name=f"xxz_n{n}_s{steps}")
    for q in range(1, n, 2):  # Neel state |0101...>
        qc.x(q)
    for _ in range(steps):
        for start in (0, 1):
            for q in range(start, n - 1, 2):
                qc.rxx(2.0 * J * dt, q, q + 1)
                qc.ryy(2.0 * J * dt, q, q + 1)
                qc.rzz(2.0 * h * J * dt, q, q + 1)
    return _to_basis(qc)


def random_xxz_params(rng: np.random.Generator, dt_range: tuple[float, float] = (0.1, 0.3)) -> dict:
    return {"J": float(rng.uniform(0.5, 1.5)), "h": float(rng.uniform(0.5, 1.5)), "dt": float(rng.uniform(*dt_range))}


FAMILIES = {
    "tfim": (tfim_trotter, random_tfim_params),
    "xxz": (xxz_trotter, random_xxz_params),
}


def make_target(family: str, n: int, steps: int, J: float, h: float, dt: float) -> QuantumCircuit:
    return FAMILIES[family][0](n, steps, J, h, dt)
