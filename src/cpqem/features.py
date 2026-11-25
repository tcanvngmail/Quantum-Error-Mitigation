"""Circuit-level features used as model inputs (device features come from NoiseParams.summary())."""
from __future__ import annotations

from qiskit import QuantumCircuit

from .clifford import count_non_clifford


def circuit_features(qc: QuantumCircuit) -> dict:
    ops = qc.count_ops()
    return {
        "n_qubits": qc.num_qubits,
        "depth": qc.depth(),
        "n_cx": int(ops.get("cx", 0)),
        "n_1q_phys": int(ops.get("sx", 0) + ops.get("x", 0)),
        "n_rz": int(ops.get("rz", 0)),
        "n_nonclifford": count_non_clifford(qc),
    }


def observable_features(label: str, qubits: tuple[int, ...], n: int) -> dict:
    return {
        "obs": label,
        "obs_weight": len(qubits),
        "obs_pos": min(qubits) / max(n - 1, 1),  # 0 = edge, ~0.5 = bulk
    }
