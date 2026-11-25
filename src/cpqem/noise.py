"""Parametric noise models seeded from IBM fake backends.

We extract per-qubit T1/T2, 1q/2q gate errors and readout errors for a linear
chain of physical qubits, then build a compact Aer NoiseModel on logical
qubits 0..n-1. Keeping the parameters explicit lets us create drift
(scale T1/T2 and error rates over time) for later phases.

Simplifications (state them in the paper):
  * reported gate error r -> depolarizing parameter lambda = r * d / (d - 1),
    composed with thermal relaxation (slightly pessimistic, no subtraction);
  * symmetric readout flips;
  * no idle noise; optional coherent errors (coh_rx over-rotation on sx/x, coh_zz residual ZZ on cx).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from qiskit.circuit.library import RXGate, RZZGate
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    coherent_unitary_error,
    depolarizing_error,
    thermal_relaxation_error,
)

from .circuits import BASIS


@dataclass
class NoiseParams:
    t1: np.ndarray          # seconds, shape (n,)
    t2: np.ndarray          # seconds, shape (n,)
    err_1q: np.ndarray      # reported 1q gate error (sx), shape (n,)
    err_2q: np.ndarray      # reported 2q gate error per bond (q, q+1), shape (n-1,)
    readout: np.ndarray     # readout error probability, shape (n,)
    time_1q: float = 36e-9
    time_2q: float = 500e-9
    source: str = "custom"
    physical_qubits: list = field(default_factory=list)
    coh_rx: float = 0.0   # coherent over-rotation (rad) after every sx / x gate
    coh_zz: float = 0.0   # coherent residual ZZ coupling (rad) after every cx gate

    @property
    def n(self) -> int:
        return len(self.t1)

    def scaled(self, t_scale: float = 1.0, err_scale: float = 1.0) -> "NoiseParams":
        """Drift helper: t_scale < 1 shortens T1/T2, err_scale > 1 increases error rates."""
        return replace(
            self,
            t1=self.t1 * t_scale,
            t2=self.t2 * t_scale,
            err_1q=np.clip(self.err_1q * err_scale, 0, 0.5),
            err_2q=np.clip(self.err_2q * err_scale, 0, 0.75),
            readout=np.clip(self.readout * err_scale, 0, 0.5),
        )

    def summary(self) -> dict:
        return {
            "dev_t1_mean_us": float(np.mean(self.t1) * 1e6),
            "dev_t2_mean_us": float(np.mean(self.t2) * 1e6),
            "dev_err1q_mean": float(np.mean(self.err_1q)),
            "dev_err2q_mean": float(np.mean(self.err_2q)) if len(self.err_2q) else 0.0,
            "dev_readout_mean": float(np.mean(self.readout)),
            **({"dev_coh_rx": self.coh_rx, "dev_coh_zz": self.coh_zz} if (self.coh_rx or self.coh_zz) else {}),
        }


def _find_chain(
    edge_err: dict[tuple[int, int], float],
    qubit_ro: dict[int, float],
    n: int,
    max_edge_err: float = 0.05,
    max_readout: float = 0.10,
    max_nodes_per_start: int = 20_000,
) -> list[int]:
    """Best linear chain of n physical qubits, avoiding broken couplers and bad-readout qubits.

    Edges with error >= max_edge_err (fake backends report ~1.0 for non-functional
    couplers) and qubits with readout error >= max_readout are excluded. A greedy-first
    DFS is run from every start qubit; the chain with the lowest mean 2q error plus
    mean readout error wins.
    """
    adj: dict[int, list[tuple[float, int]]] = {}
    for (a, b), e in edge_err.items():
        if e >= max_edge_err or qubit_ro.get(a, 1) >= max_readout or qubit_ro.get(b, 1) >= max_readout:
            continue
        adj.setdefault(a, []).append((e, b))
        adj.setdefault(b, []).append((e, a))
    for a in adj:
        adj[a] = sorted(set(adj[a]))  # try best neighbours first

    def dfs(path, budget):
        budget[0] -= 1
        if budget[0] < 0:
            return None
        if len(path) == n:
            return path
        for _, nb in adj.get(path[-1], ()):
            if nb not in path:
                res = dfs(path + [nb], budget)
                if res:
                    return res
        return None

    def score(chain):
        e2 = [edge_err.get((a, b), edge_err.get((b, a))) for a, b in zip(chain[:-1], chain[1:])]
        return float(np.mean(e2) if e2 else 0.0) + float(np.mean([qubit_ro[q] for q in chain]))

    best = None
    for start in sorted(adj):
        res = dfs([start], [max_nodes_per_start])
        if res and (best is None or score(res) < score(best)):
            best = res
    if best is None:
        raise RuntimeError(f"no chain of {n} good qubits found (relax max_edge_err / max_readout)")
    return best


def _two_qubit_gate_name(target) -> str:
    for name in ("ecr", "cz", "cx"):
        if name in target.operation_names:
            return name
    raise RuntimeError("backend has no ecr/cz/cx gate")


def from_fake_backend(backend_name: str, n: int) -> NoiseParams:
    """Read noise parameters for a chain of n qubits from qiskit_ibm_runtime.fake_provider.<backend_name>."""
    from qiskit_ibm_runtime import fake_provider

    backend = getattr(fake_provider, backend_name)()
    target = backend.target
    g2 = _two_qubit_gate_name(target)
    qubit_ro = {q: (target["measure"][(q,)].error or 0.0) for q in range(backend.num_qubits)}
    edge_err = {}
    for qargs, props in target[g2].items():
        if props is not None and len(qargs) == 2:
            a, b = qargs
            e = props.error if props.error is not None else 1.0
            key = (min(a, b), max(a, b))
            edge_err[key] = min(edge_err.get(key, 1.0), e)
    if n > 1:
        chain = _find_chain(edge_err, qubit_ro, n)
    else:
        chain = [min(qubit_ro, key=qubit_ro.get)]

    t1, t2, e1, ro = [], [], [], []
    for q in chain:
        props = backend.qubit_properties(q)
        t1.append(props.t1)
        t2.append(min(props.t2, 2 * props.t1))  # physical constraint T2 <= 2 T1
        e1.append(target["sx"][(q,)].error or 0.0)
        ro.append(target["measure"][(q,)].error or 0.0)

    e2 = []
    for a, b in zip(chain[:-1], chain[1:]):
        props = target[g2].get((a, b)) or target[g2].get((b, a))
        e2.append(props.error or 0.0)

    time_1q = target["sx"][(chain[0],)].duration or 36e-9
    if n > 1:
        p = target[g2].get((chain[0], chain[1])) or target[g2].get((chain[1], chain[0]))
        time_2q = p.duration or 500e-9
    else:
        time_2q = 500e-9

    return NoiseParams(
        t1=np.array(t1), t2=np.array(t2), err_1q=np.array(e1), err_2q=np.array(e2),
        readout=np.array(ro), time_1q=time_1q, time_2q=time_2q,
        source=backend_name, physical_qubits=chain,
    )


def build_noise_model(p: NoiseParams) -> NoiseModel:
    nm = NoiseModel(basis_gates=BASIS)
    for q in range(p.n):
        lam1 = min(2.0 * p.err_1q[q], 1.0)  # d=2: lambda = 2 r
        relax = thermal_relaxation_error(p.t1[q], p.t2[q], p.time_1q)
        err = depolarizing_error(lam1, 1).compose(relax)
        if p.coh_rx:
            err = err.compose(coherent_unitary_error(RXGate(p.coh_rx).to_matrix()))
        nm.add_quantum_error(err, ["sx", "x"], [q])
        e = float(p.readout[q])
        nm.add_readout_error(ReadoutError([[1 - e, e], [e, 1 - e]]), [q])

    for k in range(p.n - 1):
        a, b = k, k + 1
        lam2 = min(4.0 / 3.0 * p.err_2q[k], 1.0)  # d=4: lambda = 4r/3
        ra = thermal_relaxation_error(p.t1[a], p.t2[a], p.time_2q)
        rb = thermal_relaxation_error(p.t1[b], p.t2[b], p.time_2q)
        dep = depolarizing_error(lam2, 2)
        if p.coh_zz:  # symmetric in the two qubits, so the qarg order does not matter
            dep = dep.compose(coherent_unitary_error(RZZGate(p.coh_zz).to_matrix()))
        # For qargs [x, y]: A.expand(B) puts A on the first qarg and B on the second.
        nm.add_quantum_error(dep.compose(ra.expand(rb)), "cx", [a, b])
        nm.add_quantum_error(dep.compose(rb.expand(ra)), "cx", [b, a])
    return nm
