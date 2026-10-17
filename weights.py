"""Lineage-structural contribution weights, the witness store and Selective Dependency Rollback.

Weight family (ancestor-local):
    F_i = (1 - beta*Psyn_i)(1 - gamma*Pcoord_i)                    own-provenance factor
    A_r = Pind_r * F_r                                             authenticity of a lineage root r
    L_i = prod_{e in path(r(i), i)} (1 - alpha*c_e)                lineage attenuation (L_root = 1)
    W_i = A_{r(i)} * L_i * (F_i if i is not a root else 1)

W_i depends only on i's own provenance and on its ancestors, so a revision at record v can change the weights of v's
transitive descendants D(v) and of no other record. The aggregate is stored as the pair (A_c = sum_i W_i P_i(c), Z = sum_i W_i)
per analytical interval; a revision is applied as an additive delta computed from stored witnesses only.
"""
from __future__ import annotations

from collections import deque

import numpy as np


def lineage_arrays(g: dict, psyn, pcoord, alpha, beta, gamma, mode="lineage"):
    parent, c, pind = g["parent"], g["c"], g["pind"]
    n = len(parent)
    F = (1 - beta * psyn) * (1 - gamma * pcoord)
    if mode == "message":                       # no lineage: per-record authenticity weighting only
        return dict(W=F.copy(), F=F, L=np.ones(n), A=F.copy(), root=np.arange(n), parent=-np.ones(n, int), c=np.zeros(n))
    if mode == "local":                         # incident edge only, no inheritance of root authenticity
        W = np.where(parent >= 0, 1 - alpha * c, pind) * F
        return dict(W=W, F=F, L=np.ones(n), A=W.copy(), root=np.arange(n), parent=-np.ones(n, int), c=np.zeros(n))
    L = np.ones(n)
    root = np.arange(n)
    for i in range(n):
        p = parent[i]
        if p >= 0:
            L[i] = L[p] * (1 - alpha * c[i])
            root[i] = root[p]
    A = pind * F                                # meaningful for roots
    W = np.where(parent >= 0, A[root] * L * F, A)
    return dict(W=W, F=F, L=L, A=A, root=root, parent=parent.copy(), c=c.copy())


class WitnessStore:
    """Per-record witnesses + per-interval aggregate (A, Z) with optional Kahan-compensated delta application."""

    def __init__(self, arrays: dict, P: np.ndarray, interval: np.ndarray, n_int: int, kahan=True):
        self.W = arrays["W"].astype(np.float64).copy()
        self.F, self.L, self.A = arrays["F"].copy(), arrays["L"].copy(), arrays["A"].copy()
        self.root, self.parent, self.c = arrays["root"].copy(), arrays["parent"].copy(), arrays["c"].copy()
        self.P, self.iv, self.n_int, self.kahan = P, interval, n_int, kahan
        n = len(self.W)
        order = np.argsort(self.parent, kind="stable")
        ps = self.parent[order]
        self.ch_ptr = np.searchsorted(ps, np.arange(-1, n + 1))  # children of v: order[ch_ptr[v+1]:ch_ptr[v+2]]
        self.ch_idx = order
        self.Agg = np.zeros((n_int, P.shape[1])); self.Z = np.zeros(n_int)
        np.add.at(self.Agg, interval, self.W[:, None] * P); np.add.at(self.Z, interval, self.W)
        self.cA = np.zeros_like(self.Agg); self.cZ = np.zeros_like(self.Z)
        self.log = []
        self.touched_total = 0

    def children(self, v):
        return self.ch_idx[self.ch_ptr[v + 1]:self.ch_ptr[v + 2]]

    def descendants(self, v):
        """D(v) in BFS order together with the path confidence q_{v,u} = prod of c_e on path(v,u)."""
        out, q = [v], [1.0]
        dq = deque([(v, 1.0)])
        while dq:
            u, qu = dq.popleft()
            for w in self.children(u):
                qw = qu * self.c[w]
                out.append(int(w)); q.append(qw); dq.append((int(w), qw))
        return np.array(out, int), np.array(q)

    def _apply(self, idx, newW):
        d = newW - self.W[idx]
        dA = np.zeros_like(self.Agg); dZ = np.zeros_like(self.Z)
        np.add.at(dA, self.iv[idx], d[:, None] * self.P[idx]); np.add.at(dZ, self.iv[idx], d)
        if self.kahan:                          # compensated summation keeps long revision streams exact to ~1 ulp
            y = dA - self.cA; t = self.Agg + y; self.cA = (t - self.Agg) - y; self.Agg = t
            y = dZ - self.cZ; t = self.Z + y; self.cZ = (t - self.Z) - y; self.Z = t
        else:
            self.Agg += dA; self.Z += dZ
        self.log.append(dict(n=len(idx), ids=idx[:50].tolist(), old=self.W[idx][:50].tolist(), new=newW[:50].tolist()))
        self.W[idx] = newW
        self.touched_total += len(idx)
        return len(idx)

    # ---- revision events --------------------------------------------------------------------------------------
    def revise_root_authenticity(self, v, new_A, partial=True):
        """Revision of the authenticity A of lineage root v (e.g. later identified as synthetic / coordinated).
        Descendant u receives W'_u = W_u + q_{v,u} (W~_u - W_u) (confidence-weighted partial retraction) or the full
        counterfactual W~_u when partial=False."""
        assert self.parent[v] < 0, "authenticity revisions address lineage roots"
        D, q = self.descendants(v)
        Wt = np.where(D == v, new_A, new_A * self.L[D] * self.F[D])
        newW = self.W[D] + (q if partial else 1.0) * (Wt - self.W[D])
        self.A[v] = new_A
        return self._apply(D, newW)

    def revise_edge_confidence(self, v, new_c, alpha):
        """Revised confidence of v's incoming edge: L scales by (1-alpha c')/(1-alpha c) on D(v)."""
        assert self.parent[v] >= 0
        D, _ = self.descendants(v)
        r = (1 - alpha * new_c) / max(1e-12, 1 - alpha * self.c[v])
        self.L[D] *= r
        self.c[v] = new_c
        return self._apply(D, self.W[D] * r)

    def revise_own(self, v, new_F):
        """Own-provenance revision of a non-root record (synthetic / coordination evidence of v only)."""
        if self.parent[v] < 0:
            return self.revise_root_authenticity(v, self.A[v] / max(1e-12, self.F[v]) * new_F, partial=False)
        self.F[v] = new_F
        return self._apply(np.array([v]), np.array([self.A[self.root[v]] * self.L[v] * new_F]))

    # ---- read-out ------------------------------------------------------------------------------------------
    def distribution(self, intervals=None):
        A = self.Agg if intervals is None else self.Agg[intervals]
        Z = self.Z if intervals is None else self.Z[intervals]
        return A.sum(0) / Z.sum()

    def full_resum(self):
        """Reference: re-summation of all current witness weights from scratch (touches all N records)."""
        Agg = np.zeros_like(self.Agg); Z = np.zeros_like(self.Z)
        np.add.at(Agg, self.iv, self.W[:, None] * self.P); np.add.at(Z, self.iv, self.W)
        return Agg, Z
