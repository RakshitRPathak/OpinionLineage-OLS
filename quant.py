"""Weighted quantification: probabilistic classify-and-count, adjusted (PACC) and EM prior correction, and
lineage-cluster bootstrap intervals."""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def pcc(W, P):
    return (W[:, None] * P).sum(0) / W.sum()


def confusion(P_val, y_val, k=3):
    """M[y, c] = E[P(c) | true class y], estimated on held-out labelled data."""
    return np.vstack([P_val[y_val == a].mean(0) for a in range(k)])


def pacc(q, M, lam=100.0):
    """Solve q = M^T pi on the probability simplex (non-negative least squares with a heavy sum-to-one row)."""
    k = len(q)
    A = np.vstack([M.T, lam * np.ones((1, k))])
    b = np.r_[q, lam]
    pi, _ = nnls(A, b)
    return pi / pi.sum()


def em_prior(W, P, prior_train, iters=200, tol=1e-9):
    """Saerens-Latinne-Decaestecker EM, with record weights."""
    pi = prior_train.copy()
    for _ in range(iters):
        R = P * (pi / prior_train)
        R /= R.sum(1, keepdims=True)
        new = (W[:, None] * R).sum(0) / W.sum()
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return pi


def estimate(W, P, how, M=None, prior_train=None):
    if how == "pcc":
        return pcc(W, P)
    if how == "pacc":
        return pacc(pcc(W, P), M)
    if how == "em":
        return em_prior(W, P, prior_train)
    raise ValueError(how)


def tvd(a, b):
    return 0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum()


def bootstrap_ci(W, P, clusters, how, M_fn, P_val, y_val, B=300, level=0.9, rng=None, prior_train=None):
    """Percentile interval. clusters=None -> i.i.d. record bootstrap; otherwise resample whole lineages.
    The confusion matrix is re-estimated on a bootstrap resample of the validation set in every replicate."""
    rng = rng or np.random.default_rng(0)
    if clusters is None:
        clusters = np.arange(len(W))
    u, inv = np.unique(clusters, return_inverse=True)
    G = len(u)
    mass = np.zeros((G, P.shape[1])); wsum = np.zeros(G)
    np.add.at(mass, inv, W[:, None] * P); np.add.at(wsum, inv, W)
    est = []
    nv = len(y_val)
    for _ in range(B):
        cnt = rng.multinomial(G, np.full(G, 1 / G))
        q = (cnt @ mass) / max(1e-12, cnt @ wsum)
        if how == "pcc":
            est.append(q)
        else:
            vi = rng.integers(0, nv, nv)
            M = M_fn(P_val[vi], y_val[vi])
            est.append(pacc(q, M))
    est = np.array(est)
    a = (1 - level) / 2
    return np.quantile(est, a, axis=0), np.quantile(est, 1 - a, axis=0)
