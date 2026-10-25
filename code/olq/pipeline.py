"""End-to-end processing of a stream and the organic-distribution estimators compared in the paper."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import lineage as lg
from . import quant as qt
from . import textops as tx
from .models import CoordinationScorer
from .weights import lineage_arrays


@dataclass
class Components:
    F: object
    sent: object
    psyn_fn: object
    coord: CoordinationScorer
    edge: lg.EdgeModel
    M: np.ndarray
    prior_train: np.ndarray
    P_val: np.ndarray
    y_val: np.ndarray
    abg: tuple = (0.9, 0.6, 0.0)
    extra: dict = field(default_factory=dict)


def observe(df: pd.DataFrame, comp: Components, K=5):
    """All observable evidence for a stream (never reads the true_* columns)."""
    P_pairs, burst, dens = lg.similarity_pass(df, comp.F, K=K)
    df = df.assign(log_burst=np.log1p(burst))
    pcoord = comp.coord(df)
    psyn = comp.psyn_fn(df.text)
    Pp = lg.pair_features(df, P_pairs, dens, psyn, pcoord)
    P = comp.sent.predict_proba(comp.F.transform(df.text))
    return dict(df=df, pairs=Pp, psyn=psyn, pcoord=pcoord, P=P, burst=burst, dens=dens)


def infer_graph(obs, comp):
    pe = comp.edge(obs["pairs"])
    obs["pe"] = pe
    return lg.build_graph(obs["df"], obs["pairs"], pe)


def _uf_clusters(n, i, j):
    par = np.arange(n)

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a, b in zip(i, j):
        ra, rb = f(a), f(b)
        if ra != rb:
            par[max(ra, rb)] = min(ra, rb)
    return np.array([f(x) for x in range(n)])


def baseline_weights(obs, tau_nd=0.8):
    df, n = obs["df"], len(obs["df"])
    norm = df.text.map(tx.normalise)
    first = ~norm.duplicated(keep="first").to_numpy()
    pr = obs["pairs"]
    sel = pr.s_comb.to_numpy() >= tau_nd
    cl = _uf_clusters(n, pr.i.to_numpy()[sel], pr.j.to_numpy()[sel])
    size = np.bincount(cl, minlength=n)[cl]
    nd = 1.0 / size
    syn_keep = (obs["psyn"] < 0.5).astype(float)
    bot = 1 - obs["pcoord"]
    return {
        "Raw aggregation": np.ones(n),
        "Exact deduplication": first.astype(float),
        "Near-duplicate clustering": nd,
        "Synthetic-content filtering": syn_keep,
        "Bot-score weighting": bot,
        "Combined heuristic pipeline": nd * syn_keep * bot,
    }, cl


def organic_estimates(obs, g, comp, abg=None, how="pacc", g_oracle=None, include_baselines=True):
    """Return {method: estimated distribution} for one stream."""
    a, b, c = abg or comp.abg
    P = obs["P"]
    out = {}
    if include_baselines:
        bw, cl = baseline_weights(obs)
        obs["nd_clusters"] = cl
        for k, w in bw.items():
            out[k] = qt.pcc(w, P)
        out["Combined heuristic + PACC"] = qt.pacc(qt.pcc(bw["Combined heuristic pipeline"], P), comp.M)
    out["OLS, patent coefficients"] = qt.pcc(lineage_arrays(g, obs["psyn"], obs["pcoord"], 0.9, 0.6, 0.0)["W"], P)
    arr = lineage_arrays(g, obs["psyn"], obs["pcoord"], a, b, c)
    obs["W_olq"] = arr["W"]
    out["OLS (proposed)"] = qt.estimate(arr["W"], P, how, comp.M, comp.prior_train)
    if g_oracle is not None:
        ar = lineage_arrays(g_oracle, obs["psyn"], obs["pcoord"], a, b, c)
        out["OLS, oracle lineage"] = qt.estimate(ar["W"], P, how, comp.M, comp.prior_train)
    return out


def ablations(obs, g, comp, g_oracle):
    a, b, c = comp.abg
    P, ps, pc = obs["P"], obs["psyn"], obs["pcoord"]
    M, pt = comp.M, comp.prior_train
    W = lambda **kw: lineage_arrays(kw.pop("g", g), ps if kw.pop("syn", True) else 0 * ps, pc if kw.pop("coord", True) else 0 * pc,
                                    kw.pop("alpha", a), b, c, mode=kw.pop("mode", "lineage"))["W"]
    rows = {
        "Full OLS (lineage + synthetic + coordination)": qt.pcc(W(), P),
        "+ PACC label-shift adjustment": qt.pacc(qt.pcc(W(), P), M),
        "+ EM prior adjustment": qt.em_prior(W(), P, pt),
        "without lineage (message-level weighting)": qt.pcc(W(mode="message"), P),
        "incident edge only (no root inheritance)": qt.pcc(W(mode="local"), P),
        "without synthetic term (beta = 0)": qt.pcc(W(syn=False), P),
        "without coordination term (gamma = 0)": qt.pcc(W(coord=False), P),
        "oracle lineage": qt.pcc(W(g=g_oracle), P),
    }
    for al in (0.5, 0.7, 0.9, 1.0):
        rows[f"alpha = {al}"] = qt.pcc(W(alpha=al), P)
    return rows
