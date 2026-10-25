"""Opinion-Lineage Graph construction from observable evidence only (no ground-truth relationships).

For every record i, earlier records j (t_j < t_i) are candidate parents when they are among the K most similar records
(word+character TF-IDF cosine). Each candidate pair is described by semantic, lexical-containment, temporal, metadata,
synthetic-content and coordination evidence; a gradient-boosted model, isotonically calibrated on separate calibration
streams, turns that evidence into a derivation confidence c_ij = P(i derives from j's lineage). Explicit repost/quote
metadata, when exposed, fixes the parent with confidence META_CONF. The primary parent is the highest-confidence
candidate; edges always point forward in time, so the derivational graph is a forest (DAG).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier

from . import textops as tx

META_CONF = 0.99
EDGE_FEATS = ["s_comb", "s_word", "s_char", "containment", "log_dt_h", "rank", "s_gap", "log_density", "psyn_i", "psyn_j",
              "pcoord_i", "pcoord_j", "len_ratio", "meta_match", "meta_other"]
DERIVED = {"repost", "quote", "paraphrase", "ai_rewrite", "coord_copy"}


def similarity_pass(df: pd.DataFrame, F, K=5, burst_window=0.15, burst_sim=0.6, chunk=700):
    """Top-K earlier neighbours per record, burst co-posting counts and topical density."""
    X, W, C = F.transform(df.text, parts=True)
    t, acc, n = df.t.to_numpy(), df.account.to_numpy(), len(df)
    XT = X.T.tocsc()
    ci, cj, cs, rk = [], [], [], []
    burst, dens = np.zeros(n), np.zeros(n)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        S = (X[a:b] @ XT).toarray()
        rows = np.arange(a, b)
        S[np.arange(b - a), rows] = -1.0
        near = np.abs(t[None, :] - t[a:b, None]) <= burst_window
        hit = near & (S >= burst_sim)
        for r in range(b - a):
            burst[a + r] = len(np.unique(acc[hit[r]]))
        earlier = t[None, :] < t[a:b, None]
        Se = np.where(earlier, S, -1.0)
        dens[a:b] = (Se >= 0.5).sum(1)
        k = min(K, n - 1)
        top = np.argpartition(-Se, kth=k - 1, axis=1)[:, :k]
        vals = np.take_along_axis(Se, top, 1)
        o = np.argsort(-vals, axis=1)
        top, vals = np.take_along_axis(top, o, 1), np.take_along_axis(vals, o, 1)
        for r in range(b - a):
            ok = vals[r] > 0.05
            m = int(ok.sum())
            ci.extend([a + r] * m); cj.extend(top[r, ok]); cs.extend(vals[r, ok]); rk.extend(range(m))
    P = pd.DataFrame(dict(i=np.array(ci, int), j=np.array(cj, int), s_comb=np.array(cs), rank=np.array(rk)))
    if len(P):
        P["s_word"] = np.asarray(W[P.i.to_numpy()].multiply(W[P.j.to_numpy()]).sum(1)).ravel()
        P["s_char"] = np.asarray(C[P.i.to_numpy()].multiply(C[P.j.to_numpy()]).sum(1)).ravel()
        top1 = P.groupby("i").s_comb.transform("max")
        P["s_gap"] = top1 - P.s_comb
    return P, burst, dens


def pair_features(df: pd.DataFrame, P: pd.DataFrame, dens: np.ndarray, psyn: np.ndarray, pcoord: np.ndarray):
    toks = [set(tx.normalise(s).split()) for s in df.text]
    i, j = P.i.to_numpy(), P.j.to_numpy()
    P = P.copy()
    P["containment"] = [len(toks[a] & toks[b]) / max(1, len(toks[b])) for a, b in zip(i, j)]
    P["len_ratio"] = [len(toks[a]) / max(1, len(toks[b])) for a, b in zip(i, j)]
    P["log_dt_h"] = np.log1p(24 * (df.t.to_numpy()[i] - df.t.to_numpy()[j]))
    P["log_density"] = np.log1p(dens[i])
    P["psyn_i"], P["psyn_j"] = psyn[i], psyn[j]
    P["pcoord_i"], P["pcoord_j"] = pcoord[i], pcoord[j]
    mp = df.meta_parent.to_numpy()
    P["meta_match"] = (mp[i] == j).astype(float)
    P["meta_other"] = ((mp[i] >= 0) & (mp[i] != j)).astype(float)
    return P


def pair_labels(df: pd.DataFrame, P: pd.DataFrame) -> np.ndarray:
    rt, ty = df.true_root.to_numpy(), df.true_type.to_numpy()
    i, j = P.i.to_numpy(), P.j.to_numpy()
    return ((rt[i] == rt[j]) & np.isin(ty[i], list(DERIVED))).astype(int)


class EdgeModel:
    def fit(self, P: pd.DataFrame, y: np.ndarray, seed=0):
        base = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, random_state=seed)
        self.m = CalibratedClassifierCV(base, cv=3, method="isotonic").fit(P[EDGE_FEATS], y)
        return self

    def __call__(self, P: pd.DataFrame):
        return self.m.predict_proba(P[EDGE_FEATS])[:, 1] if len(P) else np.zeros(0)


def build_graph(df: pd.DataFrame, P: pd.DataFrame, pe: np.ndarray, threshold=0.5):
    """Primary-parent forest. Returns parent, c (incoming edge confidence) and pind (residual independence)."""
    n = len(df)
    parent, c, pind = -np.ones(n, int), np.zeros(n), np.ones(n)
    if len(P):
        Q = P.assign(p=pe).sort_values(["i", "p"], ascending=[True, False]).drop_duplicates("i")
        bi, bj, bp = Q.i.to_numpy(), Q.j.to_numpy(), Q.p.to_numpy()
        edge = bp >= threshold
        parent[bi[edge]], c[bi[edge]] = bj[edge], bp[edge]
        pind[bi[~edge]] = 1 - bp[~edge]
    mp = df.meta_parent.to_numpy()
    has = mp >= 0
    parent[has], c[has], pind[has] = mp[has], META_CONF, 1.0
    root = np.arange(n)
    for i in range(n):                         # records are time-ordered and parents are earlier
        if parent[i] >= 0:
            root[i] = root[parent[i]]
    return dict(parent=parent, c=c, pind=pind, root=root)


def oracle_graph(df: pd.DataFrame):
    n = len(df)
    parent = df.true_parent.to_numpy().copy()
    return dict(parent=parent, c=np.where(parent >= 0, 1.0, 0.0), pind=np.ones(n), root=df.true_root.to_numpy().copy())


def evaluate_graph(df: pd.DataFrame, g: dict) -> dict:
    ty, rt = df.true_type.to_numpy(), df.true_root.to_numpy()
    der_true = np.isin(ty, list(DERIVED))
    der_pred = g["parent"] >= 0
    tp = (der_true & der_pred).sum()
    prec = tp / max(1, der_pred.sum()); rec = tp / max(1, der_true.sum())
    auth = df.true_authentic.to_numpy()
    out = dict(root_acc=float((~der_pred[auth]).mean()), hard_neg_root_acc=float((~der_pred[auth & df.true_hard_negative.to_numpy()]).mean())
               if (auth & df.true_hard_negative.to_numpy()).any() else np.nan,
               der_precision=float(prec), der_recall=float(rec), der_f1=float(2 * prec * rec / max(1e-12, prec + rec)),
               lineage_correct=float((rt[g["root"]][der_true] == rt[der_true]).mean()) if der_true.any() else np.nan)
    for typ in sorted(DERIVED):
        m = ty == typ
        out[f"lineage_correct_{typ}"] = float((rt[g["root"]][m] == rt[m]).mean()) if m.any() else np.nan
    return out


def ece(p: np.ndarray, y: np.ndarray, bins=10) -> float:
    e = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum(abs(p[e == b].mean() - y[e == b].mean()) * (e == b).mean() for b in range(bins) if (e == b).any()))
