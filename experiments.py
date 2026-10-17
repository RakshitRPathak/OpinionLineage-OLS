"""Experiments E0-E6 of the paper. Each function returns pandas tables (and figure inputs)."""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

from . import lineage as lg
from . import models as md
from . import pipeline as pl
from . import quant as qt
from . import stream as st
from .weights import WitnessStore, lineage_arrays

METHODS = ["Raw aggregation", "Exact deduplication", "Near-duplicate clustering", "Synthetic-content filtering",
           "Bot-score weighting", "Combined heuristic pipeline", "Combined heuristic + PACC", "OLS, patent coefficients",
           "OLS (proposed)", "OLS, oracle lineage"]


def pool_nn(pool):
    v = TfidfVectorizer(sublinear_tf=True).fit_transform(pool.text.str.lower())
    _, i = NearestNeighbors(n_neighbors=2, metric="cosine").fit(v).kneighbors(v)
    return i[:, 1]


# --------------------------------------------------------------------------------------------------------------
def fit_components(pools, seed=0, log=print):
    rng = np.random.default_rng(seed)
    t0 = time.time()
    F = md.TextFeatures().fit(pools["clf"].text)
    sent, best, sent_tab, cm = md.sentiment_suite(F, pools)
    log(f"  sentiment suite done ({time.time() - t0:.0f}s), selected: {best}")
    psyn, syn_tab = md.synthetic_detector(pools, rng)
    P_val = sent.predict_proba(F.transform(pools["val"].text)); y_val = pools["val"].y.to_numpy()
    P_test = sent.predict_proba(F.transform(pools["test"].text)); y_test = pools["test"].y.to_numpy()
    M = qt.confusion(P_val, y_val)
    prior = np.bincount(pools["clf"].y, minlength=3) / len(pools["clf"])
    nn_cal, nn_test = pool_nn(pools["cal"]), pool_nn(pools["test"])
    pos = pools["clf"][pools["clf"].y == 2]
    src_cal, src_test = pos.iloc[: len(pos) // 2], pos.iloc[len(pos) // 2:]
    acc = st.Accounts(rng)
    # calibration streams (roots from the calibration pool, varied settings)
    cal_cfgs = [dict(kappa=k, rho_meta=r, h=h, paraphrase_strength=s) for k, r, h, s in
                [(0.2, 0.0, 0.2, 1.0), (0.4, 0.5, 0.2, 2.0), (0.6, 1.0, 0.4, 3.0), (0.4, 0.0, 0.0, 3.0), (0.5, 0.5, 0.4, 2.0),
                 (0.3, 0.0, 0.2, 1.0), (0.4, 1.0, 0.2, 2.0), (0.5, 0.0, 0.4, 3.0)]]
    cal = []
    for c in cal_cfgs:
        d = st.generate(st.StreamConfig(n_roots=1400, **c), pools["cal"], nn_cal, src_cal, acc, rng)
        Pp, burst, dens = lg.similarity_pass(d, F)
        cal.append(dict(df=d.assign(log_burst=np.log1p(burst)), pairs=Pp, dens=dens))
    coord = md.CoordinationScorer().fit(pd.concat([c["df"] for c in cal]))
    X, Y = [], []
    for c in cal:
        c["psyn"], c["pcoord"] = psyn(c["df"].text), coord(c["df"])
        c["pairs"] = lg.pair_features(c["df"], c["pairs"], c["dens"], c["psyn"], c["pcoord"])
        X.append(c["pairs"]); Y.append(lg.pair_labels(c["df"], c["pairs"]))
    edge = lg.EdgeModel().fit(pd.concat(X), np.concatenate(Y), seed=seed)
    log(f"  calibration streams + edge model done ({time.time() - t0:.0f}s)")
    comp = pl.Components(F, sent, psyn, coord, edge, M, prior, P_val, y_val)
    # tune alpha, beta, gamma (lineage weights) and tau_nd (heuristic baseline) on calibration streams only
    for c in cal:
        c["P"] = sent.predict_proba(F.transform(c["df"].text))
        c["g"] = pl.infer_graph(c, comp)
        c["truth"] = st.organic_truth(c["df"])
    grid = []
    for a, b, gm in itertools.product([0.5, 0.7, 0.8, 0.9, 0.95, 1.0], [0.0, 0.3, 0.6, 0.9, 1.0], [0.0, 0.35, 0.7, 1.0]):
        e = np.mean([qt.tvd(qt.pcc(lineage_arrays(c["g"], c["psyn"], c["pcoord"], a, b, gm)["W"], c["P"]), c["truth"]) for c in cal])
        grid.append(dict(alpha=a, beta=b, gamma=gm, cal_tvd=e))
    grid = pd.DataFrame(grid).sort_values("cal_tvd")
    comp.abg = tuple(grid.iloc[0][["alpha", "beta", "gamma"]])
    nd = []
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        e = np.mean([qt.tvd(qt.pcc(pl.baseline_weights(c, tau_nd=tau)[0]["Combined heuristic pipeline"], c["P"]), c["truth"]) for c in cal])
        nd.append(dict(tau_nd=tau, cal_tvd=e))
    nd = pd.DataFrame(nd).sort_values("cal_tvd")
    comp.extra.update(tau_nd=float(nd.tau_nd.iat[0]), grid=grid, nd_grid=nd, sent_tab=sent_tab, sent_best=best, cm=cm,
                      syn_tab=syn_tab, P_test=P_test, y_test=y_test, nn_test=nn_test, src_test=src_test, acc=acc,
                      cal_edge_auc=roc_auc_score(np.concatenate(Y), edge(pd.concat(X))))
    log(f"  tuned alpha,beta,gamma = {comp.abg}; tau_nd = {comp.extra['tau_nd']} ({time.time() - t0:.0f}s)")
    return comp


def make_stream(pools, comp, cfg, seed):
    return st.generate(cfg, pools["test"], comp.extra["nn_test"], comp.extra["src_test"], comp.extra["acc"],
                       np.random.default_rng(10_000 + seed))


def run_stream(df, comp, how="pcc"):
    obs = pl.observe(df, comp)
    g = pl.infer_graph(obs, comp)
    go = lg.oracle_graph(df)
    est = _with_tau(obs, g, comp, how, go, comp.extra.get("tau_nd", 0.8))
    truth = st.organic_truth(df)
    ro = qt.pcc(df.true_authentic.to_numpy().astype(float), obs["P"])
    return obs, g, go, est, truth, ro


def _with_tau(obs, g, comp, how, go, tau):
    bw, cl = pl.baseline_weights(obs, tau_nd=tau)
    obs["nd_clusters"] = cl
    obs["W_heur"] = bw["Combined heuristic pipeline"]
    P = obs["P"]
    out = {k: qt.pcc(w, P) for k, w in bw.items()}
    out["Combined heuristic + PACC"] = qt.pacc(qt.pcc(bw["Combined heuristic pipeline"], P), comp.M)
    out.update(pl.organic_estimates(obs, g, comp, how=how, g_oracle=go, include_baselines=False))
    return out


def _rows(est, truth, ro, extra):
    return [dict(method=k, tvd=qt.tvd(v, truth), prov_err=qt.tvd(v, ro), pos_err=abs(v[2] - truth[2]),
                 est_neg=v[0], est_neu=v[1], est_pos=v[2], **extra) for k, v in est.items()]


# --------------------------------------------------------------------------------------------------------------
def e2_contamination(pools, comp, seeds, kappas, log=print):
    rows, graph_rows, sci_rows, iom_rows = [], [], [], []
    for s, k in itertools.product(seeds, kappas):
        df = make_stream(pools, comp, st.StreamConfig(kappa=k), s)
        obs, g, go, est, truth, ro = run_stream(df, comp, how="pcc")
        rows += _rows(est, truth, ro, dict(seed=s, kappa=k, n=len(df)))
        graph_rows.append(dict(seed=s, kappa=k, n=len(df), **lg.evaluate_graph(df, g),
                               edge_auc=roc_auc_score(lg.pair_labels(df, obs["pairs"]), obs["pe"]) if k > 0 else np.nan,
                               edge_ece=lg.ece(obs["pe"], lg.pair_labels(df, obs["pairs"])) if k > 0 else np.nan,
                               coord_auc=roc_auc_score(df.true_campaign >= 0, obs["pcoord"]) if k > 0 else np.nan))
        sci_rows.append(dict(seed=s, kappa=k, sci_est=qt.tvd(est["Raw aggregation"], est["OLS (proposed)"]),
                             sci_true=qt.tvd(est["Raw aggregation"], ro)))
        n_auth = int(df.true_authentic.sum())
        iom_rows.append(dict(seed=s, kappa=k, n=len(df), n_auth=n_auth, iom_ols=float(obs["W_olq"].sum()),
                             iom_heur=float(obs.get("W_heur", np.ones(1)).sum()), amp_true=len(df) - n_auth))
        log(f"    E2 seed {s} kappa {k}: n={len(df)}  OLS prov {qt.tvd(est['OLS (proposed)'], ro):.4f}  "
            f"heur prov {qt.tvd(est['Combined heuristic pipeline'], ro):.4f}  raw prov {qt.tvd(est['Raw aggregation'], ro):.4f}")
    return pd.DataFrame(rows), pd.DataFrame(graph_rows), pd.DataFrame(sci_rows), pd.DataFrame(iom_rows)


def e3_robustness(pools, comp, seeds, log=print):
    rows = []
    for s, rho, h, strength in itertools.product(seeds, (0.0, 0.5, 1.0), (0.0, 0.2, 0.4), (1.0, 2.0, 3.0)):
        df = make_stream(pools, comp, st.StreamConfig(kappa=0.4, rho_meta=rho, h=h, paraphrase_strength=strength), 100 + s)
        obs, g, go, est, truth, ro = run_stream(df, comp)
        gm = lg.evaluate_graph(df, g)
        for m in ("Raw aggregation", "Near-duplicate clustering", "Combined heuristic pipeline", "OLS (proposed)", "OLS, oracle lineage"):
            rows.append(dict(seed=s, rho_meta=rho, h=h, strength=strength, method=m, prov_err=qt.tvd(est[m], ro),
                             tvd=qt.tvd(est[m], truth), root_acc=gm["root_acc"], der_f1=gm["der_f1"], lineage_correct=gm["lineage_correct"]))
        log(f"    E3 seed {s} rho {rho} h {h} strength {strength}: OLS prov {qt.tvd(est['OLS (proposed)'], ro):.4f} heur {qt.tvd(est['Combined heuristic pipeline'], ro):.4f}")
    return pd.DataFrame(rows)


def e4_ablation(pools, comp, seeds, log=print):
    rows = []
    for s in seeds:
        df = make_stream(pools, comp, st.StreamConfig(kappa=0.4), 200 + s)
        obs = pl.observe(df, comp)
        g = pl.infer_graph(obs, comp)
        go = lg.oracle_graph(df)
        truth, ro = st.organic_truth(df), qt.pcc(df.true_authentic.to_numpy().astype(float), obs["P"])
        ab = pl.ablations(obs, g, comp, go)
        ab["root oracle (true roots, same classifier)"] = ro
        for k, v in ab.items():
            rows.append(dict(seed=s, config=k, tvd=qt.tvd(v, truth), prov_err=qt.tvd(v, ro)))
        log(f"    E4 seed {s} done")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------------------
def campaign_config():
    return st.StreamConfig(n_roots=3000, kappa=0.45, families=("C",), seed_share=0.30, campaign_window=(10.0, 15.0),
                           inorganic_share=0.8, rho_meta=0.5, stealth_campaign=True, drip_days=1.5)


def e5_campaign(pools, comp, seeds, precisions=(1.0, 0.8, 0.6), discovery_day=20, log=print):
    """Stealthy campaign (unseen generator family C, stealth bots) discovered after the fact.
    Policies applied to the stored state at the discovery day:
      none               -- keep the as-published aggregate
      record-only removal-- zero the weight of flagged records only (what a filter applied post hoc would do)
      binary rollback    -- set flagged roots' authenticity to 0 when q >= 0.5 and propagate fully to D(v)
      confidence-weighted-- A' = A (1 - q), descendants corrected by q_{v,u} (Selective Dependency Rollback)"""
    a, b, gm = comp.abg
    rows, series = [], None
    for s in seeds:
        df = make_stream(pools, comp, campaign_config(), 300 + s)
        obs = pl.observe(df, comp)
        g = pl.infer_graph(obs, comp)
        P, iv = obs["P"], df.day.to_numpy()
        truth, ro = st.organic_truth(df), qt.pcc(df.true_authentic.to_numpy().astype(float), P)
        est_root = g["root"]
        seeds_true = np.where(df.true_type.eq("synthetic_root"))[0]
        auth_roots = np.where(df.true_authentic.to_numpy() & (g["parent"] < 0))[0]
        for al, prec in itertools.product(sorted({a, 0.7}, reverse=True), precisions):
            arr = lineage_arrays(g, obs["psyn"], obs["pcoord"], al, b, gm)
            rng = np.random.default_rng(500 + s + int(1000 * prec))
            # flagged lineage roots: estimated roots of true campaign seeds (recall 0.9) plus false flags
            hit = np.unique(est_root[seeds_true[rng.random(len(seeds_true)) < 0.9]])
            n_fp = int(round(len(hit) * (1 - prec) / prec))
            fp = rng.choice(auth_roots, n_fp, replace=False) if n_fp else np.array([], int)
            flagged = np.r_[hit, fp].astype(int)
            q = np.r_[rng.beta(8, 2, len(hit)), rng.beta(4, 4, n_fp)]
            res = {}
            base = WitnessStore(arr, P, iv, 30)
            res["none"] = (base.distribution(), 0)
            auth = df.true_authentic.to_numpy()
            W0a = arr["W"][auth].sum()
            lost = {"none": 0.0}
            # record-only removal
            W2 = arr["W"].copy(); W2[flagged[q >= 0.5]] = 0
            res["record-only removal"] = (qt.pcc(W2, P), int((q >= 0.5).sum()))
            lost["record-only removal"] = 1 - W2[auth].sum() / W0a
            for pol in ("binary rollback", "confidence-weighted rollback"):
                S = WitnessStore(arr, P, iv, 30)
                touched = 0
                for v, qq in zip(flagged, q):
                    if S.parent[v] >= 0:
                        continue
                    if pol == "binary rollback":
                        if qq >= 0.5:
                            touched += S.revise_root_authenticity(v, 0.0, partial=False)
                    else:
                        touched += S.revise_root_authenticity(v, S.A[v] * (1 - qq), partial=True)
                Agg, Z = S.full_resum()
                exact = float(np.abs(S.distribution() - Agg.sum(0) / Z.sum()).max())
                res[pol] = (S.distribution(), touched, exact, S)
                lost[pol] = 1 - S.W[auth].sum() / W0a
            for pol, r in res.items():
                rows.append(dict(seed=s, alpha=al, precision=prec, policy=pol, tvd=qt.tvd(r[0], truth), prov_err=qt.tvd(r[0], ro),
                                 pos_err=abs(r[0][2] - truth[2]), touched=r[1], n=len(df), n_flagged=len(flagged),
                                 max_abs_diff_vs_resum=r[2] if len(r) > 2 else np.nan, authentic_mass_lost=lost[pol]))
            if s == seeds[0] and prec == 1.0 and al == a:
                raw = np.zeros((30, 3)); np.add.at(raw, iv, P)
                tr = np.zeros(30); np.add.at(tr, iv[df.true_authentic], 1)
                tp = np.zeros(30); np.add.at(tp, iv[df.true_authentic], (df.true_y[df.true_authentic] == 2).to_numpy())
                Sb, Sc = base, res["confidence-weighted rollback"][3]
                series = pd.DataFrame(dict(day=np.arange(30), true_pos=tp / np.maximum(1, tr),
                                           raw_pos=raw[:, 2] / raw.sum(1),
                                           published_pos=Sb.Agg[:, 2] / Sb.Z, corrected_pos=Sc.Agg[:, 2] / Sc.Z,
                                           heur_pos=_daily(pl.baseline_weights(obs, comp.extra["tau_nd"])[0]["Combined heuristic pipeline"], P, iv),
                                           root_oracle_pos=_daily(df.true_authentic.to_numpy().astype(float), P, iv),
                                           n_records=np.bincount(iv, minlength=30), n_campaign=np.bincount(iv[df.true_campaign >= 0], minlength=30)))
        log(f"    E5 seed {s} done (n={len(df)})")
    return pd.DataFrame(rows), series


def _daily(W, P, iv):
    A = np.zeros((30, 3)); Z = np.zeros(30)
    np.add.at(A, iv, W[:, None] * P); np.add.at(Z, iv, W)
    return A[:, 2] / np.maximum(Z, 1e-12)


# --------------------------------------------------------------------------------------------------------------
def synthetic_forest(n, rng, root_share=0.2, alpha=1.5):
    """Time-ordered forest with Pareto-distributed cascade sizes and preferential depth (scaling experiments)."""
    n_root = int(n * root_share)
    parent = -np.ones(n, int)
    is_root = np.zeros(n, bool)
    is_root[rng.choice(np.arange(1, n), n_root - 1, replace=False)] = True
    is_root[0] = True
    roots_so_far = np.cumsum(is_root) - 1
    root_ids = np.where(is_root)[0]
    vir = rng.pareto(alpha, n_root) + 1
    cum = np.cumsum(vir)
    last_member = root_ids.copy()
    for i in np.where(~is_root)[0]:
        k = roots_so_far[i]
        r = np.searchsorted(cum[: k + 1], rng.random() * cum[k])
        parent[i] = root_ids[r] if rng.random() < 0.6 else last_member[r]
        last_member[r] = i
    c = np.where(parent >= 0, rng.beta(8, 2, n), 0.0)
    pind = np.where(parent < 0, rng.beta(9, 1, n), 1.0)
    root = np.arange(n)
    for i in range(n):
        if parent[i] >= 0:
            root[i] = root[parent[i]]
    P = rng.dirichlet(np.ones(3), n)
    return dict(parent=parent, c=c, pind=pind, root=root), P, rng.beta(1, 9, n), rng.beta(1, 9, n)


def e6_rollback_scaling(sizes, inference_s_per_record, rng_seed=0, frac=0.02, log=print):
    rows = []
    for n in sizes:
        rng = np.random.default_rng(rng_seed + n)
        g, P, ps, pc = synthetic_forest(n, rng)
        arr = lineage_arrays(g, ps, pc, 0.9, 0.6, 0.35)
        iv = rng.integers(0, 30, n)
        S = WitnessStore(arr, P, iv, 30)
        sizes_sub = np.bincount(arr["root"], minlength=n)
        roots = np.where(g["parent"] < 0)[0]
        order = rng.permutation(roots)
        tgt, acc_sz = [], 0
        for r in order:                                      # revise whole lineages until ~frac of the corpus is affected
            if acc_sz >= frac * n:
                break
            tgt.append(r); acc_sz += sizes_sub[r]
        newA = {r: S.A[r] * 0.1 for r in tgt}
        t0 = time.perf_counter()
        touched = sum(S.revise_root_authenticity(r, newA[r], partial=False) for r in tgt)
        t_sel = time.perf_counter() - t0
        t0 = time.perf_counter(); Agg, Z = S.full_resum(); t_resum = time.perf_counter() - t0
        pind2 = g["pind"].copy()
        for r in tgt:
            pind2[r] = newA[r] / arr["F"][r]
        t0 = time.perf_counter()
        arr2 = lineage_arrays(dict(g, pind=pind2), ps, pc, 0.9, 0.6, 0.35)
        Agg2 = np.zeros((30, 3)); Z2 = np.zeros(30)
        np.add.at(Agg2, iv, arr2["W"][:, None] * P); np.add.at(Z2, iv, arr2["W"])
        t_rebuild = time.perf_counter() - t0
        d_resum = float(np.abs(S.distribution() - Agg.sum(0) / Z.sum()).max())
        d_rebuild = float(np.abs(S.distribution() - Agg2.sum(0) / Z2.sum()).max())
        rows.append(dict(n=n, revisions=len(tgt), touched=touched, touched_frac=touched / n, t_selective_s=t_sel,
                         t_resum_s=t_resum, t_rebuild_s=t_rebuild, t_full_reprocess_est_s=t_rebuild + n * inference_s_per_record,
                         max_diff_vs_resum=d_resum, max_diff_vs_rebuild=d_rebuild, model_inferences_selective=0,
                         model_inferences_full=2 * n))
        log(f"    E6 n={n}: touched {touched} ({touched / n:.3%}), selective {t_sel * 1e3:.1f} ms, rebuild {t_rebuild:.2f} s, diff {d_rebuild:.2e}")
    return pd.DataFrame(rows)


def e6_drift(n=100_000, steps=100_000, every=5_000, seed=1, log=print):
    rows = []
    for kahan in (False, True):
        rng = np.random.default_rng(seed)
        g, P, ps, pc = synthetic_forest(n, rng)
        arr = lineage_arrays(g, ps, pc, 0.9, 0.6, 0.35)
        iv = rng.integers(0, 30, n)
        S = WitnessStore(arr, P, iv, 30, kahan=kahan)
        roots = np.where(g["parent"] < 0)[0]; der = np.where(g["parent"] >= 0)[0]
        for k in range(1, steps + 1):
            r = rng.random()
            if r < 0.5:
                v = roots[rng.integers(len(roots))]
                S.revise_root_authenticity(v, float(rng.random()), partial=bool(rng.random() < 0.5))
            elif r < 0.8:
                v = der[rng.integers(len(der))]
                S.revise_edge_confidence(v, float(rng.beta(5, 2)), 0.9)
            else:
                v = der[rng.integers(len(der))]
                S.revise_own(v, float(rng.random()))
            if k % every == 0:
                Ax, Zx = _exact_agg(S)
                rows.append(dict(kahan=kahan, step=k, max_abs_agg=float(np.abs(S.Agg - Ax).max()),
                                 max_rel_agg=float((np.abs(S.Agg - Ax) / np.abs(Ax)).max()),
                                 max_abs_dist=float(np.abs(S.distribution() - Ax.sum(0) / Zx.sum()).max()),
                                 touched_total=S.touched_total))
        log(f"    E6 drift kahan={kahan}: final max |dA| = {rows[-1]['max_abs_agg']:.2e}")
    return pd.DataFrame(rows)


def _exact_agg(S):
    """Correctly rounded per-cell sums (math.fsum) of the current witness contributions: the drift reference."""
    import math
    A = np.zeros_like(S.Agg); Z = np.zeros_like(S.Z)
    for k in range(S.n_int):
        m = S.iv == k
        w = S.W[m]
        Z[k] = math.fsum(w.tolist())
        for c in range(S.P.shape[1]):
            A[k, c] = math.fsum((w * S.P[m, c]).tolist())
    return A, Z


def e6_real_scale(pools, comp, n_roots=10_000, kappa=0.8, seed=7, frac=0.02, log=print):
    """Patent-scale real-text stream (~50,000 records): measured full reprocessing vs selective rollback."""
    df = make_stream(pools, comp, st.StreamConfig(n_roots=n_roots, kappa=kappa), 900 + seed)
    t0 = time.perf_counter()
    obs = pl.observe(df, comp)
    g = pl.infer_graph(obs, comp)
    t_obs = time.perf_counter() - t0
    t0 = time.perf_counter()
    P = comp.sent.predict_proba(comp.F.transform(df.text))
    t_inf = time.perf_counter() - t0
    a, b, gm = comp.abg
    arr = lineage_arrays(g, obs["psyn"], obs["pcoord"], a, b, gm)
    S = WitnessStore(arr, P, df.day.to_numpy(), 30)
    size = np.bincount(arr["root"], minlength=len(df))
    roots = np.random.default_rng(seed).permutation(np.where(g["parent"] < 0)[0])
    tgt, acc_sz = [], 0
    for r in roots:
        if acc_sz >= frac * len(df):
            break
        tgt.append(r); acc_sz += size[r]
    t0 = time.perf_counter()
    touched = sum(S.revise_root_authenticity(r, 0.0, partial=False) for r in tgt)
    t_sel = time.perf_counter() - t0
    pind2 = g["pind"].copy(); pind2[tgt] = 0.0
    arr2 = lineage_arrays(dict(g, pind=pind2), obs["psyn"], obs["pcoord"], a, b, gm)
    full = qt.pcc(arr2["W"], P)
    out = dict(n=len(df), revised_roots=len(tgt), touched=touched, touched_frac=touched / len(df), t_selective_s=t_sel,
               t_full_reprocess_s=t_obs + t_inf, t_inference_only_s=t_inf, inference_s_per_record=t_inf / len(df),
               max_diff=float(np.abs(S.distribution() - full).max()), model_inferences_selective=0, model_inferences_full=2 * len(df))
    log(f"    E6 real stream n={len(df)}: touched {touched}, selective {t_sel * 1e3:.1f} ms vs full {t_obs + t_inf:.1f} s, diff {out['max_diff']:.2e}")
    return pd.DataFrame([out]), df, obs, g
