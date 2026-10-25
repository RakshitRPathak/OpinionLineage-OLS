"""Manuscript figures (300 dpi PNG + PDF)."""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
GRAY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.6, "axes.edgecolor": MUTED,
    "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "figure.facecolor": "white"})
SINGLE, DOUBLE = 3.5, 7.16
STYLE = {
    "Raw aggregation": (GRAY, "-", "o"),
    "Exact deduplication": (YELLOW, "-", "s"),
    "Near-duplicate clustering": (AQUA, "-", "^"),
    "Synthetic-content filtering": (MAGENTA, "-", "v"),
    "Bot-score weighting": (VIOLET, "-", "D"),
    "Combined heuristic pipeline": (BLUE, "-", "P"),
    "OLS, patent coefficients": (GREEN, "--", "x"),
    "OLS (proposed)": (ORANGE, "-", "o"),
    "OLS, oracle lineage": (RED, ":", None),
}


def _save(fig, out, name):
    fig.savefig(os.path.join(out, f"{name}.png")); fig.savefig(os.path.join(out, f"{name}.pdf")); plt.close(fig)


def fig_contamination(out, R, IOM):
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, 2.35), gridspec_kw=dict(wspace=0.38))
    for m, (c, ls, mk) in STYLE.items():
        d = R[R.method == m].groupby("kappa")[["tvd", "prov_err"]].agg(["mean", "std"])
        for ax, col in zip(axs[:2], ("tvd", "prov_err")):
            mu, sd = d[(col, "mean")], d[(col, "std")]
            if col == "prov_err":                 # log axis: drop exact zeros (raw counting at kappa = 0)
                keep = mu > 1e-5
                mu, sd = mu[keep], sd[keep].clip(upper=mu[keep] * 0.95)
            ax.errorbar(mu.index * 100, mu, yerr=sd, color=c, ls=ls, marker=mk, ms=2.8, lw=1.2,
                        capsize=1.2, elinewidth=0.6, label=m)
    floor = R[R.method == "Raw aggregation"].assign(f=lambda x: x.tvd - x.prov_err)
    axs[0].set_xlabel("Contamination kappa [%]"); axs[0].set_ylabel("TVD to organic truth")
    axs[1].set_yscale("log"); axs[1].set_xlabel("Contamination kappa [%]"); axs[1].set_ylabel("Provenance error (TVD to root oracle)")
    g = IOM.groupby("kappa")
    k = g.n_auth.mean().index * 100
    axs[2].plot(k, g.n.mean(), color=GRAY, marker="o", ms=2.8, lw=1.2, label="Records N")
    axs[2].plot(k, g.n_auth.mean(), color=INK, ls=":", lw=1.2, label="True authentic roots")
    axs[2].plot(k, g.iom_heur.mean(), color=BLUE, marker="P", ms=2.8, lw=1.2, label="Heuristic weight mass")
    axs[2].plot(k, g.iom_ols.mean(), color=ORANGE, marker="o", ms=2.8, lw=1.2, label="OLS independent opinion mass")
    axs[2].set_xlabel("Contamination kappa [%]"); axs[2].set_ylabel("Count / mass"); axs[2].legend(frameon=False, loc="upper left")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 0.98), fontsize=6.2)
    _save(fig, out, "fig_contamination")


def fig_robustness(out, RB):
    strengths = sorted(RB.strength.unique())
    meths = [("OLS (proposed)", "OLS"), ("Combined heuristic pipeline", "Heuristic")]
    fig, axs = plt.subplots(2, len(strengths), figsize=(DOUBLE * 0.8, 3.0), gridspec_kw=dict(wspace=0.1, hspace=0.45), squeeze=False)
    mats = {(m, s): RB[(RB.method == m) & (RB.strength == s)].groupby(["rho_meta", "h"]).prov_err.mean().unstack()
            for m, _ in meths for s in strengths}
    vmax = max(x.to_numpy().max() for x in mats.values())
    for r, (m, lab) in enumerate(meths):
        for cidx, s in enumerate(strengths):
            ax, d = axs[r, cidx], mats[(m, s)]
            im = ax.imshow(d.to_numpy() * 100, cmap="Blues", vmin=0, vmax=vmax * 100, aspect="auto")
            for (i, j), v in np.ndenumerate(d.to_numpy() * 100):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6, color="white" if v > 0.6 * vmax * 100 else INK)
            ax.set_xticks(range(d.shape[1]), [f"{x:.1f}" for x in d.columns])
            ax.set_yticks(range(d.shape[0]), [f"{x:.1f}" for x in d.index] if cidx == 0 else [])
            if r == 1:
                ax.set_xlabel("Hard-negative share h")
            if cidx == 0:
                ax.set_ylabel(f"{lab}\nmetadata exposure rho")
            ax.set_title(f"{lab}, paraphrase s = {s:g}", fontsize=7)
            ax.grid(False)
    cb = fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02); cb.set_label("Provenance error x100", fontsize=6.5)
    _save(fig, out, "fig_robustness")


def fig_ablation(out, AB):
    g = AB.groupby("config").prov_err.agg(["mean", "std"])
    order = ["root oracle (true roots, same classifier)", "oracle lineage", "Full OLS (lineage + synthetic + coordination)",
             "without synthetic term (beta = 0)", "without coordination term (gamma = 0)", "incident edge only (no root inheritance)",
             "alpha = 1.0", "alpha = 0.9", "alpha = 0.7", "alpha = 0.5", "without lineage (message-level weighting)",
             "+ PACC label-shift adjustment", "+ EM prior adjustment"]
    order = [o for o in order if o in g.index]
    g = g.reindex(order)
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    col = [ORANGE if o.startswith("Full") else (RED if "PACC" in o or "EM" in o else (GRAY if "oracle" in o else BLUE)) for o in order]
    ax.barh(range(len(order)), g["mean"] * 100, xerr=g["std"] * 100, color=col, height=0.7, error_kw=dict(lw=0.6, capsize=1.2))
    ax.set_yticks(range(len(order)), order, fontsize=6.2); ax.invert_yaxis(); ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("Provenance error x100 (kappa = 0.4)"); ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_ablation")


def fig_campaign(out, series, CP, alpha, discovery_day=20, window=(10, 15)):
    CP = CP[np.isclose(CP.alpha, alpha)]
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.3), gridspec_kw=dict(width_ratios=[1.6, 1], wspace=0.3))
    ax = axs[0]
    ax.axvspan(*window, color="#fde3d8", alpha=0.7, lw=0)
    ax.text(window[0] + 0.2, 0.02, "campaign posted", transform=ax.get_xaxis_transform(), fontsize=6, color=ORANGE, va="bottom")
    ax.axvline(discovery_day, color=RED, ls="--", lw=0.8)
    ax.text(discovery_day + 0.3, 0.02, "discovered", transform=ax.get_xaxis_transform(), fontsize=6, color=RED, va="bottom")
    d = series
    ax.plot(d.day, d.root_oracle_pos * 100, color=INK, ls=":", lw=1.2, label="Root oracle (same classifier)")
    ax.plot(d.day, d.raw_pos * 100, color=GRAY, lw=1.2, label="Raw aggregation")
    ax.plot(d.day, d.heur_pos * 100, color=BLUE, lw=1.2, label="Combined heuristic")
    ax.plot(d.day, d.published_pos * 100, color=GREEN, lw=1.2, ls="--", label="OLS as published")
    ax.plot(d.day, d.corrected_pos * 100, color=ORANGE, lw=1.6, label="OLS after rollback")
    ax.set_xlabel("Day"); ax.set_ylabel("Positive share [%]")
    ax.legend(frameon=False, fontsize=5.8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax = axs[1]
    pols = ["none", "record-only removal", "binary rollback", "confidence-weighted rollback"]
    cols = [GRAY, YELLOW, BLUE, ORANGE]
    precs = sorted(CP.precision.unique(), reverse=True)
    w = 0.2
    for k, (p, c) in enumerate(zip(pols, cols)):
        g = CP[CP.policy == p].groupby("precision").prov_err.agg(["mean", "std"]).reindex(precs)
        ax.bar(np.arange(len(precs)) + (k - 1.5) * w, g["mean"] * 100, w * 0.92, yerr=g["std"] * 100, color=c, label=p,
               error_kw=dict(lw=0.6, capsize=1))
    ax.set_xticks(range(len(precs)), [f"{p:.1f}" for p in precs]); ax.set_xlabel("Precision of discovery flags")
    ax.set_ylabel("Provenance error x100"); ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, fontsize=5.6, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    _save(fig, out, "fig_campaign")


def fig_scaling(out, SC, DR):
    fig, axs = plt.subplots(1, 2, figsize=(DOUBLE, 2.2), gridspec_kw=dict(wspace=0.3))
    ax = axs[0]
    for col, lab, c, mk in [("t_selective_s", "Selective rollback (measured)", ORANGE, "o"), ("t_resum_s", "Re-summation of all witnesses", AQUA, "s"),
                            ("t_rebuild_s", "Weight rebuild + re-summation", BLUE, "^"),
                            ("t_full_reprocess_est_s", "Full reprocessing incl. inference (est.)", GRAY, "D")]:
        ax.plot(SC.n, SC[col], color=c, marker=mk, ms=3, lw=1.3, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("Corpus size N (records)"); ax.set_ylabel("Wall-clock time [s]")
    ax.legend(frameon=False, fontsize=5.8)
    ax = axs[1]
    for k, c, lab in [(False, GRAY, "Plain float64 deltas"), (True, ORANGE, "Kahan-compensated deltas")]:
        d = DR[DR.kahan == k]
        ax.plot(d.step, np.maximum(d.max_rel_agg, 1e-18), color=c, marker="o", ms=2.5, lw=1.3, label=lab)
    ax.axhline(np.finfo(float).eps, color=INK, ls=":", lw=0.8); ax.text(DR.step.min(), np.finfo(float).eps * 1.3, "float64 machine epsilon", fontsize=6)
    ax.set_yscale("log"); ax.set_xlabel("Sequential revision events"); ax.set_ylabel("Max relative error vs exact sum")
    ax.legend(frameon=False, fontsize=6)
    _save(fig, out, "fig_scaling")


def fig_example(out, df, comp):
    """Estimated lineage of a slice of the released stream: organic cascade and campaign cascade, edges coloured by
    whether the estimated root matches the true lineage."""
    from . import pipeline as pl
    obs = pl.observe(df, comp)
    g = pl.infer_graph(obs, comp)
    root = g["root"]
    size = pd.Series(root).value_counts()
    camp_roots = [r for r in size.index if df.true_type.iat[r] == "synthetic_root"][:1]
    org_roots = [r for r in size.index if df.true_type.iat[r] == "root"][:2]
    sel = org_roots + camp_roots
    fig, ax = plt.subplots(figsize=(DOUBLE, 2.4))
    tcol = {"root": INK, "synthetic_root": RED, "repost": GRAY, "quote": YELLOW, "paraphrase": AQUA, "ai_rewrite": VIOLET,
            "coord_copy": MAGENTA}
    y0 = 0
    for r in sel:
        mem = np.where(root == r)[0][:40]
        ys = {}
        for k, i in enumerate(mem):
            ys[i] = y0 + (k % 8) * 0.12
        for i in mem:
            p = g["parent"][i]
            if p >= 0 and p in ys:
                ok = df.true_root.iat[i] == df.true_root.iat[r]
                ax.plot([df.t.iat[p], df.t.iat[i]], [ys[p], ys[i]], color=GREEN if ok else RED, lw=0.5 + 1.2 * g["c"][i], alpha=0.8, zorder=1)
        for i in mem:
            ax.scatter(df.t.iat[i], ys[i], s=26 if i == r else 12, color=tcol[df.true_type.iat[i]], zorder=3, edgecolor="white", lw=0.3)
        lab = "campaign seed (synthetic root)" if df.true_type.iat[r] == "synthetic_root" else "authentic root"
        ax.text(df.t.iat[r], y0 + 0.98, f"{lab}: \"{df.text.iat[r][:48]}...\"", fontsize=5.4, color=INK)
        y0 += 1.25
    for t, c in tcol.items():
        ax.scatter([], [], color=c, s=14, label=t.replace("_", " "))
    ax.plot([], [], color=GREEN, lw=1, label="edge into correct lineage"); ax.plot([], [], color=RED, lw=1, label="edge into wrong lineage")
    ax.legend(frameon=False, ncol=1, fontsize=5.6, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_yticks([]); ax.set_xlabel("Time [days]  (edge width ~ derivation confidence c)"); ax.grid(axis="y", visible=False)
    _save(fig, out, "fig_example_lineage")


def make_all(out, D):
    fig_contamination(out, D["R"], D["IOM"])
    fig_robustness(out, D["RB"])
    fig_ablation(out, D["AB"])
    fig_campaign(out, D["series"], D["CP"], D["alpha"])
    fig_scaling(out, D["SC"], D["DR"])
    if D.get("example") is not None:
        df, comp = D["example"]
        fig_example(out, df, comp)
