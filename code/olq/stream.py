"""Contaminated opinion-stream generator over real TweetEval texts.

A stream contains
  * authentic independent roots  -- real TweetEval tweets written by distinct human accounts (the organic opinions);
    a fraction h of them are *hard negatives*: real tweets chosen as the nearest neighbour of another root, i.e.
    independent people saying nearly the same thing;
  * organic amplification        -- human reposts, quotes and paraphrases of authentic roots, with heavy-tailed
    (Pareto) virality that favours one sentiment class (selective amplification);
  * an inorganic campaign        -- synthetic roots (rule-based AI rewrites of on-message source tweets, families
    A/B/C) posted by bot accounts, followed by coordinated near-copies, further AI rewrites and reposts, some by
    humans who unknowingly amplify the campaign.

The estimand is the sentiment distribution of the authentic roots. Every derivative inherits its parent's sentiment.
Repost/quote metadata (the parent id) is exposed only with probability rho_meta; all other provenance must be
inferred. Ground-truth lineage is kept in columns prefixed ``true_`` and is never used by the estimators.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import textops as tx

TYPES_ORG = (("repost", 0.45), ("quote", 0.20), ("paraphrase", 0.35))
TYPES_INORG = (("coord_copy", 0.45), ("ai_rewrite", 0.30), ("repost", 0.25))


@dataclass
class StreamConfig:
    n_roots: int = 2500
    kappa: float = 0.4            # fraction of records that are NOT authentic roots
    rho_meta: float = 0.5         # probability that a repost/quote exposes its parent id
    h: float = 0.2                # hard-negative fraction among authentic roots
    inorganic_share: float = 0.6  # share of contamination that belongs to the campaign
    bias_class: int = 2           # class favoured by the campaign and by selective amplification (positive)
    amp_bias: float = 2.5         # virality multiplier of bias-class roots
    paraphrase_strength: float = 2.0
    seed_share: float = 0.06      # campaign seeds as a share of inorganic records
    families: tuple = ("A", "B", "C")
    days: float = 30.0
    campaign_window: tuple | None = None   # (start, end) day range for campaign seeds; None = whole horizon
    stealth_campaign: bool = False         # campaign uses only human-like (stealth) bot accounts
    drip_days: float = 0.15                # coordinated copies are spread over U(0, drip_days) after their seed


class Accounts:
    """Human and bot account populations with observable features (30% of bots are stealthy / human-like)."""

    def __init__(self, rng: np.random.Generator, n_human=20000, n_bot=600, stealth=0.3):
        nh, nb = n_human, n_bot
        st = rng.random(nb) < stealth
        age = np.r_[rng.normal(7.0, 1.0, nh), np.where(st, rng.normal(6.5, 1.0, nb), rng.normal(4.0, 1.0, nb))]
        ppd = np.r_[rng.normal(1.0, 1.0, nh), np.where(st, rng.normal(1.4, 1.0, nb), rng.normal(3.0, 0.7, nb))]
        reg = np.r_[rng.beta(2, 5, nh), np.where(st, rng.beta(2.5, 4, nb), rng.beta(5, 2, nb))]
        dfl = np.r_[rng.random(nh) < 0.1, np.where(st, rng.random(nb) < 0.15, rng.random(nb) < 0.6)]
        self.feat = pd.DataFrame(dict(acc_log_age=age, acc_log_ppd=ppd, acc_regularity=reg, acc_default=dfl.astype(float)))
        self.is_bot = np.r_[np.zeros(nh, bool), np.ones(nb, bool)]
        self.human_ids = np.arange(nh)
        self.bot_ids = np.arange(nh, nh + nb)
        self.stealth_ids = self.bot_ids[st]


def _pick(rng, choices):
    names, p = zip(*choices)
    return names[rng.choice(len(names), p=np.array(p) / sum(p))]


def generate(cfg: StreamConfig, roots: pd.DataFrame, root_nn: np.ndarray, source: pd.DataFrame,
             acc: Accounts, rng: np.random.Generator) -> pd.DataFrame:
    """roots/source: DataFrames with text,y. root_nn[i] = index of the nearest other tweet of roots."""
    R = []

    def add(**kw):
        R.append(kw)
        return len(R) - 1

    # ---- authentic roots (with hard negatives) -------------------------------------------------------------
    n_hard = int(round(cfg.h * cfg.n_roots))
    base = rng.choice(len(roots), cfg.n_roots - n_hard, replace=False)
    chosen = list(base)
    used = set(base.tolist())
    tries = 0
    while len(chosen) < cfg.n_roots and tries < 50 * cfg.n_roots:
        tries += 1
        j = int(root_nn[chosen[rng.integers(len(chosen))]])
        if j not in used:
            used.add(j)
            chosen.append(j)
    while len(chosen) < cfg.n_roots:                 # pool exhausted of fresh neighbours: fill at random
        j = int(rng.integers(len(roots)))
        if j not in used:
            used.add(j); chosen.append(j)
    hard_flag = np.r_[np.zeros(cfg.n_roots - n_hard, bool), np.ones(cfg.n_roots - (cfg.n_roots - n_hard), bool)]
    humans = rng.choice(acc.human_ids, cfg.n_roots, replace=False)
    auth = []
    for k, j in enumerate(chosen):
        auth.append(add(t=rng.uniform(0, cfg.days), account=int(humans[k]), text=roots.text.iat[j], y=int(roots.y.iat[j]),
                        true_type="root", true_parent=-1, hard_negative=bool(hard_flag[k]), meta_parent=-1, family="",
                        campaign=-1, src=int(j)))
    for i in auth:
        R[i]["true_root"] = i

    n_c = int(round(cfg.kappa / (1 - cfg.kappa) * cfg.n_roots)) if cfg.kappa > 0 else 0
    n_inorg = int(round(cfg.inorganic_share * n_c))
    n_org = n_c - n_inorg

    def derive(parent, typ, account, family="", campaign=-1, dt=None):
        p = R[parent]
        text = p["text"]
        if typ == "repost":
            text = tx.repost(text, rng)
        elif typ == "quote":
            text = tx.quote(text, rng)
        elif typ == "paraphrase":
            text = tx.human_paraphrase(text, rng, cfg.paraphrase_strength)
        elif typ == "ai_rewrite":
            text = tx.ai_rewrite(text, rng, family)
        elif typ == "coord_copy":
            text = tx.coordinated_copy(text, rng)
        meta = parent if typ in ("repost", "quote") and rng.random() < cfg.rho_meta else -1
        t = p["t"] + (rng.exponential(0.5) if dt is None else dt)
        i = add(t=t, account=int(account), text=text, y=p["y"], true_type=typ, true_parent=parent, hard_negative=False,
                meta_parent=meta, family=family, campaign=campaign, src=-1)
        R[i]["true_root"] = p["true_root"]
        return i

    # ---- organic amplification ------------------------------------------------------------------------------
    if n_org:
        yv = np.array([R[i]["y"] for i in auth])
        vir = (rng.pareto(1.5, len(auth)) + 1.0) * np.where(yv == cfg.bias_class, cfg.amp_bias, 1.0)
        lineage = {i: [] for i in auth}
        tgt = rng.choice(len(auth), n_org, p=vir / vir.sum())
        for k in tgt:
            r = auth[k]
            par = r if (not lineage[r] or rng.random() < 0.6) else lineage[r][rng.integers(len(lineage[r]))]
            i = derive(par, _pick(rng, TYPES_ORG), rng.choice(acc.human_ids))
            lineage[r].append(i)

    # ---- inorganic campaign -------------------------------------------------------------------------------
    if n_inorg:
        src_pos = source.index[source.y == cfg.bias_class].to_numpy()
        n_seed = max(1, int(round(cfg.seed_share * n_inorg)))
        lo, hi = cfg.campaign_window if cfg.campaign_window else (0.0, cfg.days)
        seeds, members, fam = [], {}, {}
        bots = acc.stealth_ids if cfg.stealth_campaign else acc.bot_ids
        for s in range(n_seed):
            j = int(rng.choice(src_pos))
            f = cfg.families[rng.integers(len(cfg.families))]
            i = add(t=rng.uniform(lo, hi), account=int(rng.choice(bots)), text=tx.ai_rewrite(source.text.at[j], rng, f),
                    y=int(source.y.at[j]), true_type="synthetic_root", true_parent=-1, hard_negative=False, meta_parent=-1,
                    family=f, campaign=s, src=-1)
            R[i]["true_root"] = i
            seeds.append(i); members[i] = [i]; fam[i] = f
        for _ in range(n_inorg - n_seed):
            sd = seeds[rng.integers(n_seed)]
            typ = _pick(rng, TYPES_INORG)
            if typ == "coord_copy":
                derive(sd, typ, rng.choice(bots), campaign=R[sd]["campaign"], dt=rng.uniform(0.0, cfg.drip_days))
                members[sd].append(len(R) - 1)
            elif typ == "ai_rewrite":
                par = members[sd][rng.integers(len(members[sd]))]
                f = fam[sd] if rng.random() < 0.7 else cfg.families[rng.integers(len(cfg.families))]
                derive(par, typ, rng.choice(bots), family=f, campaign=R[sd]["campaign"], dt=rng.exponential(0.3))
                members[sd].append(len(R) - 1)
            else:
                par = members[sd][rng.integers(len(members[sd]))]
                a = rng.choice(acc.human_ids) if rng.random() < 0.5 else rng.choice(bots)
                derive(par, typ, a, campaign=R[sd]["campaign"])
                members[sd].append(len(R) - 1)

    # ---- order by time, re-index ---------------------------------------------------------------------------
    df = pd.DataFrame(R)
    order = np.argsort(df.t.to_numpy(), kind="stable")
    remap = np.empty(len(df), int)
    remap[order] = np.arange(len(df))
    df = df.iloc[order].reset_index(drop=True)
    for c in ("true_parent", "meta_parent", "true_root"):
        v = df[c].to_numpy()
        df[c] = np.where(v >= 0, remap[np.maximum(v, 0)], -1)
    df.insert(0, "id", np.arange(len(df)))
    df["is_bot_account"] = acc.is_bot[df.account.to_numpy()]
    df["true_authentic"] = df.true_type.eq("root")
    df["day"] = np.floor(df.t).astype(int).clip(0, int(cfg.days) - 1)
    df = pd.concat([df, acc.feat.iloc[df.account.to_numpy()].reset_index(drop=True)], axis=1)
    return df.rename(columns={"y": "true_y", "hard_negative": "true_hard_negative", "family": "true_family",
                              "campaign": "true_campaign", "is_bot_account": "true_bot"})


def organic_truth(df: pd.DataFrame, k: int = 3) -> np.ndarray:
    y = df.loc[df.true_authentic, "true_y"].to_numpy()
    return np.bincount(y, minlength=k) / len(y)
