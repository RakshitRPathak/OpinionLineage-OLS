"""TweetEval sentiment loading with checksum verification and the fixed data partition used throughout."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pandas as pd

LABELS = ["negative", "neutral", "positive"]
K = 3


def _verify(folder: str) -> None:
    sums = os.path.join(folder, "SHA256SUMS")
    for line in open(sums):
        digest, name = line.split()
        h = hashlib.sha256(open(os.path.join(folder, name), "rb").read()).hexdigest()
        if h != digest:
            raise RuntimeError(f"checksum mismatch for {name}: TweetEval files differ from the reference copy")


def _read(folder: str, split: str) -> pd.DataFrame:
    with open(os.path.join(folder, f"sentiment_{split}_text.txt"), encoding="utf-8") as f:
        text = [t.rstrip("\n").strip() for t in f]
    y = np.loadtxt(os.path.join(folder, f"sentiment_{split}_labels.txt"), dtype=int)
    return pd.DataFrame({"text": text, "y": y})


def load_tweeteval(folder: str, n_cal_pool: int = 5615, seed: int = 2026) -> dict:
    """Return the four disjoint pools.

    clf   : 40,000 TweetEval-train tweets used to fit the sentiment and synthetic-text models
    cal   : remaining 5,615 TweetEval-train tweets: roots of the calibration streams (edge model, alpha/beta/gamma)
    val   : official 2,000-tweet validation split: model selection and the quantification confusion matrix
    test  : official 12,284-tweet test split: roots of every evaluation stream (never seen in fitting)
    """
    _verify(folder)
    tr = _read(folder, "train")
    perm = np.random.default_rng(seed).permutation(len(tr))
    cal = tr.iloc[perm[:n_cal_pool]].reset_index(drop=True)
    clf = tr.iloc[perm[n_cal_pool:]].reset_index(drop=True)
    return dict(clf=clf, cal=cal, val=_read(folder, "val"), test=_read(folder, "test"))
