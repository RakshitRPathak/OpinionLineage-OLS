"""Sentiment classifier (with validation-based selection), synthetic-text detector and coordination scorer."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from . import textops as tx


class TextFeatures:
    """Word (1-2) + character (2-5, word-bounded) TF-IDF on normalised text, fitted once on the classifier pool."""

    def __init__(self, max_word=60000, max_char=120000):
        self.w = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=max_word, sublinear_tf=True)
        self.c = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=max_char, sublinear_tf=True)

    def fit(self, texts):
        n = [tx.normalise(t) for t in texts]
        self.w.fit(n); self.c.fit(n)
        return self

    def transform(self, texts, parts=False):
        n = [tx.normalise(t) for t in texts]
        W, C = self.w.transform(n), self.c.transform(n)
        X = sparse.hstack([W, C]).tocsr() * (1 / np.sqrt(2))
        return (X, W, C) if parts else X


SENT_MODELS = {
    "SGD log-loss": lambda: SGDClassifier(loss="log_loss", alpha=2e-5, max_iter=30, tol=None, random_state=42),
    "Linear SVC (Platt)": lambda: CalibratedClassifierCV(LinearSVC(C=0.3, random_state=42), cv=3, method="sigmoid"),
    "Complement NB": lambda: ComplementNB(alpha=0.5),
    "Logistic regression": lambda: LogisticRegression(C=4.0, max_iter=2000),
}


def sentiment_suite(F: TextFeatures, pools: dict):
    """Fit four classifiers; select on validation macro-F1 (test never used for selection)."""
    Xtr = F.transform(pools["clf"].text)
    Xva, Xte = F.transform(pools["val"].text), F.transform(pools["test"].text)
    ytr, yva, yte = pools["clf"].y.to_numpy(), pools["val"].y.to_numpy(), pools["test"].y.to_numpy()
    rows, fitted = [], {}
    for name, mk in SENT_MODELS.items():
        m = mk().fit(Xtr, ytr)
        pv, pt = m.predict(Xva), m.predict(Xte)
        rows.append(dict(model=name, val_macro_f1=f1_score(yva, pv, average="macro"), test_acc=accuracy_score(yte, pt),
                         test_weighted_f1=f1_score(yte, pt, average="weighted"), test_macro_f1=f1_score(yte, pt, average="macro"),
                         test_weighted_precision=precision_score(yte, pt, average="weighted"),
                         test_weighted_recall=recall_score(yte, pt, average="weighted")))
        fitted[name] = m
    tab = pd.DataFrame(rows)
    best = tab.sort_values("val_macro_f1", ascending=False).model.iat[0]
    cm = confusion_matrix(yte, fitted[best].predict(Xte))
    return fitted[best], best, tab, cm


def synthetic_detector(pools: dict, rng: np.random.Generator, n=6000):
    """Human vs rule-based AI rewrite (families A,B for training). Evaluated on test tweets for A/B (seen) and C (unseen),
    and on human derivatives (reposts, quotes, paraphrases) that must not be flagged."""
    clf = pools["clf"]
    idx = rng.permutation(len(clf))
    hum = clf.text.iloc[idx[:n]].tolist()
    syn = [tx.ai_rewrite(t, rng, "AB"[k % 2]) for k, t in enumerate(clf.text.iloc[idx[n:2 * n]])]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=80000, sublinear_tf=True, lowercase=False)
    X = vec.fit_transform(hum + syn)
    m = LogisticRegression(C=2.0, max_iter=2000).fit(X, np.r_[np.zeros(n), np.ones(n)])
    te = pools["test"].text.sample(4000, random_state=7).tolist()
    h_te, s_te = te[:2000], te[2000:]
    ev = []
    for fam in ("A", "B", "C"):
        st = [tx.ai_rewrite(t, rng, fam) for t in s_te[:1000]]
        Xe = vec.transform(h_te[:1000] + st)
        y = np.r_[np.zeros(1000), np.ones(1000)]
        p = m.predict(Xe)
        ev.append(dict(family=fam, seen=fam != "C", acc=accuracy_score(y, p), weighted_f1=f1_score(y, p, average="weighted"),
                       synthetic_f1=f1_score(y, p)))
    for name, op in (("repost", tx.repost), ("quote", tx.quote), ("paraphrase", lambda t, r: tx.human_paraphrase(t, r))):
        d = [op(t, rng) for t in h_te[1000:2000]]
        ev.append(dict(family=f"human {name}", seen=None, acc=1 - m.predict(vec.transform(d)).mean(), weighted_f1=np.nan,
                       synthetic_f1=np.nan))
    return SyntheticScorer(vec, m), pd.DataFrame(ev)


class SyntheticScorer:
    def __init__(self, vec, m):
        self.vec, self.m = vec, m

    def __call__(self, texts):
        return self.m.predict_proba(self.vec.transform(list(texts)))[:, 1]


COORD_FEATS = ["acc_log_age", "acc_log_ppd", "acc_regularity", "acc_default", "log_burst"]


class CoordinationScorer:
    """P(record belongs to a coordinated campaign) from account features and burst co-posting (no text features)."""

    def fit(self, df: pd.DataFrame):
        self.m = LogisticRegression(C=1.0, max_iter=1000).fit(df[COORD_FEATS], (df.true_campaign >= 0).astype(int))
        return self

    def __call__(self, df: pd.DataFrame):
        return self.m.predict_proba(df[COORD_FEATS])[:, 1]
