"""Text normalisation and the derivative-content operators used by the stream generator.

All operators are deterministic given a numpy Generator. The synthetic ("AI-style") rewriter is rule-based on
purpose: it gives a controllable generator-family shift (families A and B are seen when the synthetic-text detector
is trained, family C is not), which is what the experiments need. It is not a substitute for a large language model.
"""
from __future__ import annotations

import re

import numpy as np

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_RT = re.compile(r"^\s*(rt|qt)\s+@\w+:?\s*", re.I)
_WS = re.compile(r"\s+")


def normalise(t: str) -> str:
    """Canonical form used for similarity, exact-duplicate hashing and the detectors."""
    t = t.strip().strip('"')
    t = _RT.sub("", t)
    t = _URL.sub(" ", t)
    t = _MENTION.sub(" @user ", t)
    return _WS.sub(" ", t).strip().lower()


SYN = {
    "good": ["great", "nice", "decent"], "great": ["awesome", "amazing", "good"], "bad": ["awful", "terrible", "poor"],
    "happy": ["glad", "pleased", "delighted"], "sad": ["upset", "unhappy", "down"], "love": ["adore", "really like"],
    "hate": ["can't stand", "despise", "loathe"], "like": ["enjoy", "dig"], "think": ["believe", "reckon", "feel"],
    "really": ["truly", "seriously", "so"], "very": ["really", "extremely", "super"], "big": ["huge", "large", "massive"],
    "small": ["little", "tiny", "minor"], "new": ["latest", "brand new", "fresh"], "old": ["former", "previous"],
    "tomorrow": ["tmrw", "the next day"], "tonight": ["this evening", "2nite"], "today": ["this day", "2day"],
    "going": ["heading", "gonna go"], "watch": ["see", "check out"], "see": ["watch", "catch"], "get": ["grab", "pick up"],
    "people": ["folks", "everyone", "ppl"], "best": ["greatest", "finest", "top"], "worst": ["poorest", "lowest"],
    "game": ["match", "fixture"], "movie": ["film", "flick"], "song": ["track", "tune"], "show": ["episode", "programme"],
    "excited": ["thrilled", "pumped", "hyped"], "angry": ["furious", "mad", "annoyed"], "funny": ["hilarious", "lol-worthy"],
    "wrong": ["incorrect", "mistaken"], "right": ["correct", "spot on"], "start": ["begin", "kick off"],
    "end": ["finish", "wrap up"], "win": ["victory", "triumph"], "lost": ["fell", "was beaten"], "team": ["squad", "side"],
    "fans": ["supporters", "followers"], "news": ["report", "update"], "says": ["claims", "states"], "said": ["stated", "claimed"],
    "want": ["wish", "would like"], "need": ["require", "must have"], "awesome": ["fantastic", "brilliant"],
    "amazing": ["incredible", "stunning"], "terrible": ["horrible", "dreadful"], "nice": ["lovely", "pleasant"],
    "beautiful": ["gorgeous", "stunning"], "stupid": ["dumb", "silly"], "crazy": ["insane", "wild"], "fun": ["enjoyable", "a blast"],
    "time": ["moment", "while"], "day": ["date"], "week": ["7 days"], "night": ["evening"], "morning": ["a.m.", "early hours"],
    "friday": ["fri"], "saturday": ["sat"], "sunday": ["sun"], "monday": ["mon"], "god": ["lord", "gosh"], "thanks": ["thank you", "cheers"],
    "maybe": ["perhaps", "possibly"], "probably": ["likely", "most likely"], "soon": ["shortly", "before long"],
    "help": ["assist", "support"], "buy": ["purchase", "get"], "make": ["create", "build"], "know": ["realise", "understand"],
    "looks": ["seems", "appears"], "look": ["seem", "appear"], "still": ["yet", "even now"], "also": ["too", "as well"],
    "just": ["simply", "only"], "everyone": ["everybody", "all"], "nothing": ["nada", "zero"], "problem": ["issue", "trouble"],
    "president": ["leader", "potus"], "election": ["vote", "polls"], "phone": ["mobile", "device"], "music": ["tunes", "songs"],
}
_COMMENTS_HUMAN = ["this", "so true", "exactly", "wow", "^^", "same here", "this 👆", "facts", "+1", "100%", "agreed", "yep"]
FAMILY = {
    "A": dict(open=["Honestly,", "I have to say,", "Truly,"], close=["Highly recommend!", "Absolutely worth it.",
                                                                      "What an experience."]),
    "B": dict(open=["It's worth noting that", "In my opinion,", "From my perspective,"],
              close=["Overall, a great step forward.", "All in all, very positive.", "Definitely a win."]),
    "C": dict(open=["Just wanted to share:", "Quick thought:", "Not gonna lie,"],
              close=["genuinely impressed ✨", "can't recommend it enough 🙌", "10/10 would repeat 💯"]),
}
_CONTR = {"can't": "cannot", "won't": "will not", "don't": "do not", "i'm": "I am", "it's": "it is", "isn't": "is not",
          "didn't": "did not", "you're": "you are", "that's": "that is", "doesn't": "does not", "i've": "I have"}
CAMPAIGN_TAGS = ["#gamechanger", "#mustsee", "#bestever", "#loveit", "#support", "#trending", "#winning", "#proud"]


def substitute(t: str, rng: np.random.Generator, p: float) -> str:
    """Whitespace-token synonym substitution that preserves surrounding punctuation, mentions and hashtags."""
    out = []
    for w in t.split():
        m = re.match(r"^(\W*)(\w[\w']*)(\W*)$", w)
        if m and m.group(2).lower() in SYN and rng.random() < p:
            alts = SYN[m.group(2).lower()]
            out.append(m.group(1) + alts[rng.integers(len(alts))] + m.group(3))
        else:
            out.append(w)
    return " ".join(out)


_FILLER_PRE = ["apparently", "so basically", "heard that", "can confirm", "ok so", "lowkey", "fr", "update:"]
_FILLER_POST = ["lol", "smh", "fr", "just saying", "tbh", "y'all", "!!", "no cap"]


def retention(strength: float) -> float:
    """Share of the parent's tokens kept by a derivative of the given paraphrase strength."""
    return float(np.clip(1.0 - 0.2 * strength, 0.4, 0.95))


def _keep(toks: list[str], rng: np.random.Generator, r: float) -> list[str]:
    keep = [w for w in toks if rng.random() < r]
    return keep if len(keep) >= 2 else toks[: max(2, int(len(toks) * r))]


def human_paraphrase(t: str, rng: np.random.Generator, strength: float = 2.0) -> str:
    """A human re-wording of the same opinion: token retention ~ retention(strength), synonym substitution,
    clause rotation and conversational fillers. Strength 1 keeps ~80% of tokens, strength 3 keeps ~40%."""
    s = substitute(_RT.sub("", t), rng, min(0.9, 0.3 * strength))
    toks = _keep(s.split(), rng, retention(strength))
    if len(toks) > 5 and rng.random() < 0.3 * strength:
        k = len(toks) // 3
        toks = toks[k:] + toks[:k]
    for i in range(len(toks) - 1):
        if rng.random() < 0.05 * strength:
            toks[i], toks[i + 1] = toks[i + 1], toks[i]
    s = " ".join(toks)
    if rng.random() < 0.25 * strength:
        s = f"{_FILLER_PRE[rng.integers(len(_FILLER_PRE))]} {s}"
    if rng.random() < 0.25 * strength:
        s = f"{s} {_FILLER_POST[rng.integers(len(_FILLER_POST))]}"
    return s


def repost(t: str, rng: np.random.Generator) -> str:
    return f"RT @user: {t}" if rng.random() < 0.7 else t        # some clients drop the RT marker


def quote(t: str, rng: np.random.Generator) -> str:
    """Comment plus the quoted text; half of the quotes excerpt only a leading part of the parent."""
    toks = t.split()
    if rng.random() < 0.5 and len(toks) > 4:
        toks = toks[: max(3, int(len(toks) * rng.uniform(0.4, 0.8)))]
    return f"{_COMMENTS_HUMAN[rng.integers(len(_COMMENTS_HUMAN))]} {' '.join(toks)}"


_STYLE = sorted({p for fam in FAMILY.values() for k in ("open", "close") for p in fam[k]} | set(CAMPAIGN_TAGS), key=len, reverse=True)


def strip_style(t: str) -> str:
    """Remove register markers of earlier synthetic rewrites so that rewrites of rewrites do not stack them."""
    for p in _STYLE:
        t = t.replace(p, " ").replace(p.lower(), " ")
    return _WS.sub(" ", t).strip(" .")


def ai_rewrite(t: str, rng: np.random.Generator, family: str, keep: float = 0.7) -> str:
    """Rule-based synthetic rewriting with a family-specific register (opener/closer, expanded contractions)."""
    f = FAMILY[family]
    s = strip_style(_URL.sub("", _RT.sub("", t)))
    s = re.sub(r"#\w+", "", s)
    s = _MENTION.sub("", s)
    s = substitute(s, rng, 0.6)
    s = " ".join(_CONTR.get(w.lower(), w) for w in _keep(s.split(), rng, keep) if s.split())
    s = _WS.sub(" ", s).strip(" .,!")
    if s:
        s = s[0].upper() + s[1:]
    op = f["open"][rng.integers(3)]
    cl = f["close"][rng.integers(3)]
    if family == "B":
        s = s[0].lower() + s[1:] if s else s
    return f"{op} {s}. {cl}"


def coordinated_copy(t: str, rng: np.random.Generator) -> str:
    """Near-verbatim campaign copy: one-word edit plus a rotating campaign hashtag."""
    toks = t.split()
    if len(toks) > 3 and rng.random() < 0.6:
        i = rng.integers(len(toks))
        lw = toks[i].lower().strip(".,!?")
        if lw in SYN:
            toks[i] = SYN[lw][rng.integers(len(SYN[lw]))]
    return " ".join(toks) + " " + CAMPAIGN_TAGS[rng.integers(len(CAMPAIGN_TAGS))]
