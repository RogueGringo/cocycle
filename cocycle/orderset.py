"""ORDER-SET analysis -- the session as an ordered sequence, not a bag. Decomposes each registered swing
section into PHASES (backswing / downswing / follow on the canonical base), separates SCALE (RMS amplitude)
from SHAPE variance per phase, and studies both over the shot ORDER: changepoints (regime shifts: crash,
recovery, surge), lag-1 autocorrelation (compensation dynamics), and overall-vs-isolated variance -- which
phase x element carries the session's variance, and how each single swing sits against the ensemble.
Rank statistics preferred: the order series are heavy-tailed (Pearson and Spearman disagree in SIGN on this
data -- a collapsed linear stat is not trustworthy here). numpy only."""
from __future__ import annotations
import numpy as np

PHASES = ("back", "down", "follow")


def phase_windows(P, N):
    i_top, i_imp = int(P[0] * N), int(P[1] * N)
    return {"back": (0, i_top), "down": (i_top, i_imp), "follow": (i_imp, N)}


def order_ids(ids):
    num = [int(s.replace("Shot", "")) for s in ids]
    return list(np.argsort(num))


def decompose(sheaf):
    """Per shot x phase x channel: SCALE = RMS amplitude in the phase window; SHAPE residual = distance from
    the phase's ensemble mean AFTER per-shot amplitude normalization (so scale and shape are separated)."""
    win = phase_windows(sheaf["P"], sheaf["N"])
    o = order_ids(sheaf["ids"])
    out = {"ids": [sheaf["ids"][i] for i in o], "scale": {}, "shape": {}}
    for ch in ("omega", "energy"):
        sec = sheaf[ch][o]
        for ph, (a, b) in win.items():
            seg = sec[:, a:b]
            rms = np.sqrt(np.mean(seg ** 2, 1))
            out["scale"][f"{ch}_{ph}"] = rms
            norm = seg / (rms[:, None] + 1e-9)
            mu = norm.mean(0)
            out["shape"][f"{ch}_{ph}"] = np.linalg.norm(norm - mu, axis=1) / (np.linalg.norm(mu) + 1e-9)
    return out


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-9)


def binseg(x, kmax=3, min_seg=4, thresh=1.2):
    """Binary segmentation for mean shifts on a standardized series. Returns sorted 0-based changepoint
    indices (start of the new regime). Gain = |mean diff| * sqrt(harmonic n) must exceed thresh."""
    x = zscore(np.asarray(x, float))
    cps = []

    def best_split(a, b):
        best_t, best_g = None, 0.0
        for t in range(a + min_seg, b - min_seg + 1):
            m1, m2 = x[a:t].mean(), x[t:b].mean()
            n1, n2 = t - a, b - t
            g = abs(m1 - m2) * np.sqrt(n1 * n2 / (n1 + n2))
            if g > best_g:
                best_t, best_g = t, g
        return best_t, best_g

    segs = [(0, len(x))]
    while len(cps) < kmax and segs:
        cands = [(best_split(a, b), (a, b)) for a, b in segs]
        cands = [(t, g, ab) for (t, g), ab in cands if t is not None and g > thresh]
        if not cands:
            break
        t, g, (a, b) = max(cands, key=lambda c: c[1])
        cps.append(t)
        segs.remove((a, b)); segs += [(a, t), (t, b)]
    return sorted(cps)


def lag1(x):
    x = zscore(np.asarray(x, float))
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def _midrank(x):
    x = np.asarray(x, float); o = np.argsort(x, kind="mergesort")
    r = np.empty(len(x)); r[o] = np.arange(len(x), dtype=float)
    u, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    s = np.zeros(len(u)); np.add.at(s, inv, r)
    return (s / cnt)[inv]                                     # average rank within ties (Spearman-proper)


def spearman(a, b):
    return float(np.corrcoef(_midrank(a), _midrank(b))[0, 1])


def convergence_null(elements, fcp, T=5000, seed=0):
    """The HONEST significance test the single-element p could not give: permute the shot ORDER identically
    across all (correlated) elements -- preserving cross-element correlation and each series' values, breaking
    only alignment with the FSX changepoints -- and ask how often >=2 boundaries get corroborated by chance."""
    from collections import Counter

    def corrob(series):
        v = Counter()
        for x in series:
            for c in binseg(x): v[c] += 1
        return sum(1 for c in fcp if sum(v.get(d, 0) for d in (c - 1, c, c + 1)) >= 2)

    real = corrob(elements)
    rng = np.random.default_rng(seed); ge = 0
    for _ in range(T):
        p = rng.permutation(len(elements[0]))
        if corrob([x[p] for x in elements]) >= real: ge += 1
    return real, ge / T


def variance_ledger(dec):
    """Overall vs isolated: each element's (channel x phase, scale vs shape) share of session variance,
    computed on z-scored series so elements are comparable. Returns [(name, kind, var_share, lag1)] sorted."""
    rows = []
    allv = []
    for kind in ("scale", "shape"):
        for name, v in dec[kind].items():
            allv.append((name, kind, float(np.var(zscore(v))), lag1(v), v))
    # CV-based share: relative spread each element exhibits (z-var is 1 by construction, so use raw CV)
    rows = [(n, k, float(np.std(v) / (np.abs(np.mean(v)) + 1e-9)), l) for n, k, _, l, v in allv]
    return sorted(rows, key=lambda r: -r[2])
