"""Mechanically-anchored swing model. Two things the kinematic segmentation skipped:
  1. ISOLATE the swing from the 'other' -- each 14.5s file is ~12% active in several bursts (backswing,
     downswing, and post-shot motion); isolate_swing keeps only the burst-cluster containing impact.
  2. Label phases at MECHANICALLY-defined orientation events, not arbitrary time fractions. The shaft's
     world-vertical component (from the drift-free quaternion) gives P2 (club parallel, backswing) and P6
     (club parallel, downswing = the delivery checkpoint where the article says the lower pendulum releases).
     These are the SAME physical instant across shots regardless of tempo -- the correct restriction map.
Features are measured AT these positions (delivery time, parallel-to-parallel symmetry, speed at P6). numpy only."""
from __future__ import annotations
import numpy as np
from .swing import load_shot, detect_impact, segment_swing, _quat_to_R, _mag
from .reconstruct import world_up, shaft_sensor

EVENTS = ("takeaway", "p2", "top", "p6", "impact", "follow")


def isolate_swing(shot):
    """Keep only the motion burst-cluster containing impact; report whether the file is a clean single swing
    (no other comparable burst -> a practice swing / heavy waggle flags it). Bursts are non-overlapping active
    runs; the impact cluster grows only across gaps < 0.35s (the top-of-backswing pause)."""
    tg, g = shot["GD"]; m = _mag(g); ti, _ = detect_impact(shot)
    idx = np.where(m > 0.10 * m.max())[0]
    if not len(idx): return float(tg[0]), float(tg[-1]), False
    groups = np.split(idx, np.where(np.diff(idx) > 1)[0] + 1)
    bursts = [(tg[gp[0]], tg[gp[-1]], float(m[gp].mean())) for gp in groups if tg[gp[-1]] - tg[gp[0]] > 0.03]
    ib = [b for b in bursts if b[0] <= ti <= b[1]] or [min(bursts, key=lambda b: abs((b[0] + b[1]) / 2 - ti))]
    lo, hi = ib[0][0], ib[0][1]
    changed = True
    while changed:                                              # iterative merge across <0.35s gaps only
        changed = False
        for a, b, _ in bursts:
            if b < lo and lo - b < 0.35: lo, changed = a, True
            if a > hi and a - hi < 0.35: hi, changed = b, True
    other = [(a, b, e) for a, b, e in bursts if b < lo - 0.01 or a > hi + 0.01]
    oe = max((e for _, _, e in other), default=0.0)
    ie = float(np.mean(m[(tg >= lo) & (tg <= hi)]) + 1e-9)
    return float(lo), float(hi), bool(oe < 0.5 * ie)


def _shaft_vertical(shot, seg, N=500):
    tq, q = shot["SFQ"]; q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    up = world_up(shot); s = shaft_sensor(shot, seg)
    ts = np.linspace(seg["takeaway"], seg["follow"], N)
    def Rof(tt): return _quat_to_R(q[min(max(int(np.searchsorted(tq, tt)), 0), len(q) - 1)])
    vert = np.array([(Rof(t) @ s) @ up for t in ts])
    return ts, vert, up, s


def mechanical_events(shot, seg):
    """P2 = most-horizontal shaft on the BACKSWING (argmin|vertical| over takeaway->top); P6 = most-horizontal
    on the DOWNSWING (argmin|vertical| over top->impact = the delivery position). Robust and always defined --
    a near-horizontal shaft need not cross exactly through zero, so we take the closest approach, not a sign
    crossing. Returns the shaft |vertical| at P6 so the caller can confirm the checkpoint is genuinely horizontal."""
    ts, vert, up, s = _shaft_vertical(shot, seg)
    i_top = int(np.argmin(np.abs(ts - seg["top"]))); i_imp = int(np.argmin(np.abs(ts - seg["impact"])))
    av = np.abs(vert); pad = 2
    degenerate = i_top < 2 * pad + 2 or i_imp - i_top < 2 * pad + 2   # segmentation collapsed (outlier swing)
    if degenerate:
        return {"takeaway": seg["takeaway"], "p2": None, "top": seg["top"], "p6": None,
                "impact": seg["impact"], "follow": seg["follow"], "p6_vert": np.nan, "degenerate": True}
    p2 = ts[pad + int(np.argmin(av[pad:i_top - pad]))]                # inset so P2 can't pin to takeaway
    j6 = i_top + pad + int(np.argmin(av[i_top + pad:i_imp - pad]))
    return {"takeaway": seg["takeaway"], "p2": p2, "top": seg["top"], "p6": ts[j6],
            "impact": seg["impact"], "follow": seg["follow"], "p6_vert": float(av[j6]), "degenerate": False}


def mechanical_features(shot):
    """Features measured AT the mechanically-defined positions -- coaching checkpoints, not time fractions."""
    ti, _ = detect_impact(shot); seg = segment_swing(shot, ti)
    lo, hi, clean = isolate_swing(shot)
    ev = mechanical_events(shot, seg)
    tg, g = shot["GD"]; gm = _mag(g)
    def omega_at(t): return float(gm[min(int(np.searchsorted(tg, t)), len(gm) - 1)]) if t is not None else np.nan
    f = {"clean_swing": float(clean), "swing_dur_s": hi - lo}
    if ev["p2"] is not None and ev["p6"] is not None:
        f["p2_to_top_s"] = ev["top"] - ev["p2"]                  # backswing parallel->top
        f["top_to_p6_s"] = ev["p6"] - ev["top"]                  # top->downswing parallel
        f["delivery_s"] = ev["impact"] - ev["p6"]               # P6 (parallel) -> impact: the release window
        f["parallel_symmetry"] = (ev["top"] - ev["p2"]) / max(ev["impact"] - ev["p6"], 1e-6)  # back vs down
        f["omega_p6_dps"] = omega_at(ev["p6"])                  # club speed at the delivery checkpoint
        f["omega_p2_dps"] = omega_at(ev["p2"])
    else:
        for k in ("p2_to_top_s", "top_to_p6_s", "delivery_s", "parallel_symmetry", "omega_p6_dps", "omega_p2_dps"):
            f[k] = np.nan
    return f, ev, clean


MECH_FEATURES = ("swing_dur_s", "p2_to_top_s", "top_to_p6_s", "delivery_s", "parallel_symmetry",
                 "omega_p6_dps", "omega_p2_dps")


def session_mechanics(session_dir):
    from .sheaf import _shot_paths
    ids, paths = _shot_paths(session_dir)
    out_ids, rows, cleans, evs = [], [], [], []
    for i, p in enumerate(paths):
        try:
            sh = load_shot(p); f, ev, clean = mechanical_features(sh)
        except Exception:
            continue
        out_ids.append(ids[i]); rows.append([f[k] for k in MECH_FEATURES]); cleans.append(clean); evs.append(ev)
    return out_ids, list(MECH_FEATURES), np.array(rows, float), cleans, evs
