"""The sheaf-correct analysis the collapse-to-scalars approach threw away. Each shot is a SECTION over time,
not 24 numbers. We (1) RESTRICT every shot onto one canonical phase base by impact-anchored time-warping
(takeaway->top->impact->follow warped to shared canonical fractions), so all 30 shots are comparable sections
over the same base; (2) GLUE the registered sections across shots and across streams (signed omega, accel
energy) into one functional object; (3) read the COCYCLE = each shot's residual from the glued mean section =
the obstruction to gluing = the real per-shot signal. Functional-PCA of the glued sections gives curve-shape
features that a scalar collapse cannot. numpy only."""
from __future__ import annotations
import glob, os, re
import numpy as np
from .swing import load_shot, detect_impact, segment_swing, _signed_omega, _mag


def _canon_fractions(segs):
    """Canonical positions of top & impact on the shared base = medians of the observed fractions (minimizes
    warp distortion). Returns (P_top, P_impact) in (0,1) over [takeaway, follow]."""
    span = np.array([s["follow"] - s["takeaway"] for s in segs]) + 1e-9
    ft = np.array([(s["top"] - s["takeaway"]) for s in segs]) / span
    fi = np.array([(s["impact"] - s["takeaway"]) for s in segs]) / span
    return float(np.median(ft)), float(np.median(fi))


def restrict(t, v, seg, P, N):
    """Restriction map rho: the shot's local clock -> the canonical phase base. Piecewise-linear monotone warp
    pinning (takeaway,top,impact,follow) to canonical (0, P_top, P_impact, 1), then resample v on N grid."""
    ac = [0.0, P[0], P[1], 1.0]; ar = [seg["takeaway"], seg["top"], seg["impact"], seg["follow"]]
    grid = np.linspace(0.0, 1.0, N)
    real_t = np.interp(grid, ac, ar)
    return np.interp(real_t, t, v)


def _shot_paths(session_dir):
    def num(p):
        m = re.search(r"Shot(\d+)", p); return int(m.group(1)) if m else None
    cand = [d for d in glob.glob(os.path.join(session_dir, "*Shot*")) if os.path.isdir(d)]
    cand += glob.glob(os.path.join(session_dir, "*Shot*.zip"))
    by = {}
    for p in cand:
        n = num(p)
        if n is None: continue
        if n not in by or (os.path.isdir(p) and not os.path.isdir(by[n])): by[n] = p
    return [f"Shot{n}" for n in sorted(by)], [by[n] for n in sorted(by)]


def build_sheaf(session_dir, N=256):
    """Load every shot, restrict onto the shared base, return registered sections per channel + segments.
    Channels: 'omega' (signed angular velocity, orientation-fixed), 'energy' (|accel|-1, the spectral burst)."""
    ids, paths = _shot_paths(session_dir)
    shots, segs, keep = [], [], []
    for i, p in enumerate(paths):
        try:
            sh = load_shot(p); ti, _ = detect_impact(sh); sg = segment_swing(sh, ti)
        except Exception:
            continue
        shots.append(sh); segs.append(sg); keep.append(ids[i])
    P = _canon_fractions(segs)
    omega, energy = [], []
    for sh, sg in zip(shots, segs):
        to, om = _signed_omega(sh, sg)
        omega.append(restrict(to, om, sg, P, N))
        ta, a = sh["AD"]; energy.append(restrict(ta, _mag(a) - 1.0, sg, P, N))
    return {"ids": keep, "P": P, "N": N, "omega": np.array(omega), "energy": np.array(energy), "segs": segs}


def glue(sections):
    """Glue registered sections: the global section (mean), per-phase gluing tightness (cross-shot std), and
    the COCYCLE residual per shot = ||section_i - glued_mean|| / ||glued_mean|| (the gluing obstruction)."""
    mean = sections.mean(0)
    tight = sections.std(0)
    denom = np.linalg.norm(mean) + 1e-9
    resid = np.linalg.norm(sections - mean, axis=1) / denom
    return {"mean": mean, "tightness": tight, "residual": resid}


def fpca(sections, k=4, normalize=False):
    """Functional PCA: SVD of the (optionally per-shot amplitude-normalized) centered registered curves.
    Returns scores [S,k] (curve-shape coordinates) and explained-variance ratio. normalize=True isolates
    SHAPE/timing from amplitude (speed)."""
    X = sections.copy()
    if normalize:
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    mu = X.mean(0); C = X - mu
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    k = min(k, len(s))
    scores = C @ Vt[:k].T
    evr = (s[:k] ** 2) / (np.sum(s ** 2) + 1e-12)
    return scores, evr, Vt[:k]


def quality_map(sheaf, channel="omega"):
    """The launch-monitor-free coaching object: the golfer's glued canonical section, the per-phase gluing
    tightness (their natural variation), and the per-shot per-phase obstruction z = |section - glued mean| /
    tightness -- WHERE in the phase arc each swing breaks from their own template, in units of their own sigma."""
    sec = sheaf[channel]; g = glue(sec)
    z = np.abs(sec - g["mean"]) / (g["tightness"] + 1e-9)
    worst_phase = z.argmax(1) / (sheaf["N"] - 1)                 # canonical phase of each shot's max obstruction
    return {"ids": sheaf["ids"], "P": sheaf["P"], "N": sheaf["N"], "channel": channel,
            "mean": g["mean"], "tightness": g["tightness"], "residual": g["residual"], "z": z,
            "worst_phase": worst_phase}


def joint_features(sheaf, k=4):
    """Glue channels: concatenate shape-normalized registered omega + energy, fPCA the joint object, and
    append each channel's cocycle residual. This is the section-level feature set (curve shape + obstruction)."""
    om_s, om_evr, _ = fpca(sheaf["omega"], k=k, normalize=True)
    en_s, en_evr, _ = fpca(sheaf["energy"], k=k, normalize=True)
    amp, _, _ = fpca(sheaf["omega"], k=1, normalize=False)     # 1 amplitude score = overall speed
    g_om = glue(sheaf["omega"]); g_en = glue(sheaf["energy"])
    names = ([f"omega_pc{i}" for i in range(k)] + [f"energy_pc{i}" for i in range(k)]
             + ["omega_amp", "omega_cocycle", "energy_cocycle"])
    F = np.column_stack([om_s, en_s, amp[:, 0], g_om["residual"], g_en["residual"]])
    return sheaf["ids"], names, F, {"omega_evr": om_evr, "energy_evr": en_evr}
