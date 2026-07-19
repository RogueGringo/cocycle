"""Reconstruct the ACTUAL swing motion from the drift-free fused orientation, anchored at impact (the one
certain point in space -- the ball). What IS recoverable and what ISN'T, honestly:
  - RECOVERABLE: the clubhead ARC through space. clubhead(t) - ball = L*(R(t).s_hat - R(impact).s_hat) under a
    fixed-hub approximation, from the quaternion alone (no integration -> no drift). Gives the swing plane,
    arc shape, shaft sweep -- the real geometry of the motion.
  - NOT RECOVERABLE from this single grip IMU: the sub-degree launch-delivery angles (attack angle varies ~1 deg
    across shots; IMU reconstruction noise is ~30x that). The launch monitor measures those at the clubhead
    directly and is irreplaceable for them. This module does the motion, not the delivery angles.
numpy only; s_hat (shaft direction in sensor frame) read from centripetal accel, which points down the shaft."""
from __future__ import annotations
import numpy as np
from .swing import load_shot, detect_impact, segment_swing, _quat_to_R

CLUB_LEN = 1.1                                    # m, sensor->clubhead scale (arbitrary for shape; ~iron length)


def world_up(shot):
    """Gravity/up axis in the world frame = mean over the rest window of R(t) . AD_sensor (specific force)."""
    tq, q = shot["SFQ"]; q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    ta, a = shot["AD"]
    def Rof(tt): return _quat_to_R(q[min(max(int(np.searchsorted(tq, tt)), 0), len(q) - 1)])
    up = np.mean([Rof(ta[k]) @ a[k] for k in range(min(200, len(a)))], 0)
    return up / (np.linalg.norm(up) + 1e-12)


def shaft_sensor(shot, seg):
    """Shaft direction in the SENSOR frame: centripetal acceleration points from the clubhead toward the
    rotation center, i.e. down the shaft, so s_hat = -mean(SFLA) over a clean pre-impact window."""
    tl, la = shot["SFLA"]; ti = seg["impact"]
    w = (tl >= ti - 0.030) & (tl <= ti - 0.010)
    s = -np.mean(la[w], 0) if w.any() else -la[min(np.searchsorted(tl, ti), len(la) - 1)]
    return s / (np.linalg.norm(s) + 1e-12)


def reconstruct_arc(shot, seg, L=CLUB_LEN, N=300):
    """The clubhead arc through space, anchored at the ball (impact = origin), from drift-free orientation."""
    tq, q = shot["SFQ"]; q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    s = shaft_sensor(shot, seg)
    ts = np.sort(np.append(np.linspace(seg["takeaway"], seg["follow"], N), seg["impact"]))  # exact impact knot
    def Rof(tt): return _quat_to_R(q[min(max(int(np.searchsorted(tq, tt)), 0), len(q) - 1)])
    head_imp = Rof(seg["impact"]) @ s
    arc = np.array([L * (Rof(t) @ s - head_imp) for t in ts])       # ball at origin
    phase = (ts - seg["takeaway"]) / (seg["follow"] - seg["takeaway"] + 1e-9)
    return arc, phase


def swing_plane(arc):
    """Fit the swing plane (SVD of the arc): normal, planarity (variance fraction in the plane), and the
    plane's inclination from vertical -- all scale-invariant, so robust despite the unknown lever length L."""
    c = arc - arc.mean(0)
    _, s, Vt = np.linalg.svd(c, full_matrices=False)
    normal = Vt[2]
    planarity = float((s[0] ** 2 + s[1] ** 2) / (np.sum(s ** 2) + 1e-12))
    return {"normal": normal, "planarity": planarity, "singular": s}


def plane_inclination(arc, up):
    n = swing_plane(arc)["normal"]
    return float(np.degrees(np.arccos(np.clip(abs(n @ up), 0, 1))))     # 0 = flat/horizontal plane, 90 = upright
