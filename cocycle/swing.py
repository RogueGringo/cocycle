"""Golf swing IMU adapter (the P4 'MWD transducer -> industrial application' generalization made real).
A shaft-mounted IMU bundle is just another multi-rate transducer file: high-rate accel/gyro (~1495 Hz, the
SPECTRAL substrate) + sensor-fusion quaternion (~186 Hz, a genuine path on S^3, the GEOMETRIC substrate that
COCYCLE's Hopf/Givens core already speaks). This module turns one shot into the 'model' half of
inference:model:output — a per-shot feature vector (kinematic + spectral + S^3-geometric) plus a Timeline that
transposes the swing into any regime. Launch-monitor 'output' joins later by the shot's filename timestamp.
numpy only; reuses cocycle.signal for the bake."""
from __future__ import annotations
import os, io, re, csv, glob, zipfile
import numpy as np
from .signal import bake_signal, BakeParams, Timeline

NEED = ("AD", "GD", "SFQ", "SFEA")               # accel, gyro, quaternion, euler (SFLA optional)
_TS = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})")


def _load_csv(text):
    rows = list(csv.reader(io.StringIO(text)))
    return np.array([[float(v) for v in r] for r in rows[1:] if r], np.float64)


def load_shot(path):
    """Read a shot bundle (a directory OR a .zip). Returns {stream: (t[N], data[N,c])} with a common
    zero-based clock (streams share the device clock; we subtract one global t0 to preserve alignment)."""
    raw = {}
    if path.endswith(".zip"):
        with zipfile.ZipFile(path) as z:
            for nm in z.namelist():
                key = os.path.splitext(os.path.basename(nm))[0]
                if key in NEED or key == "SFLA":
                    raw[key] = _load_csv(z.read(nm).decode("utf-8", "replace"))
    else:
        for f in glob.glob(os.path.join(path, "*.csv")):
            key = os.path.splitext(os.path.basename(f))[0]
            if key in NEED or key == "SFLA":
                raw[key] = _load_csv(open(f, encoding="utf-8", errors="replace").read())
    missing = [k for k in NEED if k not in raw]
    if missing:
        raise ValueError(f"{path}: missing streams {missing}")
    t0 = min(v[0, 0] for v in raw.values())
    return {k: (v[:, 0] - t0, v[:, 1:]) for k, v in raw.items()}


def _mag(a):
    return np.linalg.norm(a, axis=1)


def _rate(t):
    return (len(t) - 1) / (t[-1] - t[0])


def detect_impact(shot):
    """Impact = the ball-strike accel spike. NB: this IMU is +/-16 g/axis, so the magnitude SATURATES at
    16*sqrt(3)=27.7 g on nearly every strike -- impact_g is a clip constant, not a usable feature (use gyro
    for club speed). The saturation onset is still a good impact TIME marker."""
    t, a = shot["AD"]
    m = _mag(a)
    idx = int(np.argmax(m))
    return t[idx], float(m[idx])


def segment_swing(shot, t_impact):
    """Address -> takeaway -> top-of-backswing -> impact. The top is a DIRECTION REVERSAL, not just a |gyro|
    dip: the downswing ramp dwarfs the backswing in magnitude, so we project gyro onto the swing's principal
    axis to get SIGNED angular velocity omega, take the downswing sign near impact, and walk back to the zero
    crossing = top of backswing. Takeaway = back through the (opposite-sign) backswing to address quiet."""
    t, g = shot["GD"]
    m = _mag(g)
    peak = float(m.max()); qthr = 0.05 * peak

    def at(tt): return min(max(int(np.searchsorted(t, tt)), 0), len(m) - 1)
    imp = at(t_impact)
    ax0, ax1 = at(t_impact - 0.5), at(t_impact + 0.02)
    u = np.linalg.svd(g[ax0:max(ax0 + 2, ax1)], full_matrices=False)[2][0]   # swing's principal rotation axis
    omega = g @ u                                               # signed angular velocity on that axis
    gp0, gp1 = at(t_impact - 0.4), at(t_impact + 0.02)
    gp = gp0 + int(np.argmax(np.abs(omega[gp0:max(gp0 + 1, gp1)])))          # downswing peak
    S = np.sign(omega[gp]) or 1.0
    top = gp
    while top > 0 and np.sign(omega[top - 1]) == S and t[gp] - t[top - 1] < 0.8:
        top -= 1                                               # back to the zero crossing = top of backswing
    i = top
    while i > 0 and abs(omega[i - 1]) < qthr: i -= 1           # skip the trough
    while i > 0 and abs(omega[i - 1]) > qthr and t[top] - t[i - 1] < 1.6:
        i -= 1                                                 # back through the backswing to takeaway
    start = i
    end = imp
    while end < len(m) - 1 and abs(omega[end + 1]) > qthr: end += 1          # follow-through end
    back = max(t[top] - t[start], 1e-6); down = max(t[imp] - t[top], 1e-6)
    return {"takeaway": t[start], "top": t[top], "impact": t[imp], "follow": t[end],
            "backswing_s": float(back), "downswing_s": float(down), "tempo_ratio": float(back / down),
            "peak_gyro_dps": peak}


def _interp_row(t, X, tq):
    return np.array([np.interp(tq, t, X[:, c]) for c in range(X.shape[1])])


def quat_geometry(shot, seg):
    """The shaft orientation is a unit quaternion = a point on S^3; the swing is a curve on S^3.
    Geodesic(address,impact), swing-plane normal + planarity (gyro PCA), and S^3 path arc length."""
    tq, q = shot["SFQ"]
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    qa = _interp_row(tq, q, seg["takeaway"]); qa /= np.linalg.norm(qa) + 1e-12
    qi = _interp_row(tq, q, seg["impact"]);   qi /= np.linalg.norm(qi) + 1e-12
    geo = 2.0 * np.arccos(np.clip(abs(float(qa @ qi)), 0.0, 1.0))    # geodesic on S^3, rotation-double-cover safe
    win = (tq >= seg["takeaway"]) & (tq <= seg["follow"])
    qs = q[win]
    d = np.clip(np.abs(np.sum(qs[:-1] * qs[1:], axis=1)), 0, 1)
    arc = float(2.0 * np.arccos(d).sum()) if len(qs) > 1 else 0.0    # path length on S^3
    tg, g = shot["GD"]; dw = (tg >= seg["top"]) & (tg <= seg["impact"])
    gv = g[dw]
    if len(gv) > 2:
        s = np.linalg.svd(gv, compute_uv=False)
        planarity = float(s[0] ** 2 / (np.sum(s ** 2) + 1e-12))       # fraction of angular-vel variance in 1 axis
    else:
        planarity = 0.0
    te, ea = shot["SFEA"]; euler_imp = _interp_row(te, ea, seg["impact"])
    return {"geo_addr_impact_rad": geo, "s3_arc_len": arc, "swing_planarity": planarity,
            "euler_impact_x": float(euler_imp[0]), "euler_impact_y": float(euler_imp[1]),
            "euler_impact_z": float(euler_imp[2])}


def spectral_downswing(shot, seg):
    """Spectral geometry of the DOWNSWING accel burst (one windowed FFT over top->impact). The window stops
    AT impact on purpose: the +/-16 g accel saturates at the strike, and clipped samples inject spurious
    harmonics -- including the impact ring-down/follow-through would contaminate the spectral descriptors."""
    t, a = shot["AD"]; m = _mag(a) - 1.0                            # de-gravity the magnitude
    sr = _rate(t)
    w = (t >= seg["top"]) & (t <= seg["impact"])
    x = m[w]
    if len(x) < 16:
        return {"spec_centroid_hz": 0.0, "spec_bandwidth_hz": 0.0, "spec_rolloff_hz": 0.0, "spec_flatness": 0.0}
    x = (x - x.mean()) * np.hanning(len(x))
    mag = np.abs(np.fft.rfft(x)); freqs = np.fft.rfftfreq(len(x), 1 / sr); tot = mag.sum() + 1e-12
    cen = float((freqs * mag).sum() / tot)
    bw = float(np.sqrt((((freqs - cen) ** 2) * mag).sum() / tot))
    roll = float(freqs[np.searchsorted(np.cumsum(mag), 0.85 * mag.sum())])
    flat = float(np.exp(np.mean(np.log(mag + 1e-12))) / (mag.mean() + 1e-12))
    return {"spec_centroid_hz": cen, "spec_bandwidth_hz": bw, "spec_rolloff_hz": roll, "spec_flatness": flat}


FEATURES = ("impact_g", "peak_gyro_dps", "backswing_s", "downswing_s", "tempo_ratio",
            "geo_addr_impact_rad", "s3_arc_len", "swing_planarity",
            "euler_impact_x", "euler_impact_y", "euler_impact_z",
            "spec_centroid_hz", "spec_bandwidth_hz", "spec_rolloff_hz", "spec_flatness")


def swing_features(shot):
    """One shot -> the scalar feature vector (the 'model' half of inference:model:output)."""
    t_imp, imp_g = detect_impact(shot)
    seg = segment_swing(shot, t_imp)
    f = {"impact_g": imp_g, "peak_gyro_dps": seg["peak_gyro_dps"], "backswing_s": seg["backswing_s"],
         "downswing_s": seg["downswing_s"], "tempo_ratio": seg["tempo_ratio"]}
    f.update(quat_geometry(shot, seg)); f.update(spectral_downswing(shot, seg))
    return f, seg


def swing_timeline(shot, params: BakeParams | None = None):
    """Bake the accel burst into a COCYCLE Timeline so the swing transposes into any regime (Hopf/field/LED)."""
    t, a = shot["AD"]; sr = _rate(t)
    p = params or BakeParams(fps=144, nfft=1024, fmin=2, fmax=int(sr / 2), dom_fmin=2, dom_fmax=60)
    return bake_signal(_mag(a) - 1.0, sr, p)


def bake_session(session_dir, limit=None):
    """All shots in a session -> feature matrix [S, K] + ids + timestamps (the launch-monitor join key)."""
    def num(p):
        m = re.search(r"Shot(\d+)", p); return int(m.group(1)) if m else None
    cand = [d for d in glob.glob(os.path.join(session_dir, "*Shot*")) if os.path.isdir(d)]
    cand += glob.glob(os.path.join(session_dir, "*Shot*.zip"))
    by = {}                                                          # dedup by shot number; skip non-numbered
    for p in cand:
        n = num(p)
        if n is None: continue
        if n not in by or (os.path.isdir(p) and not os.path.isdir(by[n])):
            by[n] = p                                               # prefer the extracted dir over the zip
    paths = [by[n] for n in sorted(by)]
    if limit:
        paths = paths[:limit]
    ids, stamps, rows = [], [], []
    for p in paths:
        try:
            shot = load_shot(p)
            f, _ = swing_features(shot)
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}"); continue
        ids.append(re.search(r"Shot\d+", p).group())
        m = _TS.search(os.path.basename(p)); stamps.append(m.group(1) if m else "")
        rows.append([f[k] for k in FEATURES])
    return ids, stamps, list(FEATURES), np.array(rows, np.float64)
