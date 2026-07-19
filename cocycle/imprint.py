"""Swing IMPRINTS -- the render-as-representation experiment ("renders:time for known outputs... another
image to extract"). Each swing's reconstructed clubhead arc is projected onto its own swing plane, anchored
at the ball, and rasterized into a canonical speed-weighted image: a standardized visual fingerprint of the
motion. Two uses, tested with the same skeptical harness as every layer: (1) image-space geometry as a mishit
separator (must agree with the cocycle), (2) image-PCA features screened against the known FSX outputs with
ALL-vs-GOOD honesty -- representation #4 in the series scalars/curves/fPCA/images. numpy only."""
from __future__ import annotations
import numpy as np
from .reconstruct import reconstruct_arc, swing_plane, world_up
from .swing import load_shot, detect_impact, segment_swing
from .sheaf import _shot_paths

K = 64            # raster size (KxK)
SPAN = 2.4        # metres from ball to image edge (fixed metric window -> sizes are comparable)


def plane_coords(arc, up):
    """Project the 3D arc onto its own swing plane in a CANONICAL, gravity-aligned frame: v-axis = world up
    projected into the plane (stable and physically meaningful), u = n x v, ball at origin, normal sky-side.
    (An arc-extreme-based basis is NOT canonical -- it rotates arbitrarily between swings and reduces
    cross-image PCA to noise; that failure mode is why this frame is gravity-anchored.)"""
    sp = swing_plane(arc); n = sp["normal"]
    if n @ up < 0: n = -n                                   # normal points sky-side
    v = up - (up @ n) * n; v /= np.linalg.norm(v) + 1e-12    # in-plane vertical
    u = np.cross(n, v)
    c = arc - arc.mean(0)
    if np.sum((c @ u)[: len(c) // 3]) < 0: u = -u            # backswing on a consistent side
    return np.stack([arc @ u, arc @ v], 1)                  # ball at origin by construction


def rasterize(uv, speed, k=K, span=SPAN):
    """Splat the plane curve into a kxk image, intensity ~ local speed (the render:time encoding).
    NO per-image normalization -- absolute speed is signal (a slow mishit must look faint)."""
    img = np.zeros((k, k), np.float32)
    ij = np.clip(((uv + span) / (2 * span) * (k - 1)).round().astype(int), 0, k - 1)
    for p in range(len(ij) - 1):                            # short line splats between samples
        a, b = ij[p], ij[p + 1]
        m = max(abs(b[0] - a[0]), abs(b[1] - a[1]), 1)
        for s in range(m + 1):
            x, y = a + ((b - a) * s / m).round().astype(int)
            img[y, x] = max(img[y, x], speed[p])
    return img


def swing_imprint(shot, seg, k=K):
    arc, phase = reconstruct_arc(shot, seg)
    uv = plane_coords(arc, world_up(shot))
    d = np.linalg.norm(np.diff(uv, axis=0), axis=1)
    speed = np.append(d, d[-1])                             # in-plane speed proxy per sample
    return rasterize(uv, speed, k)


def session_imprints(session_dir, k=K):
    ids, paths = _shot_paths(session_dir)
    out_ids, imgs = [], []
    for i, p in enumerate(paths):
        try:
            sh = load_shot(p); ti, _ = detect_impact(sh); sg = segment_swing(sh, ti)
            imgs.append(swing_imprint(sh, sg, k)); out_ids.append(ids[i])
        except Exception:
            continue
    imgs = np.array(imgs)
    return out_ids, imgs / (imgs.max() + 1e-9)              # SESSION-global scale: speed stays comparable


def image_features(imgs, k=6):
    """PCA scores of the flattened imprints (centered), plus each image's distance from the mean imprint
    (the image-space cocycle analogue)."""
    F = imgs.reshape(len(imgs), -1)
    mu = F.mean(0); C = F - mu
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    k = min(k, len(s))
    scores = C @ Vt[:k].T
    evr = (s[:k] ** 2) / (np.sum(s ** 2) + 1e-12)
    dist = np.linalg.norm(C, axis=1) / (np.linalg.norm(mu) + 1e-9)
    return scores, evr, dist
