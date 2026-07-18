"""Projector — 7-D → 2-D through the C(7,2)=21 rotation planes (the 21 "angles of viewing").

Maintains 21 plane angles, integrated each frame at speeds modulated by prosody (pitch sets the
base spin, brightness biases the w-planes, onset jolts). Builds the composed 7×7 rotation from 21
Givens rotations, then perspective-projects 7→3→2. The 4th rotated coordinate becomes hue, so the
palette turns as the object rotates through the higher dimensions.

Output ctx['proj'] = {'xy': float32[N,2] in ~[-1,1], 'near': float32[N], 'w': float32[N], 'band': int[N]}
"""
from __future__ import annotations
import numpy as np
from ..engine import Component

PLANES = [(i, j) for i in range(7) for j in range(i + 1, 7)]   # 21
FOCAL = 3.5
# viewing axis: how much each of dims 2..6 tilts "into the screen" (single perspective stage)
DEPTH_W = np.array([0.0, 0.0, 0.6, 0.4, 0.3, 0.2, 0.1], np.float32)


class Projector(Component):
    name = "proj"

    def __init__(self, scale=0.62, base=0.35, seed=3):
        self.scale, self.base = scale, base
        rng = np.random.default_rng(seed)
        self.ang = np.zeros(21, dtype=np.float64)
        self.rate = (0.4 + 0.9 * rng.random(21))          # per-plane base rate
        # which planes involve the higher dims (3..6) — biased by brightness
        self.hi = np.array([1.0 if (i >= 3 or j >= 3) else 0.4 for (i, j) in PLANES])

    def _rot(self):
        R = np.eye(7)
        c, s = np.cos(self.ang), np.sin(self.ang)
        for k, (i, j) in enumerate(PLANES):
            G = np.eye(7)
            G[i, i] = c[k]; G[j, j] = c[k]; G[i, j] = -s[k]; G[j, i] = s[k]
            R = G @ R
        return R.astype(np.float32)

    def update(self, ctx):
        f = ctx.get("features")
        f0 = f["f0"] if f else 0.3
        cen = f["centroid"] if f else 0.3
        flux = f["flux"] if f else 0.0
        speed = self.base * (0.5 + 1.5 * f0) * (1 + 2.0 * flux)
        self.ang += ctx.dt * speed * self.rate * (0.6 + cen * self.hi)

        X = ctx.get("field")["X"]
        band = ctx.get("field")["band"]
        p = X @ self._rot().T                            # (N,7) rotated

        # single-stage perspective: dims 2..6 tilt into the screen, project onto (0,1)
        depth = p @ DEPTH_W                              # (N,)
        persp = FOCAL / np.clip(FOCAL - depth, 0.8, None)
        xy = p[:, :2] * persp[:, None] * self.scale
        ctx.put(self.name, {
            "xy": xy.astype(np.float32),
            "near": np.clip(persp * 0.4, 0.05, 2.0).astype(np.float32),
            "w": p[:, 3].astype(np.float32),
            "band": band,
        })
