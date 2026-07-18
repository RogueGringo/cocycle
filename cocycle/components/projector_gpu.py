"""GivensProjector — the P1 GPU/differentiable projector (spec: .claude/specs/p1-givens-projector.md).

Drop-in for the numpy Projector: same output contract (ctx['proj'] = {xy, near, w, band}) and the same
single-stage perspective, but the 7-D rotation is the differentiable, orthogonal-by-construction GivensRotor
whose 21 plane-angles are advected live by prosody. Device-agnostic: runs on the RTX 5070 when torch sees CUDA,
falls back to CPU otherwise. torch is imported HERE only — the core engine stays torch-free.
"""
from __future__ import annotations
import itertools
import numpy as np
import torch
from ..engine import Component
from ..givens import GivensRotor

PLANES = list(itertools.combinations(range(7), 2))          # 21
FOCAL = 3.5
DEPTH_W = np.array([0.0, 0.0, 0.6, 0.4, 0.3, 0.2, 0.1], np.float32)


class GivensProjector(Component):
    name = "proj"

    def __init__(self, scale=0.62, base=0.35, seed=3, device=None):
        self.scale, self.base = scale, base
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rotor = GivensRotor(7, learnable=False).to(self.device)
        rng = np.random.default_rng(seed)
        self.ang = np.zeros(21, dtype=np.float32)                       # integrated angles (numpy state)
        self.rate = (0.4 + 0.9 * rng.random(21)).astype(np.float32)     # per-plane base rate
        self.hi = np.array([1.0 if (i >= 3 or j >= 3) else 0.4 for (i, j) in PLANES], np.float32)
        self.plane_band = np.array([k % 7 for k in range(21)])          # route the 7 bands onto the 21 planes
        self._depth = torch.tensor(DEPTH_W, device=self.device)
        self.last_R = None                                              # exposed for verification (A3)

    def setup(self, ctx):
        ctx.meta["proj_device"] = self.device

    def update(self, ctx):
        f = ctx.get("features")
        f0 = f["f0"] if f else 0.3
        cen = f["centroid"] if f else 0.3
        flux = f["flux"] if f else 0.0
        bands = f["bands7"] if f else np.full(7, 0.3, np.float32)
        # per-plane angular velocity from prosody: base spin (f0) x brightness-biased w-planes + band drive + transient jolt
        vel = (self.base * (0.5 + 1.5 * f0) * self.rate * (0.6 + cen * self.hi)
               + 0.8 * bands[self.plane_band] * self.rate
               + 2.0 * flux * self.rate)
        self.ang = self.ang + ctx.dt * vel
        self.rotor.set_angles(torch.tensor(self.ang, device=self.device))
        R = self.rotor.matrix()                                         # [7,7] orthogonal, on device
        self.last_R = R.detach()

        fld = ctx.get("field")
        X = torch.as_tensor(fld["X"], dtype=torch.float32, device=self.device)
        p = X @ R.transpose(0, 1)                                       # rotate on the GPU
        depth = p @ self._depth
        persp = FOCAL / torch.clamp(FOCAL - depth, min=0.8)
        xy = (p[:, :2] * persp[:, None] * self.scale)
        near = torch.clamp(persp * 0.4, 0.05, 2.0)
        ctx.put(self.name, {
            "xy": xy.cpu().numpy().astype(np.float32),
            "near": near.cpu().numpy().astype(np.float32),
            "w": p[:, 3].cpu().numpy().astype(np.float32),
            "band": fld["band"],
        })

    def orthogonality(self):
        """(max|RRᵀ−I|, det) of the last frame's rotation — the A3 check."""
        R = self.last_R
        err = (R @ R.transpose(0, 1) - torch.eye(7, device=R.device)).abs().max().item()
        return err, torch.linalg.det(R).item()
