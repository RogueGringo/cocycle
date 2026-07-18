"""TopologyField — the rank-7 particle field (the v0.10 object, repurposed as a representation space).

N particles live in R^7, sampled on a 7-sphere shell. They are advected by a divergence-free flow
built from random solenoidal Fourier modes (v(x)=Σ A_m sin(K_m·x+φ_m) with A_m·K_m=0), whose gains
are driven by the seven audio bands — so each band forces one region of the seven-dimensional flow.
Prosody adds global behaviour: onset (flux) kicks particles outward, loudness (rms) sets flow speed.
A soft radial spring keeps the cloud on the manifold. Pure numpy so the math is verifiable headless;
the GPU-shader advection to millions of particles is the P1 swap behind this same interface.

Output ctx['field'] = {'X': float32[N,7], 'band': int[N]}
"""
from __future__ import annotations
import numpy as np
from ..engine import Component


class TopologyField(Component):
    name = "field"

    def __init__(self, n=60000, modes=28, radius=1.5, speed=0.55, seed=7):
        self.n, self.M, self.R0, self.speed = n, modes, radius, speed
        self.seed = seed

    def setup(self, ctx):
        rng = np.random.default_rng(self.seed)
        X = rng.standard_normal((self.n, 7)).astype(np.float32)
        X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        self.X = X * self.R0
        self.band = (np.arange(self.n) % 7).astype(np.int32)      # stable colour/band membership
        # solenoidal Fourier modes
        self.K = (rng.standard_normal((self.M, 7)) * 1.7).astype(np.float32)
        A = rng.standard_normal((self.M, 7)).astype(np.float32)
        A -= (np.sum(A * self.K, axis=1, keepdims=True) /
              (np.sum(self.K * self.K, axis=1, keepdims=True) + 1e-9)) * self.K   # A ⟂ K -> div-free
        self.A = A
        self.phi = (rng.random(self.M) * 2 * np.pi).astype(np.float32)
        self.mode_band = (np.arange(self.M) % 7)
        ctx.meta["N"] = self.n

    def update(self, ctx):
        f = ctx.get("features")
        bands = f["bands7"] if f else np.full(7, 0.3, np.float32)
        rms, flux = (f["rms"], f["flux"]) if f else (0.3, 0.0)
        dt = ctx.dt

        proj = self.X @ self.K.T                      # (N,M)
        s = np.sin(proj + self.phi)                   # (N,M)
        gain = (0.35 + 1.1 * bands[self.mode_band]).astype(np.float32)   # audio drives each mode
        v = (s * gain) @ self.A                        # (N,7) divergence-free advection

        # per-axis breathing from the seven bands (each band pushes along its axis)
        v += 0.6 * (bands - bands.mean())[None, :]
        # onset kick: outward impulse on transients
        v += (0.8 * flux) * self.X

        spd = self.speed * (0.5 + 1.3 * rms)
        self.X += v * (dt * spd)

        # soft spring back onto the manifold shell (keeps it bounded, never rigid)
        r = np.linalg.norm(self.X, axis=1, keepdims=True) + 1e-9
        self.X += (self.R0 - r) / r * self.X * (0.08 + 0.10 * rms)
        np.clip(self.X, -6.0, 6.0, out=self.X)

        ctx.put(self.name, {"X": self.X, "band": self.band})
