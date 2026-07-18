"""GivensRotor — the differentiable, orthogonal-by-construction N-D rotation at the heart of COCYCLE.

The research thread's one durable structural fact: the payload is the *orthogonal rotation itself* — rotation
preserves (gradient resolution, distances, angles) where compression starves. This module makes that operation
first-class: a rank-N rotation parameterized by its C(N,2) Givens plane angles, composed as a product of plane
rotations so the result is orthogonal for ANY angles (no projection, no re-orthogonalization). Fully
differentiable in the angles (autograd-clean: each Givens is built additively from constant basis matrices, no
in-place writes), so the same object serves the audio-driven projector today and a *learned* Stiefel-manifold
transform (put a gradient through it) tomorrow.

    rot = GivensRotor(7)              # 21 planes for the rank-7 field
    R   = rot.matrix()               # [7,7] orthogonal, differentiable in rot.angles
    y   = rot(x)                     # rotate points x[...,7]
    rot.set_angles(audio_driven)     # drive the 21 angles from features (no grad)
"""
from __future__ import annotations
import itertools
import torch
import torch.nn as nn


class GivensRotor(nn.Module):
    def __init__(self, n: int = 7, learnable: bool = False, init=None):
        super().__init__()
        self.n = n
        self.planes = list(itertools.combinations(range(n), 2))          # C(n,2) planes
        a = torch.zeros(len(self.planes)) if init is None else torch.as_tensor(init, dtype=torch.float32)
        self.angles = nn.Parameter(a, requires_grad=learnable)
        # constant basis matrices per plane (registered as buffers so .to(device) moves them)
        Pd = torch.zeros(len(self.planes), n, n)                          # e_i e_i^T + e_j e_j^T
        Pa = torch.zeros(len(self.planes), n, n)                          # e_i e_j^T - e_j e_i^T
        for k, (i, j) in enumerate(self.planes):
            Pd[k, i, i] = 1.0; Pd[k, j, j] = 1.0
            Pa[k, i, j] = 1.0; Pa[k, j, i] = -1.0
        self.register_buffer("Pd", Pd)
        self.register_buffer("Pa", Pa)
        self.register_buffer("I", torch.eye(n))

    def matrix(self) -> torch.Tensor:
        """Composed N×N orthogonal matrix R = ∏_k G_k, differentiable in the angles."""
        c = torch.cos(self.angles); s = torch.sin(self.angles)
        R = self.I
        for k in range(len(self.planes)):
            # G_k = I + (cos-1)(e_i e_i^T + e_j e_j^T) + sin(e_i e_j^T - e_j e_i^T)  — exact plane rotation, no in-place
            G = self.I + (c[k] - 1.0) * self.Pd[k] + s[k] * self.Pa[k]
            R = G @ R
        return R

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.matrix().transpose(-1, -2)

    @torch.no_grad()
    def set_angles(self, vals):
        self.angles.copy_(torch.as_tensor(vals, dtype=self.angles.dtype, device=self.angles.device))


if __name__ == "__main__":
    torch.manual_seed(0)
    rot = GivensRotor(7, learnable=True)
    rot.set_angles(torch.randn(21) * 1.3)
    R = rot.matrix()
    orth = (R @ R.T - torch.eye(7)).abs().max().item()
    det = torch.linalg.det(R).item()
    # gradient flows to the angles through an angle-DEPENDENT objective (norm-preserving losses are
    # rotation-invariant -> zero grad by construction; use a linear functional of R instead)
    rot.angles.grad = None; rot.angles.requires_grad_(True)
    target = torch.randn(7, 7)
    loss = (rot.matrix() - target).pow(2).sum()
    loss.backward()
    gnorm = rot.angles.grad.norm().item()
    print(f"[GivensRotor] orthogonality max|R R^T - I| = {orth:.2e}  (want ~0)")
    print(f"[GivensRotor] det(R) = {det:.6f}  (want +1)")
    print(f"[GivensRotor] gradient to 21 angles: |g| = {gnorm:.3f}  (want >0 -- differentiable)")
    print("PASS" if orth < 1e-5 and abs(det - 1) < 1e-4 and gnorm > 0 else "FAIL")
