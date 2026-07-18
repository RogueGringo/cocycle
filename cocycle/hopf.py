"""Hopf fibration scene (spec p2-hopf-scene.md). A NEW visual object: S³→S², each S² base point lifting to a
great circle in S³. We upload the S³ fiber points once, then per frame the vertex shader rotates them by an
audio-driven SO(4) (6-plane Givens) rotation, stereographically projects S³→R³, and perspective-projects to
2D — drawing each fiber as a glowing closed loop (the interlocking, never-intersecting circles). torch-free.
"""
from __future__ import annotations
import itertools
import numpy as np
import moderngl

PLANES4 = list(itertools.combinations(range(4), 2))                 # 6 planes for SO(4)


def hopf_fibers(n_fibers=500, n_pts=256):
    """Return S³ fiber points [n_fibers, n_pts, 4] and a per-fiber hue. The Hopf map is constant along each."""
    i = np.arange(n_fibers) + 0.5
    phi = np.arccos(1 - 2 * i / n_fibers)                           # Fibonacci sphere base points on S²
    theta = np.pi * (1 + 5 ** 0.5) * i
    bz = np.clip(np.cos(phi), -0.999, 0.999)
    bx, by = np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta)
    z1 = np.sqrt((1 + bz) / 2).astype(complex)                      # preimage (z1 real, z2 complex)
    z2 = (bx + 1j * by) / np.sqrt(2 * (1 + bz))
    th = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    e = np.exp(1j * th)                                             # e^{iθ} sweep = the fiber circle
    Z1, Z2 = z1[:, None] * e[None, :], z2[:, None] * e[None, :]     # [B,M]
    S3 = np.stack([Z1.real, Z1.imag, Z2.real, Z2.imag], -1).astype(np.float32)   # [B,M,4] on S³
    hue = ((theta / (2 * np.pi)) % 1.0).astype(np.float32)
    return S3, hue


def hopf_map(x):
    """S³ → S² (for verification): p is invariant along a fiber."""
    x0, x1, x2, x3 = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    return np.stack([2 * (x0 * x2 + x1 * x3), 2 * (x1 * x2 - x0 * x3), x0**2 + x1**2 - x2**2 - x3**2], -1)


VERT = """#version 330
in vec4 s3; in float hue;
uniform float R[16]; uniform float scale; uniform float focal;
out vec3 v_col;
void main(){
    vec4 y;
    for (int i=0;i<4;i++){ float a=0.0; for (int j=0;j<4;j++) a += R[i*4+j]*s3[j]; y[i]=a; }
    float d = 1.0 - y.w;                                            // stereographic from north pole
    d = abs(d) < 0.06 ? (d < 0.0 ? -0.06 : 0.06) : d;
    vec3 q = clamp(y.xyz / d, -12.0, 12.0);
    float persp = focal / max(focal - q.z * 0.4, 0.6);             // R³ -> 2D
    gl_Position = vec4(q.xy * persp * scale, 0.0, 1.0);
    float h = hue;
    v_col = clamp(vec3(abs(h*6.0-3.0)-1.0, 2.0-abs(h*6.0-2.0), 2.0-abs(h*6.0-4.0)), 0.0, 1.0);
}
"""
FRAG = """#version 330
in vec3 v_col; out vec4 f_col;
void main(){ f_col = vec4(v_col * 0.55, 0.55); }
"""


class HopfRenderer:
    def __init__(self, ctx: moderngl.Context, n_fibers=500, n_pts=256):
        self.ctx = ctx; self.B, self.M = n_fibers, n_pts
        S3, hue = hopf_fibers(n_fibers, n_pts)
        self.S3 = S3
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.vbo = ctx.buffer(S3.reshape(-1, 4).tobytes())
        self.hbo = ctx.buffer(np.repeat(hue, n_pts).astype(np.float32).tobytes())
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "4f", "s3"), (self.hbo, "1f", "hue")])

    def draw(self, R_flat, scale=0.5, focal=3.0):
        self.prog["R"].value = tuple(float(v) for v in R_flat)
        self.prog["scale"].value = scale; self.prog["focal"].value = focal
        self.ctx.enable(moderngl.BLEND); self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        try: self.ctx.line_width = 1.4
        except Exception: pass
        for f in range(self.B):                                     # each fiber = one closed loop
            self.vao.render(mode=moderngl.LINE_LOOP, vertices=self.M, first=f * self.M)


class ProsodyGivens4:
    """numpy SO(4) rotor: 6 Givens angles advected by prosody, exposed as flat 16-float R for the shader."""
    def __init__(self, base=0.4, seed=5):
        rng = np.random.default_rng(seed)
        self.base = base; self.ang = np.zeros(6); self.rate = 0.5 + 0.8 * rng.random(6)

    def update(self, f, dt):
        f0 = f["f0"] if f else 0.3; flux = f["flux"] if f else 0.0
        bands = f["bands7"] if f else np.full(7, 0.3)
        vel = self.base * (0.5 + 1.5*f0) * self.rate + 0.7 * bands[:6] * self.rate + 2.0 * flux * self.rate
        self.ang = self.ang + dt * vel

    def R_flat(self):
        c, s = np.cos(self.ang), np.sin(self.ang); R = np.eye(4)
        for k, (i, j) in enumerate(PLANES4):
            G = np.eye(4); G[i, i] = c[k]; G[j, j] = c[k]; G[i, j] = -s[k]; G[j, i] = s[k]
            R = G @ R
        return R.astype(np.float32).ravel()
