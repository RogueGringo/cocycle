"""GPU-resident particle renderer — closes the compute loop (spec p1b follow-on).

The vertex shader reads each particle's 7-D position straight from the compute field's SSBO (vertex-pulling by
gl_VertexID — no vertex buffer, no CPU roundtrip), applies the audio-driven 7x7 Givens rotation (uniform) and
the single-stage perspective, and emits a glowing point. So the million-particle swarm advected by the compute
shader is drawn without the positions ever leaving the GPU. Render path is moderngl+numpy only (torch-free):
the rotation matrix is built in numpy from the same prosody->angle mapping as the differentiable GivensProjector.
"""
from __future__ import annotations
import itertools
import numpy as np
import moderngl

PLANES = list(itertools.combinations(range(7), 2))                 # 21

VERT = """#version 430
layout(std430, binding=0) buffer Pos { float p[]; };
uniform float R[49];                                                // row-major 7x7 orthogonal
uniform float scale; uniform float focal;
out vec3 v_col;
const float DW[7] = float[7](0.0, 0.0, 0.6, 0.4, 0.3, 0.2, 0.1);
void main(){
    uint b = uint(gl_VertexID) * 8u;
    float x[7]; for (int k=0;k<7;k++) x[k]=p[b+uint(k)];
    float y[7];
    for (int i=0;i<7;i++){ float a=0.0; for (int j=0;j<7;j++) a += R[i*7+j]*x[j]; y[i]=a; }
    float depth=0.0; for (int k=0;k<7;k++) depth += DW[k]*y[k];
    float persp = focal / max(focal - depth, 0.8);
    gl_Position = vec4(vec2(y[0], y[1]) * persp * scale, 0.0, 1.0);
    float near = clamp(persp*0.4, 0.05, 2.0);
    gl_PointSize = clamp(near*3.0, 1.0, 14.0);
    float h = 0.52 + 0.42*tanh(y[3]*0.6);                          // hue from the 4th rotated coord
    vec3 rgb = clamp(vec3(abs(h*6.0-3.0)-1.0, 2.0-abs(h*6.0-2.0), 2.0-abs(h*6.0-4.0)), 0.0, 1.0);
    v_col = rgb * near;
}
"""
FRAG = """#version 430
in vec3 v_col; out vec4 f_col;
void main(){ vec2 d=gl_PointCoord-vec2(0.5); float a=exp(-dot(d,d)*7.0); if(a<0.01) discard; f_col=vec4(v_col*a,a); }
"""


class GPURenderer:
    def __init__(self, ctx: moderngl.Context):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.vao = ctx.vertex_array(self.prog, [])                  # attributeless — positions from the SSBO
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    def draw(self, pos_ssbo, R_flat, n, scale=0.62, focal=3.5):
        pos_ssbo.bind_to_storage_buffer(0)
        self.prog["R"].value = tuple(float(v) for v in R_flat)
        self.prog["scale"].value = scale; self.prog["focal"].value = focal
        self.ctx.enable(moderngl.BLEND); self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)
        self.vao.render(mode=moderngl.POINTS, vertices=n)


class ProsodyGivens:
    """numpy 7-D Givens rotor: 21 angles advected by prosody, exposed as a flat 49-float R for the shader."""
    def __init__(self, base=0.35, seed=3):
        rng = np.random.default_rng(seed)
        self.base = base
        self.ang = np.zeros(21)
        self.rate = 0.4 + 0.9 * rng.random(21)
        self.hi = np.array([1.0 if (i >= 3 or j >= 3) else 0.4 for (i, j) in PLANES])
        self.pb = np.array([k % 7 for k in range(21)])

    def update(self, f, dt):
        f0 = f["f0"] if f else 0.3; cen = f["centroid"] if f else 0.3; flux = f["flux"] if f else 0.0
        bands = f["bands7"] if f else np.full(7, 0.3)
        vel = (self.base * (0.5 + 1.5*f0) * self.rate * (0.6 + cen*self.hi)
               + 0.8 * bands[self.pb] * self.rate + 2.0 * flux * self.rate)
        self.ang = self.ang + dt * vel

    def R_flat(self):
        c, s = np.cos(self.ang), np.sin(self.ang); R = np.eye(7)
        for k, (i, j) in enumerate(PLANES):
            G = np.eye(7); G[i, i] = c[k]; G[j, j] = c[k]; G[i, j] = -s[k]; G[j, i] = s[k]
            R = G @ R
        return R.astype(np.float32).ravel()
