"""GLRenderer — moderngl additive point renderer for the projected particles.

Decoupled from windowing: it takes any moderngl Context, so the same renderer runs in the live
window (moderngl-window) and in a headless standalone context (offscreen FBO) for smoke tests.
Each particle is a soft additive glow; colour comes from the 4th rotated coordinate.
"""
from __future__ import annotations
import numpy as np
import moderngl

VERT = """
#version 330
in vec2 in_pos; in float in_size; in vec3 in_col;
out vec3 v_col;
void main(){ v_col = in_col; gl_Position = vec4(in_pos, 0.0, 1.0); gl_PointSize = in_size; }
"""
FRAG = """
#version 330
in vec3 v_col; out vec4 f_col;
void main(){
    vec2 d = gl_PointCoord - vec2(0.5);
    float a = exp(-dot(d, d) * 7.0);
    if (a < 0.01) discard;
    f_col = vec4(v_col * a, a);
}
"""


def hue_rgb(h):
    h6 = (h % 1.0) * 6.0
    x = 1 - np.abs(h6 % 2 - 1); z = np.zeros_like(h6)
    r = np.select([h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5], [1, x, z, z, x], default=1)
    g = np.select([h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5], [x, 1, 1, x, z], default=z)
    b = np.select([h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5], [z, z, x, 1, 1], default=x)
    return np.stack([r, g, b], axis=1).astype(np.float32)


def proj_to_gl(pr, size_scale=3.0, hue_base=0.52, hue_span=0.42):
    """turn a Projector output into (xy, size, col) arrays for the renderer."""
    xy = pr["xy"]; near = pr["near"]; w = pr["w"]
    hue = hue_base + hue_span * np.tanh(w * 0.6)
    col = hue_rgb(hue) * np.clip(near[:, None] * 0.9, 0.05, 1.0)
    size = np.clip(near * size_scale, 1.0, 14.0).astype(np.float32)
    return xy.astype(np.float32), size, col


class GLRenderer:
    def __init__(self, ctx: moderngl.Context, max_points: int):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=VERT, fragment_shader=FRAG)
        self.vbo = ctx.buffer(reserve=max_points * 6 * 4)
        self.vao = ctx.vertex_array(self.prog, [(self.vbo, "2f 1f 3f", "in_pos", "in_size", "in_col")])
        ctx.enable(moderngl.PROGRAM_POINT_SIZE)

    def draw(self, xy, size, col):
        n = len(xy)
        buf = np.empty((n, 6), np.float32)
        buf[:, :2] = xy; buf[:, 2] = size; buf[:, 3:] = col
        self.vbo.write(buf.tobytes())
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE)   # additive glow
        self.vao.render(mode=moderngl.POINTS, vertices=n)
