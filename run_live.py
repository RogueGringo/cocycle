"""Live viewer for the authoring rig — real window, real GPU, real-time.

    python run_live.py                 # built-in "moan" synth
    python run_live.py --audio mic     # live input
    python run_live.py --audio file --file track.wav

Renders the projected rank-7 field as additive glow. This is the P0 preview; the millions-of-
particles GPU-shader advection and the card-2 live-embedding source drop in behind the same
component interfaces (P1). I cannot see this window from the build environment — it's for your rig.
"""
import argparse
import numpy as np
import moderngl
import moderngl_window as mglw
from cocycle import default_composition
from cocycle.components.render_gl import GLRenderer, proj_to_gl

P = argparse.ArgumentParser()
P.add_argument("--audio", default="synth", choices=["synth", "mic", "file"])
P.add_argument("--file", default=None)
P.add_argument("--n", type=int, default=200000)
ARGS, _ = P.parse_known_args()


class App(mglw.WindowConfig):
    title = "COCYCLE — rank-7 field"
    gl_version = (3, 3)
    window_size = (1280, 800)
    aspect_ratio = None

    def __init__(self, **kw):
        super().__init__(**kw)
        self.comp = default_composition(n=ARGS.n, audio=ARGS.audio, path=ARGS.file)
        self.comp.setup()
        self.renderer = GLRenderer(self.ctx, ARGS.n)
        self.fade = self.ctx.program(
            vertex_shader="#version 330\nin vec2 p; void main(){ gl_Position=vec4(p,0,1);}",
            fragment_shader="#version 330\nout vec4 c; void main(){ c=vec4(0.02,0.02,0.05,0.16);}")
        quad = np.array([-1, -1, 3, -1, -1, 3], np.float32)
        self.fq = self.ctx.vertex_array(self.fade, [(self.ctx.buffer(quad.tobytes()), "2f", "p")])

    def _frame(self):
        # motion-trail fade instead of hard clear
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.fq.render(mode=moderngl.TRIANGLES, vertices=3)
        self.comp.step()
        xy, size, col = proj_to_gl(self.comp.ctx.get("proj"))
        self.renderer.draw(xy, size, col)

    def on_render(self, time, frametime):   # moderngl-window >=2.4
        self._frame()

    def render(self, time, frametime):       # older moderngl-window
        self._frame()


if __name__ == "__main__":
    mglw.run_window_config(App)
