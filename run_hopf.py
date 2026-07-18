"""Live Hopf viewer — the interlocking-circles scene. S³ fibers rotated by an audio-driven SO(4), stereographic
+ perspective projected, drawn as glowing loops. torch-free, GL 3.3.

    python run_hopf.py                       # moan synth, 500 fibers
    python run_hopf.py --fibers 900 --audio mic
    python run_hopf.py --audio file --file track.wav

Verified offscreen (verify_hopf.py: genuine Hopf fibers + render); this is the live path for the rig.
"""
import argparse
import numpy as np
import moderngl
import moderngl_window as mglw
from cocycle.engine import Context
from cocycle.components.audio import AudioSource
from cocycle.components.features import FeatureExtractor
from cocycle.hopf import HopfRenderer, ProsodyGivens4

P = argparse.ArgumentParser()
P.add_argument("--audio", default="synth", choices=["synth", "mic", "file"])
P.add_argument("--file", default=None)
P.add_argument("--fibers", type=int, default=500)
P.add_argument("--pts", type=int, default=256)
ARGS, _ = P.parse_known_args()


class App(mglw.WindowConfig):
    title = "COCYCLE — Hopf fibration"
    gl_version = (3, 3)
    window_size = (1000, 1000)
    aspect_ratio = 1.0

    def __init__(self, **kw):
        super().__init__(**kw)
        self.hr = HopfRenderer(self.ctx, ARGS.fibers, ARGS.pts)
        self.pg = ProsodyGivens4()
        self.audio = AudioSource(mode=ARGS.audio, path=ARGS.file)
        self.feat = FeatureExtractor()
        self.c = Context(); self.audio.setup(self.c); self.feat.setup(self.c)
        self.fade = self.ctx.program(
            vertex_shader="#version 330\nin vec2 p; void main(){ gl_Position=vec4(p,0,1);}",
            fragment_shader="#version 330\nout vec4 c; void main(){ c=vec4(0.02,0.02,0.05,0.12);}")
        self.fq = self.ctx.vertex_array(self.fade,
                    [(self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes()), "2f", "p")])
        print(f"[COCYCLE-HOPF] fibers={ARGS.fibers} x {ARGS.pts} pts  {self.ctx.info['GL_RENDERER']}")

    def _frame(self):
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.fq.render(mode=moderngl.TRIANGLES, vertices=3)
        self.audio.update(self.c); self.feat.update(self.c)
        self.pg.update(self.c.get("features"), self.c.dt)
        self.hr.draw(self.pg.R_flat())
        self.c.t += self.c.dt; self.c.frame += 1

    def on_render(self, time, frametime):
        self._frame()

    def render(self, time, frametime):
        self._frame()


if __name__ == "__main__":
    mglw.run_window_config(App)
