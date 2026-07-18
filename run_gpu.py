"""Live GPU viewer — the million-particle mode. Compute field advects particles in an SSBO; the vertex
shader reads them in place and projects with the audio-driven Givens rotation. Fully on-GPU, torch-free.

    python run_gpu.py                      # built-in moan synth, 1M particles
    python run_gpu.py --n 4000000 --audio mic
    python run_gpu.py --audio file --file track.wav

Requires GL 4.3+ (SSBO in the vertex shader). I cannot see this window from the build env — verified
offscreen by verify_p1c.py; this is the live path for the rig.
"""
import argparse
import numpy as np
import moderngl
import moderngl_window as mglw
from cocycle.engine import Context
from cocycle.components.audio import AudioSource
from cocycle.components.features import FeatureExtractor
from cocycle.compute_field import ComputeField
from cocycle.gpu_render import GPURenderer, ProsodyGivens

P = argparse.ArgumentParser()
P.add_argument("--audio", default="synth", choices=["synth", "mic", "file"])
P.add_argument("--file", default=None)
P.add_argument("--n", type=int, default=1_000_000)
ARGS, _ = P.parse_known_args()


class App(mglw.WindowConfig):
    title = "COCYCLE — GPU rank-7 field"
    gl_version = (4, 3)                                            # SSBO in vertex shader
    window_size = (1280, 800)
    aspect_ratio = None

    def __init__(self, **kw):
        super().__init__(**kw)
        self.cf = ComputeField(self.ctx, n=ARGS.n)
        self.gr = GPURenderer(self.ctx)
        self.pg = ProsodyGivens()
        self.audio = AudioSource(mode=ARGS.audio, path=ARGS.file)
        self.feat = FeatureExtractor()
        self.c = Context(); self.audio.setup(self.c); self.feat.setup(self.c)
        self.fade = self.ctx.program(
            vertex_shader="#version 330\nin vec2 p; void main(){ gl_Position=vec4(p,0,1);}",
            fragment_shader="#version 330\nout vec4 c; void main(){ c=vec4(0.02,0.02,0.05,0.14);}")
        self.fq = self.ctx.vertex_array(self.fade,
                    [(self.ctx.buffer(np.array([-1, -1, 3, -1, -1, 3], np.float32).tobytes()), "2f", "p")])
        print(f"[COCYCLE-GPU] particles={ARGS.n:,}  {self.ctx.info['GL_RENDERER']}  GL {self.ctx.info['GL_VERSION'][:12]}")

    def _frame(self):
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        self.fq.render(mode=moderngl.TRIANGLES, vertices=3)        # motion-trail fade
        self.audio.update(self.c); self.feat.update(self.c)
        f = self.c.get("features")
        self.cf.step(f["bands7"], rms=f["rms"], flux=f["flux"], dt=self.c.dt)
        self.pg.update(f, self.c.dt)
        self.gr.draw(self.cf.pos, self.pg.R_flat(), ARGS.n)
        self.c.t += self.c.dt; self.c.frame += 1

    def on_render(self, time, frametime):
        self._frame()

    def render(self, time, frametime):
        self._frame()


if __name__ == "__main__":
    mglw.run_window_config(App)
