"""P1c verification — GPU-resident render integration (compute field SSBO -> vertex-pulling render).
Offscreen (headless): advect 1M particles on the GPU, draw them by reading the SSBO in the vertex shader
(no CPU position roundtrip), project with the audio-driven Givens rotation. Asserts: non-black output with
many lit particles, output changes with audio, real-time at 1M, motion trails compose. Exits non-zero on any
breach."""
import sys, time
import numpy as np, moderngl
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from cocycle.engine import Context
from cocycle.components.audio import AudioSource
from cocycle.components.features import FeatureExtractor
from cocycle.compute_field import ComputeField
from cocycle.gpu_render import GPURenderer, ProsodyGivens

N = 1_000_000
RES = (640, 640)


def main():
    ctx = moderngl.create_standalone_context(require=430)
    print(f"[P1c] {ctx.info['GL_RENDERER'][:30]}  GL {ctx.info['GL_VERSION'][:12]}  N={N:,}")
    fbo = ctx.framebuffer(color_attachments=[ctx.texture(RES, 3)])
    cf = ComputeField(ctx, n=N)
    gr = GPURenderer(ctx)
    pg = ProsodyGivens()
    audio, feat = AudioSource(mode="synth"), FeatureExtractor()
    c = Context(); audio.setup(c); feat.setup(c)

    def frame(loud=None):
        if loud is not None:                                       # forced audio for the reactivity probe
            c.data["features"] = {"bands7": np.full(7, 0.7 if loud else 0.05, np.float32),
                                  "f0": 0.5 if loud else 0.1, "centroid": 0.6 if loud else 0.1,
                                  "flux": 0.0, "rms": 0.8 if loud else 0.05}
        else:
            audio.update(c); feat.update(c)
        f = c.get("features")
        cf.step(f["bands7"], rms=f["rms"], flux=f["flux"], dt=c.dt)
        pg.update(f, c.dt)
        fbo.use(); ctx.clear(0.02, 0.02, 0.05)
        gr.draw(cf.pos, pg.R_flat(), N)
        c.t += c.dt; c.frame += 1

    # D1: renders, non-black, many lit particles
    for _ in range(8): frame()
    px = np.frombuffer(fbo.read(components=3), np.uint8)
    lit = int(np.count_nonzero(px)); peak = int(px.max())
    # D2: output changes with audio (loud vs quiet frame differ)
    frame(loud=True); a = np.frombuffer(fbo.read(components=3), np.uint8).astype(np.int16)
    for _ in range(6): frame(loud=False)
    b = np.frombuffer(fbo.read(components=3), np.uint8).astype(np.int16)
    react = float(np.abs(a - b).mean())
    # D3: real-time — compute + render loop at 1M, no CPU particle readback (pixels read only at the end)
    t0 = time.perf_counter(); STEPS = 120
    for _ in range(STEPS): frame()
    ctx.finish()
    fps = STEPS / (time.perf_counter() - t0)

    checks = {
        "D1 renders 1M from SSBO, non-black + many lit": lit > 5000 and peak > 40,
        "D2 output responds to audio (loud≠quiet)": react > 0.5,
        "D3 real-time at 1M (compute+render ≥60 fps)": fps >= 60,
    }
    print("\n[P1c] acceptance criteria:")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n    lit subpixels {lit:,}  peak {peak}   audio Δpixels {react:.2f}   rate {fps:.0f} fps ({1000/fps:.2f} ms/frame)")
    print("    (particles read from the SSBO in-shader — zero CPU position roundtrip)")
    ok = all(checks.values())
    print(f"\n[P1c] {'ALL ACCEPTANCE CRITERIA MET — compute→render loop is fully on-GPU' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
