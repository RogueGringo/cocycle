"""P2 verification — Hopf fibration scene (spec p2-hopf-scene.md). Checks the GEOMETRY (fibers are genuine
Hopf fibers, on S³) numpy-side, then renders offscreen and checks non-black + audio reactivity. Headless."""
import sys, time
import numpy as np, moderngl
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from cocycle.hopf import hopf_fibers, hopf_map, HopfRenderer, ProsodyGivens4

B, M, RES = 500, 256, (640, 640)


def main():
    # E2 geometry (no GL needed): fibers on S³, Hopf map constant along each fiber
    S3, hue = hopf_fibers(B, M)
    norms = np.linalg.norm(S3, axis=-1)
    on_s3 = float(np.abs(norms - 1).max())
    p = hopf_map(S3)                                                # [B,M,3]
    fiber_var = float(p.var(axis=1).max())                         # variance of the S² image WITHIN each fiber
    print(f"[P2] geometry: {B} fibers x {M} pts   on-S³ max|‖x‖−1| = {on_s3:.2e}   "
          f"max within-fiber Hopf-image variance = {fiber_var:.2e}")

    ctx = moderngl.create_standalone_context(require=330)
    print(f"[P2] {ctx.info['GL_RENDERER'][:30]}  GL {ctx.info['GL_VERSION'][:12]}")
    fbo = ctx.framebuffer(color_attachments=[ctx.texture(RES, 3)])
    hr = HopfRenderer(ctx, B, M); pg = ProsodyGivens4()
    feat = lambda loud: {"bands7": np.full(7, 0.7 if loud else 0.05, np.float32),
                         "f0": 0.5 if loud else 0.1, "flux": 0.0}

    def render(loud, n=6):
        for _ in range(n):
            pg.update(feat(loud), 1 / 60); fbo.use(); ctx.clear(0.02, 0.02, 0.05)
            hr.draw(pg.R_flat())
        return np.frombuffer(fbo.read(components=3), np.uint8)

    a = render(True).astype(np.int16); lit = int(np.count_nonzero(a)); peak = int(a.max())
    b = render(False).astype(np.int16); react = float(np.abs(a - b).mean())
    t0 = time.perf_counter()
    for _ in range(60): pg.update(feat(True), 1 / 60); fbo.use(); ctx.clear(0.02, 0.02, 0.05); hr.draw(pg.R_flat())
    ctx.finish(); fps = 60 / (time.perf_counter() - t0)

    checks = {
        "E2a all fiber points lie on S³ (‖x‖≈1)": on_s3 < 1e-4,
        "E2b Hopf map constant along each fiber (genuine fibers)": fiber_var < 1e-4,
        "E1 renders offscreen non-black + many lit line pixels": lit > 5000 and peak > 40,
        "E3 audio-reactive (loud≠quiet)": react > 0.5,
    }
    print("\n[P2] acceptance criteria:")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n    lit {lit:,}  peak {peak}   audio Δpixels {react:.2f}   rate {fps:.0f} fps ({B} line-loops)")
    ok = all(checks.values())
    print(f"\n[P2] {'ALL ACCEPTANCE CRITERIA MET — genuine Hopf fibers, audio-reactive, rendered' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
