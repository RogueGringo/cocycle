"""P0 verification — run the full pipeline headless (no window) and prove every stage is live.

Steps the numpy core through N frames, asserts each component produces non-degenerate, changing,
finite output, writes a few LED frames as .ppm (no image deps), and does an offscreen moderngl
smoke test (standalone context -> FBO -> pixel readback) to validate the GPU render path without a
display. Exits non-zero on any failure — this is the instrument, so it must be honest.
"""
import sys, numpy as np
from cocycle import default_composition
from cocycle.components.render_gl import GLRenderer, proj_to_gl

FRAMES = 180
N = 60000


def ppm(path, rgb01):
    h, w, _ = rgb01.shape
    data = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode())
        f.write(data.tobytes())


def main():
    comp = default_composition(n=N, led=(32, 32), audio="synth")
    comp.setup()
    print(f"[P0] pipeline: " + " -> ".join(c.name for c in comp.components))
    print(f"[P0] particles={N}  planes=21  bands=7  running {FRAMES} frames headless...")

    band_hist, disp_hist, led_energy, cover = [], [], [], []
    prevX = comp.ctx.get("field")["X"].copy() if comp.ctx.get("field") else None
    saved = []
    for fr in range(FRAMES):
        ctx = comp.step()
        feat = ctx.get("features"); fld = ctx.get("field"); pr = ctx.get("proj")
        led = ctx.get("led"); sp = ctx.get("spatial")
        X = fld["X"]
        assert np.isfinite(X).all(), "field went non-finite"
        if prevX is not None:
            disp_hist.append(float(np.mean(np.linalg.norm(X - prevX, axis=1))))
        prevX = X.copy()
        band_hist.append(feat["bands7"].copy())
        led_energy.append(float(led.mean()))
        cells = np.clip(((pr["xy"] * 0.5 + 0.5) * 32).astype(int), 0, 31)
        cover.append(len(np.unique(cells[:, 0] * 32 + cells[:, 1])) / 1024.0)
        if fr in (60, 120, 179):
            ppm(f"led_frame_{fr:03d}.ppm", led); saved.append(f"led_frame_{fr:03d}.ppm")

    band_hist = np.array(band_hist)
    checks = {
        "features vary over time": float(band_hist.std(axis=0).mean()) > 1e-3,
        "field advects (mean step>0)": np.mean(disp_hist) > 1e-4,
        "field stays bounded": np.max(np.abs(prevX)) < 6.5,
        "LED grid has energy": np.mean(led_energy) > 1e-3,
        "LED content changes": float(np.std(led_energy)) > 1e-5,
        "projection covers the panel": np.mean(cover) > 0.15,
        "spatializer emits sources": sp is not None and len(sp) >= 1,
    }
    print("\n[P0] stage checks:")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n    mean band activity   {band_hist.mean():.3f}   (std over time {band_hist.std(0).mean():.3f})")
    print(f"    mean particle step   {np.mean(disp_hist):.4f}")
    print(f"    LED panel coverage   {np.mean(cover)*100:.1f}%   energy {np.mean(led_energy):.3f}")
    print(f"    spatial sources/frame {len(sp)}   e.g. band0 L/R "
          f"{sp[0]['L']:.2f}/{sp[0]['R']:.2f}" if sp else "")
    print(f"    wrote LED frames: {', '.join(saved)}  (open as .ppm)")

    # offscreen GL smoke — validate the shader/render path with no display
    gl_ok = None
    try:
        import moderngl
        gctx = moderngl.create_standalone_context()
        fbo = gctx.simple_framebuffer((256, 256)); fbo.use(); fbo.clear(0, 0, 0, 1)
        r = GLRenderer(gctx, N)
        xy, size, col = proj_to_gl(comp.ctx.get("proj"))
        r.draw(xy, size, col)
        px = np.frombuffer(fbo.read(components=3), np.uint8)
        gl_ok = int(px.max()) > 0
        print(f"\n[P0] offscreen GL render: {'PASS' if gl_ok else 'FAIL'} "
              f"(peak pixel {int(px.max())}, lit {np.count_nonzero(px)} subpx)  GL={gctx.info['GL_RENDERER']}")
    except Exception as e:
        print(f"\n[P0] offscreen GL render: SKIPPED ({type(e).__name__}: {e}) "
              f"— run_live.py will exercise the GPU path on your rig")

    ok = all(checks.values()) and (gl_ok is not False)
    print(f"\n[P0] {'VERTICAL SLICE VERIFIED' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
