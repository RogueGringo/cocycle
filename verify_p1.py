"""P1 verification — acceptance criteria A1–A4 for the GPU Givens projector (spec p1-givens-projector.md).
Runs the composition with GivensProjector over synthetic audio, asserts torch active, angles advect AND
respond to audio, R stays orthogonal + det+1 every frame, and the LED pipeline stays valid. Exits non-zero
on any breach."""
import sys, numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from cocycle.engine import Composition
from cocycle.components import AudioSource, FeatureExtractor, TopologyField, LEDMapper, Spatializer
from cocycle.components.projector_gpu import GivensProjector

FRAMES = 150


def main():
    import torch
    gp = GivensProjector()
    comp = Composition([AudioSource(mode="synth"), FeatureExtractor(), TopologyField(n=40000),
                        gp, LEDMapper(w=32, h=32), Spatializer()])
    comp.setup()
    dev = comp.ctx.meta.get("proj_device")
    print(f"[P1] torch {torch.__version__} active — projector device: {dev}  (CUDA available: {torch.cuda.is_available()})")

    ang_hist, orth_err, dets, led_energy, vels, energy = [], [], [], [], [], []
    for fr in range(FRAMES):
        prev = gp.ang.copy()
        ctx = comp.step()
        ang_hist.append(gp.ang.copy())
        e, d = gp.orthogonality(); orth_err.append(e); dets.append(d)
        led_energy.append(float(ctx.get("led").mean()))
        f = ctx.get("features")
        vels.append(float(np.abs(gp.ang - prev).mean()))               # per-frame angular velocity
        energy.append(float(f["rms"] + f["bands7"].mean() + f["flux"]))  # audio activity

    ang_hist = np.array(ang_hist)
    vv, ee = np.array(vels[2:]), np.array(energy[2:])                   # skip init transient
    resp_corr = float(np.corrcoef(vv, ee)[0, 1]) if vv.std() > 0 and ee.std() > 0 else 0.0
    checks = {
        "A1 torch active + device reported": dev in ("cuda", "cpu"),
        "A2a angles advect (change over frames)": ang_hist.std(0).mean() > 1e-3,
        "A2b angle velocity tracks audio activity (|corr|>0.2)": abs(resp_corr) > 0.2,
        "A3a orthogonality max|RRᵀ−I| < 1e-6 every frame": max(orth_err) < 1e-6,
        "A3b det(R) ~ +1 every frame": max(abs(np.array(dets) - 1)) < 1e-4,
        "A4 LED pipeline valid + changing": np.mean(led_energy) > 1e-3 and np.std(led_energy) > 1e-5,
    }
    print("\n[P1] acceptance criteria:")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n    orthogonality residual: max {max(orth_err):.2e} over {FRAMES} frames")
    print(f"    det(R) range: [{min(dets):.6f}, {max(dets):.6f}]")
    print(f"    angle advection std {ang_hist.std(0).mean():.3f}   velocity↔audio corr {resp_corr:+.2f}")
    ok = all(checks.values())
    print(f"\n[P1] {'ALL ACCEPTANCE CRITERIA MET' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
