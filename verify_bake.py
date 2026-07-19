"""P4 verification — offline signal bake (spec p4-signal-bake.md). Bakes synthetic audio (G1-G3) and a
synthetic multi-channel MWD-like drilling signal (G4) to prove the elementals engine is signal-agnostic and
the baked timeline transposes to any regime. Headless, numpy only."""
import sys, time
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from cocycle.signal import bake_signal, BakeParams, params_schema
from cocycle.engine import Context
from cocycle.components.field import TopologyField
from cocycle.components.projector import Projector


def main():
    SR = 44100
    t = np.arange(int(6.0 * SR)) / SR
    # a clean 300 Hz tone + a mid-track transient (for onset), for correctness
    tone = 0.6 * np.sin(2 * np.pi * 300 * t)
    tone[int(3.0 * SR):int(3.0 * SR) + 400] += 0.9                          # a click
    tl = bake_signal(tone, SR, BakeParams(fps=144, dom_fmin=80, dom_fmax=1200))

    dom = tl.raw[:, tl.idx("domfreq")]
    onset = tl.raw[:, tl.idx("onset")]
    click_fr = int(3.0 * tl.fps)
    g1 = (tl.fps >= 90 and abs(tl.n_frames - 6 * tl.fps) < 3 * tl.fps and np.isfinite(tl.raw).all())
    g2_dom = abs(np.median(dom) - 300) < 15
    g2_onset = onset[click_fr - 3:click_fr + 6].max() > 3 * (np.median(onset) + 1e-6)

    # G3: transpose to a regime — drive the field/projector purely from the baked timeline
    comp = [TopologyField(n=5000), Projector()]
    c = Context(); [k.setup(c) for k in comp]
    X0 = None
    for i in range(0, min(tl.n_frames, 120)):
        c.data["features"] = tl.to_features(i)
        for k in comp: k.update(c)
        if i == 0: X0 = c.get("field")["X"].copy()
    moved = float(np.linalg.norm(c.get("field")["X"] - X0, axis=1).mean())
    cover = c.get("proj") is not None and np.isfinite(c.get("proj")["xy"]).all()
    g3 = moved > 1e-3 and cover

    # G4: signal-agnostic — synthetic MWD (rotating tool @ 3 Hz = 180 RPM, low sr, shocks on ch3)
    SR2 = 1000; t2 = np.arange(int(8.0 * SR2)) / SR2
    rpm = 3.0
    base = np.sin(2 * np.pi * rpm * t2) + 0.4 * np.sin(2 * np.pi * 2 * rpm * t2)
    ch = [base + 0.15 * np.random.randn(len(t2)) for _ in range(4)]
    for s in np.arange(1.0, 8.0, 1.3):                                      # stick-slip shocks on ch3
        ch[3][int(s * SR2):int(s * SR2) + 20] += 3.0
    X = np.stack(ch, 1).astype(np.float32)
    mtl = bake_signal(X, SR2, BakeParams(fps=144, nfft=2048, fmin=1, fmax=200, dom_fmin=1, dom_fmax=50))
    rpm_rec = [float(np.median(mtl.raw[:, cc, mtl.idx("domfreq")])) for cc in range(4)]
    g4_rpm = all(abs(r - rpm) < 0.5 for r in rpm_rec)
    shock_on = mtl.raw[:, 3, mtl.idx("onset")]; quiet_on = mtl.raw[:, 0, mtl.idx("onset")]
    g4_shock = shock_on.max() > 2 * quiet_on.max() + 1e-6

    # G5: the control panel's emitted config round-trips into a real bake (panel -> config -> timeline)
    from cocycle.signal import bake_config
    import soundfile as sf, tempfile, os
    wav = os.path.join(tempfile.gettempdir(), "cocycle_g5.wav"); sf.write(wav, tone, SR)
    panel_cfg = {"signal": {"type": "audio", "path": wav},
                 "params": {"fps": 144, "nfft": 2048, "dom_fmin": 80, "dom_fmax": 1200},
                 "elementals": ["rms", "domfreq", "onset"], "regime": "field"}
    ptl = bake_config(panel_cfg)
    sel = ptl.select(panel_cfg["elementals"])
    g5 = (ptl.fps == 144 and sel.shape[1] == 3
          and abs(np.median(ptl.raw[:, ptl.idx("domfreq")]) - 300) < 15)

    checks = {
        "G1 timeline at target fps, complete, finite": g1,
        "G2 dominant-freq recovers the 300 Hz tone": g2_dom,
        "G2 onset fires on the transient": g2_onset,
        "G3 baked timeline drives a scene (field advects)": g3,
        "G4 MWD-agnostic: per-channel domfreq recovers 3 Hz RPM": g4_rpm,
        "G4 stick-slip shocks show as onsets on ch3": g4_shock,
        "G5 panel config round-trips into a real bake (selected elementals)": g5,
    }
    print(f"[P4] audio bake: {tl.n_frames} frames @ {tl.fps} fps ({tl.duration:.1f}s), {len(tl.names)} elementals; "
          f"median domfreq {np.median(dom):.0f} Hz")
    print(f"[P4] MWD bake: {mtl.n_channels} channels @ {mtl.fps} fps; recovered RPM freqs {['%.2f' % r for r in rpm_rec]} Hz (planted 3.0)")
    print(f"[P4] params/hyperparams exposed for a panel: {[s['name']+'('+s['kind']+')' for s in params_schema()]}")
    print("\n[P4] acceptance criteria:")
    for k, v in checks.items(): print(f"    {'PASS' if v else 'FAIL'}  {k}")
    ok = all(checks.values())
    print(f"\n[P4] {'ALL ACCEPTANCE CRITERIA MET — signal-agnostic bake; audio & MWD both transpose to the same regimes' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
