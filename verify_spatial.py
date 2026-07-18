"""P3 verification — binaural spatializer (spec p3-binaural.md). Asserts ITD/ILD correctness + a coherent
orbit numerically, then renders a listenable binaural WAV (orbiting moan). Headless."""
import sys
import numpy as np
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
from cocycle.binaural import itd_ild, binauralize

SR = 44100


def moan(seconds=6.0):
    t = np.arange(int(seconds * SR)) / SR
    f0 = 165 + 22 * np.sin(2 * np.pi * 0.55 * t) + 6 * np.sin(2 * np.pi * 5.2 * t)
    ph = np.cumsum(2 * np.pi * f0 / SR)
    sig = sum(w * np.sin(h * ph) for h, w in enumerate([1, .9, .55, .7, .35, .28, .5, .2], 1))
    env = 0.4 + 0.5 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.14 * t))
    return (0.5 * env * sig / 4.4).astype(np.float32)


def main():
    # F1: ITD/ILD correctness at the cardinal azimuths
    dLr, dRr, gLr, gRr = itd_ild(np.pi / 2, SR)     # hard right
    dLl, dRl, gLl, gRl = itd_ild(-np.pi / 2, SR)    # hard left
    dLc, dRc, gLc, gRc = itd_ild(0.0, SR)           # center
    right_ok = dLr > dRr and gRr > gLr              # right: LEFT delayed, RIGHT louder
    left_ok = dRl > dLl and gLl > gRl               # left: mirror
    center_ok = abs(dLc - dRc) < 1e-6 and abs(gLc - gRc) < 1e-6
    print(f"[P3] ITD/ILD  right(dL={dLr:.1f} dR={dRr:.1f} gL={gLr:.2f} gR={gRr:.2f})  "
          f"left mirror={left_ok}  center symmetric={center_ok}")

    # F2/F3: render the orbit, check the L-R balance sweeps both sides
    audio = moan(6.0)
    spin = lambda tt: 2 * np.pi * 0.5 * tt           # 0.5 Hz orbit (full rotation every 2 s)
    L, R = binauralize(audio, SR, spin)
    # running L-R balance over 50ms windows
    w = int(0.05 * SR); bal = []
    for s in range(0, len(L) - w, w):
        el = float(np.sum(L[s:s + w] ** 2) + 1e-9); er = float(np.sum(R[s:s + w] ** 2) + 1e-9)
        bal.append((er - el) / (er + el))
    bal = np.array(bal)
    crossings = int(np.sum(np.diff(np.sign(bal)) != 0))
    finite = np.isfinite(L).all() and np.isfinite(R).all()
    swing = float(bal.max() - bal.min())

    import soundfile as sf
    sf.write("orbit_binaural.wav", np.stack([L, R], 1), SR)

    checks = {
        "F1a right source -> LEFT delayed + RIGHT louder": right_ok,
        "F1b left source -> mirror": left_ok,
        "F1c center -> symmetric": center_ok,
        "F2 coherent orbit (L-R balance sweeps both sides)": crossings >= 4 and swing > 0.4,
        "F3 valid finite stereo output": finite and len(L) == len(audio),
    }
    print("\n[P3] acceptance criteria:")
    for k, v in checks.items():
        print(f"    {'PASS' if v else 'FAIL'}  {k}")
    print(f"\n    orbit: {crossings} L/R zero-crossings, balance swing {swing:.2f}   wrote orbit_binaural.wav "
          f"({len(L)/SR:.1f}s stereo)")
    ok = all(checks.values())
    print(f"\n[P3] {'ALL ACCEPTANCE CRITERIA MET — binaural orbit verified' if ok else 'FAILURES ABOVE'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
