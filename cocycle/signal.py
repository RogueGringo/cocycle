"""Offline signal bake (spec p4-signal-bake.md). Parses ANY time-series signal file into a dense,
renderer-agnostic timeline of "elementals" at a target framerate, with global per-track normalization.
Audio is one source; raw multi-channel sensor data (MWD drilling transducers) is another — same elementals
(RMS, dominant-frequency=RPM, spectral content, onset=shock), same code. The timeline drives ANY scene via
Timeline.to_features(). BakeParams is introspectable so a control panel can be generated over it. numpy only.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
import numpy as np

ELEMENTALS = (["rms", "peak"] + [f"band{b}" for b in range(7)]
              + ["centroid", "rolloff", "flux", "flatness", "zcr", "domfreq", "onset"])


@dataclass
class BakeParams:
    fps: int = 144            # target framerate (visualization); floor 90 enforced   [param]
    nfft: int = 2048          # analysis window (samples) — scale up for low-rate sensors  [hyperparam]
    n_bands: int = 7          # log spectral bands                                    [hyperparam]
    fmin: float = 50.0        # lowest spectral band edge (Hz)                        [param]
    fmax: float = 0.0         # highest band edge; 0 => sr/2                          [param]
    dom_fmin: float = 40.0    # dominant-frequency search low (Hz): audio pitch / MWD RPM range  [param]
    dom_fmax: float = 1600.0  # dominant-frequency search high (Hz)                   [param]
    norm_lo: float = 2.0      # global-normalization low percentile                  [hyperparam]
    norm_hi: float = 98.0     # global-normalization high percentile                 [hyperparam]


PARAM_META = {  # for a generated panel: (kind, min, max, step)
    "fps": ("param", 90, 240, 1), "nfft": ("hyperparam", 256, 16384, 256),
    "n_bands": ("hyperparam", 3, 16, 1), "fmin": ("param", 1.0, 500.0, 1.0),
    "fmax": ("param", 0.0, 24000.0, 100.0), "dom_fmin": ("param", 0.5, 500.0, 0.5),
    "dom_fmax": ("param", 5.0, 8000.0, 5.0), "norm_lo": ("hyperparam", 0.0, 20.0, 0.5),
    "norm_hi": ("hyperparam", 80.0, 100.0, 0.5),
}


def params_schema(p: BakeParams | None = None):
    """Introspectable schema a control panel is generated from."""
    p = p or BakeParams()
    return [{"name": f.name, "value": getattr(p, f.name), "type": str(f.type),
             "kind": PARAM_META[f.name][0], "min": PARAM_META[f.name][1],
             "max": PARAM_META[f.name][2], "step": PARAM_META[f.name][3]} for f in fields(p)]


class Timeline:
    def __init__(self, raw, fps, sr, names, n_channels=1):
        self.raw = raw                      # [F, C, E] (or [F, E] for mono)
        self.fps = fps; self.sr = sr; self.names = names; self.n_channels = n_channels
        self.duration = raw.shape[0] / fps
        r = raw.reshape(raw.shape[0], -1)
        lo = np.percentile(r, 2, axis=0); hi = np.percentile(r, 98, axis=0)
        self.normed = (np.clip((r - lo) / (hi - lo + 1e-9), 0, 1)).reshape(raw.shape).astype(np.float32)

    @property
    def n_frames(self): return self.raw.shape[0]

    def idx(self, name): return self.names.index(name)

    def to_features(self, i, ch=0):
        """FeatureExtractor-compatible dict from the baked, normalized elementals — drives any scene."""
        n = self.normed[i, ch] if self.raw.ndim == 3 else self.normed[i]
        bands = np.array([n[self.idx(f"band{b}")] for b in range(7)], np.float32)
        return {"bands7": bands, "f0": float(n[self.idx("domfreq")]),
                "centroid": float(n[self.idx("centroid")]), "flux": float(n[self.idx("flux")]),
                "rms": float(n[self.idx("rms")])}

    def select(self, names):
        """Normalized timeline restricted to a super-user's chosen elementals (the panel's selection)."""
        cols = [self.idx(nm) for nm in names]
        return self.normed[..., cols]

    def save_npz(self, path, selected=None):
        names = selected or self.names
        np.savez_compressed(path, data=self.select(names), names=np.array(names),
                            fps=self.fps, sr=self.sr, n_channels=self.n_channels)


def _elementals(x, sr, p: BakeParams):
    fmax = p.fmax or sr / 2
    hop = max(1, round(sr / p.fps))
    nfft = min(p.nfft, len(x))
    win = np.hanning(nfft).astype(np.float32)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    edges = np.logspace(np.log10(max(p.fmin, 1)), np.log10(fmax), p.n_bands + 1)
    bidx = [np.where((freqs >= edges[b]) & (freqs < edges[b + 1]))[0] for b in range(p.n_bands)]
    lo, hi = int(sr / p.dom_fmax), int(sr / p.dom_fmin)
    starts = range(0, max(1, len(x) - nfft), hop)
    out = np.zeros((len(list(starts)), len(ELEMENTALS)), np.float32)
    prev_mag = None
    for fr, s in enumerate(range(0, max(1, len(x) - nfft), hop)):
        blk = x[s:s + nfft].astype(np.float32)
        if len(blk) < nfft: blk = np.pad(blk, (0, nfft - len(blk)))
        mag = np.abs(np.fft.rfft(blk * win)); tot = mag.sum() + 1e-9
        e = out[fr]
        e[0] = np.sqrt(np.mean(blk ** 2)); e[1] = np.abs(blk).max()
        for b in range(min(p.n_bands, 7)):
            e[2 + b] = np.log1p(mag[bidx[b]].sum()) if len(bidx[b]) else 0.0
        e[9] = (freqs * mag).sum() / tot                                    # centroid
        cs = np.cumsum(mag); e[10] = freqs[np.searchsorted(cs, 0.85 * cs[-1])]  # rolloff
        e[11] = float(np.maximum(mag - prev_mag, 0).sum()) if prev_mag is not None else 0.0  # flux
        e[12] = float(np.exp(np.mean(np.log(mag + 1e-9))) / (mag.mean() + 1e-9))            # flatness
        e[13] = float(np.mean(np.abs(np.diff(np.sign(blk))) > 0))            # zcr
        # dominant frequency via FFT-autocorr + cumulative-mean-normalized difference (YIN; unbiased in lag)
        if hi > lo:
            acf = np.fft.irfft(np.abs(np.fft.rfft(blk, 2 * nfft)) ** 2)[:hi + 1]
            dfn = 2.0 * (acf[0] - acf)                                       # difference function ~ 2(r0 - r_k)
            cs = np.cumsum(dfn[1:hi + 1]) + 1e-9
            cmnd = np.empty(hi + 1); cmnd[0] = 1.0
            cmnd[1:hi + 1] = dfn[1:hi + 1] * np.arange(1, hi + 1) / cs
            below = np.where(cmnd[lo:hi] < 0.2)[0]
            if len(below):
                tau = lo + int(below[0])                                     # descend to the valley BOTTOM,
                while tau + 1 < hi and cmnd[tau + 1] < cmnd[tau]: tau += 1   # not the first crossing (YIN step 4)
            else:
                tau = lo + int(np.argmin(cmnd[lo:hi]))
            if 0 < tau < hi:                                                 # parabolic sub-sample refinement
                a, b, c = cmnd[tau - 1], cmnd[tau], cmnd[tau + 1]; den = a - 2 * b + c
                tau = tau + (0.5 * (a - c) / den if den != 0 else 0.0)
            e[14] = sr / tau if tau > 0 else 0.0
        e[15] = e[11]                                                       # onset = pos spectral flux
        prev_mag = mag
    return out


def bake_signal(x, sr, params: BakeParams | None = None):
    p = params or BakeParams()
    p.fps = max(90, p.fps)                                                  # floor
    x = np.asarray(x, np.float32)
    if x.ndim == 1:
        return Timeline(_elementals(x, sr, p), p.fps, sr, list(ELEMENTALS), 1)
    chans = [_elementals(x[:, c], sr, p) for c in range(x.shape[1])]        # multi-channel (MWD)
    F = min(c.shape[0] for c in chans)
    return Timeline(np.stack([c[:F] for c in chans], 1), p.fps, sr, list(ELEMENTALS), x.shape[1])


def bake_audio(path, params: BakeParams | None = None):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return bake_signal(x.mean(1), sr, params)


def bake_config(cfg: dict):
    """Bake from a config dict as emitted by the control panel: {signal, params, elementals, regime}.
    Closes the round-trip panel -> config.json -> timeline; `regime`/`elementals` are consumed downstream."""
    p = BakeParams(**{k: v for k, v in cfg.get("params", {}).items() if k in {f.name for f in fields(BakeParams)}})
    src = cfg.get("signal", {})
    kind = src.get("type", "audio")
    if kind == "audio":
        return bake_audio(src["path"], p)
    if kind == "multichannel":
        import soundfile as sf
        x, sr = sf.read(src["path"], dtype="float32", always_2d=True)
        return bake_signal(x, sr, p)
    raise ValueError(f"unknown signal type {kind!r}")


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser(description="Offline signal bake -> elementals timeline (drives any regime).")
    ap.add_argument("--config", required=True, help="JSON config as emitted by the control panel")
    ap.add_argument("--out", default="timeline.npz", help="output .npz (selected elementals, normalized)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    tl = bake_config(cfg)
    sel = cfg.get("elementals") or list(ELEMENTALS)
    tl.save_npz(a.out, selected=sel)
    print(f"baked {tl.n_frames} frames x {tl.n_channels} ch @ {tl.fps} fps ({tl.duration:.1f}s); "
          f"{len(sel)} elementals -> {a.out}")
