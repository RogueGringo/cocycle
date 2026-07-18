"""FeatureExtractor — prosody-aware, not a bare FFT bandpass.

Emits the 7 log-spaced band energies that drive the seven axes of the topology field, plus the
prosodic / voice-like descriptors a "moaning theme" needs: pitch (f0), brightness (spectral
centroid), onset (flux), loudness (rms envelope), rolloff, the first spectral peaks (formant-ish),
and crude arousal/valence contours. All values EMA-smoothed and normalized to ~[0,1].

Output ctx['features'] = {bands7, f0, centroid, flux, rms, rolloff, formants, arousal, valence}
"""
from __future__ import annotations
import numpy as np
from ..engine import Component

NB = 7  # seven bands -> seven axes


class FeatureExtractor(Component):
    name = "features"

    def __init__(self, fmin=45.0, fmax=None, smooth=0.6):
        self.fmin, self.fmax, self.smooth = fmin, fmax, smooth
        self._prev_mag = None
        self._ema = {}
        self._win = None
        self._band_idx = None

    def setup(self, ctx):
        sr = ctx.meta.get("sr", 44100)
        self.fmax = self.fmax or sr * 0.5

    def _norm(self, key, val, adapt=0.01):
        # adaptive peak-normalize + EMA smooth, per feature
        mx = self._ema.get(key + "_mx", 1e-6)
        mx = max(mx * (1 - adapt), abs(val), 1e-6)
        self._ema[key + "_mx"] = mx
        v = np.clip(val / mx, 0.0, 1.0)
        s = self._ema.get(key, v)
        s = self.smooth * s + (1 - self.smooth) * v
        self._ema[key] = s
        return float(s)

    def update(self, ctx):
        au = ctx.get("audio")
        x = au["window"]; sr = au["sr"]; n = len(x)
        if self._win is None or len(self._win) != n:
            self._win = np.hanning(n).astype(np.float32)
            freqs = np.fft.rfftfreq(n, 1 / sr)
            edges = np.logspace(np.log10(self.fmin), np.log10(self.fmax), NB + 1)
            self._band_idx = [np.where((freqs >= edges[b]) & (freqs < edges[b + 1]))[0] for b in range(NB)]
            self._freqs = freqs

        mag = np.abs(np.fft.rfft(x * self._win))
        total = mag.sum() + 1e-9

        bands = np.array([mag[idx].sum() if len(idx) else 0.0 for idx in self._band_idx])
        bands7 = np.array([self._norm(f"b{b}", np.log1p(bands[b])) for b in range(NB)], dtype=np.float32)

        centroid = float((self._freqs * mag).sum() / total)
        csum = np.cumsum(mag); rolloff = float(self._freqs[np.searchsorted(csum, 0.85 * csum[-1])])
        flux = 0.0 if self._prev_mag is None else float(np.maximum(mag - self._prev_mag, 0).sum())
        self._prev_mag = mag
        rms = au["rms"]

        # f0 via autocorrelation in the 60–500 Hz band
        ac = np.correlate(x, x, mode="full")[n - 1:]
        lo, hi = int(sr / 500), int(sr / 60)
        f0 = 0.0
        if hi < len(ac):
            lag = lo + int(np.argmax(ac[lo:hi])); f0 = sr / lag if lag else 0.0

        # formant-ish: top-3 spectral peaks below 4 kHz
        band = mag.copy(); band[self._freqs > 4000] = 0
        peaks = np.argsort(band)[-3:]
        formants = sorted(float(self._freqs[p]) for p in peaks)

        arousal = self._norm("arousal", rms * (1 + centroid / 4000) * (1 + flux / total))
        valence = self._norm("valence", centroid) - self._norm("darkness", bands[0])

        ctx.put(self.name, {
            "bands7": bands7,
            "f0": self._norm("f0", f0), "f0_hz": f0,
            "centroid": self._norm("centroid", centroid),
            "flux": self._norm("flux", flux),
            "rms": self._norm("rms", rms),
            "rolloff": self._norm("rolloff", rolloff),
            "formants": formants,
            "arousal": arousal, "valence": float(np.clip(valence, -1, 1)),
        })
