"""Binaural spatializer (spec p3-binaural.md). Splits audio into the 7 bands and places each as a spatial
voice on a ring whose rotation is driven by the same prosody that spins the visuals; renders each voice with
interaural time (ITD) + level (ILD) differences so the 7 band-voices orbit the listener in lockstep with the
geometry. numpy only (torch-free). Honest scope: ITD/ILD nails left-right; front-back/elevation need HRTF.
"""
from __future__ import annotations
import numpy as np

ITD_MAX = 0.0007          # seconds, ~Woodworth max for a human head


def itd_ild(az, sr):
    """az in radians (0=front, +=right). Returns (delay_L, delay_R) in samples and (gain_L, gain_R)."""
    sa = np.sin(az)
    itd = ITD_MAX * sa                                   # +right -> right ear leads -> LEFT ear delayed
    dL = max(itd, 0.0) * sr
    dR = max(-itd, 0.0) * sr
    gL = np.sqrt(0.5 * (1 - sa))                         # equal-power ILD; source-side ear louder
    gR = np.sqrt(0.5 * (1 + sa))
    return dL, dR, gL, gR


def _add_delayed(buf, start, sig, delay):
    """overlap-add sig into buf at start with a fractional-sample delay (linear interp)."""
    di = int(delay); fr = delay - di
    a, b = start + di, start + di + len(sig)
    buf[a:b] += sig * (1 - fr)
    buf[a + 1:b + 1] += sig * fr


def binauralize(audio, sr, spin_fn, n_bands=7, nfft=1024, hop=512, fmin=60.0):
    """Spatialize a mono buffer: 7 band-voices on a ring rotating by Φ=spin_fn(t). Returns (L, R)."""
    audio = np.asarray(audio, np.float32)
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    edges = np.logspace(np.log10(fmin), np.log10(sr / 2), n_bands + 1)
    band_bins = [np.where((freqs >= edges[b]) & (freqs < edges[b + 1]))[0] for b in range(n_bands)]
    win = np.hanning(nfft).astype(np.float32)
    N = len(audio)
    L = np.zeros(N + nfft + 4, np.float32); R = np.zeros_like(L)
    for s in range(0, N - nfft, hop):
        spec = np.fft.rfft(audio[s:s + nfft] * win)
        phi = spin_fn((s + nfft / 2) / sr)
        for b in range(n_bands):
            bspec = np.zeros_like(spec); bspec[band_bins[b]] = spec[band_bins[b]]
            bsig = np.fft.irfft(bspec, nfft).astype(np.float32)
            az = np.arctan2(np.sin(2 * np.pi * b / n_bands + phi), np.cos(2 * np.pi * b / n_bands + phi))
            dL, dR, gL, gR = itd_ild(az, sr)
            _add_delayed(L, s, bsig * gL, dL)
            _add_delayed(R, s, bsig * gR, dR)
    peak = max(np.abs(L).max(), np.abs(R).max(), 1e-6)
    return (L[:N] / peak * 0.9).astype(np.float32), (R[:N] / peak * 0.9).astype(np.float32)
