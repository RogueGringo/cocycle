"""AudioSource — feeds a rolling analysis window each frame.

Modes:
  'synth' : a built-in voice-like "moan" generator (swelling formant tone) — needs no hardware,
            used for headless smoke tests and for a default live scene.
  'file'  : streams a WAV/FLAC/OGG via soundfile (blocks advanced at frame rate).
  'mic'   : live line/mic input via sounddevice (optional; imported lazily).

Output ctx['audio'] = {'window': float32[WIN], 'sr': int, 'rms': float}
"""
from __future__ import annotations
import numpy as np
from ..engine import Component

SR = 44100
WIN = 2048


class AudioSource(Component):
    name = "audio"

    def __init__(self, mode="synth", path=None, sr=SR, win=WIN, fps=60):
        self.mode = mode
        self.path = path
        self.sr = sr
        self.win = win
        self.hop = max(1, sr // fps)
        self.ring = np.zeros(win, dtype=np.float32)
        self._ph = 0.0          # synth phase
        self._file = None
        self._stream = None
        self._mic_buf = None

    def setup(self, ctx):
        ctx.meta["sr"] = self.sr
        if self.mode == "file":
            import soundfile as sf
            self._file = sf.SoundFile(self.path)
            self.sr = self._file.samplerate
        elif self.mode == "mic":
            import sounddevice as sd
            self._mic_buf = np.zeros(self.win, dtype=np.float32)

            def cb(indata, frames, t, status):
                x = indata[:, 0] if indata.ndim > 1 else indata
                n = len(x)
                self._mic_buf = np.roll(self._mic_buf, -n)
                self._mic_buf[-n:] = x
            self._stream = sd.InputStream(channels=1, samplerate=self.sr,
                                          blocksize=self.hop, callback=cb)
            self._stream.start()

    def _synth_block(self, ctx, n):
        # voice-like moan: low f0 with vibrato, a few formant-weighted harmonics, slow swell
        t = ctx.t + np.arange(n) / self.sr
        f0 = 165.0 + 22.0 * np.sin(2 * np.pi * 0.55 * t) + 6.0 * np.sin(2 * np.pi * 5.2 * t)
        self._ph = self._ph + np.cumsum(2 * np.pi * f0 / self.sr)
        ph = self._ph
        self._ph = ph[-1] % (2 * np.pi)
        # formant emphasis on harmonics 1..8
        forms = np.array([1.0, 0.9, 0.55, 0.7, 0.35, 0.28, 0.5, 0.2])
        sig = np.zeros(n)
        for h, w in enumerate(forms, start=1):
            sig += w * np.sin(h * ph)
        # dynamic phrasing: a slow swell crossed with a slower phrase gate (quiet -> loud passages)
        swell = 0.10 + 0.65 * (0.5 + 0.5 * np.sin(2 * np.pi * 0.14 * t))
        phrase = 0.4 + 0.6 * np.clip(np.sin(2 * np.pi * 0.06 * t) + 0.3, 0, 1)
        tremolo = 1.0 + 0.25 * np.sin(2 * np.pi * 3.1 * t)
        env = swell * phrase * tremolo
        breath = 0.04 * np.random.randn(n)
        return (0.5 * env * (sig / forms.sum()) + breath * env).astype(np.float32)

    def update(self, ctx):
        n = self.hop
        if self.mode == "synth":
            block = self._synth_block(ctx, n)
        elif self.mode == "file":
            block = self._file.read(n, dtype="float32", always_2d=True)[:, 0]
            if len(block) < n:                    # loop
                self._file.seek(0)
                more = self._file.read(n - len(block), dtype="float32", always_2d=True)[:, 0]
                block = np.concatenate([block, more])
        else:  # mic
            self.ring = self._mic_buf.copy()
            ctx.put(self.name, {"window": self.ring, "sr": self.sr,
                                "rms": float(np.sqrt(np.mean(self.ring ** 2)) + 1e-9)})
            return
        self.ring = np.roll(self.ring, -n)
        self.ring[-n:] = block
        ctx.put(self.name, {"window": self.ring.copy(), "sr": self.sr,
                            "rms": float(np.sqrt(np.mean(self.ring ** 2)) + 1e-9)})

    def teardown(self, ctx):
        if self._stream:
            self._stream.stop(); self._stream.close()
        if self._file:
            self._file.close()
