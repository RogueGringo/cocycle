"""LEDMapper — transpose the high-dim projection onto a physical LED layout.

P0 targets a simulated 2-D matrix (e.g. 32×32). It bins the projected particles into LED cells,
accumulating colour (hue from the 4th rotated coord) weighted by nearness, then tone-maps to [0,1].
The output grid IS the frame that a real driver (rpi-rgb-led-matrix / WS2812 SPI / DMX) pushes on
the Pi — swapping this sink for a hardware writer is the only change needed at P2.

Output ctx['led'] = float32[H,W,3] in [0,1]   (also .as_uint8() helper on the component)
"""
from __future__ import annotations
import numpy as np
from ..engine import Component


def hue_rgb(h):
    """vectorized hue([0,1]) -> rgb, full sat/val. h: (N,) -> (N,3)."""
    h6 = (h % 1.0) * 6.0
    x = 1 - np.abs(h6 % 2 - 1)
    z = np.zeros_like(h6)
    r = np.select([h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5], [1, x, z, z, x], default=1)
    g = np.select([h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5], [x, 1, 1, x, z], default=z)
    b = np.select([h6 < 1, h6 < 2, h6 < 3, h6 < 4, h6 < 5], [z, z, x, 1, 1], default=x)
    return np.stack([r, g, b], axis=1)


class LEDMapper(Component):
    name = "led"

    def __init__(self, w=32, h=32, gain=0.9, hue_base=0.52, hue_span=0.42):
        self.W, self.H = w, h
        self.gain, self.hue_base, self.hue_span = gain, hue_base, hue_span
        self.grid = np.zeros((h, w, 3), np.float32)

    def update(self, ctx):
        pr = ctx.get("proj")
        xy, near, w = pr["xy"], pr["near"], pr["w"]
        gx = np.clip(((xy[:, 0] * 0.5 + 0.5) * self.W).astype(np.int32), 0, self.W - 1)
        gy = np.clip(((xy[:, 1] * 0.5 + 0.5) * self.H).astype(np.int32), 0, self.H - 1)
        hue = self.hue_base + self.hue_span * np.tanh(w * 0.6)
        col = hue_rgb(hue) * (near[:, None] * self.gain)

        acc = np.zeros((self.H, self.W, 3), np.float32)
        flat = gy * self.W + gx
        for c in range(3):
            np.add.at(acc.reshape(-1, 3)[:, c], flat, col[:, c])
        acc /= (acc.max() + 1e-6)
        acc = acc / (0.55 + acc)                      # tone map -> luminous, dark floor
        self.grid = 0.55 * self.grid + 0.45 * acc     # temporal smoothing (LED persistence)
        ctx.put(self.name, self.grid)

    def as_uint8(self):
        return (np.clip(self.grid, 0, 1) * 255).astype(np.uint8)
