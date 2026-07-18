"""Spatializer — the auditory projection of the same field (the co-projection with the light).

Per band, it takes the centroid of that band's particles in the projected plane and reads it as a
sound source: horizontal position -> azimuth, nearness -> gain. P0 emits equal-power stereo pan
gains per source (the values a multichannel/ambisonic or HRTF renderer consumes at P2). One field,
one projection, two coherent senses — the coherence is what fuses the percept.

Output ctx['spatial'] = list of dicts: {'band', 'az', 'gain', 'L', 'R'}
"""
from __future__ import annotations
import numpy as np
from ..engine import Component


class Spatializer(Component):
    name = "spatial"

    def update(self, ctx):
        pr = ctx.get("proj")
        band = pr["band"]; xy = pr["xy"]; near = pr["near"]
        f = ctx.get("features")
        bands = f["bands7"] if f else np.full(7, 0.3)
        out = []
        for b in range(7):
            m = band == b
            if not m.any():
                continue
            cx = float(np.average(xy[m, 0], weights=near[m] + 1e-6))
            az = float(np.clip(cx, -1, 1)) * (np.pi / 2)          # -90°..+90°
            gain = float(np.clip(bands[b], 0, 1))
            # equal-power pan
            th = (az / (np.pi / 2) * 0.5 + 0.5) * (np.pi / 2)
            out.append({"band": b, "az": az, "gain": gain,
                        "L": float(np.cos(th) * gain), "R": float(np.sin(th) * gain)})
        ctx.put(self.name, out)
