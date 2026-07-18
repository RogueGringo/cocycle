"""COCYCLE engine core — a lean, Unreal-style component runtime.

A Composition is an ordered list of Components sharing one per-frame Context (a blackboard).
Each Component reads upstream outputs from the Context and writes its own. The same Composition
runs at full fidelity on the authoring rig and, later, at LED resolution on the Pi — only the
sink Components (LEDMapper / Spatializer) change. This module has no GPU / audio dependency so it
imports and smoke-tests anywhere.
"""
from __future__ import annotations
import time


class Context:
    """Per-frame blackboard passed down the component chain."""
    __slots__ = ("t", "dt", "frame", "data", "meta")

    def __init__(self, dt: float = 1 / 60):
        self.t = 0.0
        self.dt = dt
        self.frame = 0
        self.data: dict = {}      # component outputs, keyed by component.name
        self.meta: dict = {}      # static config shared across components

    def get(self, key, default=None):
        return self.data.get(key, default)

    def put(self, key, value):
        self.data[key] = value
        return value


class Component:
    """Base class for a subderived component. Override setup/update/teardown."""
    name = "component"

    def setup(self, ctx: Context) -> None:      # allocate buffers, read meta
        pass

    def update(self, ctx: Context) -> None:     # one frame of work; write to ctx.data[self.name]
        raise NotImplementedError

    def teardown(self, ctx: Context) -> None:   # release resources
        pass


class Composition:
    """Wires components into a runnable scene graph."""
    def __init__(self, components, dt: float = 1 / 60, meta: dict | None = None):
        self.components = list(components)
        self.ctx = Context(dt=dt)
        if meta:
            self.ctx.meta.update(meta)
        self._setup = False

    def setup(self):
        for c in self.components:
            c.setup(self.ctx)
        self._setup = True
        return self

    def step(self):
        if not self._setup:
            self.setup()
        for c in self.components:
            c.update(self.ctx)
        self.ctx.t += self.ctx.dt
        self.ctx.frame += 1
        return self.ctx

    def run(self, frames: int, realtime: bool = False):
        self.setup()
        for _ in range(frames):
            t0 = time.perf_counter()
            self.step()
            if realtime:
                slack = self.ctx.dt - (time.perf_counter() - t0)
                if slack > 0:
                    time.sleep(slack)
        return self.ctx

    def teardown(self):
        for c in self.components:
            c.teardown(self.ctx)
