"""COCYCLE — an audio-reactive engine that renders a signal on a rank-7 topology field.

A lean, component-based SDK: compose subderived Components (audio -> prosody features -> rank-7
field -> 21-plane projection -> LED transposition + spatial audio) into one Composition that runs
at full fidelity on the authoring rig and, later, at LED resolution on a Raspberry Pi.
"""
from .engine import Component, Context, Composition
from . import components

__version__ = "0.0.1"
__all__ = ["Component", "Context", "Composition", "components", "default_composition"]


def default_composition(n=60000, led=(32, 32), audio="synth", path=None, dt=1 / 60):
    """The P0 vertical slice, wired end to end."""
    from .components import (AudioSource, FeatureExtractor, TopologyField,
                             Projector, LEDMapper, Spatializer)
    return Composition([
        AudioSource(mode=audio, path=path),
        FeatureExtractor(),
        TopologyField(n=n),
        Projector(),
        LEDMapper(w=led[0], h=led[1]),
        Spatializer(),
    ], dt=dt)
