"""COCYCLE — an audio-reactive engine that renders a signal on a rank-7 topology field.

A lean, component-based SDK: compose subderived Components (audio -> prosody features -> rank-7
field -> 21-plane projection -> LED transposition + spatial audio) into one Composition that runs
at full fidelity on the authoring rig and, later, at LED resolution on a Raspberry Pi.
"""
from .engine import Component, Context, Composition
from . import components

__version__ = "0.0.1"
__all__ = ["Component", "Context", "Composition", "components", "default_composition"]


def default_composition(n=60000, led=(32, 32), audio="synth", path=None, dt=1 / 60, projector="auto"):
    """The vertical slice, wired end to end.

    projector: 'auto' uses the GPU/differentiable GivensProjector when torch is importable and falls back to
    the numpy Projector otherwise; 'gpu' forces it (raises without torch); 'cpu' forces the numpy Projector.
    The GivensProjector import is lazy so `import cocycle` never requires torch.
    """
    from .components import (AudioSource, FeatureExtractor, TopologyField,
                             Projector, LEDMapper, Spatializer)
    proj = None
    if projector in ("auto", "gpu"):
        try:
            from .components.projector_gpu import GivensProjector
            proj = GivensProjector()
        except Exception:
            if projector == "gpu":
                raise
            proj = None
    if proj is None:
        proj = Projector()
    return Composition([
        AudioSource(mode=audio, path=path),
        FeatureExtractor(),
        TopologyField(n=n),
        proj,
        LEDMapper(w=led[0], h=led[1]),
        Spatializer(),
    ], dt=dt)
