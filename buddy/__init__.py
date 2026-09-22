"""
buddy: FISH image-analysis pipeline.

Import the pieces you need directly from their submodules, e.g.::

    from buddy.experiment import Experiment
    from buddy.image import Image

This module is intentionally empty so that ``import buddy`` stays cheap:
the heavy dependencies (big-fish, omnipose/torch, bio-formats) live in
``buddy.image`` / ``buddy.experiment`` and are only loaded when those
submodules are actually imported.
"""
