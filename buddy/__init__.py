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

import warnings as _warnings

# third-party warnings that are not actionable for a buddy user
_warnings.filterwarnings("ignore", category=FutureWarning, module=r"bigfish(\.|$)")        # skimage deprecations inside big-fish
_warnings.filterwarnings("ignore", message=r"chan to seg' has value range of ZERO")       # Omnipose: the empty channels of the composite
_warnings.filterwarnings("ignore", message=r".*torch\.load.*weights_only.*", category=FutureWarning)
_warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"aicsimageio(\.|$)")  # old aicsimageio pulled in by Omnipose
