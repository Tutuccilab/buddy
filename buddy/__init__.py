"""
buddy: FISH image-analysis pipeline.

Import the pieces you need directly from their submodules, e.g.::

    from buddy.experiment import Experiment
    from buddy import pipeline

Where is what: ``buddy.where()`` prints, per pipeline step, the function in buddy/pipeline.py, the module that does
the work, the block of the config file it reads and the files it writes (``buddy.where("segment")`` for one step).
The same table is in README.md ("Where things are").

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


def where(step=None):
    """Print where each pipeline step lives (see buddy/pipeline.py CODE_MAP); step: "segment", 3, "spots", ... or None for all."""
    from buddy.pipeline import where as _where
    _where(step)


# napari-buddy versions that work with this buddy (napari-buddy 0.1.0 imports modules that moved in buddy 0.2.0;
# napari-buddy 0.2.0 shows the cell-cycle phases of buddy 0.3.0)
MIN_NAPARI_BUDDY = (0, 2, 0)


def _version_tuple(text):
    import re
    return tuple(int(x) for x in re.findall(r"\d+", text)[:3])


def _check_napari_buddy():
    """Warn (once, at import) when the installed napari-buddy is too old for this buddy."""
    from importlib.metadata import PackageNotFoundError, version
    try:
        installed = version("napari-buddy")
    except PackageNotFoundError:
        return
    if _version_tuple(installed) < MIN_NAPARI_BUDDY:
        _warnings.warn(f"napari-buddy {installed} does not work with this buddy version: update napari-buddy to "
                       f"{'.'.join(map(str, MIN_NAPARI_BUDDY))} or newer (git pull in the napari-buddy folder, then "
                       f"pip install -e <napari-buddy folder>)", stacklevel=2)


try:
    _check_napari_buddy()
except Exception:          # a version check must never stop buddy from importing
    pass
