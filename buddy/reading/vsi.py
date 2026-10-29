"""
Reading microscope files through Bio-Formats (Java) without bioio: the bioformats_jar package ships
bioformats_package.jar 6.7.0 (the version bioio-bioformats used), JPype starts a JVM on it and hands the planes over.
No Maven and nothing downloaded at first use any more (buddy <= 0.5.1 fetched Bio-Formats with jgo into ~/.jgo).

    from buddy.reading.vsi import BioFormatsFile
    with BioFormatsFile(path) as f:
        f.series_names                       # e.g. ['001 CY5, CY3.5 NAR, CY3, DAPI', 'macro image']
        f.shape(0)                           # (T, C, Z, Y, X) of series 0
        stack = f.read(series=0, channel=2)  # ZYX numpy array of one channel (all z, t = 0)

Speed: every plane is read into one reused Java byte array that numpy views through the buffer protocol (one copy
per plane, no Python bytes objects in between).
"""
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger("buddy")

JAVA_HELP = ("Java is needed to read .vsi files (Bio-Formats). The installers put it in the environment "
             "(conda-forge openjdk=11, JAVA_HOME is set when the environment is active); see INSTALL.md -> 'Java not found'.")

_loci = None


def loci():
    """The loci Java package (Bio-Formats 6.7.0 from bioformats_jar), starting the JVM on first use."""
    global _loci
    if _loci is None:
        try:
            import bioformats_jar
            import jpype
        except ImportError as e:
            raise ImportError("jpype1 / bioformats_jar are not installed (pip install bioformats_jar); " + JAVA_HELP) from e
        if not jpype.isJVMStarted():
            jar = os.environ.get("BIOFORMATS_JAR", bioformats_jar.JAR)
            # Python's faulthandler (on under pytest) must be off while the JVM runs, not only while it starts: the
            # JVM uses SIGSEGV for its own purposes (safepoints before a garbage collection, implicit null checks in
            # compiled code) and handles it itself; with faulthandler's handler in place instead, the process aborts
            # at the first such signal (on Linux typically during a GC while reading a large file; Windows reports
            # "access violation" at start-up). JPype's documentation says the same. So it stays off from here on.
            import faulthandler
            if faulthandler.is_enabled():
                faulthandler.disable()
                log.debug("Python's faulthandler switched off: it conflicts with the Java VM's signal handling")
            try:
                jpype.startJVM(classpath=[jar], convertStrings=False)
            except jpype.JVMNotFoundException as e:
                raise RuntimeError(JAVA_HELP) from e
            log.info(f"Java started for Bio-Formats ({Path(jar).name})")
        _loci = jpype.JPackage("loci")
        _loci.common.DebugTools.setRootLevel("ERROR")
    return _loci


def bioformats_version():
    return str(loci().formats.FormatTools.VERSION)


def _dtype(reader):
    FT = loci().formats.FormatTools
    kinds = {int(FT.INT8): "i1", int(FT.UINT8): "u1", int(FT.INT16): "i2", int(FT.UINT16): "u2", int(FT.INT32): "i4",
             int(FT.UINT32): "u4", int(FT.FLOAT): "f4", int(FT.DOUBLE): "f8"}
    return np.dtype(("<" if reader.isLittleEndian() else ">") + kinds[int(reader.getPixelType())])


class BioFormatsFile:
    """One microscope file opened with Bio-Formats' ImageReader (every format it knows: .vsi, .nd2, .czi, ...)."""

    def __init__(self, path):
        self.path = str(Path(path))
        L = loci()
        service = L.common.services.ServiceFactory().getInstance(L.formats.services.OMEXMLService)
        self._meta = service.createOMEXMLMetadata()
        self._reader = L.formats.ImageReader()
        self._reader.setFlattenedResolutions(False)      # one series per image; sub-resolutions stay inside it
        self._reader.setMetadataStore(self._meta)
        self._reader.setId(self.path)
        self.series_names = [str(self._meta.getImageName(i)) for i in range(int(self._reader.getSeriesCount()))]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self._reader.close()
        except Exception:                                  # noqa: BLE001 - the JVM may already be gone
            pass

    def shape(self, series):
        """(T, C, Z, Y, X) of a series."""
        r = self._reader
        r.setSeries(series)
        return tuple(int(n) for n in (r.getSizeT(), r.getEffectiveSizeC(), r.getSizeZ(), r.getSizeY(), r.getSizeX()))

    def read(self, series, channel, t=0):
        """All z-planes of one channel of a series as a (Z, Y, X) numpy array."""
        import jpype
        r = self._reader
        r.setSeries(series)
        if r.isRGB():
            raise ValueError(f"{self.path}: RGB series {series!r} - buddy reads one grey channel per fluorophore")
        _, nc, nz, ny, nx = self.shape(series)
        if not 0 <= channel < nc:
            raise ValueError(f"{self.path}: series {series} has {nc} channels, no channel {channel}")
        dtype = _dtype(r)
        out = np.empty((nz, ny, nx), dtype)
        buf = jpype.JArray(jpype.JByte)(ny * nx * dtype.itemsize)   # one Java array, reused for every plane
        for z in range(nz):
            r.openBytes(int(r.getIndex(z, channel, t)), buf)
            out[z] = np.frombuffer(memoryview(buf), dtype).reshape(ny, nx)
        return out
