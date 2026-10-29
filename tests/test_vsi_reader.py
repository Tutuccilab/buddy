"""buddy's Bio-Formats reader (buddy/reading/vsi.py) on an OME-TIFF stack: series names, shape, planes, dtype."""
import numpy as np
import pytest
import tifffile


@pytest.fixture(scope="module")
def stack(tmp_path_factory):
    folder = tmp_path_factory.mktemp("bf")
    data = (np.random.default_rng(0).random((3, 7, 64, 80)) * 4000).astype(np.uint16)
    path = folder / "stack.ome.tif"
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "CZYX", "Channel": {"Name": ["CY5", "CY3", "DAPI"]}})
    return path, data


def test_reads_every_channel_plane_by_plane(stack):
    path, data = stack
    jpype = pytest.importorskip("jpype")
    try:
        jpype.getDefaultJVMPath()
    except jpype.JVMNotFoundException:
        pytest.skip("no Java on this machine")
    from buddy.reading.vsi import BioFormatsFile, bioformats_version
    assert bioformats_version() == "6.7.0"
    with BioFormatsFile(path) as f:
        assert len(f.series_names) == 1
        assert f.shape(0) == (1, 3, 7, 64, 80)
        for c in range(3):
            assert np.array_equal(f.read(0, c), data[c])
        assert f.read(0, 1).dtype == np.uint16
        with pytest.raises(ValueError):
            f.read(0, 3)
