"""
Shared test setup.

Test data: a 512 x 512 px crop of one processed image (all z planes, channels ASH1CLB2 / SRL1 / SUN4 / DAPI,
the DIC image, the Omnipose masks and the hand-corrected masks in manual/) plus the reference outputs of the
pipeline on that crop in reference/. It lives in tests/data/field_A/ (not in git, ~75 MB); when missing it is downloaded from the GitHub
release TEST_DATA_RELEASE (plain download; for a private repository through the GitHub CLI, `gh`, when it is installed
and logged in). tests/make_test_data.py makes it, tests/make_reference.py the reference outputs.

Markers:  slow   - Omnipose / GPU / bud-model tests (minutes; skipped with `pytest -m "not slow"`)
"""
import io
import os
import urllib.request
import zipfile
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent
DATA_DIR = TESTS_DIR / "data"
FIELD = DATA_DIR / "field_A"
TEST_DATA_RELEASE, TEST_DATA_REPO = "test-data-v3", "Tutuccilab/buddy"
TEST_DATA_URL = f"https://github.com/{TEST_DATA_REPO}/releases/download/{TEST_DATA_RELEASE}/field_A.zip"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs Omnipose, a GPU or the bud-model weights (minutes)")


VSI_DIR = DATA_DIR / "vsi"                 # one real .vsi of the example set (+ its _..._ folder and _DIC.tif), ~400 MB:
VSI_RELEASE, VSI_ZIP = "test-data-vsi", "vsi.zip"    # fetched only for the slow .vsi test (test_vsi_file.py)
_DOWNLOAD_ERROR = []


def _fetch(release, asset, ready):
    """
    tests/data/<asset unzipped> from the GitHub release: a plain download, then (a private repository) the GitHub
    CLI, which downloads with your login. ready: a path that exists once the data is there.
    """
    if ready.exists():
        return True
    if os.environ.get("BUDDY_TEST_NO_DOWNLOAD"):
        return False
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = f"https://github.com/{TEST_DATA_REPO}/releases/download/{release}/{asset}"
    try:
        with urllib.request.urlopen(url, timeout=600) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR)
    except Exception as e:                     # no network, no release yet, a private repository, ...
        _DOWNLOAD_ERROR.append(f"{type(e).__name__}: {e}")
        _download_with_gh(release, asset)
    return ready.exists()


def _download_with_gh(release, asset):
    """`gh release download` (GitHub CLI, logged in) can fetch the release of a private repository."""
    import shutil
    import subprocess
    if not shutil.which("gh"):
        return
    try:
        subprocess.run(["gh", "release", "download", release, "--repo", TEST_DATA_REPO, "--pattern", asset,
                        "--dir", str(DATA_DIR), "--clobber"], check=True, capture_output=True, timeout=1800)
        zipfile.ZipFile(DATA_DIR / asset).extractall(DATA_DIR)
        (DATA_DIR / asset).unlink()
    except Exception as e:  # noqa: BLE001 - not logged in, no such release, ...: reported with the skip
        _DOWNLOAD_ERROR.append(f"gh release download: {type(e).__name__}: {getattr(e, 'stderr', b'').decode(errors='replace') or e}")


def field_a_available():
    return _fetch(TEST_DATA_RELEASE, "field_A.zip", FIELD / "img.json")


def _skip_or_fail(what, release, asset):
    why = f" (download failed: {_DOWNLOAD_ERROR[-1]})" if _DOWNLOAD_ERROR else ""
    message = (f"test data ({what}) not available{why}: unzip {asset} of release {release} into tests/data "
               f"(private repository: log in to the GitHub CLI, gh auth login, and rerun)")
    if os.environ.get("BUDDY_REQUIRE_TEST_DATA"):   # set in the GitHub workflows: no silent green run
        pytest.fail(message)
    pytest.skip(message)


@pytest.fixture(scope="session")
def field_a():
    """Path of the test field (skips the test when the data is not available)."""
    if not field_a_available():
        _skip_or_fail("tests/data/field_A", TEST_DATA_RELEASE, "field_A.zip")
    return FIELD


@pytest.fixture(scope="session")
def vsi_folder():
    """tests/data/vsi: one real .vsi with its _..._ folder, _DIC.tif and vsi_reference.json (skips when not there)."""
    if not _fetch(VSI_RELEASE, VSI_ZIP, VSI_DIR / "vsi_reference.json"):
        _skip_or_fail("tests/data/vsi, one real .vsi", VSI_RELEASE, VSI_ZIP)
    return VSI_DIR


@pytest.fixture
def work_copy(field_a, tmp_path):
    """
    A fresh copy of the test field to run the pipeline on: tmp/output/field_A (the image folder) and
    tmp/config.json pointing at it. Returns (config_file, image_folder). reference/ is not copied.
    """
    import json
    import shutil
    out = tmp_path / "output" / FIELD.name
    shutil.copytree(field_a, out, ignore=shutil.ignore_patterns("reference"))
    cfg = json.load(open(field_a / "config.json"))
    cfg["inputdir"] = str(tmp_path / "input")
    cfg["outputdir"] = str(tmp_path / "output")
    cfg_file = tmp_path / "config.json"
    json.dump(cfg, open(cfg_file, "w"), indent=1)
    meta = json.load(open(out / "img.json"))
    meta["savepath"] = str(out)
    json.dump(meta, open(out / "img.json", "w"), indent=1)
    return cfg_file, out
