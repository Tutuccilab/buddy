"""
Shared test setup.

Test data: a 512 x 512 px crop of one processed image (all z planes, channels ASH1CLB2 / SRL1 / SUN4 / DAPI,
the DIC image, the Omnipose masks and the hand-corrected masks in manual/) plus the reference outputs of the
pipeline on that crop in reference/. It lives in tests/data/field_A/ (not in git, ~75 MB); when missing it is downloaded from the GitHub
release named in TEST_DATA_URL. tests/make_test_data.py makes it, tests/make_reference.py the reference outputs.

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
TEST_DATA_URL = "https://github.com/Tutuccilab/buddy/releases/download/test-data-v1/field_A.zip"


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs Omnipose, a GPU or the bud-model weights (minutes)")


def field_a_available():
    if (FIELD / "img.json").is_file():
        return True
    if os.environ.get("BUDDY_TEST_NO_DOWNLOAD"):
        return False
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(TEST_DATA_URL, timeout=120) as r:
            zipfile.ZipFile(io.BytesIO(r.read())).extractall(DATA_DIR)
    except Exception as e:                     # no network, no release yet, a private repository, ...
        _DOWNLOAD_ERROR.append(f"{type(e).__name__}: {e}")
        return False
    return (FIELD / "img.json").is_file()


_DOWNLOAD_ERROR = []


@pytest.fixture(scope="session")
def field_a():
    """Path of the test field (skips the test when the data is not available)."""
    if not field_a_available():
        why = f" (download failed: {_DOWNLOAD_ERROR[-1]})" if _DOWNLOAD_ERROR else ""
        message = f"test data (tests/data/field_A) not available{why}: run tests/make_test_data.py or set TEST_DATA_URL"
        if os.environ.get("BUDDY_REQUIRE_TEST_DATA"):   # set in the GitHub workflows: no silent green run
            pytest.fail(message)
        pytest.skip(message)
    return FIELD


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
