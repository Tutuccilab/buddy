# Installing buddy

## Quick install (recommended)

From the `buddy` repo root:

**Windows (Anaconda Prompt):**
```
powershell -ExecutionPolicy Bypass -File install.ps1
```

**macOS / Linux (Terminal):**
```
bash install.sh
```

The script creates a `buddy` conda environment (Python 3.10), installs
buddy with all its dependencies (including the napari viewer and Jupyter),
adds the `bioio-ome-tiff` writer plugin, installs the `napari-buddy` plugin
if it sits next to this repo, and runs a quick import check.

## Manual install

```
conda create --name buddy python=3.10
conda activate buddy
python -m pip install -e ".[test,napari,notebooks]"   # buddy + deps + pytest + napari + jupyter
python -m pip install bioio-ome-tiff                   # OME-TIFF writer (see note below)
python -m pip install -e ../napari-buddy               # the napari plugin (adjust the path)
```

Python **must** be 3.9 or 3.10 — omnipose does not work on 3.11+.
Use `python -m pip` so installs land in the active env, not base conda.

## Test it

```
conda activate buddy
pytest buddy/tests/test_image.py -v
python -m napari      # Plugins -> buddy   (never bare 'napari')
python -m notebook    # run the analysis notebooks (from the repo root)
```

The pipeline test runs on the bundled example data and should finish with
no errors.

## Good to know

- **bioio OME-TIFF writer.** bioio 3.x split its writers into separate
  packages, so `OmeTiffWriter` now lives in `bioio-ome-tiff`, and
  `buddy/image.py` imports it as
  `from bioio_ome_tiff.writers import OmeTiffWriter`. The installers pull
  this in automatically; if you install by hand, run
  `pip install bioio-ome-tiff`.
- **Launch with `python -m`.** Always use `python -m napari` and
  `python -m notebook`, never bare `napari` / `jupyter`. The bare commands
  can bounce to a broken base-conda copy that happens to be first on your
  PATH (the cause of the numpy/`_ssl` DLL errors).
- **First `.vsi` read is slow.** Reading Olympus `.vsi` files needs Java.
  bioio-bioformats fetches a JRE and Maven automatically the first time
  (`Fetching zulu-jre:11 using cjdk...`). This needs internet once; after
  that it's cached. On locked-down/offline machines, uncomment
  `openjdk=11` and `maven` in `environment.yml` and build the env with
  `conda env create -f environment.yml` instead.
- **No GPU?** Set `use_GPU = False` in `experiment.py` / the test scripts.
- **Apple Silicon + omnipose.** If the omnipose step fails to build,
  comment `omnipose` out of `pyproject.toml`, reinstall, then:
  ```
  pip install git+https://github.com/kevinjohncutler/omnipose.git
  pip install git+https://github.com/kevinjohncutler/cellpose-omni.git
  ```
```
