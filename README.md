# buddy

An end-to-end pipeline for quantifying single-molecule FISH (smFISH) in
yeast cells — from raw microscope files to per-cell mRNA counts.
`buddy` orchestrates:

- **Read** — VSI fluorescence stacks + DIC cell images
- **Segment** — cells and nuclei (Omnipose/Cellpose)
- **Detect** — spot detection and dense-region decomposition (Big-FISH, https://big-fish.readthedocs.io/)
- **Align** — registers the DIC/segmentation frame to the fluorescence frame
- **Correct** — interactive mother/bud annotation in napari (via the
  companion [`napari-buddy`](https://github.com/Tutuccilab/napari-buddy) plugin)
- **Quantify** — per-cell and per-compartment (mother vs bud) counts of
  mRNAs, dense regions, nascent RNAs, transcription sites, and nuclei,
  with cell morphology
- **Track** - Processing parameters are recorded per image (JSON) so runs stay reproducible.

Goal is to automate processing and record processing parameters.

# Expected inputs
The module expects a directory containing the following files:
- Fluorescent channels: .vsi microscopy files and the corresponding vsi directories
- Cell outlines: 2D .tif files (e.g. DIC images)
- Configuration: a json file containing channel and processing parameters.

# Donwload code and example data
Go to https://doi.org/10.5281/zenodo.22900927 to dowload napari, napari-buddy and some test data. 
The folder structure of the zipped file should allow you to directly start with the tutorial after installation.
Note: newer versions of the code can be downloaded by cloning this repo.

# Installation

From the **`buddy` repo root**:

**Windows (Anaconda Prompt):**
```
powershell -ExecutionPolicy Bypass -File install.ps1
```

**macOS / Linux (Terminal):**
```
bash install.sh
```
This will install buddy and all its dependencies.

# Dependencies

Full list in `pyproject.toml`. The key ones:

**Core processing**
- `big-fish` — smFISH spot detection and dense-region decomposition
- `omnipose` + `cellpose-omni` — cell and nucleus segmentation
- `scikit-image` — image I/O, morphology, region properties
- `numpy`, `scipy` — arrays; alignment (distance transform) and focus (Savitzky–Golay)
- `pandas` — per-cell quantification tables
- `jsonpickle` — serialising image objects and parameters to JSON
- `natsort` — natural sorting of image lists

**Reading VSI microscopy files** (requires a working Java environment)
- `bioio` + `bioio-bioformats`
- `scyjava`

**Interactive mask/bud correction**
- `napari` (with a Qt backend, e.g. `napari[pyqt6]`)
- `napari-buddy` — the companion plugin (its reader is imported in the workflow notebook)

**Development**
- `pytest` — test suite
Details:
 - Omnipose

   On MacOS: install from git source or git clone on MacOS in a conda env with python<=3.10.11 (see [this issue](https://github.com/kevinjohncutler/omnipose/issues/14))

   First add `jupyterlab` to `omnipose_mac_environment.yml` from [Omnipose](https://omnipose.readthedocs.io/installation.html) repository, then:
   ```
   conda env create --name myenv --file /Volumes/DataDrive/omnipose_mac_environment.yml
   conda activate myenv
   pip install git+https://github.com/kevinjohncutler/omnipose.git
   pip install git+https://github.com/kevinjohncutler/cellpose-omni.git
   ```

- Omnipose GUI (nice to have, not required)

   Run `omnipose` from the command line.

Omnipose has to be installed with GPU support (https://omnipose.readthedocs.io/installation.html#gpu-support). Alternatively, the variable use_GPU can be set to False in experiment.py and the test python scripts.  

- Other
   ```
   conda install -c conda-forge scyjava
   pip install "napari[pyqt6]"
   pip install bioio bioio_bioformats big-fish jsonpickle pathlib  
   ```
- Might be needed:
   - `export JAVA_HOME=/Users/elton/miniconda3/envs/myenv/lib/jvm`
   - deactivate and reactivate myenv for `scyjava` to work

# Typical workflow
1. Batch process inputs to TIF files: `buddy/FISH bulk processing.ipynb`
2. Find good segmentation parameters using Omnipose GUI
3. Batch segment DIC and DAPI pictures: `buddy/FISH bulk processing.ipynb`
4. Find good spot detection parameters using napari-buddy plugin
   1. Load an img.json file in napari
   2. Tweak spot detection parameters
5. Batch detect spots: `buddy/FISH bulk processing.ipynb`

## Issues

If you encounter a problem, please [file an issue] with a detailed description.

[napari]: https://github.com/napari/napari
[pytest]: https://docs.pytest.org/
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[file an issue]: https://github.com/Tutuccilab/buddy/issues