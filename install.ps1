#  installer for Windows (run from the Anaconda Prompt, in the buddy repo root)
#
#   powershell -ExecutionPolicy Bypass -File install.ps1 -NapariBuddyPath 
#
# Optional args:
#   -NapariBuddyPath "..\napari-buddy"   path to the napari-buddy repo
#   -EnvName "buddy"                        conda env name
#   cd buddy 

param(
    [string]$NapariBuddyPath = "..\napari-buddy",
    [string]$EnvName = "buddy"
)
$ErrorActionPreference = "Stop"

Write-Host "==> Creating conda environment '$EnvName' (Python 3.10)"
conda create --name $EnvName python=3.10 -y

Write-Host "==> Installing buddy (editable, with test + napari + notebooks extras)"
# The 'napari' extra pulls in napari[pyqt6] - the viewer + Qt backend -
# INTO this env, so 'python -m napari' launches from here and never falls
# back to a base-conda napari.
conda run --no-capture-output -n $EnvName pip install -e ".[test,napari,notebooks]"

Write-Host "==> Ensuring bioio OME-TIFF writer plugin"
# bioio 3.x split its writers into separate plugin packages, so
# buddy\image.py imports OmeTiffWriter from bioio_ome_tiff. Install it
# explicitly so this works even if the on-disk pyproject.toml predates
# that dependency.
conda run --no-capture-output -n $EnvName pip install "bioio-ome-tiff"

if (Test-Path $NapariBuddyPath) {
    Write-Host "==> Installing napari-buddy plugin from $NapariBuddyPath"
    conda run --no-capture-output -n $EnvName pip install -e $NapariBuddyPath
} else {
    Write-Host "==> napari-buddy not found at '$NapariBuddyPath' - skipping."
    Write-Host "    Install it later with: conda activate $EnvName; pip install -e <path-to-napari-buddy>"
}

Write-Host "==> Smoke test (imports only)"
conda run --no-capture-output -n $EnvName python -c "import buddy, bigfish, omnipose, bioio_ome_tiff; print('buddy imports OK')"

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  conda activate $EnvName"
Write-Host "  pytest buddy/tests/test_image.py -v    # full pipeline test on example data"
Write-Host "  python -m napari                       # launch viewer; check Plugins -> buddy"
Write-Host "  python -m notebook                    # run the analysis notebooks"
Write-Host ""
Write-Host "Tip: always launch with 'python -m napari' (not bare 'napari') so it"
Write-Host "runs THIS env's napari and can't get bounced to base conda."
