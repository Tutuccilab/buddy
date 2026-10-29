# buddy installer for Windows (run from the Anaconda Prompt, in the buddy repo root)
#
#   powershell -ExecutionPolicy Bypass -File install.ps1
#
# Optional args:
#   -NapariBuddyPath "..\napari-buddy"      path to the napari-buddy repo
#   -EnvName "buddy"                        conda env name
#   -CPU                                    do not install the GPU (CUDA) version of PyTorch
#
# An existing environment of that name is removed first, so the script can be run again after a failed install.
# Every package version comes from constraints\windows.txt, so every install gets the same packages.
# macOS / Linux: install.sh (same steps). What the script does, and what to do when it fails: INSTALL.md.

param(
    [string]$NapariBuddyPath = "..\napari-buddy",
    [string]$EnvName = "buddy",
    [switch]$CPU          # add -CPU to skip the GPU version of PyTorch
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path "pyproject.toml")) { throw "Run this from the buddy repo root (the folder with pyproject.toml)." }
if (-not (Get-Command conda -ErrorAction SilentlyContinue)) { throw "conda not found: run this in the Anaconda Prompt (INSTALL.md)." }

$Constraints = "constraints\windows.txt"
$Omnipose = (Get-Content $Constraints | Where-Object { $_ -match '^omnipose==' } | Select-Object -First 1).Trim()

function Run { conda run --no-capture-output -n $EnvName @args; if ($LASTEXITCODE -ne 0) { throw "step failed: $args" } }

if ((conda env list) -match "^$EnvName\s") {
    Write-Host "==> Removing the existing conda environment '$EnvName'"
    conda env remove --name $EnvName -y
}

Write-Host "==> Creating conda environment '$EnvName' (Python 3.12, Java 11 for reading .vsi files)"
# Java comes from conda-forge (activating the env sets JAVA_HOME); Bio-Formats itself is a pip package (bioformats_jar).
conda create --name $EnvName python=3.12 conda-forge::openjdk=11 -y

Write-Host "==> Installing buddy (editable, with test + napari + notebooks extras; versions from $Constraints)"
# The 'napari' extra pulls the napari viewer + Qt backend INTO this env, so 'python -m napari' launches
# from here and never falls back to a base-conda napari.
Run pip install -c $Constraints -e ".[test,napari,notebooks]"

Write-Host "==> Installing Omnipose ($Omnipose, without its own dependency list: what it needs is in pyproject.toml)"
Run pip install --no-deps $Omnipose

if (Test-Path $NapariBuddyPath) {
    Write-Host "==> Installing napari-buddy plugin from $NapariBuddyPath"
    Run pip install -c $Constraints -e $NapariBuddyPath
} else {
    Write-Host "==> napari-buddy not found at '$NapariBuddyPath' - skipping."
    Write-Host "    Install it later with: conda activate $EnvName; pip install -e <path-to-napari-buddy>"
}

# GPU: on Windows the PyPI PyTorch is CPU-only. With an NVIDIA GPU, swap in the CUDA 12.1 build (same version,
# self-contained wheel) so that Omnipose, the bud model and spot detection run on the GPU.
$hasGpu = $null -ne (Get-Command nvidia-smi -ErrorAction SilentlyContinue)
if ($hasGpu -and -not $CPU) {
    Write-Host "==> NVIDIA GPU found: installing PyTorch with CUDA support"
    # --no-deps: only swap torch (+ torchvision), keep everything else; retries: the file is ~2.5 GB
    Run pip install --force-reinstall --no-deps --no-cache-dir --timeout 120 --retries 10 -c $Constraints torch torchvision --index-url https://download.pytorch.org/whl/cu121
} else {
    Write-Host "==> No NVIDIA GPU found (or -CPU given): keeping the CPU version of PyTorch"
}

Write-Host "==> Smoke test (imports only)"
Run python -c "import buddy, bigfish, bioformats_jar, torch, torchvision; from buddy.segmentation import masks; masks.check_omnipose(); print('buddy imports OK | GPU available:', torch.cuda.is_available())"
Run java -version

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  conda activate $EnvName"
Write-Host "  pytest -m 'not slow'                   # fast tests (about a minute)"
Write-Host "  python -m napari                       # the viewer; Plugins -> Napari Buddy"
Write-Host "  python -m notebook notebooks           # the analysis notebooks"
Write-Host ""
Write-Host "Tip: always launch with 'python -m napari' (not bare 'napari') so it runs THIS env's napari."
