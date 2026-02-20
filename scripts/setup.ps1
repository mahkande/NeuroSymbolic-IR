param(
    [switch]$Lite
)

$ErrorActionPreference = "Stop"

Write-Host "[setup] Starting NeuroGraph OS setup..."

if (!(Test-Path ".venv")) {
    Write-Host "[setup] Creating virtual environment (.venv)"
    python -m venv .venv
}

Write-Host "[setup] Activating virtual environment"
. .\.venv\Scripts\Activate.ps1

Write-Host "[setup] Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

$req = if ($Lite) { "requirements-lite.txt" } else { "requirements.txt" }
Write-Host "[setup] Installing dependencies from $req"
python -m pip install -r $req

Write-Host "[setup] Downloading required NLTK data"
@'
import nltk
nltk.download("punkt_tab", quiet=True)
'@ | python -

Write-Host ""
Write-Host "[setup] Done."
Write-Host "[setup] Run dashboard with:"
Write-Host "        streamlit run ui/dashboard.py"
