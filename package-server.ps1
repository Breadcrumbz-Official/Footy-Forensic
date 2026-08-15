# Package the server for deployment to another machine
# Run from the soccer-form-ai/ directory

param(
    [string]$Dest = "server-deployment"
)

$src = "server"

# Create destination
New-Item -ItemType Directory -Path $Dest -Force | Out-Null

# Copy Python files (not __pycache__)
@(
    "main.py",
    "worker.py",
    "video.py",
    "pose.py",
    "ball_detection.py",
    "biomechanics.py",
    "analysis.py",
    "scoring.py",
    "models_cache.py",
    "check_env.py",
    "requirements.txt"
) | ForEach-Object {
    $file = Join-Path $src $_
    if (Test-Path $file) {
        Copy-Item $file $Dest
        Write-Host "[ok] $_"
    } else {
        Write-Error "$_ not found"
    }
}

# Copy models/ if it exists (optional)
$models = Join-Path $src "models"
if (Test-Path $models) {
    Copy-Item $models $Dest -Recurse -Force
    Write-Host "[ok] models/ (54MB)"
} else {
    Write-Host "[skip] models/ (will download on first run)"
}

# Create a README for deployment
@"
# Soccer Form AI Server

Run this on the machine that does the analysis:

    python -m venv sfai_env
    sfai_env\Scripts\activate  # Windows
    source sfai_env/bin/activate  # macOS/Linux

    pip install -r requirements.txt
    python check_env.py  # verify setup
    python -m uvicorn main:app --host 0.0.0.0 --port 8000

Expose to browser with:
    ngrok http 8000

See ../DEPLOY.md for full instructions.
"@ | Out-File "$Dest\README.txt"

Write-Host "`n✓ Server packaged to $Dest/"
Write-Host "  11 Python files (650 KB)"
if (Test-Path $models) {
    Write-Host "  models/ 54 MB (pre-cached)"
} else {
    Write-Host "  models/ will download on first run (requires internet)"
}
Write-Host "`nCopy this folder to another machine and run:"
Write-Host "  pip install -r requirements.txt"
Write-Host "  python check_env.py"
Write-Host "  python -m uvicorn main:app --host 0.0.0.0 --port 8000"
