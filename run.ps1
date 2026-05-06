# run.ps1 - setup and launch the Lake District Trip Planner
#
# Safe to run multiple times: skips steps whose output already exists.
#
# If PowerShell blocks this script, run once from an admin terminal:
#     Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot

function Write-Step { param($msg) Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-OK   { param($msg) Write-Host "   [ok] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "   [!]  $msg" -ForegroundColor Yellow }
function Write-Fail { param($msg) Write-Host "   [x] $msg" -ForegroundColor Red; exit 1 }

Write-Host ""
Write-Host "  Lake District Trip Planner" -ForegroundColor White
Write-Host "  ===========================" -ForegroundColor DarkGray

# --- 1. Python ---

Write-Step "Checking Python..."
try {
    $pyver = & python --version 2>&1
    Write-OK "$pyver"
} catch {
    Write-Fail "Python not found. Install Python 3.10+ and re-run."
}

# --- 2. Virtual environment ---

$VenvDir   = Join-Path $Root ".venv"
$PipExe    = Join-Path $VenvDir "Scripts\pip.exe"
$PythonExe = Join-Path $VenvDir "Scripts\python.exe"

Write-Step "Checking virtual environment..."
if (-not (Test-Path $VenvDir)) {
    Write-Host "   Creating .venv (first run only)..." -ForegroundColor Gray
    & python -m venv "$VenvDir"
    Write-OK "Created .venv"
} else {
    Write-OK ".venv already exists"
}

# --- 3. Dependencies ---

Write-Step "Installing / verifying dependencies..."
Write-Host "   (this is fast after the first run)" -ForegroundColor Gray
& $PipExe install -r "$Root\requirements.txt" --quiet
Write-OK "Dependencies up to date"

# --- 4. .env check ---

Write-Step "Checking .env..."
$EnvFile = Join-Path $Root ".env"
if (-not (Test-Path $EnvFile)) {
    Write-Warn ".env not found. Create it in the project root with:"
    Write-Host ""
    Write-Host "      OPENROUTER_API_KEY=your_key_here" -ForegroundColor Yellow
    Write-Host "      MODEL=anthropic/claude-haiku-4-5-20251001" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "   The server will start but the chat assistant will not work." -ForegroundColor Yellow
} else {
    Write-OK ".env found"
}

# --- 5. Walking route graph ---

$WalkGraph = Join-Path $Root "output\lake_district_walking_graph.graphml"
Write-Step "Checking walking route graph..."
if (-not (Test-Path $WalkGraph)) {
    Write-Host "   Not found - downloading from OpenStreetMap (~50 MB, 1-3 min)..." -ForegroundColor Gray
    & $PythonExe "$Root\build_walking_graph.py"
    Write-OK "Walking graph built"
} else {
    $sizeMB = [math]::Round((Get-Item $WalkGraph).Length / 1MB, 1)
    Write-OK "Walking graph already exists ($($sizeMB) MB)"
}

# --- 6. Driving route graph ---

$DriveGraph = Join-Path $Root "output\lake_district_driving_graph.graphml"
Write-Step "Checking driving route graph..."
if (-not (Test-Path $DriveGraph)) {
    Write-Host "   Not found - downloading from OpenStreetMap (~8 MB)..." -ForegroundColor Gray
    & $PythonExe "$Root\build_driving_graph.py"
    Write-OK "Driving graph built"
} else {
    $sizeMB = [math]::Round((Get-Item $DriveGraph).Length / 1MB, 1)
    Write-OK "Driving graph already exists ($($sizeMB) MB)"
}

# --- 7. Launch ---

Write-Step "Starting server..."
Write-Host ""
Write-Host "   http://localhost:8000" -ForegroundColor Green
Write-Host "   Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

& $PythonExe -m uvicorn server.main:app --host 0.0.0.0 --port 8000
