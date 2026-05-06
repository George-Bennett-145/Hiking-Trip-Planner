# deploy.ps1 - Deploy the Lake District Trip Planner to Google Cloud Run.
#
# Idempotent: ensures the secret has a value, grants Cloud Run access to it,
# then builds the image with Cloud Build and deploys to Cloud Run in one shot.
#
# First run takes 8-12 minutes. Subsequent runs are 3-5 minutes.

$Root = $PSScriptRoot

# --- Config ---
$Project = "hiking-trip-planner-495520"
$Region  = "europe-west2"   # London
$Service = "lake-district-planner"
$Secret  = "openrouter-api-key"

function Write-Step { param($msg) Write-Host "`n>> $msg" -ForegroundColor Cyan }
function Write-OK   { param($msg) Write-Host "   [ok] $msg" -ForegroundColor Green }
function Write-Warn { param($msg) Write-Host "   [!]  $msg" -ForegroundColor Yellow }
function Stop-Script { param($msg) Write-Host "   [x] $msg" -ForegroundColor Red; exit 1 }

function Invoke-Gcloud {
    # Run gcloud and stop the script if exit code is non-zero. Avoids the
    # PS 5.1 quirk where redirecting native stderr triggers spurious aborts.
    param([string[]]$GcloudArgs, [string]$ErrMsg = "gcloud command failed")
    & gcloud @GcloudArgs
    if ($LASTEXITCODE -ne 0) { Stop-Script $ErrMsg }
}

Write-Host ""
Write-Host "  Lake District Trip Planner - Cloud Run deploy" -ForegroundColor White
Write-Host "  =============================================" -ForegroundColor DarkGray

# --- 1. Ensure project is selected ---

Write-Step "Selecting project $Project..."
Invoke-Gcloud @("config", "set", "project", $Project, "--quiet") "Could not set project."
Write-OK "Project active"

# --- 2. Ensure secret has a value (populate from .env on first run) ---

Write-Step "Checking secret '$Secret' has a value..."
$versions = & gcloud secrets versions list $Secret --format="value(name)"
if ($LASTEXITCODE -ne 0) { Stop-Script "Could not list secret versions." }

if (-not $versions) {
    Write-Warn "Secret has no versions. Populating from .env..."
    $envFile = Join-Path $Root ".env"
    if (-not (Test-Path $envFile)) { Stop-Script ".env not found in project root." }
    $line = Get-Content $envFile | Where-Object { $_ -match '^OPENROUTER_API_KEY\s*=' } | Select-Object -First 1
    if (-not $line) { Stop-Script "OPENROUTER_API_KEY not in .env" }
    $value = ($line -replace '^OPENROUTER_API_KEY\s*=\s*', '').Trim('"', "'", ' ')
    if (-not $value) { Stop-Script "OPENROUTER_API_KEY in .env is empty" }
    $tmp = Join-Path $env:TEMP "or_key.tmp"
    [System.IO.File]::WriteAllText($tmp, $value)
    & gcloud secrets versions add $Secret --data-file="$tmp" --quiet | Out-Null
    $rc = $LASTEXITCODE
    Remove-Item $tmp -Force
    if ($rc -ne 0) { Stop-Script "Failed to upload secret value." }
    Write-OK "Secret populated"
} else {
    Write-OK "Secret has at least one version"
}

# --- 3. Grant Cloud Run service account access to the secret ---

Write-Step "Ensuring Cloud Run can read the secret..."
$projectNumber = & gcloud projects describe $Project --format="value(projectNumber)"
if ($LASTEXITCODE -ne 0) { Stop-Script "Could not read project number." }
$runSA = "$projectNumber-compute@developer.gserviceaccount.com"
& gcloud secrets add-iam-policy-binding $Secret `
    --member="serviceAccount:$runSA" `
    --role="roles/secretmanager.secretAccessor" `
    --quiet | Out-Null
if ($LASTEXITCODE -ne 0) { Stop-Script "Could not grant secret access." }
Write-OK "Service account: $runSA"

# --- 4. Build & deploy in one command ---

Write-Step "Building image and deploying to Cloud Run..."
Write-Host "   (Cloud Build takes a few minutes; output streams below)" -ForegroundColor Gray
Write-Host ""

& gcloud run deploy $Service `
    --source $Root `
    --region $Region `
    --allow-unauthenticated `
    --memory 2Gi `
    --cpu 2 `
    --min-instances 0 `
    --max-instances 1 `
    --timeout 300 `
    --update-secrets="OPENROUTER_API_KEY=${Secret}:latest" `
    --quiet

if ($LASTEXITCODE -ne 0) { Stop-Script "Deployment failed. See output above." }

# --- 5. Show URL ---

Write-Step "Deployment complete."
$url = & gcloud run services describe $Service --region $Region --format="value(status.url)"
Write-Host ""
Write-Host "   $url" -ForegroundColor Green
Write-Host ""
Write-Host "   First request after idle takes ~10s (cold start + graph load)." -ForegroundColor Gray
Write-Host "   To make it always-on (avoids cold starts):" -ForegroundColor Gray
Write-Host "       gcloud run services update $Service --region $Region --min-instances=1" -ForegroundColor Gray
Write-Host ""
