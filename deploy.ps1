# Fast Browser Use - Windows one-click deploy for the LiquidAI d1-3B decision
# model on llama.cpp CPU inference.
#
#   .\deploy.ps1            (or double-click deploy.bat)
#   .\deploy.ps1 -NoChat    (skip the interactive chat loop at the end)
#
# What it does:
#   1. Checks Python 3.10+ and creates a local .venv
#   2. Installs the Python environment and Playwright Chromium
#   3. Verifies models\d1-3B-Q4_K_M.gguf (auto-download from ModelScope if missing)
#   4. Locates or downloads a prebuilt llama.cpp llama-server (Windows CPU or Vulkan build)
#   5. Runs a live real-browser verification task (example.com -> IANA)
#   6. Drops you into the interactive chat loop with a visible browser
#
# Model already downloaded? Just place it at:
#   <project>\models\d1-3B-Q4_K_M.gguf
# and the script picks it up automatically.

param(
  [switch]$NoChat
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$Root = $PSScriptRoot
Set-Location $Root
$GGUF = "d1-3B-Q4_K_M.gguf"
$ModelPath = Join-Path $Root "models\$GGUF"
$ToolsDir = Join-Path $Root "tools\llama.cpp"
$ServerBin = Join-Path $ToolsDir "llama-server.exe"
$GGUF_URL_MS = "https://modelscope.cn/models/LiquidAI/d1-3B-GGUF/resolve/master/$GGUF"
$GGUF_URL_HF = "https://huggingface.co/LiquidAI/d1-3B-GGUF/resolve/main/$GGUF"

function Log($msg)  { Write-Host "[deploy] $msg" -ForegroundColor Cyan }
function Warn($msg) { Write-Host "[deploy] WARNING: $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "[deploy] ERROR: $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 1. python env
Log "Step 1/5: checking Python"
$Python = $null
foreach ($candidate in @("python", "py")) {
  try {
    $v = & $candidate --version 2>$null
    if ($LASTEXITCODE -eq 0) { $Python = $candidate; Write-Host "[deploy]   found: $v"; break }
  } catch {}
}
if (-not $Python) { Fail "Python 3.10+ not found. Install from https://www.python.org/downloads/ (check 'Add python.exe to PATH')." }
if ($Python -eq "py") { $Python = "py -3" }

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
  Log "Creating virtual environment .venv"
  Invoke-Expression "$Python -m venv .venv"
  if (-not (Test-Path $venvPython)) { Fail "Failed to create .venv" }
}
Log "Installing Python dependencies (pip install -e .)"
& $venvPython -m pip install --quiet --upgrade pip
& $venvPython -m pip install --quiet -e .
if ($LASTEXITCODE -ne 0) { Fail "pip install failed" }

# ---------------------------------------------------------------- 2. browser
Log "Step 2/5: installing Playwright Chromium"
& $venvPython -m fast_browser_use.cli install-browser
if ($LASTEXITCODE -ne 0) { Warn "Chromium install reported an error; if the verification step fails, run: .venv\Scripts\fbu.exe install-browser" }

# ---------------------------------------------------------------- 3. model weights
Log "Step 3/5: checking d1-3B GGUF weights"
$haveModel = (Test-Path $ModelPath) -and ((Get-Item $ModelPath).Length -gt 1MB)
if ($haveModel) {
  Log "Model found: models\$GGUF ($([math]::Round((Get-Item $ModelPath).Length/1MB,1)) MB)"
} else {
  Warn "Model not found at models\$GGUF"
  Log "If you already downloaded d1-3B-Q4_K_M.gguf: copy it into the 'models' folder of this project, then rerun."
  Log "Otherwise it will now be downloaded from ModelScope (~1.7 GB)..."
  New-Item -ItemType Directory -Force -Path (Join-Path $Root "models") | Out-Null
  $ok = $false
  try {
    & curl.exe -L -C - --retry 5 --retry-delay 3 --progress-bar -o $ModelPath $GGUF_URL_MS
    $ok = ($LASTEXITCODE -eq 0) -and (Test-Path $ModelPath)
  } catch { }
  if (-not $ok) {
    Warn "ModelScope download failed; falling back to Hugging Face"
    & curl.exe -L -C - --retry 5 --retry-delay 3 --progress-bar -o $ModelPath $GGUF_URL_HF
    if ($LASTEXITCODE -ne 0) { Fail "Model download failed. Download $GGUF manually and place it at models\$GGUF" }
  }
}

# ---------------------------------------------------------------- 4. llama-server
Log "Step 4/5: locating llama-server (llama.cpp CPU build)"
$envFBU_BIN = $env:FBU_LLAMACPP_BIN
$serverFound = $null
if ($envFBU_BIN -and (Test-Path $envFBU_BIN)) { $serverFound = $envFBU_BIN }
elseif (Test-Path $ServerBin) { $serverFound = $ServerBin }
elseif (Get-Command llama-server -ErrorAction SilentlyContinue) { $serverFound = (Get-Command llama-server).Source }
elseif (Test-Path (Join-Path $Root "build\bin\llama-server.exe")) { $serverFound = Join-Path $Root "build\bin\llama-server.exe" }

# Also pick up a previously downloaded/extracted release (cpu OR vulkan build).
# Vulkan builds run fine on CPU; both contain llama-server.exe.
if (-not $serverFound) {
  $searchDirs = @($Root, (Join-Path $env:USERPROFILE "Downloads")) | Where-Object { Test-Path $_ }
  foreach ($dir in $searchDirs) {
    $hit = Get-ChildItem -Path $dir -Recurse -Depth 4 -Filter "llama-server.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($hit) {
      Log "Found an existing llama-server: $($hit.FullName)"
      Log "Copying it (plus its dlls) into tools\llama.cpp for future runs"
      New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
      Copy-Item $hit.FullName $ServerBin -Force
      Get-ChildItem -Path $hit.DirectoryName -Filter "*.dll" -ErrorAction SilentlyContinue | Copy-Item -Destination $ToolsDir -Force -ErrorAction SilentlyContinue
      $serverFound = $ServerBin
      break
    }
  }
}

if ($serverFound) {
  Log "llama-server ready: $serverFound"
} else {
  Log "No llama-server found; downloading the prebuilt Windows CPU release from GitHub..."
  New-Item -ItemType Directory -Force -Path $ToolsDir | Out-Null
  try {
    $release = Invoke-RestMethod -Uri "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest" -TimeoutSec 60
    # Prefer the pure CPU build; the Vulkan build (works on CPU + any GPU) is the fallback.
    $asset = $release.assets | Where-Object { $_.name -match "^llama-.*-bin-win-cpu-x64\.zip$" } | Select-Object -First 1
    if (-not $asset) { $asset = $release.assets | Where-Object { $_.name -match "^llama-.*-bin-win-vulkan-x64\.zip$" } | Select-Object -First 1 }
    if (-not $asset) { throw "no win-cpu-x64 or win-vulkan-x64 asset in latest release" }
    $zipPath = Join-Path $env:TEMP $asset.name
    Log "Downloading $($asset.name) ($([math]::Round($asset.size/1MB,0)) MB)"
    & curl.exe -L --retry 5 --retry-delay 3 --progress-bar -o $zipPath $asset.browser_download_url
    if ($LASTEXITCODE -ne 0) { throw "curl download failed" }
    Log "Extracting to tools\llama.cpp"
    Expand-Archive -Path $zipPath -DestinationPath $ToolsDir -Force
    # The zip nests everything under llama-*/; lift llama-server.exe (+ dlls) up.
    $nested = Get-ChildItem -Path $ToolsDir -Recurse -Filter "llama-server.exe" | Select-Object -First 1
    if ($nested -and ($nested.FullName -ne $ServerBin)) {
      Copy-Item $nested.FullName $ServerBin -Force
      Get-ChildItem -Path $nested.Directory -Filter "*.dll" | Copy-Item -Destination $ToolsDir -Force -ErrorAction SilentlyContinue
    }
    Remove-Item $zipPath -Force -ErrorAction SilentlyContinue
  } catch {
    Warn "Automatic llama-server download failed: $($_.Exception.Message)"
    Fail @"
Could not fetch llama-server automatically. Manual fix (2 minutes):
  1. Use the zip you already downloaded (cpu or vulkan build both work,
     e.g. llama-b11849-bin-win-vulkan-x64.zip).
  2. Extract llama-server.exe (and any *.dll next to it) into:  tools\llama.cpp\
  3. Rerun this script.
"@
  }
  if (-not (Test-Path $ServerBin)) { Fail "llama-server.exe still missing after download." }
  $serverFound = $ServerBin
  Log "llama-server ready: $serverFound"
}

# ---------------------------------------------------------------- 5. live verification
Log "Step 5/5: live real-browser verification (example.com -> IANA)"
$env:FBU_LLAMACPP_BIN = $serverFound
$env:FBU_MODEL = $ModelPath
$trace = Join-Path $Root "artifacts\deploy-verification.json"
& $venvPython -m fast_browser_use.cli run https://example.com `
  --goal 'Open the "More information" link so the IANA example-domains page is shown.' `
  --expect-url 'https://www.iana.org/help/example-domains' `
  --expect-title 'Example Domains' `
  --trace $trace
if ($LASTEXITCODE -ne 0) {
  Warn "Verification task did not pass on the first attempt; see $trace"
  Warn "d1-3B is a 3B decision model on CPU: retry, or phrase the goal closer to the visible page labels."
}

Write-Host ""
Write-Host "=============================================================" -ForegroundColor Green
Write-Host " Fast Browser Use deployed successfully (d1-3B + llama.cpp)" -ForegroundColor Green
Write-Host "=============================================================" -ForegroundColor Green
Write-Host ""
Write-Host " Weights : models\$GGUF"
Write-Host " Server  : $serverFound  (auto-started on demand, port 8765)"
Write-Host ""
Write-Host " Run a task (open a new terminal in the project folder):"
Write-Host '   .venv\Scripts\fbu.exe run <url> --goal "..." ' -ForegroundColor White
Write-Host '     [--expect-url ...] [--expect-title ...] [--expect-text ...]' -ForegroundColor White
Write-Host ""
Write-Host " Example:"
Write-Host '   .venv\Scripts\fbu.exe run https://example.com --goal "Open the More information link"' -ForegroundColor White
Write-Host ""
Write-Host " Env knobs (optional):"
Write-Host "   FBU_BACKEND=llamacpp            (default)"
Write-Host "   FBU_LLAMACPP_BIN=$serverFound"
Write-Host "   FBU_MODEL=$ModelPath"
Write-Host "   FBU_LLAMACPP_URL=http://127.0.0.1:8765"
Write-Host "   FBU_LLAMACPP_THREADS=N          (defaults to all cores)"
Write-Host ""

# ---------------------------------------------------------------- 6. interactive chat
if ($NoChat) {
  Log "Skipping the interactive chat (-NoChat). Start it later with: .\fbu.bat chat"
} else {
  Log "Starting the interactive chat loop (type /help inside; /exit to quit)"
  & $venvPython -m fast_browser_use.cli chat
}
