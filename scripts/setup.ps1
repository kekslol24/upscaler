# Windows setup: installs ffmpeg via winget, creates the Python venv, and
# downloads the realesrgan-ncnn-vulkan engine binary + models.
# Run from a PowerShell prompt:  powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir

function Find-Python {
    foreach ($candidate in @("py -3", "python", "python3")) {
        $exe, $argsRest = $candidate -split " ", 2
        try {
            $v = & $exe $argsRest --version 2>&1
            if ($LASTEXITCODE -eq 0 -and "$v" -match "Python 3\.(1[0-9]|[2-9][0-9])") { return $candidate }
        } catch {}
    }
    return $null
}

Write-Host "== ffmpeg =="
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host "already installed: $((Get-Command ffmpeg).Source)"
} else {
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    # winget updates the user PATH, but the current shell keeps its old copy.
    $env:Path = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [Environment]::GetEnvironmentVariable("Path", "User")
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        Write-Warning "ffmpeg installed but not found on PATH yet; open a new terminal after this script finishes."
    }
}

Write-Host "== Python venv =="
$py = Find-Python
if (-not $py) {
    Write-Error "Python 3.10+ not found. Install it from https://www.python.org/downloads/ (tick 'Add python.exe to PATH') or run: winget install Python.Python.3.12"
}
$venvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $exe, $argsRest = $py -split " ", 2
    & $exe $argsRest -m venv .venv
}
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r requirements.txt

Write-Host "== Real-ESRGAN ncnn-vulkan engine binary =="
$EngineDir = Join-Path $RootDir "engines_bin\realesrgan"
$Binary = Join-Path $EngineDir "realesrgan-ncnn-vulkan.exe"
if (Test-Path $Binary) {
    Write-Host "already installed at $EngineDir"
} else {
    New-Item -ItemType Directory -Force -Path $EngineDir | Out-Null
    $zip = Join-Path $env:TEMP "realesrgan-ncnn-vulkan.zip"
    Invoke-WebRequest -Uri "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesrgan-ncnn-vulkan-20220424-windows.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $EngineDir -Force
    Remove-Item $zip -Force
    Write-Host "installed to $EngineDir"
}

Write-Host "== RIFE ncnn-vulkan frame interpolation binary =="
$RifeDir = Join-Path $RootDir "engines_bin\rife"
$RifeBinary = Join-Path $RifeDir "rife-ncnn-vulkan.exe"
if ((Test-Path $RifeBinary) -and (Test-Path (Join-Path $RifeDir "rife-v4.6"))) {
    Write-Host "already installed at $RifeDir"
} else {
    New-Item -ItemType Directory -Force -Path $RifeDir | Out-Null
    $zip = Join-Path $env:TEMP "rife-ncnn-vulkan.zip"
    $tmp = Join-Path $env:TEMP "rife-ncnn-vulkan-extract"
    # ~430 MB: the release bundles every RIFE model; only v4.6 is kept
    Invoke-WebRequest -Uri "https://github.com/nihui/rife-ncnn-vulkan/releases/download/20221029/rife-ncnn-vulkan-20221029-windows.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $extracted = Get-ChildItem -Directory $tmp | Select-Object -First 1
    Copy-Item (Join-Path $extracted.FullName "rife-ncnn-vulkan.exe") $RifeDir -Force
    Get-ChildItem (Join-Path $extracted.FullName "*.dll") -ErrorAction SilentlyContinue | Copy-Item -Destination $RifeDir -Force
    Copy-Item (Join-Path $extracted.FullName "rife-v4.6") $RifeDir -Recurse -Force
    Remove-Item $zip, $tmp -Recurse -Force
    Write-Host "installed to $RifeDir"
}

Write-Host ""
Write-Host "== Checking Vulkan sees a GPU =="
$vk = Get-Command vulkaninfo, vulkaninfoSDK -ErrorAction SilentlyContinue | Select-Object -First 1
if ($vk) {
    # vulkaninfo prints loader warnings to stderr; with ErrorActionPreference=Stop
    # PowerShell would turn those into a terminating error, so relax it here.
    $prevEap = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $devices = & $vk.Source --summary 2>$null | Select-String "deviceName"
    $ErrorActionPreference = $prevEap
    if ($devices) { $devices | ForEach-Object { Write-Host $_.Line.Trim() } }
    else { Write-Warning "vulkaninfo reported no device; update your GPU driver." }
} else {
    Write-Host "vulkaninfo not found (optional). The engine will use the default Vulkan device; the GPU picker in the UI will be empty."
}

Write-Host ""
Write-Host "Setup done. Start the server with:"
Write-Host "  .venv\Scripts\activate"
Write-Host "  uvicorn backend.main:app --reload"
Write-Host "Then open http://127.0.0.1:8000"
