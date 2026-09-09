# Builds the Windows desktop package: dist\Upscaler\ (one-folder app you can
# zip and run anywhere) and, if Inno Setup is installed, dist\UpscalerSetup.exe.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_exe.ps1
#
# Prerequisites: scripts\setup.ps1 has been run (venv, engines_bin, models)
# and, for the installer, Inno Setup 6 (winget install JRSoftware.InnoSetup).
$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $PSScriptRoot
Set-Location $RootDir
$venvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { Write-Error "run scripts\setup.ps1 first (no .venv found)" }

$Dist = Join-Path $RootDir "dist\Upscaler"

Write-Host "== Build dependencies =="
& $venvPython -m pip install --upgrade pyinstaller pyinstaller-hooks-contrib

Write-Host "== PyInstaller =="
if (Test-Path $Dist) { Remove-Item $Dist -Recurse -Force }
& $venvPython -m PyInstaller --noconfirm --clean upscaler.spec
if (-not (Test-Path (Join-Path $Dist "Upscaler.exe"))) { Write-Error "PyInstaller did not produce dist\Upscaler\Upscaler.exe" }

Write-Host "== Engines and models =="
foreach ($dir in @("engines_bin", "models")) {
    $src = Join-Path $RootDir $dir
    if (-not (Test-Path $src)) { Write-Error "$dir missing — run scripts\setup.ps1 (and export the ONNX models) first" }
    Copy-Item $src (Join-Path $Dist $dir) -Recurse -Force
}
# the ncnn release zips carry every model; only the ones the app uses are shipped
$rifeDir = Join-Path $Dist "engines_bin\rife"
if (Test-Path $rifeDir) {
    Get-ChildItem $rifeDir -Directory | Where-Object { $_.Name -ne "rife-v4.6" } | Remove-Item -Recurse -Force
}

Write-Host "== ffmpeg =="
$bin = Join-Path $Dist "bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
$localFfmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue
$localFfprobe = Get-Command ffprobe -ErrorAction SilentlyContinue
if ($localFfmpeg -and $localFfprobe -and ($localFfmpeg.Source -notlike "*WindowsApps*")) {
    Copy-Item $localFfmpeg.Source, $localFfprobe.Source $bin -Force
    Write-Host "bundled ffmpeg from $($localFfmpeg.Source)"
} else {
    $zip = Join-Path $env:TEMP "ffmpeg-release-essentials.zip"
    $tmp = Join-Path $env:TEMP "ffmpeg-release-essentials"
    Invoke-WebRequest -Uri "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" -OutFile $zip
    Expand-Archive -Path $zip -DestinationPath $tmp -Force
    $found = Get-ChildItem $tmp -Recurse -Filter ffmpeg.exe | Select-Object -First 1
    Copy-Item (Join-Path $found.DirectoryName "ffmpeg.exe"), (Join-Path $found.DirectoryName "ffprobe.exe") $bin -Force
    Remove-Item $zip, $tmp -Recurse -Force
    Write-Host "bundled ffmpeg release-essentials build"
}

Write-Host "== Smoke test =="
$env:UPSCALER_DATA_DIR = Join-Path $env:TEMP "upscaler-smoke"
$proc = Start-Process -FilePath (Join-Path $Dist "Upscaler.exe") -ArgumentList "--no-browser" -PassThru
try {
    $ok = $false
    foreach ($i in 1..30) {
        Start-Sleep -Seconds 1
        try {
            $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:8000/api/engines" -TimeoutSec 2
            if ($r.StatusCode -eq 200) { $ok = $true; break }
        } catch {}
    }
    if ($ok) { Write-Host "app starts and answers /api/engines" } else { Write-Warning "app did not answer within 30 s — check $env:UPSCALER_DATA_DIR\upscaler.log" }
} finally {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Remove-Item Env:UPSCALER_DATA_DIR
}

Write-Host "== Installer =="
$iscc = Get-Command ISCC.exe -ErrorAction SilentlyContinue
if (-not $iscc) {
    $candidate = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (Test-Path $candidate) { $iscc = Get-Command $candidate }
}
if ($iscc) {
    & $iscc.Source (Join-Path $RootDir "installer\upscaler.iss")
    Write-Host "installer written to dist\UpscalerSetup.exe"
} else {
    Write-Host "Inno Setup not found — skipping installer. dist\Upscaler\ is a complete portable build."
    Write-Host "(winget install JRSoftware.InnoSetup, then re-run to get dist\UpscalerSetup.exe)"
}
