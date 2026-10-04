[CmdletBinding()]
param(
    [switch]$Activate,
    [switch]$SkipLfs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ProjectRoot = $PSScriptRoot
$PythonVersionFile = Join-Path $ProjectRoot ".python-version"
$LockFile = Join-Path $ProjectRoot "uv.lock"
$ProjectFile = Join-Path $ProjectRoot "pyproject.toml"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Executable,
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Write-Host "> $Executable $($Arguments -join ' ')" -ForegroundColor DarkGray
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $Executable $($Arguments -join ' ')"
    }
}

function Find-Uv {
    $Command = Get-Command uv -CommandType Application -ErrorAction SilentlyContinue
    if ($null -ne $Command) {
        return $Command.Source
    }

    $Candidates = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")
    )
    foreach ($Candidate in $Candidates) {
        if (Test-Path -LiteralPath $Candidate) {
            return $Candidate
        }
    }
    return $null
}

function Install-Uv {
    Write-Host "uv was not found; downloading the official Astral installer..." -ForegroundColor Yellow
    [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    $Installer = Join-Path (
        [IO.Path]::GetTempPath()
    ) ("uv-install-{0}.ps1" -f [Guid]::NewGuid().ToString("N"))
    try {
        Invoke-WebRequest `
            -UseBasicParsing `
            -Uri "https://astral.sh/uv/install.ps1" `
            -OutFile $Installer
        $PowerShellExecutable = (Get-Process -Id $PID).Path
        Invoke-Checked $PowerShellExecutable @(
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            $Installer
        )
    }
    finally {
        if (Test-Path -LiteralPath $Installer) {
            Remove-Item -LiteralPath $Installer -Force
        }
    }
}

if (-not (Test-Path -LiteralPath $ProjectFile)) {
    throw "pyproject.toml is missing from project root: $ProjectRoot"
}
if (-not (Test-Path -LiteralPath $LockFile)) {
    throw "uv.lock is missing from project root: $ProjectRoot"
}
if (-not (Test-Path -LiteralPath $PythonVersionFile)) {
    throw ".python-version is missing from project root: $ProjectRoot"
}

$UvExecutable = Find-Uv
if ($null -eq $UvExecutable) {
    Install-Uv
    $UvExecutable = Find-Uv
}
if ($null -eq $UvExecutable) {
    throw "uv installation completed but uv.exe could not be located. Open a new shell and rerun."
}

$UvDirectory = Split-Path -Parent $UvExecutable
if (($env:Path -split ";") -notcontains $UvDirectory) {
    $env:Path = "$UvDirectory;$env:Path"
}

$PythonRequest = (Get-Content -Raw -Encoding UTF8 $PythonVersionFile).Trim()
if ([string]::IsNullOrWhiteSpace($PythonRequest)) {
    throw ".python-version is empty"
}

Write-Host "Project root: $ProjectRoot" -ForegroundColor Cyan
Invoke-Checked $UvExecutable @("--version")
Invoke-Checked $UvExecutable @("python", "install", $PythonRequest)

if (-not $SkipLfs) {
    $GitCommand = Get-Command git -CommandType Application -ErrorAction SilentlyContinue
    if ($null -eq $GitCommand) {
        throw "Git is required. Install Git for Windows, clone the repository, then rerun."
    }
    & $GitCommand.Source "lfs" "version"
    if ($LASTEXITCODE -ne 0) {
        throw (
            "Git LFS is required for dataset artifacts. Git for Windows normally includes it; " +
            "install/update Git for Windows and rerun."
        )
    }
    if (Test-Path -LiteralPath (Join-Path $ProjectRoot ".git")) {
        Invoke-Checked $GitCommand.Source @("-C", $ProjectRoot, "lfs", "install", "--local")
        Invoke-Checked $GitCommand.Source @("-C", $ProjectRoot, "lfs", "pull")
    }
    else {
        Write-Warning ".git is absent; skipped `git lfs pull`. Use a Git clone to restore LFS data."
    }
}

$OriginalLocation = Get-Location
try {
    Set-Location -LiteralPath $ProjectRoot
    Invoke-Checked $UvExecutable @(
        "sync",
        "--python",
        $PythonRequest,
        "--extra",
        "embedding",
        "--extra",
        "eda",
        "--dev",
        "--locked"
    )
    Invoke-Checked $UvExecutable @("run", "--locked", "pe-research", "--help")
}
finally {
    Set-Location -LiteralPath $OriginalLocation
}

Write-Host ""
Write-Host "Bootstrap complete." -ForegroundColor Green
Write-Host "Run project commands from the repository with:"
Write-Host "  uv run --locked pe-research --help" -ForegroundColor Cyan
Write-Host "  uv run --locked pytest" -ForegroundColor Cyan

if ($Activate) {
    $ActivationScript = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"
    if (-not (Test-Path -LiteralPath $ActivationScript)) {
        throw "Virtual-environment activation script is missing: $ActivationScript"
    }
    if ($MyInvocation.InvocationName -ne ".") {
        Write-Warning (
            "To retain activation in the current shell, dot-source the bootstrap: " +
            ". .\bootstrap.ps1 -Activate"
        )
    }
    Set-Location -LiteralPath $ProjectRoot
    . $ActivationScript
    Write-Host "Project virtual environment activated: $env:VIRTUAL_ENV" -ForegroundColor Green
}
