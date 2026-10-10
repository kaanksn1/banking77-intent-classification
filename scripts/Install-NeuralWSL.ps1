#Requires -RunAsAdministrator
[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'

# No automatic restart, driver replacement, or change to existing distributions.
$repoDirectory = Split-Path -Parent $PSScriptRoot
$logDirectory = Join-Path $repoDirectory '.tools'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
Start-Transcript -Path (Join-Path $logDirectory 'wsl-install.log') -Append
try {
    & wsl.exe --install --distribution Ubuntu-24.04 --no-launch
    if ($LASTEXITCODE -notin @(0, 3010)) {
        throw "WSL installation returned exit code $LASTEXITCODE. Read the Windows installation output."
    }
    Write-Host 'Save your work and restart Windows when convenient if the installer requests it.'
    Write-Host 'After restart, launch Ubuntu-24.04 once to create your Linux user.'
    Write-Host 'Then follow docs/NEURAL_BASELINES.md for the AMD runtime and GPU preflight.'
} finally {
    Stop-Transcript
}
