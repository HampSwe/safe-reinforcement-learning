$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class SeedBatchPowerRequest
{
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@

$esContinuous = [uint32]2147483648
$esSystemRequired = [uint32]0x00000001
$projectDir = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectDir ".venv\Scripts\python.exe"
$trainPath = Join-Path $PSScriptRoot "train.py"
$config = "experiment_05_no_bootstrap_t_reset_critic_epochs.yaml"
$logsDir = Join-Path $PSScriptRoot "logs"

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Virtual-environment Python was not found at: $pythonPath"
}

$requestResult = [SeedBatchPowerRequest]::SetThreadExecutionState(
    $esContinuous -bor $esSystemRequired
)
if ($requestResult -eq 0) {
    throw "Windows rejected the keep-awake request."
}

Push-Location $PSScriptRoot
try {
    foreach ($seed in 1..5) {
        $completedRun = Get-ChildItem -LiteralPath $logsDir -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match "_t_no_bootstrap_reset_critic_epochs_seed_${seed}$" } |
            Where-Object {
                $configPath = Join-Path $_.FullName "config_used.json"
                $jitPath = Join-Path $_.FullName "exported\policy.pt"
                $onnxPath = Join-Path $_.FullName "exported\policy.onnx"
                if (-not ((Test-Path -LiteralPath $configPath -PathType Leaf) -and
                    (Test-Path -LiteralPath $jitPath -PathType Leaf) -and
                    (Test-Path -LiteralPath $onnxPath -PathType Leaf))) {
                    return $false
                }
                $used = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
                return $used.seed -eq $seed -and
                    $used.config -eq $config -and
                    $used.iterations -eq 4000
            } |
            Select-Object -First 1

        if ($completedRun) {
            Write-Host ("Seed {0}/5 already completed in {1}; skipping." -f $seed, $completedRun.Name)
            continue
        }

        Write-Host ""
        Write-Host "======================================================================="
        Write-Host ("T + no bootstrap + reset: seed {0}/5" -f $seed)
        Write-Host "======================================================================="

        & $pythonPath $trainPath `
            --config $config `
            --iterations 4000 `
            --device gpu `
            --run-name ("t_no_bootstrap_reset_critic_epochs_seed_{0}" -f $seed) `
            --seed $seed `
            --no-show-dashboard-after

        if ($LASTEXITCODE -ne 0) {
            throw "Seed $seed failed with exit code $LASTEXITCODE."
        }
    }

    Write-Host ""
    Write-Host "All five seeded experiments finished successfully."
}
finally {
    Pop-Location
    [SeedBatchPowerRequest]::SetThreadExecutionState($esContinuous) | Out-Null
}
