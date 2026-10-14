$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class PpoGradPowerRequest
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
$logsDir = Join-Path $PSScriptRoot "logs"

$config = "experiment_10_ppo_grad_t_bootstrap_reset.yaml"
$runName = "ppo_grad_t_bootstrap_reset_seed_1"
$iterations = 4000
$seed = 1

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Virtual-environment Python was not found at: $pythonPath"
}

$requestResult = [PpoGradPowerRequest]::SetThreadExecutionState(
    $esContinuous -bor $esSystemRequired
)
if ($requestResult -eq 0) {
    throw "Windows rejected the keep-awake request."
}

Push-Location $projectDir
try {
    $completedRun = Get-ChildItem -LiteralPath $logsDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "_${runName}$" } |
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
                $used.iterations -eq $iterations
        } |
        Select-Object -First 1

    if ($completedRun) {
        Write-Host ("Run already completed in {0}; skipping." -f $completedRun.Name)
    } else {
        Write-Host ""
        Write-Host "======================================================================="
        Write-Host ("RLSGF_PPOGrad: {0}, seed {1}, {2} iterations" -f $config, $seed, $iterations)
        Write-Host "======================================================================="

        & $pythonPath $trainPath `
            --config $config `
            --iterations $iterations `
            --device gpu `
            --run-name $runName `
            --seed $seed `
            --no-show-dashboard-after

        if ($LASTEXITCODE -ne 0) {
            throw "Run '$runName' failed with exit code $LASTEXITCODE."
        }

        Write-Host ""
        Write-Host "Finished."
    }
}
finally {
    Pop-Location
    [PpoGradPowerRequest]::SetThreadExecutionState($esContinuous) | Out-Null
}
