$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class ExperimentBatchPowerRequest
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

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Virtual-environment Python was not found at: $pythonPath"
}

$experiments = @(
    [pscustomobject]@{
        Name = "T + bootstrap + continue episodes + critic epochs"
        RunName = "t_bootstrap_continue_critic_epochs"
        Config = "experiment_01_bootstrap_t_critic_epochs.yaml"
        Iterations = 4000
    },
    [pscustomobject]@{
        Name = "T + bootstrap + reset after update + critic epochs"
        RunName = "t_bootstrap_reset_critic_epochs"
        Config = "experiment_02_bootstrap_t_reset_critic_epochs.yaml"
        Iterations = 4000
    },
    [pscustomobject]@{
        Name = "2*T + bootstrap + reset after update + critic epochs"
        RunName = "2t_bootstrap_reset_critic_epochs"
        Config = "experiment_03_bootstrap_2t_reset_critic_epochs.yaml"
        Iterations = 2000
    },
    [pscustomobject]@{
        Name = "2*T + no bootstrap + reset after update + critic epochs"
        RunName = "2t_no_bootstrap_reset_critic_epochs"
        Config = "experiment_04_legacy_2t_reset_critic_epochs.yaml"
        Iterations = 2000
    }
)

$requestResult = [ExperimentBatchPowerRequest]::SetThreadExecutionState(
    $esContinuous -bor $esSystemRequired
)
if ($requestResult -eq 0) {
    throw "Windows rejected the keep-awake request."
}

Push-Location $projectDir
try {
    for ($index = 0; $index -lt $experiments.Count; $index++) {
        $experiment = $experiments[$index]
        Write-Host ""
        Write-Host "======================================================================="
        Write-Host ("Experiment {0}/{1}: {2}" -f ($index + 1), $experiments.Count, $experiment.Name)
        Write-Host ("Log name: {0}; config: {1}; iterations: {2}" -f $experiment.RunName, $experiment.Config, $experiment.Iterations)
        Write-Host "======================================================================="

        & $pythonPath $trainPath `
            --config $experiment.Config `
            --iterations $experiment.Iterations `
            --device gpu `
            --run-name $experiment.RunName `
            --no-show-dashboard-after

        if ($LASTEXITCODE -ne 0) {
            throw "Experiment '$($experiment.Name)' failed with exit code $LASTEXITCODE."
        }
    }

    Write-Host ""
    Write-Host "All four experiments finished successfully."
}
finally {
    Pop-Location
    [ExperimentBatchPowerRequest]::SetThreadExecutionState($esContinuous) | Out-Null
}
