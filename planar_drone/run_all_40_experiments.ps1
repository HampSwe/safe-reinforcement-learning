$ErrorActionPreference = "Stop"

Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;

public static class FullMatrixPowerRequest
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

if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Virtual-environment Python was not found at: $pythonPath"
}

# Full 2x2x2 factorial: rollout length (T / 2T) x bootstrap final state x reset after
# update. Iteration counts hold total sampled environment steps constant (1,600,000
# steps/env): T runs use 4000 iterations at 400 steps/update, 2T runs use 2000
# iterations at 800 steps/update. Case 1 (T, bootstrap, reset) is run first per
# request; the rest follow the order from SEEDED_EXPERIMENT_HANDOFF.md. Case 6
# (T, no-bootstrap, reset / experiment_05) already has all 5 seeds completed and
# will be skipped by the completion check below.
$experiments = @(
    [pscustomobject]@{ RunName = "t_bootstrap_reset_critic_epochs";         Config = "experiment_02_bootstrap_t_reset_critic_epochs.yaml";           Iterations = 4000 },
    [pscustomobject]@{ RunName = "t_bootstrap_continue_critic_epochs";      Config = "experiment_01_bootstrap_t_critic_epochs.yaml";                 Iterations = 4000 },
    [pscustomobject]@{ RunName = "2t_bootstrap_continue_critic_epochs";     Config = "experiment_06_bootstrap_2t_continue_critic_epochs.yaml";       Iterations = 2000 },
    [pscustomobject]@{ RunName = "2t_bootstrap_reset_critic_epochs";        Config = "experiment_03_bootstrap_2t_reset_critic_epochs.yaml";          Iterations = 2000 },
    [pscustomobject]@{ RunName = "t_no_bootstrap_continue_critic_epochs";   Config = "experiment_07_no_bootstrap_t_continue_critic_epochs.yaml";     Iterations = 4000 },
    [pscustomobject]@{ RunName = "t_no_bootstrap_reset_critic_epochs";      Config = "experiment_05_no_bootstrap_t_reset_critic_epochs.yaml";        Iterations = 4000 },
    [pscustomobject]@{ RunName = "2t_no_bootstrap_continue_critic_epochs";  Config = "experiment_08_no_bootstrap_2t_continue_critic_epochs.yaml";    Iterations = 2000 },
    [pscustomobject]@{ RunName = "2t_no_bootstrap_reset_critic_epochs";     Config = "experiment_04_legacy_2t_reset_critic_epochs.yaml";             Iterations = 2000 }
)

$requestResult = [FullMatrixPowerRequest]::SetThreadExecutionState(
    $esContinuous -bor $esSystemRequired
)
if ($requestResult -eq 0) {
    throw "Windows rejected the keep-awake request."
}

function Test-RunComplete {
    param($RunName, $Seed, $Config, $Iterations)

    $suffix = "_{0}_seed_{1}" -f $RunName, $Seed
    $candidate = Get-ChildItem -LiteralPath $logsDir -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match [regex]::Escape($suffix) + '$' } |
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
            return $used.seed -eq $Seed -and
                $used.config -eq $Config -and
                $used.iterations -eq $Iterations
        } |
        Select-Object -First 1

    return $candidate
}

Push-Location $projectDir
try {
    $totalRuns = $experiments.Count * 5
    $runIndex = 0

    foreach ($experiment in $experiments) {
        foreach ($seed in 1..5) {
            $runIndex++

            $completedRun = Test-RunComplete -RunName $experiment.RunName -Seed $seed -Config $experiment.Config -Iterations $experiment.Iterations
            if ($completedRun) {
                Write-Host ("[{0}/{1}] {2} seed {3}: already completed in {4}; skipping." -f $runIndex, $totalRuns, $experiment.RunName, $seed, $completedRun.Name)
                continue
            }

            Write-Host ""
            Write-Host "======================================================================="
            Write-Host ("[{0}/{1}] {2}: seed {3}/5" -f $runIndex, $totalRuns, $experiment.RunName, $seed)
            Write-Host ("Config: {0}; iterations: {1}" -f $experiment.Config, $experiment.Iterations)
            Write-Host "======================================================================="

            & $pythonPath $trainPath `
                --config $experiment.Config `
                --iterations $experiment.Iterations `
                --device gpu `
                --run-name ("{0}_seed_{1}" -f $experiment.RunName, $seed) `
                --seed $seed `
                --no-show-dashboard-after

            if ($LASTEXITCODE -ne 0) {
                throw "Experiment '$($experiment.RunName)', seed $seed failed with exit code $LASTEXITCODE."
            }
        }
    }

    Write-Host ""
    Write-Host "All 40 experiments (8 cases x 5 seeds) finished successfully."
}
finally {
    Pop-Location
    [FullMatrixPowerRequest]::SetThreadExecutionState($esContinuous) | Out-Null
}
