# Train PPOGrad with T/16 rollouts, summarize against T/4, T (exp51) and the T baselines,
# then commit + push the summary (plain push, never force).
$ErrorActionPreference = "Continue"
$projectDir = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectDir ".venv\Scripts\python.exe"
$summary = Join-Path $PSScriptRoot "results\ppo_grad_t16_summary.md"

& (Join-Path $PSScriptRoot "run_ppo_grad_t16.ps1")
$trainExit = $LASTEXITCODE
Write-Host "Training script exit code: $trainExit"

$notes = @"
Experiment: RLSGF_PPOGrad with rollout length T/16 (25 steps/update, 64000 iterations, reset_after_update=false,
curriculum pinned to 400 env steps per curriculum-iteration), seed 1, compared with T/4 (exp52), T+reset (exp51)
and the plain RL-SGF T baselines (reset seeds exp15-19, continue seeds exp20-24). All runs use the same total
sample budget (3.28e9 env steps). Training script exit code: $trainExit.
"@

Push-Location $projectDir
& $python (Join-Path $PSScriptRoot "summarize_runs.py") `
    --title "PPOGrad rollout-length sweep: T/16 vs T/4 vs T" `
    --out $summary `
    --notes $notes `
    "exp*_ppo_grad_t16_bootstrap_continue_seed_1" `
    "exp52_ppo_grad_t4_bootstrap_continue_seed_1" `
    "exp51_ppo_grad_t_bootstrap_reset_seed_1" `
    "exp1[5-9]_t_bootstrap_reset_critic_epochs_seed_*" `
    "exp2[0-4]_t_bootstrap_continue_critic_epochs_seed_*"

git add .
git commit -m "Add PPOGrad T/16 rollout experiment (config, launcher, results summary) and run summarizer"
git push origin HEAD
Write-Host "git push exit code: $LASTEXITCODE"
Pop-Location
New-Item -ItemType File -Force -Path (Join-Path $PSScriptRoot "results\.t16_pipeline_done") | Out-Null
