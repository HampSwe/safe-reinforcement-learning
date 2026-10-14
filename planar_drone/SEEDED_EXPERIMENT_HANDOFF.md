# Seeded planar-drone experiment handoff

## Current status

The five runs for **T + no bootstrap + reset after update + 3 critic epochs**
finished successfully on 2026-08-12. They used seeds 1 through 5 and 4,000
learning iterations per seed.

The completed run directories are:

- `logs/exp10_t_no_bootstrap_reset_critic_epochs_seed_1`
- `logs/exp11_t_no_bootstrap_reset_critic_epochs_seed_2`
- `logs/exp12_t_no_bootstrap_reset_critic_epochs_seed_3`
- `logs/exp13_t_no_bootstrap_reset_critic_epochs_seed_4`
- `logs/exp14_t_no_bootstrap_reset_critic_epochs_seed_5`

The combined console logs are:

- `logs/t_no_bootstrap_seeds_20260812_123533.out.log`
- `logs/t_no_bootstrap_seeds_20260812_123533.err.log`

The Windows scheduled task was named `SURF_Missing_T_NoBootstrap_Seeds`. Its
last result was `0`, meaning success. It is currently idle (`Ready`).

## Code and configuration used

The following working-tree changes were made for this experiment and may still
be uncommitted:

- `train.py`: adds `--seed` and seeds Python, NumPy, CPU PyTorch, and all CUDA
  PyTorch generators before the environment and models are constructed.
- `config/experiment_05_no_bootstrap_t_reset_critic_epochs.yaml`: the missing
  T/no-bootstrap/reset configuration.
- `run_t_no_bootstrap_seeds.ps1`: restart-safe sequential runner for seeds 1-5.

The effective configuration for every completed run was:

```yaml
runner:
  num_steps_per_env_multiplier: 1  # T; train.py resolves this to 400 steps
  reset_after_update: true
  save_interval: 100
  algorithm:
    bootstrap_final_state: false
    critic_use_epochs: true
    critic_epochs: 3
    critic_minibatch: 131072
```

Each invocation was equivalent to:

```powershell
& ..\.venv\Scripts\python.exe .\train.py `
  --config experiment_05_no_bootstrap_t_reset_critic_epochs.yaml `
  --iterations 4000 `
  --device gpu `
  --run-name t_no_bootstrap_reset_critic_epochs_seed_1 `
  --seed 1 `
  --no-show-dashboard-after
```

The batch changed both `--seed` and the run-name suffix for seeds 1 through 5.
It ran seeds sequentially on one GPU. Do not run multiple seeds concurrently on
this GPU unless memory use and the scientific consequences have been checked.

The runner prevents Windows sleep with `SetThreadExecutionState`. It also checks
for `config_used.json`, `exported/policy.pt`, and `exported/policy.onnx` before
skipping a completed seed. If interrupted, a completed seed is skipped but an
incomplete seed starts again from iteration zero in a new `expN_*` directory.
There is no checkpoint resume in this batch.

`config_used.json` is the authoritative record for seed and effective settings.
For example, each completed run should contain:

```json
{
  "config": "experiment_05_no_bootstrap_t_reset_critic_epochs.yaml",
  "num_steps_per_env": 400,
  "num_steps_per_env_multiplier": 1,
  "reset_after_update": true,
  "iterations": 4000,
  "seed": 1,
  "critic_epochs": 3,
  "bootstrap_final_state": false
}
```

The seeding controls the RNG initial state, but the code does not enable
`torch.use_deterministic_algorithms(True)`. CUDA kernels may therefore prevent
bit-for-bit reproduction. The seeds are still appropriate for replicated
training comparisons.

## Full eight-case experiment matrix

Yes, there are eight cases if the experiment is the full factorial product of
three binary choices:

1. rollout length: T or 2T;
2. bootstrap the final unfinished rollout state: true or false;
3. reset all environments after every update: true or false.

Keep the critic settings fixed at 3 epochs and minibatch 131,072 throughout.

| Case | Rollout | Bootstrap | Reset after update | Iterations | Config status |
|---:|---:|---:|---:|---:|---|
| 1 | T | true | false | 4000 | Existing `experiment_01_bootstrap_t_critic_epochs.yaml` |
| 2 | T | true | true | 4000 | Existing `experiment_02_bootstrap_t_reset_critic_epochs.yaml` |
| 3 | 2T | true | false | 2000 | Create config |
| 4 | 2T | true | true | 2000 | Existing `experiment_03_bootstrap_2t_reset_critic_epochs.yaml` |
| 5 | T | false | false | 4000 | Create config |
| 6 | T | false | true | 4000 | Existing `experiment_05_no_bootstrap_t_reset_critic_epochs.yaml`; seeds 1-5 done |
| 7 | 2T | false | false | 2000 | Create config |
| 8 | 2T | false | true | 2000 | Existing `experiment_04_legacy_2t_reset_critic_epochs.yaml` |

The different iteration counts deliberately hold the total sampled environment
steps constant:

```text
T:  400 steps/update * 4000 updates = 1,600,000 steps/environment
2T: 800 steps/update * 2000 updates = 1,600,000 steps/environment
```

All runs use 2,048 parallel environments, as currently hard-coded by
`make_env()` in `train.py`.

The three missing configs should copy an existing experiment YAML and change
only these fields:

| Suggested config | Multiplier | Bootstrap | Reset |
|---|---:|---:|---:|
| `experiment_06_bootstrap_2t_continue_critic_epochs.yaml` | 2 | true | false |
| `experiment_07_no_bootstrap_t_continue_critic_epochs.yaml` | 1 | false | false |
| `experiment_08_no_bootstrap_2t_continue_critic_epochs.yaml` | 2 | false | false |

Every config must retain:

```yaml
runner:
  num_steps_per_env: 0  # train.py replaces this
  obs_groups: {"actor": ["policy"], "critic": ["policy", "privileged"]}
  save_interval: 100
  algorithm:
    class_name: rsl_rl.algorithms.rl_sgf:RLSGF
    critic_use_epochs: true
    critic_epochs: 3
    critic_minibatch: 131072
  actor:
    class_name: MLPModel
    distribution_cfg:
      class_name: GaussianDistribution
      std_type: log
      init_std: 1.0
      std_range: [0.2, 2.0]
  critic:
    class_name: MLPModel
```

The no-bootstrap/continue cases are scientifically valid factorial cells, but
their behavior needs careful interpretation. In `rl_sgf.py`, no-bootstrap mode
sets the final value to zero and marks only trajectories that ended inside the
rollout as usable. With `reset_after_update: false`, an episode can cross an
update boundary, so this combination can discard unfinished segments and may
introduce a different sampling bias. That is a reason to measure the cells, not
to silently omit them.

## Running seeds 1-5 for the other seven cases

Use one sequential PowerShell batch. For each case, use the same seeds 1-5 and
include both the condition and seed in `--run-name`. A suitable experiment list
is:

```powershell
$experiments = @(
    [pscustomobject]@{ RunName="t_bootstrap_continue_critic_epochs"; Config="experiment_01_bootstrap_t_critic_epochs.yaml"; Iterations=4000 },
    [pscustomobject]@{ RunName="t_bootstrap_reset_critic_epochs"; Config="experiment_02_bootstrap_t_reset_critic_epochs.yaml"; Iterations=4000 },
    [pscustomobject]@{ RunName="2t_bootstrap_continue_critic_epochs"; Config="experiment_06_bootstrap_2t_continue_critic_epochs.yaml"; Iterations=2000 },
    [pscustomobject]@{ RunName="2t_bootstrap_reset_critic_epochs"; Config="experiment_03_bootstrap_2t_reset_critic_epochs.yaml"; Iterations=2000 },
    [pscustomobject]@{ RunName="t_no_bootstrap_continue_critic_epochs"; Config="experiment_07_no_bootstrap_t_continue_critic_epochs.yaml"; Iterations=4000 },
    # Case 6 is already complete for seeds 1-5.
    [pscustomobject]@{ RunName="2t_no_bootstrap_continue_critic_epochs"; Config="experiment_08_no_bootstrap_2t_continue_critic_epochs.yaml"; Iterations=2000 },
    [pscustomobject]@{ RunName="2t_no_bootstrap_reset_critic_epochs"; Config="experiment_04_legacy_2t_reset_critic_epochs.yaml"; Iterations=2000 }
)

foreach ($experiment in $experiments) {
    foreach ($seed in 1..5) {
        & $pythonPath $trainPath `
            --config $experiment.Config `
            --iterations $experiment.Iterations `
            --device gpu `
            --run-name ("{0}_seed_{1}" -f $experiment.RunName, $seed) `
            --seed $seed `
            --no-show-dashboard-after

        if ($LASTEXITCODE -ne 0) {
            throw "Experiment $($experiment.RunName), seed $seed failed with exit code $LASTEXITCODE."
        }
    }
}
```

Before each invocation, add the same completion check used in
`run_t_no_bootstrap_seeds.ps1`. A run counts as complete only if:

- its suffix exactly matches `_${RunName}_seed_${seed}`;
- `config_used.json` agrees with config, seed, and iteration count;
- both `exported/policy.pt` and `exported/policy.onnx` exist.

This makes scheduler retries safe. Do not treat the presence of an experiment
directory or a final checkpoint alone as completion because export occurs only
after learning finishes.

Before launching the batch:

1. Check there is no active `train.py` process and no running training task.
2. Validate every YAML with `yaml.safe_load`.
3. Validate the PowerShell file with the PowerShell parser.
4. Run `python train.py --help` and confirm `--seed` is present.
5. Confirm the RTX GPU is visible with `nvidia-smi`.
6. Commit or deliberately preserve the current uncommitted seed/config scripts.

For persistence, register the batch in Windows Task Scheduler with these
properties (the completed seed batch is the model to copy):

- run as the current interactive user;
- trigger at logon and start it manually for the first run;
- `ExecutionTimeLimit = PT0S` (no timeout);
- `MultipleInstances = IgnoreNew`;
- allow starting on battery and do not stop when switching to battery;
- wake the computer;
- restart up to 5 times at 1-minute intervals;
- redirect stdout and stderr to timestamped files under `planar_drone/logs`;
- keep the script's `SetThreadExecutionState` call so Windows does not sleep.

A reboot interrupts the current seed. The at-logon trigger starts the batch
again, the completion checks skip finished seeds, and the interrupted seed is
rerun from iteration zero. Closing Codex or a terminal does not affect a
scheduler-owned process.

After starting, verify all of the following rather than relying only on the task
state:

- process command line contains the expected config, seed, and iterations;
- the new run's `config_used.json` contains those exact values;
- its TensorBoard event file grows over several seconds;
- GPU memory/utilization increases;
- stderr has no traceback.

## Analysis protocol

For each individual run, average metrics over its final 50 iterations:

- T/4,000-update runs: iterations 3950-3999;
- 2T/2,000-update runs: iterations 1950-1999.

Important: first compute one tail mean per seed. Then report the mean and sample
standard deviation (`ddof=1`) of those five seed-level means for each condition.
Do not pool all 250 iteration values and present them as 250 independent
observations; iterations within a training run are temporally correlated.

For direct condition comparisons, seeds are paired. For example, compare
bootstrap seed 1 with no-bootstrap seed 1, and report the mean/uncertainty of the
five paired differences. This removes some between-seed noise.

At minimum extract:

- mean reward;
- episode length;
- action standard deviation;
- fraction of unsafe steps;
- obstacle, crash, and out-of-bounds termination fractions;
- critic V0 and V1 explained variance.

The large combined stdout logs contain one block per iteration. TensorBoard
event files in each run directory contain the same scalar history and may be
easier to parse programmatically.

## Runtime estimate

The completed five seeded T/no-bootstrap/reset runs took:

| Seed | Runtime |
|---:|---:|
| 1 | 1.868 h |
| 2 | 1.886 h |
| 3 | 1.863 h |
| 4 | 2.056 h |
| 5 | 1.863 h |

Average: approximately **1.91 hours per run**. Earlier single runs ranged from
about 1.86 to 2.57 hours depending on condition; T/bootstrap/continue was the
slowest observed case.

There are 40 runs in the full design (8 cases x 5 seeds). Five are complete, so
35 remain. On one RTX 5080 with sequential execution:

- optimistic estimate at 1.91 h/run: about **67 hours** remaining;
- estimate allowing for slower conditions and export overhead: **73-85 hours**;
- practical wall-clock budget: approximately **3 to 4 uninterrupted days**.

The full matrix from scratch would take roughly **80-95 hours**. Running cases
concurrently on the same GPU is not recommended: it risks memory pressure,
changes throughput, and weakens comparability.
