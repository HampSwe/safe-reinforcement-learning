# Planar-drone safe-RL experiments

Three algorithms compared on the same 2-D quadrotor obstacle-avoidance task with the same
per-env environment-step budget and the same five seeds (1–5):

| Sweep | Algorithm | Config | Launcher | Run dirs |
|---|---|---|---|---|
| A | **RL-SGF** (safe gradient flow) | `config/experiment_05_no_bootstrap_t_reset_critic_epochs.yaml` | `run_t_no_bootstrap_seeds.sh 7500 1 2 3 4 5` | `logs/exp{1..5}_t_no_bootstrap_reset_critic_epochs_7500it_seed_{1..5}` |
| B | **Pure PPO** (task reward only, no safety term) | `config/experiment_13_pure_ppo_t4_no_r1.yaml` | `run_pure_ppo_seeds.sh 30000 1 2 3 4 5` | `logs/exp{6..10}_pure_ppo_t4_no_r1_30000it_seed_{1..5}` |
| C | **CPO** (Constrained Policy Optimization) | `config/experiment_14_cpo_t4.yaml` | `run_cpo_seeds.sh 30000 1 2 3 4 5` | `logs/exp{11..15}_cpo_t4_30000it_seed_{1..5}` |

All three were run sequentially (one seed at a time, GPU fully owned by that run) on a single
RTX 3070 with the Linux venv `../.venv-linux` (torch 2.7.0+cu128) and the rsl_rl fork in
`../rsl_rl_lib` (v5.4.1 + `rl_sgf.py` + `cpo.py`). The launchers are restart-safe: a seed whose
run dir already has `exported/policy.onnx` is skipped. Console output per seed goes to
`logs/console/<run_name>.log`.

---

## 1. Task (`environment.py`, `VecDrone2D`)

* **World**: 20 × 20 box, x ∈ [−10, 10], y ∈ [0, 20]. A circular obstacle of radius 0.5 sits at
  (0, 10) (+ 0.05 jitter). The *unsafe region* used by the safety signal is 1.3 × that radius.
* **Dynamics**: planar quadrotor state [x, y, θ, vx, vy, ω]; actions [thrust, moment] ∈ [−1, 1]²
  mapped to F = hover + a₀·hover, M = 2·a₁. Physics dt = 1 ms with decimation 20 → control step
  20 ms. Episode length **T = 400 control steps (8 s)**.
* **Spawn**: drone random in the box, target on the opposite side of the obstacle at distance
  0.5–8 from the centre; both must clear the obstacle. A **curriculum** rotates the target off the
  drone→obstacle→target line: easy (obstacle out of the way) at iteration 0, fully collinear by
  iteration 500 (`RAMP_ITERS = 500`; five discrete stages). The curriculum is measured in
  units of 400 env steps for every algorithm (see §3).
* **Observations**: actor gets 15-D `policy` (target error, sin/cos θ, velocities, previous
  action, obstacle offset, distances to the four walls); the critics additionally get the 2-D
  `privileged` group (raw θ, mass). Same for all three algorithms.
* **Rewards** (both emitted every step):
  * **r0 (task)**: quadratic/linear distance-to-target terms, a Gaussian target bonus, penalties on
    thrust, moment, action rate, ω and speed — all × 10⁻³. No obstacle term.
  * **r1 (safety cost)**: β(e^{−k·d} − 1) with β = 0.05, k = 0.5 when outside the unsafe region
    (d = clearance, so r1 ≤ 0 and → 0 far away), and 1 − β = 0.95 when inside it.
    "V1 ≤ 0" (mean cost ≤ 0) is the safety constraint.
* **Terminations**: hitting the obstacle, hitting the floor (y ≤ 0), leaving the box, or timeout at
  T. `environment.py` logs per-episode rates `Episode/ep_term_obstacle`, `ep_term_crashed`,
  `ep_term_oob`, `ep_term_any`, `ep_timeout` (exact fractions of episodes that ended that way in
  the iteration; with 2048 envs × 400 steps that is ~2050 episodes per iteration).
* **2048 parallel envs** (hard-coded in `train.py:make_env`).

Policy / value networks for all sweeps: MLP 256-256-256, ELU; Gaussian policy with a
log-parameterised std, init 1.0, clamped to [0.2, 2.0].

---

## 2. Algorithms

### A. RL-SGF (`rsl_rl/algorithms/rl_sgf.py`)
Episodic safe-gradient-flow update. Each iteration:
1. Reset every env, roll out exactly one episode (1 × T = 400 steps) under the current policy
   (`reset_after_update: true`, `bootstrap_final_state: false` → V0/V1 are pure Monte-Carlo
   episodic returns, no bootstrapping).
2. Estimate V0 = −E[R0], V1 = E[R1] and their policy gradients (REINFORCE with V0/V1 critics as
   baselines, γ = 0.99).
3. Solve the QCQP for the blend weight u and step
   θ ← θ − h·(∇V0 + u·∇V1)/(1 + u) with α = 5, h_ref = 0.01 (adaptive in the policy std),
   step clipped to ‖Δθ‖ ≤ 0.5. u = 0 when the safety constraint is slack, so the update is then
   plain policy gradient on the task.
4. Fit the two critics with 3 minibatched epochs (minibatch 131072).

### B. Pure PPO (`rsl_rl/algorithms/ppo.py`, stock fork PPO)
Standard PPO on **r0 only** (`subtract_r1_from_r0: false`; the safety cost is logged but never
enters the objective). 100-step continuing rollouts (T/4, the same rollout:episode ratio as the
Isaac hover task's 250/1000), GAE γ = 0.99, λ = 0.95, 5 epochs × 4 minibatches, clip 0.2, entropy
0.01, adaptive LR (desired KL 0.01, init 1e-3), value-loss coef 1.0, grad-norm 1.0.
This is the "what does a pure task learner do" baseline.

### C. CPO (`rsl_rl/algorithms/cpo.py`)
Achiam et al. 2017, ported from the Isaac-side implementation (see the header of the config and
`cpo.py`). Same rollout/GAE/critic setup as PPO (100-step rollouts, γ = λ-pairs 0.99/0.95 for both
reward and cost, two MLP critics fitted 5 epochs × 4 minibatches, lr 1e-3). Each iteration takes
**one** trust-region step: conjugate gradient (15 iters, damping 0.1) on the Fisher matrix
(estimated on every 5th sample, `fvp_subsample: 5`), the CPO dual solve (optim cases 0–4), and a
15-step backtracking line search (decay 0.8) that must satisfy KL ≤ 0.01 and the cost constraint.
Constraint: **mean per-step r1 ≤ `cost_limit` = 0**, i.e. the same V1 ≤ 0 condition RL-SGF
enforces. Objective: r0 only.

---

## 3. Fair budget

| | RL-SGF | PPO | CPO |
|---|---|---|---|
| iterations | 7 500 | 30 000 | 30 000 |
| steps / env / iteration | 400 (1 episode) | 100 | 100 |
| **env steps per env** | **3.0 M** | **3.0 M** | **3.0 M** |
| total env steps (2048 envs) | 6.1 G | 6.1 G | 6.1 G |
| policy updates | 7 500 QCQP steps | 30 000 × (5 epochs × 4 mb) | 30 000 trust-region steps |
| curriculum unit | 400 steps | 400 steps (`curriculum_steps_per_env: 400`) | 400 steps |
| checkpoint interval | 100 it | 500 it | 500 it |
| wall time / seed | ~230 min (1.8 s/it) | ~325 min (0.65 s/it) | ~615 min (1.2 s/it) |
| total wall time | ~19 h | ~27 h | ~51 h |

Everything is matched on environment interaction (3 M steps per env), not on iterations or
gradient steps. `curriculum_steps_per_env: 400` pins the curriculum to "one episode = one
curriculum iteration" so PPO/CPO see the same difficulty schedule as RL-SGF in env-step terms.

### Reproducing
```bash
cd ~/caltech/surf/planar_drone
bash run_t_no_bootstrap_seeds.sh 7500 1 2 3 4 5     # RL-SGF
bash run_pure_ppo_seeds.sh 30000 1 2 3 4 5         # PPO
bash run_cpo_seeds.sh 30000 1 2 3 4 5              # CPO
```
Each launcher exports `CC`/`CXX` from the conda `drone` env because `torch.compile` needs a C
compiler that the venv does not ship. Single run:
```bash
../.venv-linux/bin/python train.py --config <cfg>.yaml --iterations <N> --device gpu \
    --run-name <name> --seed <s> --no-show-dashboard-after
```
Run dirs contain `config_used.json`, TensorBoard events, `model_<it>.pt` and
`exported/policy.{pt,onnx}`.

---

## 4. Results (as of 2026-09-16)

Metrics are means over the final 500 (RL-SGF) / 2000 (PPO, CPO) iterations, i.e. the last
~200k env steps per env in each case. "Obstacle" = fraction of episodes ending in an obstacle
hit; "any" = fraction ending in any termination (obstacle/floor/out-of-bounds).

### A. RL-SGF — 7 500 it, 5/5 seeds done
| seed | reward | obstacle | any |
|---|---|---|---|
| 1 | 126.1 | 1.71 % | 2.16 % |
| 2 | 124.7 | 1.91 % | 2.37 % |
| 3 | 126.7 | 1.76 % | 2.20 % |
| 4 | 135.1 | **14.35 %** | 14.45 % |
| 5 | 128.4 | 1.97 % | 2.36 % |
| mean (1,2,3,5) | 126.5 | 1.84 % | 2.27 % |

Seed 4 is an outlier: it never learned obstacle avoidance (obstacle rate ~18 % → 14 % across the
run) and got the highest task reward by ignoring the obstacle. Not yet diagnosed.

### B. Pure PPO — 30 000 it, 5/5 seeds done
| seed | reward | obstacle | any |
|---|---|---|---|
| 1 | 166.5 | 1.47 % | 1.47 % |
| 2 | 166.4 | 1.47 % | 1.47 % |
| 3 | 166.3 | 1.33 % | 1.52 % |
| 4 | 165.4 | 1.90 % | 1.90 % |
| 5 | 165.6 | 1.67 % | 1.67 % |
| mean ± std | 166.0 ± 0.5 | 1.57 ± 0.20 % | 1.61 ± 0.16 % |

PPO converges within the first ~2 000 iterations (~200k steps/env). Even without any safety term
its collision rate ends slightly *below* RL-SGF's, and its task reward is ~30 % higher.

### C. CPO — 30 000 it, 5/5 seeds done (2026-09-16)
| seed | reward | obstacle | any |
|---|---|---|---|
| 1 | 165.8 | 2.11 % | 2.11 % |
| 2 | 163.2 | 4.63 % | 4.63 % |
| 3 | 162.8 | 4.72 % | 4.72 % |
| 4 | 157.2 | 8.63 % | 8.63 % |
| 5 | 158.4 | 8.15 % | 8.15 % |
| mean ± std | 161.5 ± 3.6 | 5.65 ± 2.72 % | 5.65 ± 2.72 % |

~613 min per seed. **The constraint never became active in any seed**: `Loss/optim_case` = 3
(feasible, constraint slack) and ν = 0 on all 5 × 30 000 iterations, because the mean per-step r1
sits at ≈ −0.037 (the −β(1 − e^{−kd}) shaping dominates; the 0.95 hits are too rare) and never
crosses the limit 0 even at 13 % collisions early on. These runs are therefore effectively TRPO on
r0 — the same caveat noted in the original CPO_DESIGN.md §7 for the Isaac task — and the large
seed-to-seed spread in collision rate (2–9 %) is just how well plain TRPO happens to avoid the
obstacle. Possible follow-ups: `cost_limit ≈ −0.04`, or an indicator cost (1 on collision, limit =
target rate). CPO also converges much more slowly than PPO (13 % collisions still at 5k iterations
vs PPO converged by ~2k).

### D. SafePPO — trained elsewhere, 5/5 seeds
PPO + the RL-SGF closed-form safety projection (`experiment_15_safe_ppo_t4.yaml`, 30 000 × 100
steps, seeds 1–5), trained on another machine and imported as
`logs/external/planar_safeppo_5seeds/` (README with full details inside). Last 2000 it:
reward **171.4 ± 0.6**, obstacle **1.29 ± 0.16 %**. Caveat: its PPO block uses lr 1e-4 /
entropy 0.005 (Isaac SafePPO values) vs 1e-3 / 0.01 in sweep B.

### Learning-curve figures
`plot_seeded_curves.py` → `results/seeded_curves_{reward,collision}.{png,pdf}`: per-seed
smoothing (~75 M env steps) then mean ± 95 % t-CI across seeds, x = cumulative env steps, cut at
6.144 G. RL-SGF seed 4 is excluded by default. Labels: SafePPO = RSGF-RL₁, RL-SGF = RSGF-RL₂.

---

## 5. Diagnostics logged (TensorBoard)

* All algorithms: `Train/mean_reward`, `Train/mean_episode_length`, `Policy/mean_std`,
  `Episode/ep_term_{obstacle,crashed,oob,any}`, `Episode/ep_timeout`, `Episode/ep_ends_per_step`,
  `Episode/frac_steps_unsafe`, `Episode/r1_mean`, `Episode/difficulty` (curriculum),
  `Perf/*`.
* RL-SGF: `Loss/V0`, `Loss/V1`, `Loss/u`, `Loss/A`, `Loss/B`, `Loss/C`, `Loss/delta`, `Loss/h`,
  `Loss/N_eff`, `Loss/{raw_,}step_norm`, `Loss/clipped`, `Loss/critic_v{0,1}_{ev,loss}`.
* PPO: `Loss/surrogate`, `Loss/value`, `Loss/entropy`, `Loss/learning_rate`.
* CPO: `Loss/optim_case`, `Loss/lambda`, `Loss/nu`, `Loss/cost_mean`, `Loss/cost_violation`,
  `Loss/line_search_step`, `Loss/line_search_accepted`, `Loss/kl`, `Loss/step_frac`,
  `Loss/critic_r_ev`, `Loss/critic_c_ev`, `Loss/{value,cost_value}_function`.

Summaries: `summarize_runs.py`, or read the event files directly with
`tensorboard.backend.event_processing.event_accumulator.EventAccumulator`.
