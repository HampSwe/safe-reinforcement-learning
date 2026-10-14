# SafePPO on the planar drone — findings (exp45–54, seed 1, 2026-09-23/24)

Companion to `experiments.md` (the RL-SGF / PPO / CPO sweeps). This file records what we learned while
tuning **SafePPO** (`../rsl_rl_lib/rsl_rl/algorithms/safe_ppo.py`: PPO's task step pushed through the
RL-SGF closed-form QCQP projection) against the **RL-SGF reference run** `exp16`
(`config/experiment_16_rlsgf_no_obst_term.yaml`, `logs/exp16_rlsgf_no_obst_term_6000it_seed_1`) on the
no-obstacle-termination task with the **cumulative per-step safety cost** (`runner.r1_mode: step`).

All numbers are seed 1, means over 100-iteration blocks read from TensorBoard. SafePPO runs use 100
steps/env per iteration, RL-SGF 400, so **RL-SGF iterations are quoted ×4** to compare at equal env steps.
Run-dir numbers are auto-assigned and lag the config numbers (config `experiment_N` → `logs/exp(N-2)_…`).

---

## 1. The update, and what each knob does

Per iteration, PPO runs its usual 5 epochs × 4 minibatches on the task advantage and yields a raw
parameter step `Δθ_PPO` (Adam, fixed lr). That step is then treated as the task gradient and projected:

```
g0 = -z · Δθ_PPO                         task "gradient" on RL-SGF's scale
g1 = ∇V1  (cost critic, split-half norm) safety gradient
u  = closed-form QCQP multiplier of  min |Δ - h·g0'|²  s.t.  g1·Δ + (L/2)|Δ|² ≤ -α h V1,   L = κ/h
Δθ = task_gain · Δθ_PPO/(1+u)  -  (h/κ) · u/(1+u) · g1,     task_gain = h·z
```

* **z** — rescales `Δθ_PPO` so that `|g0|` is comparable to RL-SGF's REINFORCE gradient. `C ∝ |g0|²`
  in the QCQP, so z decides whether the constraint ever binds. z = 20 (= 1/h_ref) reproduces the old
  `g0 = -Δθ/h` at h = h_ref; z = 200 puts `|g0|` (11–34) on RL-SGF's range (8–75).
* **h (h_ref) + σ schedule** — `h = h_ref (σ_min/σ_ref)^2`, floor `h_min 1e-7`. Scales *both* channels
  (task `h z`, safety `h/κ`); u is h-independent (Lemma 3). With z fixed, h_ref sets the initial
  task gain `h·z`: 1 for (h_ref 0.05, z 20) and (0.005, 200); 10 for (0.05, 200).
* **κ** — curvature of the constraint's quadratic term, `L = κ/h`. The paper has κ = 1. κ = 0.25
  (exp45–51) multiplied the safety step by 4 *and* scaled `g0` inside the solver by 0.25, so the logged
  `V0_grad_norm` was κ·z·|Δθ_PPO| and C stayed > 0 (constraint effectively never bound). κ = 1 fixes both.
* **project_std** — rsl-rl's `ActorCritic` keeps the action std as a separate state-independent
  parameter vector. With `project_std: false` that block received PPO's raw Adam step every iteration,
  outside the QCQP scaling; with `true` it is scaled by `h z/(1+u)` and pushed by `g1` like every other
  parameter (`rl_sgf.py` parity).
* **KL halving (`kl_backoff_steps`)** — TRPO-style backtracking: on constraint-active iterations, halve
  the applied step until the probe KL vs θ_old ≤ `kl_budget` 0.08. **Dropped on Varun's instruction**
  (`kl_backoff_steps: 1`), see §3.
* **max_theta_step** — L2 clip on the whole applied step (std row included), RL-SGF's disaster guard
  (`rl_sgf.py` uses 0.5). Added as the replacement for the halving. Logs `raw_step_norm`, `clipped`.
* **PPO lr / schedule** — rsl-rl's adaptive-KL schedule (`desired_kl 0.01`) is a multiplicative
  ×1.5 / ÷1.5 controller clamped to [1e-5, 1e-2]. See §2.

---

## 2. The adaptive lr schedule is not what the config says

`learning_rate: 1e-4, schedule: adaptive` (exp46) actually runs at **1e-2** — the cap — from iteration
0 (the early KL is far below 0.01, so the controller multiplies up immediately), holds it ~25
iterations, then collapses to the **1e-5 floor by iteration ~60**, and ≥ 90 % of all later iterations sit
at the floor. So exp46's task channel was effectively frozen after the first few hundred iterations: raw
PPO step 0.02, applied step 0.001 after `h z = 0.042`. Its reward-64 plateau (best of the SafePPO runs) is
therefore a *frozen* policy, not a converged one.

Fixed-lr runs with halving still on:

| run | lr | outcome |
|---|---|---|
| exp46 (`logs/exp44_…`) | adaptive (1e-2 → 1e-5) | reward 64 from it 4k, 3 % hits, 0 % oob — frozen at the lr floor |
| exp47 (`logs/exp45_…`) | 1e-4 | reward 56 at 2k, 13 % hits, oob excursion to 82 % at it 1250 |
| exp48 (`logs/exp46_…`) | 5e-5 | peak 61 at 3k, then the oob exploit (45–54 %), avg 44 |
| exp49 (`logs/exp47_…`) | 1e-2 | `|g0|²` so large the constraint is forced active (u 2–6); 2.7 % hits, reward flat 45; the halving acts as an lr controller |

Take-away: with `project_std: false` every fixed lr eventually drives σ to the 0.2 floor, and the run
then falls into the exploit of §4. The adaptive schedule only "worked" by switching the task channel off.

---

## 3. KL halving dropped → `max_theta_step` clip (exp50)

Varun: do not use the KL back-off. Without it we expected wild steps on constraint-active iterations, so
we added RL-SGF's post-QCQP L2 clip (`max_theta_step 0.5`; the halving-guarded runs had never taken a
step above 0.17) and moved the fixed lr to **3e-4** (middle of the range tried).

exp50 (`logs/exp48_…`, κ 0.25, z 20, project_std false): stable, the clip fired on 2 % of early
iterations only, the constraint never became active (u ≈ 0, C > 0), reward 50 at it 1000, hits 3–4 %,
oob recovered to 0.4 %. But σ fell 0.84 → 0.20 in 600 iterations because the std block still received
PPO's unscaled step, and `h ∝ σ²` (and with it the task step) collapsed along with it.

---

## 4. The cumulative-r1 exploit

With `r1_mode: step` the safety return is the discounted sum of per-step costs, so **ending the episode
early lowers the cost**. Policies with a slack constraint (V1 ≈ −4) drift toward leaving the box (oob) or
hitting the floor (crash) while obstacle hits stay ~3 %. Seen in exp47/48 (oob 45–82 %), exp51 (oob 82 %
/ crash 38 % after KL spikes) and exp53 (crash 4 % → 42 %). RL-SGF exp16 goes through the same phase
early (oob 83 % / crash 53 % in the first ~300 equivalent iterations) but grows out of it as the task
reward takes over (oob < 1 %, crash < 0.5 % from ~2k). It appears whenever the task channel becomes too weak or too noisy relative to the
safety channel — it is a symptom, not something to patch in the cost.

---

## 5. project_std, κ, z, h_ref (exp51–54)

| run (config → log dir) | change | result |
|---|---|---|
| **exp51** (`logs/exp49_…`) | `project_std: true` | σ now decays gently (0.98 → 0.55 at it 500), **but** on the 5–10 % of iterations where the constraint was active the raw safety step was 0.4–0.8 (4–8× PPO's own) with KL 9–125 even after the 0.5 clip; each spike was followed by an oob 82 % / crash 38 % excursion. Reward 14–28. |
| **exp52** (`logs/exp50_…`) | κ 0.25 → **1** (paper's L = 1/h) | Fully stable: 911/1238 iterations constraint-active, u 0.02–0.6, C < 0, median KL ≈ 0, clip fired once. Hits **3.0–3.6 % from it 400** (RL-SGF: 22–24 % at the same env-step count), oob 10–19 %. Reward stalls at ~47 from it 300: with h ≈ 0.004 the task step `h z Δθ_PPO` ≈ 0.008 and `|g0|` ≈ 2 vs RL-SGF's 20–25 at the same h. |
| **exp53** (`logs/exp51_…`) | z 20 → **200** | `|g0|` 11–34, u 1.3–6, C −30 … −1600 — RL-SGF-like QCQP geometry. But task_gain started at `h z = 10` (4.4 in the first 100-it block): σ 0.25 by it 1000 (RL-SGF 0.53), per-iteration KL median 0.105 / max 16, crash exploit 4 % → 42 %, reward peak 46 at it 300 (exactly where task_gain crossed 1) then 34. |
| **exp54** (`logs/exp52_…`, **running**, started 2026-09-24 01:15, ~23 h to 100k) | h_ref 0.05 → **0.005** | task_gain starts at 1 (QCQP task step = PPO's own), g1 coefficient `h/κ` starts at 0.005 (RL-SGF's h at it 0 ≈ 0.008). |

RL-SGF exp16 for reference (iterations ×4): reward 56 @ 250 → 72 @ 1200 → 131 @ 24k, hits 23 % → 3.6 %,
σ 0.92 → 0.21, u 0.3 → 3.5, C always negative, `max_theta_step` never fires, `|g0|` 8–75, `|g1|` 2–17.

---

## 6. Current state / what we believe

* **Safety works**: with κ = 1 and project_std the QCQP is well-behaved (C < 0, u in RL-SGF's range,
  no KL spikes) and SafePPO reaches ~3 % hits 6–8× faster (in env steps) than RL-SGF.
* **Task learning is the open problem**: the projected PPO step is either too small (z 20: reward
  stalls at 47 while RL-SGF keeps climbing to 131) or, if rescaled to RL-SGF's gradient magnitude (z 200
  with h_ref 0.05), too large for PPO's Adam step and it collapses σ / triggers the exploit. exp54 tests
  the middle (z 200, task_gain 1 at start).
* A PPO Adam step and a REINFORCE gradient are not the same object: `Δθ_PPO` is already a normalised,
  clipped, multi-epoch step, so `z` is doing the job of converting it back into gradient units. The right
  z is whatever makes `|g0|` ≈ RL-SGF's, but then the *task_gain* `h z` must be ~1 (not 10) or the
  applied step exceeds what PPO's trust region was tuned for. `h_ref = 1/z` keeps both.
* The cumulative cost's early-termination exploit is real and shows up first as oob/crash, not hits;
  watch `Episode/ep_term_oob` / `ep_term_crashed` on every run.
* Iteration-matched comparison against RL-SGF must use env steps (×4), and RL-SGF's numbers in the
  first ~1k SafePPO-iterations are *worse* on hits and *better* on reward — the two methods trade off
  differently early on.

### Standard "full logs" table for future runs
`rew, hit%, oob%, crash%, eplen, V1_start_rows, u, constraint_active, V0_grad_norm, V1_grad_norm,
V1_grad_std_norm, A, C, mean_std, h, task_gain, ppo_step_norm, raw_step_norm, step_norm, clipped,
kl_projected` — 100-it block means, alongside exp16 (×4) and the best previous SafePPO run.

### Code changes in this batch
* `safe_ppo.py`: `z` (task-channel scale), `kappa_fixed`, `project_std`, `max_theta_step` (+ logged
  `raw_step_norm`, `clipped`), bootstrapped cost returns, V1 from episode-start rows, split-half `|∇V1|²`.
  Previous versions kept as `safe_ppo.py.bak_pre_{v1_start,kappa,cost_gae,z,max_theta_step}`.
* `train.py`: `runner.r1_mode` (`step` / `episode`), `r1_walls`, `stagger_initial_episodes`;
  `config_used.json` records `z`, `kappa`, `max_theta_step`, ….
* `environment.py`: `r1_walls` (wall band as part of the unsafe set, exp42/43), `terminate_on_oob`,
  `stagger_episodes()`, per-episode unsafe-entry flags.
* `rl_sgf.py`: separate `gamma1` for the r1 channel (1.0 needed for `r1_mode: episode`).
