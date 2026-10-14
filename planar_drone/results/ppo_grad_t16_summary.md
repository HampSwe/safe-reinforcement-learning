# PPOGrad rollout-length sweep: T/16 vs T/4 vs T

Generated 2026-08-17 14:14.

Experiment: RLSGF_PPOGrad with rollout length T/16 (25 steps/update, 64000 iterations, reset_after_update=false,
curriculum pinned to 400 env steps per curriculum-iteration), seed 1, compared with T/4 (exp52), T+reset (exp51)
and the plain RL-SGF T baselines (reset seeds exp15-19, continue seeds exp20-24). All runs use the same total
sample budget (3.28e9 env steps). Training script exit code: 0.

Values are means over the final 1.64e+08 env steps of each run (same sample window regardless of rollout length); 'final' is the last logged iteration.

| run | iters | steps/iter | reward (mean / final) | unsafe frac (mean / final) | obstacle rate (mean / final) | ~obstacle hits per 819k steps |
|---|---|---|---|---|---|---|
| exp53_ppo_grad_t16_bootstrap_continue_seed_1 | 63999 | 51200 | 126.4 / 125.9 | 2.68e-03 / 2.29e-03 | 5.14e-04 / 4.69e-04 | 421 |
| exp52_ppo_grad_t4_bootstrap_continue_seed_1 | 16000 | 204800 | 124.3 / 123.2 | 1.01e-03 / 4.59e-04 | 9.17e-05 / 7.32e-05 | 75 |
| exp51_ppo_grad_t_bootstrap_reset_seed_1 | 4000 | 819200 | 140.5 / 145.0 | 2.60e-03 / 2.05e-03 | 3.96e-04 / 3.87e-04 | 324 |
| exp15_t_bootstrap_reset_critic_epochs_seed_1 | 4000 | 819200 | 120.2 / 122.4 | 1.66e-03 / 1.53e-03 | 5.63e-05 / 5.62e-05 | 46 |
| exp16_t_bootstrap_reset_critic_epochs_seed_2 | 4000 | 819200 | 116.8 / 108.8 | 1.77e-03 / 3.08e-03 | 6.16e-05 / 8.06e-05 | 50 |
| exp17_t_bootstrap_reset_critic_epochs_seed_3 | 4000 | 819200 | 120.3 / 116.3 | 1.65e-03 / 1.95e-03 | 5.51e-05 / 5.62e-05 | 45 |
| exp18_t_bootstrap_reset_critic_epochs_seed_4 | 4000 | 819200 | 126.7 / 135.5 | 2.52e-03 / 2.28e-03 | 3.81e-04 / 4.02e-04 | 312 |
| exp19_t_bootstrap_reset_critic_epochs_seed_5 | 4000 | 819200 | 121.5 / 123.2 | 1.60e-03 / 1.29e-03 | 5.91e-05 / 7.08e-05 | 48 |
| exp20_t_bootstrap_continue_critic_epochs_seed_1 | 4000 | 819200 | 111.5 / 122.1 | 2.44e-03 / 2.40e-03 | 3.80e-04 / 3.48e-04 | 311 |
| exp21_t_bootstrap_continue_critic_epochs_seed_2 | 4000 | 819200 | 108.6 / 108.9 | 1.54e-03 / 1.27e-03 | 4.98e-05 / 5.62e-05 | 41 |
| exp22_t_bootstrap_continue_critic_epochs_seed_3 | 4000 | 819200 | 110.8 / 115.8 | 1.48e-03 / 1.40e-03 | 4.63e-05 / 3.30e-05 | 38 |
| exp23_t_bootstrap_continue_critic_epochs_seed_4 | 4000 | 819200 | 107.8 / 96.9 | 2.55e-03 / 3.31e-03 | 3.75e-04 / 3.98e-04 | 307 |
| exp24_t_bootstrap_continue_critic_epochs_seed_5 | 4000 | 819200 | 111.9 / 106.5 | 2.44e-03 / 2.30e-03 | 3.82e-04 / 3.97e-04 | 313 |
