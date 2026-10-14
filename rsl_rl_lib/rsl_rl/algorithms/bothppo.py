"""BothPPO -- SafePPO with a PPO-like step in BOTH channels (task V0 and safety V1), frozen at the settings of
planar_drone experiment_77 (2026-09-28), the first configuration that beat pure RL-SGF on the bands/disc
environment (reward 152 vs 146, obstacle entry 3.7 % vs 16.7 % at equal env steps, seed 1).

This is a thin subclass of :class:`rsl_rl.algorithms.safe_ppo.SafePPO`: the code is unchanged, only the
defaults are pinned so a config needs nothing beyond ``class_name: rsl_rl.algorithms.bothppo:BothPPO``.
Every keyword can still be overridden from the yaml.

What "both-PPO" means (see safe_ppo.py for the full derivation):

    g0 = -z  * (displacement of a clipped-surrogate PPO loop on the task advantage)        [SafePPO, z set]
    g1 = -z1 * (displacement of a clipped-surrogate PPO loop on the NEGATED cost advantage) [grad_v1_mode "ppo"]
    theta <- theta - h (g0 + u g1) / (1 + u),   u from the Lemma-3 QCQP with V1_hat, alpha

Both loops use the same epochs / minibatches / learning rate / clip / grad-norm; the V1 loop keeps its own
persistent Adam state per env half (the halves give the split-half cross estimate of |g1|^2 used in A).
Because g0 and g1 are then in the same units (z = z1 = 120 x an Adam displacement), dir_scale is 1 and the
QCQP geometry is honest (A ~ 10-40 instead of the ~10^3 a raw score-function |g1|^2 gives).

Pinned choices and why (planar_drone exp72-82, seed 1, 24000 x 100-step iterations, 2048 envs):

  * constant h = 0.1 (``h_schedule False``): the sigma^2 schedule pins h at 0.004 once the std hits its floor
    and the run plateaus (exp74/82); with constant h the policy keeps moving (exp75+).
  * KL halving ON (``kl_backoff_steps 5``, ``kl_budget 0.08``): constant h without a trust region drifted
    (actor weight norms 9 -> 24, exp75); with the halving the projected step's Gaussian KL stays <= 0.08 on
    99.8 % of iterations (exp77). ``max_theta_step 0.5`` is the disaster guard behind it.
  * ``degenerate_cap True``: the A <= eps branch is scaled to |g0|/|g1| so a noisy cross estimate cannot
    kick the parameters 5-8x (exp72 vs exp74).
  * ``v1_estimate "start"`` + ``bootstrap_cost_returns True``: V1_hat = mean over episode-start rows of the
    in-window cost-to-go plus the cost critic's tail (an n-step TD estimate of V1(s_0), NOT a complete-episode
    Monte-Carlo return like rl_sgf.py's -- see the note in SafePPO).
  * ``project_std True``, ``kappa_fixed 1.0`` (L = 1/h as in the paper), ``alpha 1.0``, ``gamma1 0.99``.

Action-std floor: this is an ACTOR setting (``actor.distribution_cfg.std_range``), not an algorithm
argument, so it cannot be pinned here; :data:`STD_RANGE` and :func:`recommended_config` record the default.
The default floor is 0.2 (exp77). A LOWER floor may well be better: 0.15 gave the same reward with ~40 %
fewer obstacle entries (exp78: 2.3 % vs 3.7 %), but 0.10 went back up to ~10 % at higher reward (exp81) --
the floor changes how precisely the policy exploits the cost budget, not the budget itself, so re-check
``Episode/ep_hit_obstacle`` if you move it. If the constructed actor's floor differs from STD_RANGE[0] a
note is printed at start-up so the run's setting is visible in the console log.
"""

from __future__ import annotations

from typing import Any

from rsl_rl.algorithms.safe_ppo import SafePPO
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage

STD_RANGE: tuple[float, float] = (0.2, 2.0)
"""Default action-std clamp for the actor (exp77). See the module docstring: a lower floor (0.15) may be better."""

# Algorithm keyword defaults = experiment_77's ``runner.algorithm`` block minus ``class_name``.
DEFAULTS: dict[str, Any] = {
    # --- PPO task channel (stock PPO kwargs) ---
    "num_learning_epochs": 5,
    "num_mini_batches": 4,
    "clip_param": 0.2,
    "gamma": 0.99,
    "lam": 0.95,
    "value_loss_coef": 1.0,
    "entropy_coef": 0.005,
    "learning_rate": 3.0e-4,
    "max_grad_norm": 1.0,
    "use_clipped_value_loss": True,
    "schedule": "fixed",
    "desired_kl": 0.01,
    "normalize_advantage_per_mini_batch": False,
    # --- safe gradient flow (Lemma-3 QCQP) ---
    "h": 0.1,
    "z": 120.0,
    "alpha": 1.0,
    "eps": 1.0e-8,
    "gamma1": 0.99,
    "h_schedule": False,   # constant h (fixed step); True = h_ref (sigma_min/sigma_ref)^h_exp
    "h_exp": 2.0,
    "h_min": 1.0e-7,
    # --- safety channel ---
    "r1_scale": 1.0,
    "dir_scale": 1.0,
    "grad_v1_divide_by_T": True,
    "grad_v1_denom": "episode",
    "v1_estimate": "start",
    "split_half_norm": True,
    "degenerate_cap": True,
    "grad_v1_mode": "ppo",   # the "both" in BothPPO
    "z1": 120.0,
    "kappa_calibrate": False,
    "kappa_fixed": 1.0,
    "kappa_warmup": 300,
    "kappa_cal_skip": 100,
    "kappa_ema": 0.995,
    "kappa_min": 0.01,
    "kappa_max": 100.0,
    "bootstrap_cost_returns": True,
    "project_std": True,
    # --- cost critic (V1 baseline) ---
    "cost_critic_lr": 3.0e-4,
    "cost_critic_epochs": 4,
    "cost_critic_mini_batches": 4,
    "cost_critic_max_grad_norm": 3.0,
    # --- trust region on the projected step ---
    "kl_budget": 0.08,
    "kl_backoff_steps": 5,   # KL halving ON (1 = off)
    "max_theta_step": 0.5,
    "memory_chunk_size": 50,
}


class BothPPO(SafePPO):
    """SafePPO with PPO-like steps in both channels, constant h, KL halving on (experiment_77 defaults)."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        cost_critic: MLPModel,
        **kwargs: Any,
    ) -> None:
        merged = {**DEFAULTS, **kwargs}
        super().__init__(actor, critic, storage, cost_critic, **merged)
        self._report_std_floor(actor)

    @staticmethod
    def _report_std_floor(actor: MLPModel) -> None:
        dist = getattr(actor, "distribution", None)
        rng = getattr(dist, "std_range", None)
        if rng is None:
            log_rng = getattr(dist, "log_std_range", None)
            if log_rng is not None:
                import math

                rng = (math.exp(float(log_rng[0])), math.exp(float(log_rng[1])))
        if rng is None:
            return
        lo = float(rng[0])
        if abs(lo - STD_RANGE[0]) > 1e-9:
            print(f"BothPPO: actor std floor is {lo:g} (default {STD_RANGE[0]:g}; 0.15 was safer than 0.2 in exp78).")
        else:
            print(f"BothPPO: actor std floor {lo:g} (default; a lower floor, e.g. 0.15, may be better -- see bothppo.py).")


def recommended_config(num_steps_per_env: int = 100, std_range: tuple[float, float] = STD_RANGE) -> dict[str, Any]:
    """The experiment_77 ``runner`` block (planar_drone train.py layout) for BothPPO, as a dict.

    Environment keys (wall/floor bands, obstacle cost, terminations) are the planar bands/disc task and belong
    to planar_drone's runner, not the algorithm; they are included so the reference setting is reproducible.
    """
    return {
        "num_steps_per_env": num_steps_per_env,
        "curriculum_steps_per_env": 400,
        "subtract_r1_from_r0": False,
        "terminate_on_obstacle": False,
        "terminate_on_oob": True,
        "terminate_on_crash": True,
        "r1_walls": True,
        "wall_margin": 1.0,
        "unsafe_cost": 3.0,
        "band_unsafe_cost": 3.0,
        "r1_floor": True,
        "floor_margin": 1.0,
        "r1_mode": "step",
        "obs_groups": {"actor": ["policy"], "critic": ["policy", "privileged"]},
        "save_interval": 500,
        "algorithm": {"class_name": "rsl_rl.algorithms.bothppo:BothPPO", **DEFAULTS},
        "actor": {
            "class_name": "MLPModel",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "std_type": "log",
                "init_std": 1.0,
                # Floor 0.2 = exp77. 0.15 gave ~40 % fewer obstacle entries at equal reward (exp78); consider it.
                "std_range": list(std_range),
            },
        },
        "critic": {"class_name": "MLPModel"},
    }


__all__ = ["BothPPO", "DEFAULTS", "STD_RANGE", "recommended_config"]
