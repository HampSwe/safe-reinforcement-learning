"""SafePPO -- PPO with the RL-SGF closed-form safety projection, for the rsl_rl fork (v5.4.x API).

Port (2026-09-14) of ``isaac_drone_racer/algorithms/safe_ppo.py`` (branch ``feat/safe_ppo``, written for
stock rsl-rl-lib 3.0.1) to this fork's ``MLPModel`` actor / critic, ``RolloutStorage`` and
``construct_algorithm`` conventions -- the same port pattern as ``cpo.py``.

It reproduces the code path exercised by the Isaac SafePPO 5-seed sweep (seeds 51-55,
``*_spe250_h0.05_a0.0333_ds1_r1scale30_lr1e-4_ent0.005_boot``). Options that sweep had switched OFF
(h_ratio, h_lr_gain, lr_from_applied_kl, safety_warmup_iters) are not ported; ``grad_v1_mode="ppo"`` (a PPO-like
step in the V1 channel too) was added here 2026-09-28 -- see ``bothppo.py`` for the pinned "both-PPO" defaults.

Each iteration:

  1. Fit the cost critic V1(s) to the discounted cost-to-go G1 (bootstrapped past the window edge), then
     use it as the REINFORCE baseline.                                  [in-sample, as on the Isaac side]
  2. V1_hat = mean(G1);  grad V1 = sum_t,n grad log pi(a|s) * (G1 - b) / (T*N)  [grad_v1_divide_by_T]
  3. Run the stock PPO update and read its actor-parameter displacement as the task gradient,
         g0_eff := -(theta_after_ppo - theta_before) / h          (``z`` None, legacy)
         g0_eff := -z * (theta_after_ppo - theta_before)          (``z`` set)
     With ``z`` the QCQP has the paper's structure: u and the direction do not depend on h, and h scales
     BOTH channels of the applied step  h z dtheta_ppo/(1+u) - (h/kappa) u g1/(1+u)  (also the inactive
     step, h z dtheta_ppo). z = 1/h_ref reproduces the legacy update exactly while h = h_ref; with the
     sigma schedule on, the task step then shrinks with h instead of staying fixed while |g0| blows up.
  4. Solve the RL-SGF QCQP (Lemma 3) with g0_eff, grad V1, V1_hat, alpha. If the constraint is inactive
     the PPO step stands unchanged; if active, the projected step -h(g0 + u g1)/(1+u) replaces it, halved
     up to ``kl_backoff_steps`` times until its Gaussian KL <= ``kl_budget`` (``kl_backoff_steps=1``
     disables the halving). Finally the whole step is L2-clipped to ``max_theta_step`` (None = off), the
     same disaster guard rl_sgf.py applies; ``clipped`` / ``raw_step_norm`` are logged.

With ``project_std=False`` the action-std parameter is excluded from the projection and keeps PPO's own
update (it is also never shrunk by the back-off).

``r1_scale`` multiplies the r1 the safety channel sees, so V1 and grad V1 scale together. The Isaac
sweep set it with the Hydra override ``env.commands.hover_clf.r1_scale=30``; here it is applied to
``extras["r1"]`` inside the algorithm, which is the same operation. ``r1_scale = k, alpha = a/k`` is an
exact reparameterisation of ``dir_scale = k, alpha = a``: every Lemma-3 quantity is identical.

Reference: Pol Mestres, Arnau Marzabal, Jorge Cortes, "Anytime Safe Reinforcement Learning",
https://arxiv.org/abs/2504.16417
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups


# ---------------------------------------------------------------------- #
# Lemma-3 closed form -- verbatim logic of isaac_drone_racer/algorithms/sgf_update.py
# ---------------------------------------------------------------------- #
@dataclass
class SgfUpdateInfo:
    u_hat: float            # Lagrange multiplier (0 = unconstrained, nan = degenerate A <= eps)
    v1_hat: float
    A_hat: float
    C_hat: float
    constraint_active: bool
    grad_v0_norm: float
    grad_v1_norm: float
    degenerate: bool = False


def sgf_closed_form_update(
    grad_v0_flat: torch.Tensor, grad_v1_flat: torch.Tensor, v1_hat: float, alpha: float, h: float, eps: float,
    g1_norm_sq: float | None = None, degenerate_cap: bool = False,
) -> tuple[torch.Tensor, SgfUpdateInfo]:
    """RL-SGF closed-form step  -h (g0 + u g1) / (1 + u).

    This is the KKT solution of   min |xi + g0|^2  s.t.  |xi + g1|^2 <= |g1|^2 - 2 alpha V1,
    i.e. the constraint   g1'xi + 1/2 |xi|^2 <= -alpha V1   (linear model of V1 plus a unit-curvature term).

        A = ||g1||^2 - 2 alpha V1,   B = 2A,   C = 2 g1'g0 - ||g0||^2 - 2 alpha V1,   Delta = 4 ||g1 - g0||^2 A
        A <= eps : degenerate, pure safety step -h g1 (with ``degenerate_cap``: its norm is clipped to that of
                   the u=0 step, h ||g0||; the raw -h g1 has no u/(1+u) damping and came out 5-8x a normal
                   step in exp72, KL 20-110 per hit)
        C >= 0   : constraint inactive, u = 0, step -h g0
        else     : u = (-B + sqrt(Delta)) / (2A), clipped at 0

    ``g1_norm_sq`` optionally replaces ||g1||^2 everywhere it enters (A and ||g1 - g0||^2) by an external
    estimate, e.g. the split-half cross product <g1_a, g1_b>, which is unbiased for the true ||grad V1||^2
    whereas ||g1_hat||^2 = ||grad V1||^2 + (sampling variance) is inflated in high dimension.
    """
    g0, g1 = grad_v0_flat, grad_v1_flat
    if g1_norm_sq is None:
        g1_norm_sq = (g1 * g1).sum().item()
    # an external (cross-product) estimate can come out negative when the true gradient is ~0 -> clamp
    g1_norm_sq = max(g1_norm_sq, 0.0)
    g0_norm_sq = (g0 * g0).sum().item()
    g1_dot_g0 = (g1 * g0).sum().item()

    A_hat = g1_norm_sq - 2.0 * alpha * v1_hat
    B_hat = 2.0 * A_hat
    C_hat = 2.0 * g1_dot_g0 - g0_norm_sq - 2.0 * alpha * v1_hat
    Delta_hat = 4.0 * max(g1_norm_sq - 2.0 * g1_dot_g0 + g0_norm_sq, 0.0) * A_hat

    degenerate = False
    if A_hat <= eps:
        u_hat, constraint_active, degenerate = float("nan"), True, True
        step = -h * g1
        if degenerate_cap:
            g1_raw_norm = math.sqrt((g1 * g1).sum().item())
            if g1_raw_norm > 0.0:
                step = step * min(1.0, math.sqrt(g0_norm_sq) / g1_raw_norm)
    elif C_hat >= 0.0:
        u_hat, constraint_active = 0.0, False
        step = -h * g0
    else:
        constraint_active = True
        u_hat = max((-B_hat + math.sqrt(max(Delta_hat, 0.0))) / (2.0 * A_hat), 0.0)
        step = -h * (g0 + u_hat * g1) / (1.0 + u_hat)

    return step, SgfUpdateInfo(
        u_hat=u_hat, v1_hat=v1_hat, A_hat=A_hat, C_hat=C_hat, constraint_active=constraint_active,
        grad_v0_norm=math.sqrt(g0_norm_sq), grad_v1_norm=math.sqrt(g1_norm_sq), degenerate=degenerate,
    )


class SafePPO(PPO):
    """PPO + the RL-SGF closed-form safety projection (Lemma 3), single cost channel r1."""

    cost_critic: MLPModel

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        cost_critic: MLPModel,
        *,
        # --- safe gradient flow (QCQP) ---
        h: float = 0.05,
        alpha: float = 1.0 / 30.0,
        eps: float = 1e-8,
        gamma1: float = 0.99,
        # --- optional sigma schedule h = h_ref * (sigma_min / sigma_ref) ** h_exp ---
        h_schedule: bool = False,
        h_exp: float = 2.0,
        h_min: float = 1e-7,
        z: float | None = None,
        # --- safety channel ---
        r1_scale: float = 30.0,
        dir_scale: float = 1.0,
        grad_v1_divide_by_T: bool = True,
        grad_v1_denom: str = "legacy",
        v1_estimate: str = "mean",
        split_half_norm: bool = False,
        degenerate_cap: bool = False,
        grad_v1_mode: str = "reinforce",
        z1: float | None = None,
        kappa_calibrate: bool = False,
        kappa_fixed: float = 1.0,
        cost_adv: str = "mc",
        cost_lam: float = 0.95,
        kappa_warmup: int = 300,
        kappa_cal_skip: int = 100,
        kappa_ema: float = 0.995,
        kappa_min: float = 1e-2,
        kappa_max: float = 1e2,
        bootstrap_cost_returns: bool = True,
        project_std: bool = False,
        # --- cost critic (the V1 baseline) ---
        cost_critic_lr: float = 3.0e-4,
        cost_critic_epochs: int = 4,
        cost_critic_mini_batches: int = 4,
        cost_critic_max_grad_norm: float = 3.0,
        # --- trust region on the projected step ---
        kl_budget: float = 0.08,
        kl_backoff_steps: int = 5,
        max_theta_step: float | None = None,
        # --- memory ---
        memory_chunk_size: int = 50,
        **ppo_kwargs,
    ) -> None:
        if ppo_kwargs.get("multi_gpu_cfg") is not None:
            raise NotImplementedError("SafePPO is single-GPU only.")
        if ppo_kwargs.get("rnd_cfg") is not None or ppo_kwargs.get("symmetry_cfg") is not None:
            raise NotImplementedError("SafePPO does not support RND or symmetry augmentation.")
        super().__init__(actor, critic, storage, **ppo_kwargs)

        self.cost_critic = cost_critic.to(self.device)
        self._raw_cost_critic = self.cost_critic
        self.cost_critic_optimizer = torch.optim.Adam(self.cost_critic.parameters(), lr=cost_critic_lr)

        self.h_ref = float(h)
        self.alpha = alpha
        self.eps = eps
        self.gamma1 = gamma1
        self.h_schedule = h_schedule
        self.h_exp = h_exp
        self.h_min = h_min
        self.z = None if z is None else float(z)
        self.h_cap = 1.0 / alpha if alpha > 0 else float("inf")
        self.sigma_ref: float | None = None
        self.r1_scale = r1_scale
        self.dir_scale = dir_scale
        self.grad_v1_divide_by_T = grad_v1_divide_by_T
        # 2026-09-21: units of the V1 channel. The QCQP compares |grad V1|^2 with 2 alpha V1, which only means
        # "a step -h grad V1 can restore the required contraction" if V1 and grad V1 are the level and the
        # gradient of the SAME quantity: the expected r1-return of one complete episode, V1(s_0).
        #   v1_estimate  "mean"  : rtg1 averaged over ALL buffer rows (legacy) = mean remaining cost of a
        #                          visited state, not V1(s_0); with the episode r1 (credit at t=1) it is
        #                          structurally > 0 and made every update degenerate (exp32).
        #                "start" : rtg1 averaged over the rows where an episode STARTS (row after a done, or
        #                          row 0 if the env was done at the end of the previous window). Each such
        #                          rtg1 = MC cost inside the window + cost-critic tail = one sample of V1(s_0).
        #   grad_v1_denom "legacy" : sum over all rows / (T*N if grad_v1_divide_by_T else N).
        #                 "episode": sum over all rows / (number of episodes in the window), estimated by the
        #                          number of dones (EMA over windows so early windows without any done are
        #                          not divided by ~0). This is the policy-gradient theorem for V1(s_0):
        #                          E[sum_t grad log pi rtg_t] per episode, i.e. the gradient of the quantity
        #                          "start" estimates -- so h |grad V1|^2 ~ dV1 in the same units as V1.
        if grad_v1_denom not in ("legacy", "episode"):
            raise ValueError(f"grad_v1_denom must be 'legacy' or 'episode', got {grad_v1_denom!r}")
        if v1_estimate not in ("mean", "start"):
            raise ValueError(f"v1_estimate must be 'mean' or 'start', got {v1_estimate!r}")
        self.grad_v1_denom = grad_v1_denom
        self.v1_estimate = v1_estimate
        self._episodes_per_window_ema: float | None = None
        # 2026-09-22: units of the QCQP. Its constraint g1'xi + 1/2|xi|^2 <= -alpha V1 carries an implicit
        # curvature constant of 1 for V1 in parameter space (per unit h). Writing the true model as
        # dV1 ~ g1'dtheta + (L/2)|dtheta|^2 and kappa := L h, the closed form applies to the rescaled problem
        #     closed_form(kappa g0, g1, V1, alpha kappa, h / kappa)
        # (substitute xi = zeta / kappa); the unconstrained step -h g0 = dtheta_PPO is unchanged, and so is the
        # contraction demanded per update, alpha h |V1|.
        #   split_half_norm : estimate |grad V1|^2 by <g1_a, g1_b> from the two env halves (unbiased) instead of
        #                     |g1_hat|^2 (noise-inflated: exp34 read |g1| ~ 10 for a V1 in [-0.05, 0.95]).
        #   kappa_calibrate : regress the residual (dV1_obs - g1'dtheta) on |dtheta|^2 / 2 over an EMA window to
        #                     get L_hat, and use kappa = clamp(L_hat h, kappa_min, kappa_max) after kappa_warmup
        #                     updates (kappa = 1, the legacy assumption, before that).
        self.split_half_norm = split_half_norm
        self.degenerate_cap = degenerate_cap
        # grad_v1_mode "ppo" (2026-09-28): the V1 direction is the displacement of a PPO-style update on the
        # COST advantage (clipped surrogate, same epochs / minibatches / lr / clip as the task channel, its own
        # persistent Adam), g1 := -z1 (theta_after - theta_old), run separately on the two env halves so the
        # split-half cross estimate stays available. Makes g0 and g1 the same kind of object (Adam-preconditioned
        # PPO steps) instead of comparing a PPO step with a raw score-function gradient in the QCQP.
        if grad_v1_mode not in ("reinforce", "ppo"):
            raise ValueError(f"grad_v1_mode must be 'reinforce' or 'ppo', got {grad_v1_mode!r}")
        self.grad_v1_mode = grad_v1_mode
        self.z1 = (self.z if self.z is not None else 1.0) if z1 is None else float(z1)
        self.cost_actor_optimizers: list[torch.optim.Optimizer] = []
        if grad_v1_mode == "ppo":
            self.cost_actor_optimizers = [
                torch.optim.Adam(self._policy_parameters(), lr=self.learning_rate) for _ in range(2)
            ]
        self.kappa_calibrate = kappa_calibrate
        self.kappa_fixed = kappa_fixed   # kappa = L h used when kappa_calibrate is False (1.0 = legacy closed form)
        # 2026-09-22: weight of grad log pi in the grad V1 estimator.
        #   "mc"  (legacy): G1_t - V1_hat(s_t), the bootstrapped Monte-Carlo cost-to-go minus the critic baseline;
        #                   unbiased given the critic tail, but carries the full return variance.
        #   "gae" : GAE(gamma1, cost_lam) on the cost channel, the same estimator PPO uses on the task side,
        #           built from one-step TD residuals r1_t + gamma1 V1_hat(s_t+1) - V1_hat(s_t). cost_lam = 1
        #           recovers "mc"; lower lambda trades critic bias for variance. The V1 LEVEL is untouched (still
        #           the bootstrapped MC return on episode-start rows), only the gradient estimator changes.
        if cost_adv not in ("mc", "gae"):
            raise ValueError(f"cost_adv must be 'mc' or 'gae', got {cost_adv!r}")
        self.cost_adv = cost_adv
        self.cost_lam = cost_lam
        self._last_obs: TensorDict | None = None
        self.kappa_warmup = kappa_warmup
        self.kappa_cal_skip = kappa_cal_skip   # updates skipped before feeding the regression (cost-critic warm-up)
        self._update_count = 0
        self.kappa_ema = kappa_ema
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self._kappa = 1.0
        self._cal_sry = 0.0    # EMA of residual * |dtheta|^2
        self._cal_syy = 0.0    # EMA of |dtheta|^4
        self._cal_n = 0
        self._prev_v1: float | None = None
        self._prev_lin = 0.0
        self._prev_dtheta_sq = 0.0
        self.last_g1_a: torch.Tensor | None = None
        self.last_g1_b: torch.Tensor | None = None
        self.bootstrap_cost_returns = bootstrap_cost_returns
        self.project_std = project_std
        self.cost_critic_epochs = cost_critic_epochs
        self.cost_critic_mini_batches = cost_critic_mini_batches
        self.cost_critic_max_grad_norm = cost_critic_max_grad_norm
        self.kl_budget = kl_budget
        self.kl_backoff_steps = kl_backoff_steps
        self.max_theta_step = None if max_theta_step is None else float(max_theta_step)
        self.memory_chunk_size = memory_chunk_size

        # Preallocated (T, N, 1): compute_returns runs under the runner's inference_mode, and tensors
        # allocated there could not be used in update()'s autograd graph.
        T, N = storage.num_transitions_per_env, storage.num_envs
        self.r1 = torch.zeros(T, N, 1, device=self.device)
        self.rtg1 = torch.zeros(T, N, 1, device=self.device)
        # Row 0 of a window starts an episode iff the env was done on the last row of the previous window;
        # every env is freshly reset before the first window.
        self._prev_last_done = torch.ones(N, 1, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ #
    # parameter helpers (actor only; the PPO critic keeps its own update)
    # ------------------------------------------------------------------ #
    def _std_param(self) -> nn.Parameter:
        dist = self._raw_actor.distribution
        return dist.log_std_param if dist.std_type == "log" else dist.std_param

    def _policy_parameters(self) -> list[nn.Parameter]:
        """Actor MLP parameters with the action-std parameter LAST, so it is the tail of the flat vector."""
        std = self._std_param()
        rest = [p for p in self._raw_actor.parameters() if p.requires_grad and p is not std]
        return rest + [std]

    @staticmethod
    def _flat(params: list[nn.Parameter]) -> torch.Tensor:
        return torch.cat([p.data.reshape(-1) for p in params])

    @staticmethod
    def _set_flat(params: list[nn.Parameter], flat: torch.Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.data.copy_(flat[offset : offset + n].view_as(p))
                offset += n

    def _sigma_min(self) -> float:
        dist = self._raw_actor.distribution
        with torch.no_grad():
            if dist.std_type == "log":
                return dist.log_std_param.clamp(*dist.log_std_range).exp().min().item()
            return dist.std_param.clamp(*dist.std_range).min().item()

    def _current_h(self) -> float:
        if not self.h_schedule:
            return self.h_ref
        sigma_min = self._sigma_min()
        if self.sigma_ref is None:
            self.sigma_ref = sigma_min
        h = self.h_ref * (sigma_min / self.sigma_ref) ** self.h_exp
        return float(max(min(h, self.h_ref, self.h_cap), self.h_min))

    def _probe_distribution(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.actor(obs, stochastic_output=True)
            return self.actor.distribution.mean.clone(), self.actor.distribution.std.clone()

    @staticmethod
    def _gaussian_kl(mean0, std0, mean1, std1) -> float:
        kl = torch.log(std1 / std0) + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(dim=-1).mean().item()

    # ------------------------------------------------------------------ #
    # rollout hooks
    # ------------------------------------------------------------------ #
    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        step = self.storage.step  # captured before super() advances it
        if "r1" in extras:
            self.r1[step] = (self.r1_scale * extras["r1"].to(self.device)).reshape(-1, 1)
        else:
            self.r1[step] = 0.0
        self.cost_critic.update_normalization(obs)
        super().process_env_step(obs, rewards, dones, extras)

    def compute_returns(self, obs: TensorDict) -> None:
        """PPO's GAE returns, plus the discounted cost-to-go G1 (Monte Carlo within the window; bootstrapped
        from the cost critic at the final observation when ``bootstrap_cost_returns``)."""
        super().compute_returns(obs)
        st = self.storage
        self._last_obs = obs   # final observation of the window, for the GAE bootstrap value in update()
        if self.bootstrap_cost_returns:
            prev = self.cost_critic(obs).detach().reshape(st.num_envs, 1)
        else:
            prev = torch.zeros(st.num_envs, 1, device=self.device)
        for t in reversed(range(st.num_transitions_per_env)):
            not_done = 1.0 - st.dones[t].float().reshape(st.num_envs, 1)
            self.rtg1[t].copy_(self.r1[t] + self.gamma1 * prev * not_done)
            prev = self.rtg1[t]

    # ------------------------------------------------------------------ #
    # the V1 channel
    # ------------------------------------------------------------------ #
    def _episode_start_mask(self) -> torch.Tensor:
        """(T, N, 1) bool: True on the rows where an episode starts (see v1_estimate 'start')."""
        st = self.storage
        dones = st.dones.reshape(st.num_transitions_per_env, st.num_envs, 1).bool()
        start = torch.zeros_like(dones)
        start[0] = self._prev_last_done
        start[1:] = dones[:-1]
        return start

    def _train_cost_critic(self) -> float:
        """Regress the cost critic onto G1 (minibatched epochs)."""
        st = self.storage
        M = st.num_transitions_per_env * st.num_envs
        obs = st.observations.flatten(0, 1)
        targets = self.rtg1.reshape(-1)
        n_mb = max(1, self.cost_critic_mini_batches)
        batch_size = max(1, M // n_mb)
        total, n_updates = 0.0, 0
        for _ in range(self.cost_critic_epochs):
            perm = torch.randperm(M, device=self.device)
            for i in range(n_mb):
                idx = perm[i * batch_size : (i + 1) * batch_size]
                if idx.numel() == 0:
                    continue
                loss = (self.cost_critic(obs[idx]).squeeze(-1) - targets[idx]).pow(2).mean()
                self.cost_critic_optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.cost_critic_max_grad_norm)
                self.cost_critic_optimizer.step()
                total += loss.item()
                n_updates += 1
        return total / max(n_updates, 1)

    def _cost_baseline(self) -> torch.Tensor:
        st = self.storage
        T, N = st.num_transitions_per_env, st.num_envs
        out = torch.zeros(T, N, 1, device=self.device)
        with torch.no_grad():
            for s in range(0, T, self.memory_chunk_size):
                e = min(s + self.memory_chunk_size, T)
                out[s:e] = self.cost_critic(st.observations[s:e].flatten(0, 1)).reshape(e - s, N, 1)
        return out

    def _cost_advantage(self, baseline: torch.Tensor) -> torch.Tensor:
        """(T, N, 1) weight of grad log pi in the grad V1 estimator: G1 - b ("mc") or GAE on r1 ("gae")."""
        st = self.storage
        T, N = st.num_transitions_per_env, st.num_envs
        if self.cost_adv == "mc":
            return (self.rtg1 - baseline).detach()
        with torch.no_grad():
            if self.bootstrap_cost_returns and self._last_obs is not None:
                next_value = self.cost_critic(self._last_obs).reshape(N, 1)
            else:
                next_value = torch.zeros(N, 1, device=self.device)
            adv = torch.zeros(T, N, 1, device=self.device)
            gae = torch.zeros(N, 1, device=self.device)
            for t in reversed(range(T)):
                not_done = 1.0 - st.dones[t].float().reshape(N, 1)
                delta = self.r1[t] + self.gamma1 * next_value * not_done - baseline[t]
                gae = delta + self.gamma1 * self.cost_lam * not_done * gae
                adv[t] = gae
                next_value = baseline[t]
        return adv

    def _estimate_grad_v1_reinforce(self, params: list[nn.Parameter], baseline: torch.Tensor) -> torch.Tensor:
        """grad V1 = sum_{t,n} grad log pi(a|s) A1_t / denom, chunked in time (A1 per ``cost_adv``)."""
        st = self.storage
        T, N = st.num_transitions_per_env, st.num_envs
        adv = self._cost_advantage(baseline)
        if self.grad_v1_denom == "episode":
            n_ep = st.dones.float().sum().item()
            self._episodes_per_window_ema = (n_ep if self._episodes_per_window_ema is None
                                             else 0.9 * self._episodes_per_window_ema + 0.1 * n_ep)
            denom = max(self._episodes_per_window_ema, 1.0)
        else:
            denom = (T * N) if self.grad_v1_divide_by_T else N
        self.last_grad_v1_denom = float(denom)
        D = sum(p.numel() for p in params)
        half = N // 2
        # g1_a / g1_b: the same estimator on the two env halves (independent rollouts), each normalised by
        # half the episode count so that g1 = (g1_a + g1_b) / 2 is the full-batch estimate.
        g1_a = torch.zeros(D, device=self.device)
        g1_b = torch.zeros(D, device=self.device)
        for s in range(0, T, self.memory_chunk_size):
            e = min(s + self.memory_chunk_size, T)
            self.actor(st.observations[s:e].flatten(0, 1), stochastic_output=True)
            logp = self.actor.get_output_log_prob(st.actions[s:e].flatten(0, 1)).reshape(e - s, N, 1)
            weighted = logp * adv[s:e]
            surr_a = weighted[:, :half].sum() / (0.5 * denom)
            surr_b = weighted[:, half:].sum() / (0.5 * denom)
            grads_a = torch.autograd.grad(surr_a, params, allow_unused=True, retain_graph=True)
            grads_b = torch.autograd.grad(surr_b, params, allow_unused=True)
            g1_a += torch.cat([
                (g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads_a, params)
            ])
            g1_b += torch.cat([
                (g if g is not None else torch.zeros_like(p)).reshape(-1) for g, p in zip(grads_b, params)
            ])
        self.last_g1_a = g1_a.detach() * self.dir_scale
        self.last_g1_b = g1_b.detach() * self.dir_scale
        g1 = 0.5 * (self.last_g1_a + self.last_g1_b)
        self.last_v1_grad_norm_raw = g1.norm().item()
        return g1

    def _cost_ppo_displacement(
        self, params: list[nn.Parameter], theta_old: torch.Tensor, env_sl: slice, adv: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> torch.Tensor:
        """Run the clipped-surrogate PPO loop on the cost advantage ``adv`` (minimising V1) over the envs in
        ``env_sl``, starting from ``theta_old``; return theta_after - theta_old and restore theta_old."""
        st = self.storage
        obs = st.observations[:, env_sl].flatten(0, 1)
        actions = st.actions[:, env_sl].flatten(0, 1)
        old_logp = st.actions_log_prob[:, env_sl].flatten(0, 1).squeeze(-1)
        # PPO maximises E[ratio * A]; the cost channel minimises E[ratio * A1], i.e. PPO with A := -A1
        a = -adv[:, env_sl].flatten(0, 1).squeeze(-1)
        M = obs.batch_size[0]
        mb = max(1, M // self.num_mini_batches)
        for p in optimizer.param_groups:
            p["lr"] = self.learning_rate
        for _ in range(self.num_learning_epochs):
            perm = torch.randperm(self.num_mini_batches * mb, device=self.device)
            for i in range(self.num_mini_batches):
                idx = perm[i * mb : (i + 1) * mb]
                self.actor(obs[idx], stochastic_output=True)
                logp = self.actor.get_output_log_prob(actions[idx]).squeeze(-1)
                ratio = torch.exp(logp - old_logp[idx])
                surr = -a[idx] * ratio
                surr_clipped = -a[idx] * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                loss = torch.max(surr, surr_clipped).mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                optimizer.step()
        delta = self._flat(params) - theta_old
        self._set_flat(params, theta_old)
        return delta

    def _estimate_grad_v1_ppo(
        self, params: list[nn.Parameter], baseline: torch.Tensor, theta_old: torch.Tensor
    ) -> torch.Tensor:
        """grad_v1_mode 'ppo': g1_a / g1_b = -z1 x (cost-PPO displacement) on the two env halves (each half
        gets its own Adam state), g1 = (g1_a + g1_b) / 2. Same total compute as one full-batch PPO pass."""
        st = self.storage
        N = st.num_envs
        half = N // 2
        adv = self._cost_advantage(baseline)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)   # as PPO normalises its advantages
        d_a = self._cost_ppo_displacement(params, theta_old, slice(0, half), adv, self.cost_actor_optimizers[0])
        d_b = self._cost_ppo_displacement(params, theta_old, slice(half, N), adv, self.cost_actor_optimizers[1])
        self.last_grad_v1_denom = float("nan")
        self.last_g1_a = (-self.z1 * d_a).detach() * self.dir_scale
        self.last_g1_b = (-self.z1 * d_b).detach() * self.dir_scale
        g1 = 0.5 * (self.last_g1_a + self.last_g1_b)
        self.last_v1_grad_norm_raw = g1.norm().item()
        return g1

    # ------------------------------------------------------------------ #
    # update
    # ------------------------------------------------------------------ #
    def update(self) -> dict[str, float]:  # noqa: C901
        st = self.storage
        params = self._policy_parameters()
        theta_old = self._flat(params).clone()

        # --- V1 channel at theta_old (everything that needs the buffer, before PPO clears it) ---
        cost_critic_loss = self._train_cost_critic()
        baseline = self._cost_baseline()
        v1_mean = self.rtg1.mean().item()
        start_mask = self._episode_start_mask()
        n_start = int(start_mask.sum().item())
        v1_start = self.rtg1[start_mask].mean().item() if n_start > 0 else float("nan")
        if self.v1_estimate == "start":
            if n_start == 0:
                raise RuntimeError("v1_estimate 'start': no episode started in this window; cannot estimate V1")
            v1_hat = v1_start
        else:
            v1_hat = v1_mean
        self._prev_last_done = st.dones[-1].reshape(-1, 1).bool().clone()

        # --- curvature calibration from the previous update: dV1_obs vs its linear prediction ---
        dv1_obs, cal_res, l_hat = float("nan"), float("nan"), float("nan")
        if self._prev_v1 is not None and math.isfinite(v1_hat) and math.isfinite(self._prev_v1):
            dv1_obs = v1_hat - self._prev_v1
            cal_res = dv1_obs - self._prev_lin
            if self._update_count > self.kappa_cal_skip:
                d = self.kappa_ema
                self._cal_sry = d * self._cal_sry + (1.0 - d) * cal_res * self._prev_dtheta_sq
                self._cal_syy = d * self._cal_syy + (1.0 - d) * self._prev_dtheta_sq ** 2
                self._cal_n += 1
            if self._cal_syy > 0.0:
                l_hat = 2.0 * self._cal_sry / self._cal_syy
        self._update_count += 1
        with torch.no_grad():
            g = self.rtg1.reshape(-1)
            cost_critic_ev = (1.0 - (g - baseline.reshape(-1)).var() / (g.var() + 1e-8)).item()
            r1_flat = self.r1.reshape(-1)
            r1_stats = {
                "r1_mean": r1_flat.mean().item(),
                "r1_max": r1_flat.max().item(),
                "r1_unsafe_frac": (r1_flat > 0.0).float().mean().item(),
            }
        if self.grad_v1_mode == "ppo":
            grad_v1 = self._estimate_grad_v1_ppo(params, baseline, theta_old)
        else:
            grad_v1 = self._estimate_grad_v1_reinforce(params, baseline)

        # Reference distribution for the KL back-off: pi_old on the first rollout step's observations.
        obs_probe = st.observations[0].clone()
        old_mean, old_std = self._probe_distribution(obs_probe)

        # --- task channel: the stock PPO update (also updates the PPO critic, and clears storage) ---
        loss_dict = super().update()
        delta_ppo = self._flat(params) - theta_old
        ppo_step_norm = delta_ppo.norm().item()

        # --- keep the action-std parameter out of the projection (project_std=False) ---
        n_std = self._std_param().numel()
        D = delta_ppo.numel()
        std_sl = slice(D - n_std, D)
        v1_grad_std_norm = grad_v1[std_sl].norm().item()
        if self.project_std:
            grad_v1_proj, delta_proj = grad_v1, delta_ppo
        else:
            grad_v1_proj = grad_v1.clone()
            grad_v1_proj[std_sl] = 0.0
            delta_proj = delta_ppo.clone()
            delta_proj[std_sl] = 0.0

        h_now = self._current_h()
        if self.z is None:
            g0_eff = -(delta_proj / h_now)
            task_gain = 1.0                    # legacy: h z == 1, the PPO step is applied as taken
        else:
            g0_eff = -(self.z * delta_proj)
            task_gain = h_now * self.z         # the QCQP's -h g0 = h z dtheta_ppo
        kappa_raw = l_hat * h_now if math.isfinite(l_hat) else float("nan")
        if self.kappa_calibrate and self._cal_n >= self.kappa_warmup and math.isfinite(kappa_raw):
            self._kappa = min(max(kappa_raw, self.kappa_min), self.kappa_max)
        kappa = self._kappa if self.kappa_calibrate else self.kappa_fixed
        h_eff = h_now / kappa
        g1_sq_raw = grad_v1_proj.dot(grad_v1_proj).item()
        # split-half cross estimate <g1_a, g1_b>: always computed for logging; only fed to the QCQP when
        # split_half_norm is on. (2026-09-26: with it on, the clamp-to-0 on disagreeing halves flipped A<0
        # and fired the degenerate -h g1 branch on ~45% of iterations -> 5-8x step kicks; see exp72.)
        g1_sq_est = None
        if self.last_g1_a is not None:
            ga, gb = self.last_g1_a.clone(), self.last_g1_b.clone()
            if not self.project_std:
                ga[std_sl] = 0.0
                gb[std_sl] = 0.0
            g1_sq_est = ga.dot(gb).item()
        g1_sq_qcqp = g1_sq_est if self.split_half_norm else None
        g1_sq_used = g1_sq_raw if g1_sq_qcqp is None else max(g1_sq_qcqp, 0.0)
        step, info = sgf_closed_form_update(
            kappa * g0_eff, grad_v1_proj, v1_hat, self.alpha * kappa, h_eff, self.eps, g1_norm_sq=g1_sq_qcqp,
            degenerate_cap=self.degenerate_cap,
        )

        # --- apply the projection, with a KL trust region on the projected part only ---
        std_part = torch.zeros_like(step)
        proj_part = step.clone()
        proj_ppo = delta_ppo.clone()           # PPO step without the std parameter (== delta_proj)
        if not self.project_std:
            std_part[std_sl] = delta_ppo[std_sl]
            proj_part[std_sl] = 0.0
            proj_ppo[std_sl] = 0.0
        u_val = info.u_hat if math.isfinite(info.u_hat) else 0.0

        scale = 1.0
        if info.constraint_active:
            kl = float("nan")
            trial_scale = scale
            for _ in range(max(1, self.kl_backoff_steps)):
                trial_scale = scale
                self._set_flat(params, theta_old + proj_part * trial_scale + std_part)
                new_mean, new_std = self._probe_distribution(obs_probe)
                kl = self._gaussian_kl(old_mean, old_std, new_mean, new_std)
                if kl <= self.kl_budget or not math.isfinite(kl):
                    break
                scale *= 0.5
            # If every halving failed, the LAST trial is what gets applied (as on the Isaac side); report that.
            scale = trial_scale
            applied = proj_part * scale + std_part
        else:
            # Constraint inactive: the PPO step stands as taken (scaled by h z when z is set; std part untouched).
            applied = task_gain * proj_ppo + std_part

        # --- L2 clip on the total step (RL-SGF's max_theta_step; disaster guard, the KL back-off's replacement
        # when kl_backoff_steps == 1). Applied to the whole step, std row included, as in rl_sgf.py. ---
        raw_step_norm = applied.norm().item()
        clipped = self.max_theta_step is not None and raw_step_norm > self.max_theta_step
        if clipped:
            applied = applied * (self.max_theta_step / raw_step_norm)
            scale *= self.max_theta_step / raw_step_norm
        self._set_flat(params, theta_old + applied)
        step_norm = applied.norm().item()
        new_mean, new_std = self._probe_distribution(obs_probe)
        kl = self._gaussian_kl(old_mean, old_std, new_mean, new_std)

        # --- channel split of the applied step (diagnostics, as in the Isaac sweep) ---
        if info.constraint_active:
            if info.degenerate:
                task_norm, safety_norm = 0.0, h_eff * grad_v1_proj.norm().item() * scale
                b_g1 = -h_eff * scale
            else:
                task_norm = task_gain * delta_proj.norm().item() / (1.0 + u_val) * scale
                safety_norm = h_eff * u_val * grad_v1_proj.norm().item() / (1.0 + u_val) * scale
                b_g1 = -h_eff * u_val / (1.0 + u_val) * scale
        else:
            task_norm, safety_norm = task_gain * delta_proj.norm().item() * scale, 0.0
            b_g1 = 0.0
        # Linear prediction g1'dtheta of this update's dV1, with the noise-inflated self-term b |g1_hat|^2
        # (the applied step contains b * g1_hat) replaced by b <g1_a, g1_b> when available.
        lin_pred = grad_v1.dot(applied).item()
        if g1_sq_est is not None:
            lin_pred -= b_g1 * (g1_sq_raw - g1_sq_est)
        self._prev_v1 = v1_hat
        self._prev_lin = lin_pred
        self._prev_dtheta_sq = applied.dot(applied).item()

        # Leave the actor's distribution set at theta_new (the runner logs get_policy().output_std).
        self._probe_distribution(obs_probe[:1])

        loss_dict.update({
            "V1": v1_hat,
            "V1_mean_rows": v1_mean,
            "V1_start_rows": v1_start,
            "V1_start_n": float(n_start),
            "grad_v1_denom_used": self.last_grad_v1_denom,
            "g1_norm_sq_raw": g1_sq_raw,
            "g1_norm_sq_cross": g1_sq_est if g1_sq_est is not None else float("nan"),
            "g1_norm_sq_used": g1_sq_used,
            "kappa_used": kappa,
            "kappa_raw": kappa_raw,
            "L_hat": l_hat,
            "dV1_obs": dv1_obs,
            "dV1_lin_prev": self._prev_lin if self._prev_v1 is not None else float("nan"),
            "cal_residual": cal_res,
            "dV1_lin_pred": lin_pred,
            "u": info.u_hat,
            "A": info.A_hat,
            "C": info.C_hat,
            "h": h_now,
            "task_gain": task_gain,
            "degenerate": float(info.degenerate),
            "constraint_active": float(info.constraint_active),
            "V0_grad_norm": info.grad_v0_norm,
            "V1_grad_norm": info.grad_v1_norm,
            "V1_grad_norm_raw": self.last_v1_grad_norm_raw,
            "V1_grad_std_norm": v1_grad_std_norm,
            "ppo_step_norm": ppo_step_norm,
            "step_norm": step_norm,
            "raw_step_norm": raw_step_norm,
            "clipped": float(clipped),
            "kl_projected": kl,
            "backoff_scale": scale,
            "g1_share": safety_norm / (task_norm + safety_norm + 1e-12),
            "step_g0_norm": task_norm,
            "step_g1_norm": safety_norm,
            "dV1_pred": torch.dot(grad_v1_proj, applied).item(),
            "dV1_allow": self.alpha * h_now * abs(v1_hat),   # invariant under kappa: (alpha kappa)(h / kappa)
            "sigma_min": self._sigma_min(),
            "cost_critic_loss": cost_critic_loss,
            "cost_critic_ev": cost_critic_ev,
            "cost_baseline": baseline.mean().item(),
            **r1_stats,
        })
        return loss_dict

    # ------------------------------------------------------------------ #
    # runner contract
    # ------------------------------------------------------------------ #
    def train_mode(self) -> None:
        super().train_mode()
        self.cost_critic.train()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.cost_critic.eval()

    def save(self) -> dict:
        saved = super().save()
        saved["cost_critic_state_dict"] = self._raw_cost_critic.state_dict()
        saved["cost_critic_optimizer_state_dict"] = self.cost_critic_optimizer.state_dict()
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        resume = super().load(loaded_dict, load_cfg, strict)
        if (load_cfg is None or load_cfg.get("critic")) and "cost_critic_state_dict" in loaded_dict:
            self._raw_cost_critic.load_state_dict(loaded_dict["cost_critic_state_dict"], strict=strict)
        if (load_cfg is None or load_cfg.get("optimizer")) and "cost_critic_optimizer_state_dict" in loaded_dict:
            self.cost_critic_optimizer.load_state_dict(loaded_dict["cost_critic_optimizer_state_dict"])
        return resume

    def compile(self, mode: str | None = None) -> None:
        super().compile(mode)
        self.cost_critic = compile_model(self._raw_cost_critic, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> SafePPO:
        """Same shape as PPO.construct_algorithm, plus a cost critic with the critic's architecture."""
        alg_class: type[SafePPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        # The runner reads cfg["algorithm"]["rnd_cfg"] for logging; SafePPO has no RND.
        cfg["algorithm"].setdefault("rnd_cfg", None)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        cost_critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Models (reward, cost): {critic}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg: SafePPO = alg_class(
            actor, critic, storage, cost_critic=cost_critic, device=device, **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    def broadcast_parameters(self) -> None:
        raise NotImplementedError("SafePPO is single-GPU only.")

    def reduce_parameters(self) -> None:
        raise NotImplementedError("SafePPO is single-GPU only.")
