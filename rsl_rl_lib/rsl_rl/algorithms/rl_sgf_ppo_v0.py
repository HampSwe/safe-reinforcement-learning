# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, Symmetry, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups, resolve_optimizer


class RLSGF_PPO_V0:
    """RL-SGF with V0's actor gradient replaced by a full PPO update.

    This is a variant of :class:`rsl_rl.algorithms.rl_sgf.RLSGF`. Read that file
    first; only the differences are documented here.

    In the original RL-SGF, both ``V0_grad`` (reward objective) and ``V1_grad``
    (safety objective) are single vanilla policy-gradient (REINFORCE with a
    value baseline) estimates, computed once per update from the current
    rollout, then combined analytically via a closed-form QCQP into one small
    step ``theta - h*(V0_grad + u*V1_grad)/(1+u)``. That closed form assumes
    both inputs are true first-order gradients of V0/V1 at the *same* theta.

    Here, V1_grad is computed exactly as before. V0_grad is instead derived
    from a full PPO update: several clipped-surrogate, importance-sampled,
    minibatched epochs (standard GAE(lambda) advantages, critic_v0 as the
    value function) are run on the *same* rollout, moving the actor from
    theta_start to theta_ppo. The net displacement is converted back into an
    "effective gradient" ``V0_grad = (theta_start - theta_ppo) / h`` so it can
    be substituted into the *unchanged* QCQP formula. With u = 0 (the safety
    constraint inactive) this exactly reproduces the pure-PPO update; with
    u > 0 it blends toward the safety direction using the same combination as
    the original algorithm.

    Caveat (why this is not covered by the paper's guarantee): V0_grad here is
    a secant/finite-difference direction from several off-policy-corrected PPO
    epochs, not a single on-policy gradient at theta_start. The two vectors fed
    into the QCQP are no longer the same kind of object. Treat any resulting
    safety behavior as empirical, not proven.

    This variant is intended to be run with ``num_steps_per_env_multiplier: 1``
    (rollout length T, i.e. one episode length) so that a single PPO phase's
    "trust region" reasoning applies to a batch that never spans more than one
    episode length per environment before the next update.

    Reference:
        - Pol Mestres, Arnau Marzabal, Jorge Cortés "Anytime Safe Reinforcement Learning", https://arxiv.org/abs/2504.16417
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: MLPModel,
        critic_v0: MLPModel,
        critic_v1: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        h: float = 1e-2,
        alpha: float = 5.0,
        max_theta_step: float = 0.5,
        env: VecEnv | None = None,
        critic_use_epochs: bool = False,
        critic_epochs: int = 3,
        critic_minibatch: int = 131072,
        bootstrap_final_state: bool = False,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND / symmetry are accepted for interface parity with PPO/RLSGF but are
        # not wired into the V0 PPO phase below -- fail loudly instead of silently
        # ignoring them.
        if rnd_cfg is not None:
            raise NotImplementedError("RLSGF_PPO_V0 does not support rnd_cfg.")
        if symmetry_cfg is not None:
            raise NotImplementedError("RLSGF_PPO_V0 does not support symmetry_cfg.")
        self.rnd = None
        self.symmetry = None

        # Components
        self.actor = actor.to(self.device)
        self.critic_v0 = critic_v0.to(self.device)
        self.critic_v1 = critic_v1.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critic_v0`` / ``self.critic_v1``.
        self._raw_actor = self.actor
        self._raw_critic_v0 = self.critic_v0
        self._raw_critic_v1 = self.critic_v1

        # V0 phase: standard PPO optimizer over the actor + critic_v0 (mirrors
        # PPO's single joint optimizer over actor + critic).
        self.actor_optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic_v0.parameters()), lr=learning_rate
        )  # type: ignore

        # V1 phase: unchanged from RLSGF -- post-hoc value-fn regression, its own optimizer.
        self.critic_lr = 1e-3
        self.critic_v1_optimizer = torch.optim.Adam(self.critic_v1.parameters(), lr=self.critic_lr)

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # RL-SGF extras (V1 side only -- V0 now goes through storage.returns/advantages)
        self.r1 = torch.zeros(storage.num_transitions_per_env, storage.num_envs, 1, device=self.device)
        self.rtg0 = torch.zeros(storage.num_transitions_per_env, storage.num_envs, 1, device=self.device)
        self.rtg1 = torch.zeros(storage.num_transitions_per_env, storage.num_envs, 1, device=self.device)
        # Per-step timeout flags, needed to bootstrap V0's GAE the same way PPO does
        # (added to the reward at the timeout step since `dones` masks out the next
        # value there otherwise). Captured during process_env_step, since `extras`
        # is not available later in compute_returns/update.
        self.time_outs = torch.zeros(storage.num_transitions_per_env, storage.num_envs, 1, device=self.device)

        # RL-SGF parameters
        self.h = h
        self.alpha = alpha

        # ---- __init__ ----
        self.h_ref = self.h  # the h you already tuned; valid at the starting sigma
        self.sigma_ref = None  # captured lazily on the first update
        self.h_exp = 2.0
        self.h_cap = 1 / self.alpha  # Lemma 1 needs h < 1/alpha
        self.h_min = 1e-7  # optional: don't let it vanish entirely

        self.weight_decay = 0
        # Unlike RLSGF's hand-rolled decay mask (weights only, not biases/log_std),
        # here weight decay -- if enabled -- is applied via the actor optimizer's
        # standard (undifferentiated) decay, since V0_grad is a secant direction,
        # not a plain gradient one can cleanly add a decay term to.

        self.max_theta_step = max_theta_step
        self.memory_chunk_size = 50

        self.env = env
        self.critic_use_epochs = critic_use_epochs
        self.critic_epochs = critic_epochs
        self.critic_minibatch = critic_minibatch
        self.bootstrap_final_state = bootstrap_final_state

        # PPO parameters (drive the V0 phase)
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        if self.weight_decay:
            # Re-create with decay if a subclass/config ever sets it.
            self.actor_optimizer = resolve_optimizer(optimizer)(
                chain(self.actor.parameters(), self.critic_v0.parameters()),
                lr=learning_rate,
                weight_decay=self.weight_decay,
            )  # type: ignore

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic_v0.get_hidden_state())
        # Compute the actions and V0 value estimate (needed for GAE/PPO)
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic_v0(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.actor.update_normalization(obs)
        self.critic_v0.update_normalization(obs)
        self.critic_v1.update_normalization(obs)

        # Record the rewards and dones (unmutated -- rtg0/rtg1 stay comparable to
        # the base RLSGF's definitions; the GAE timeout-bootstrap is applied
        # separately in compute_returns using self.time_outs).
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        self.r1[self.storage.step] = extras["r1"].to(self.device).unsqueeze(-1)
        if "time_outs" in extras:
            self.time_outs[self.storage.step] = extras["time_outs"].to(self.device).unsqueeze(-1).float()

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic_v0.reset(dones)
        self.critic_v1.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute RL-SGF's Monte-Carlo returns (V1's gradient, V0 logging) and
        V0's GAE(lambda) returns/advantages (V0's PPO phase)."""
        st = self.storage

        if self.bootstrap_final_state:
            # The observation passed by the runner is s_{T}. Preserve recurrent
            # state because evaluating the critics must not advance the rollout.
            critic_v0_hidden_state = self.critic_v0.get_hidden_state()
            critic_v1_hidden_state = self.critic_v1.get_hidden_state()
            final_v0 = self.critic_v0(obs).detach()
            prev_r1 = self.critic_v1(obs).detach()
            self.critic_v0.reset(hidden_state=critic_v0_hidden_state)
            self.critic_v1.reset(hidden_state=critic_v1_hidden_state)
            prev_r0 = final_v0
            last_values = final_v0
        else:
            prev_r0 = torch.zeros(st.num_envs, 1, device=self.device)
            prev_r1 = torch.zeros(st.num_envs, 1, device=self.device)
            critic_v0_hidden_state = self.critic_v0.get_hidden_state()
            last_values = self.critic_v0(obs).detach()
            self.critic_v0.reset(hidden_state=critic_v0_hidden_state)

        for step in reversed(range(st.num_transitions_per_env)):
            next_is_not_terminal = 1.0 - st.dones[step].float()
            self.rtg0[step] = st.rewards[step] + self.gamma * prev_r0 * next_is_not_terminal
            self.rtg1[step] = self.r1[step] + self.gamma * prev_r1 * next_is_not_terminal
            prev_r0 = self.rtg0[step]
            prev_r1 = self.rtg1[step]

        self.rtg0 = self.rtg0.detach()
        self.rtg1 = self.rtg1.detach()

        # ---- GAE(lambda) for V0's PPO phase (standard PPO.compute_returns, plus
        # the timeout bootstrap PPO normally folds into the reward at collection
        # time -- done here instead since we don't mutate the shared reward buffer). ----
        T = st.num_transitions_per_env
        advantage = 0
        for step in reversed(range(T)):
            next_values = last_values if step == T - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            step_reward = st.rewards[step] + self.gamma * st.values[step] * self.time_outs[step]
            delta = step_reward + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def _actor_sigma_min(self):
        """Smallest per-dimension std (thrust/moment) -- governs L_0, L_1."""
        dist = self.actor.distribution
        if dist.std_type == "log":
            log_std = dist.log_std_param.clamp(dist.log_std_range[0], dist.log_std_range[1])
            return log_std.exp().min().item()
        return dist.std_param.clamp(dist.std_range[0], dist.std_range[1]).min().item()

    def _current_h(self):
        with torch.no_grad():
            sigma_min = self._actor_sigma_min()
        if self.sigma_ref is None:
            self.sigma_ref = sigma_min  # sigma at iteration 0
        h = self.h_ref * (sigma_min / self.sigma_ref) ** self.h_exp
        h = min(h, self.h_ref, self.h_cap)  # never exceed the tuned value
        h = max(h, self.h_min)
        return float(h), sigma_min

    def _ppo_v0_phase(self) -> dict[str, float]:
        """Run PPO's clipped-surrogate, multi-epoch actor+critic_v0 update for
        the V0 (reward) objective. Mutates self.actor and self.critic_v0 in
        place. Mirrors rsl_rl.algorithms.ppo.PPO.update() (RND/symmetry/
        recurrent paths omitted -- see the constructor's NotImplementedError
        for RND/symmetry, and this project's actor/critic are feedforward)."""
        st = self.storage
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        n_kl_updates = 0

        if self.actor.is_recurrent or self.critic_v0.is_recurrent:
            generator = st.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = st.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (
                        batch.advantages - batch.advantages.mean()
                    ) / (batch.advantages.std() + 1e-8)  # type: ignore

            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic_v0(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.actor_optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

                    mean_kl += kl_mean.item()
                    n_kl_updates += 1

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss (critic_v0)
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            self.actor_optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic_v0.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        return {
            "ppo_v0_value_loss": mean_value_loss / num_updates,
            "ppo_v0_surrogate_loss": mean_surrogate_loss / num_updates,
            "ppo_v0_entropy": mean_entropy / num_updates,
            "ppo_v0_kl": mean_kl / max(n_kl_updates, 1),
            "ppo_v0_lr": self.learning_rate,
        }

    def update(self) -> dict[str, float]:
        st = self.storage
        T, N = st.num_transitions_per_env, st.num_envs

        dones = st.dones.squeeze(-1).bool()  # (T,N): episode ended AT this row
        rows = torch.arange(T, device=self.device).unsqueeze(1).expand(T, N)

        starts = torch.zeros_like(dones)
        starts[0] = True
        starts[1:] = dones[:-1]

        start_row = torch.where(starts, rows, torch.full_like(rows, -1))
        start_row = torch.cummax(start_row, dim=0).values
        tau = rows - start_row

        if self.bootstrap_final_state:
            usable = torch.ones_like(dones)
            n_ep = starts.sum().clamp(min=1).float()
        else:
            ended = torch.flip(torch.cummax(torch.flip(dones, [0]).long(), 0).values, [0]).bool()
            usable = ended
            n_ep = dones.sum().clamp(min=1).float()

        gamma_tau = (self.gamma ** tau.float()).unsqueeze(-1)
        keep = usable.unsqueeze(-1).float()

        rtg0 = self.rtg0.squeeze(-1)
        rtg1 = self.rtg1.squeeze(-1)

        good_start = starts & usable
        self.V0 = -rtg0[good_start].mean()  # logging only -- not used by the QCQP below
        self.V1 = rtg1[good_start].mean()

        obs_all = st.observations
        act_all = st.actions

        with torch.no_grad():
            # critic_v0 predictions BEFORE this iteration's updates, purely for the
            # explained-variance diagnostic (comparable to RLSGF's ev0).
            v0_b = self.critic_v0(obs_all.flatten(0, 1)).reshape(T, N, 1)
            v1_b = self.critic_v1(obs_all.flatten(0, 1)).reshape(T, N, 1)

            g0_flat = self.rtg0.reshape(-1)
            g1_flat = self.rtg1.reshape(-1)
            p0 = v0_b.reshape(-1)
            p1 = v1_b.reshape(-1)
            ev0 = (1.0 - (g0_flat - p0).var() / (g0_flat.var() + 1e-8)).item()
            ev1 = (1.0 - (g1_flat - p1).var() / (g1_flat.var() + 1e-8)).item()

        w1 = self.rtg1 - v1_b

        # ---- V1_grad: unchanged RLSGF vanilla policy gradient, computed at theta_start ----
        d = sum(p.numel() for p in self.actor.parameters())
        g1 = torch.zeros(d, device=self.device)

        chunk = self.memory_chunk_size
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            obs_c = obs_all[start:end].flatten(0, 1)
            act_c = act_all[start:end].flatten(0, 1)

            self.actor(obs_c, stochastic_output=True)
            logp_c = self.actor.get_output_log_prob(act_c).reshape(end - start, N, 1)

            gt_c = gamma_tau[start:end] * keep[start:end]
            J1_c = (gt_c * logp_c * w1[start:end]).sum() / n_ep

            g1 += torch.nn.utils.parameters_to_vector(torch.autograd.grad(J1_c, self.actor.parameters()))

        self.V1_grad = g1

        theta_start = torch.nn.utils.parameters_to_vector(self.actor.parameters()).detach().clone()
        # h characterizes the exploration noise of the policy BEING improved
        # (theta_start), so it must be read before the PPO phase moves the actor.
        self.h, sigma_min = self._current_h()

        # ---- V0_grad: run full PPO on the same rollout, then read off the secant ----
        ppo_stats = self._ppo_v0_phase()
        theta_ppo = torch.nn.utils.parameters_to_vector(self.actor.parameters()).detach().clone()
        self.V0_grad = (theta_start - theta_ppo) / self.h

        # Restore the actor to theta_start: the PPO move was only used to define
        # V0_grad's direction; the actual step taken is the QCQP-combined one below.
        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(theta_start, self.actor.parameters())
        self.theta = theta_start

        # ---- QCQP (unchanged from RLSGF) ----
        self.A = self.V1_grad.norm() ** 2 - 2 * self.alpha * self.V1
        self.B = 2 * self.A
        self.C = 2 * self.V1_grad.dot(self.V0_grad) - self.V0_grad.norm() ** 2 - 2 * self.alpha * self.V1
        self.delta = 4 * ((self.V1_grad - self.V0_grad).norm() ** 2) * self.A

        if self.A > 0:
            if self.C < 0:
                self.u = (-self.B + torch.sqrt(self.delta)) / (2 * self.A)
            else:
                self.u = torch.zeros((), device=self.device)
            self.p_theta = self.theta - self.h * (self.V0_grad + self.u * self.V1_grad) / (1 + self.u)
        else:
            self.u = torch.zeros((), device=self.device)
            self.p_theta = self.theta - self.h * self.V1_grad

        step = self.p_theta - self.theta
        raw_step_norm = step.norm()
        clipped = raw_step_norm > self.max_theta_step
        if clipped:
            step = step * (self.max_theta_step / raw_step_norm)
        self.p_theta = self.theta + step

        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(self.p_theta.detach(), self.actor.parameters())

        # ---- critic_v1 update: unchanged from RLSGF (post-hoc regression against rtg1) ----
        if not self.critic_use_epochs:
            self.critic_v1_optimizer.zero_grad()
            v1_loss_total = 0.0
            for start in range(0, T, chunk):
                end = min(start + chunk, T)
                obs_c = obs_all[start:end].flatten(0, 1)
                v1_c = self.critic_v1(obs_c).reshape(end - start, N, 1)
                frac = (end - start) / T
                v1_loss_c = ((v1_c - self.rtg1[start:end]) ** 2).mean() * frac
                v1_loss_c.backward()
                v1_loss_total += v1_loss_c.item()
            self.critic_v1_optimizer.step()
        else:
            obs_flat = obs_all.reshape(T * N, *obs_all.shape[2:])
            rtg1_flat = self.rtg1.reshape(T * N, 1)
            n_samples = T * N
            v1_loss_total = 0.0
            n_batches = 0
            for _ in range(self.critic_epochs):
                perm = torch.randperm(n_samples, device=self.device)
                for mb in range(0, n_samples, self.critic_minibatch):
                    idx = perm[mb : mb + self.critic_minibatch]
                    self.critic_v1_optimizer.zero_grad()
                    v1_c = self.critic_v1(obs_flat[idx])
                    v1_loss_c = ((v1_c - rtg1_flat[idx]) ** 2).mean()
                    v1_loss_c.backward()
                    self.critic_v1_optimizer.step()
                    v1_loss_total += v1_loss_c.item()
                    n_batches += 1
            v1_loss_total /= max(n_batches, 1)

        # ---- reset storage AND per-iteration buffers so the next rollout starts fresh ----
        self.storage.clear()
        self.r1.zero_()
        self.time_outs.zero_()

        result = {
            "V0": self.V0.item(), "V1": self.V1.item(),
            "A": self.A.item(), "B": self.B.item(), "C": self.C.item(),
            "delta": self.delta.item(),
            "u": self.u.item() if torch.is_tensor(self.u) else float(self.u),
            "V0_grad_norm": self.V0_grad.norm().item(),
            "V1_grad_norm": self.V1_grad.norm().item(),
            "critic_v0_loss": ppo_stats["ppo_v0_value_loss"],
            "critic_v1_loss": v1_loss_total,
            "clipped": float(clipped),
            "raw_step_norm": raw_step_norm.item(),
            "step_norm": (self.p_theta - self.theta).norm().item(),
            "h": self.h,
            "theta_norm": self.theta.norm().item(),
            "N_eff": n_ep.item(),
            "critic_v0_ev": ev0,
            "critic_v1_ev": ev1,
        }
        result.update(ppo_stats)
        return result

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic_v0.train()
        self.critic_v1.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic_v0.eval()
        self.critic_v1.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_v0_state_dict": self._raw_critic_v0.state_dict(),
            "critic_v1_state_dict": self._raw_critic_v1.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_v1_optimizer_state_dict": self.critic_v1_optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}

        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic_v0.load_state_dict(loaded_dict["critic_v0_state_dict"], strict=strict)
            self._raw_critic_v1.load_state_dict(loaded_dict["critic_v1_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.actor_optimizer.load_state_dict(loaded_dict["actor_optimizer_state_dict"])
            self.critic_v1_optimizer.load_state_dict(loaded_dict["critic_v1_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critics with ``torch.compile``."""
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic_v0 = compile_model(self._raw_critic_v0, mode)
        self.critic_v1 = compile_model(self._raw_critic_v1, mode)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> RLSGF_PPO_V0:
        """Construct the algorithm."""
        alg_class: type[RLSGF_PPO_V0] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        default_sets = ["actor", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Not supported by this algorithm (see __init__), but resolve_*_config still
        # needs to run so it stamps a `None` default into cfg["algorithm"] -- Logger
        # unconditionally reads cfg["algorithm"]["rnd_cfg"] / ["symmetry_cfg"].
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic_v0: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        critic_v1: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg: RLSGF_PPO_V0 = alg_class(
            actor, critic_v0, critic_v1, storage, device=device, env=env, **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        model_params = [self._raw_actor.state_dict(), self._raw_critic_v0.state_dict(), self._raw_critic_v1.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic_v0.load_state_dict(model_params[1])
        self._raw_critic_v1.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them (best-effort; untested in this project, which is single-GPU)."""
        all_params = list(chain(self.actor.parameters(), self.critic_v0.parameters(), self.critic_v1.parameters()))
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
