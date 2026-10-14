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


class RLSGF_PPOGrad:
    """RL-SGF with V0's gradient replaced by the PPO-surrogate gradient.

    Minimal-diff variant of :class:`rsl_rl.algorithms.rl_sgf.RLSGF`; read that
    file first, only the differences are documented here.

    The PPO surrogate gradient, evaluated at the current theta (importance
    ratio == 1, clipping inactive), is exactly the policy gradient weighted by
    GAE(lambda) advantages. So V0's gradient keeps the SAME estimator template
    as the original (gamma^tau discounting, /n_ep normalization, chunked
    accumulation) and only the per-step weight changes:

        original:  w0 = rtg0 - V0(s)        (Monte-Carlo return residual)
        here:      w0 = A_GAE(s, a)         (GAE(lambda) advantage from critic_v0)

    V1's gradient is untouched, so both gradients are the same kind of object
    in the same units and the QCQP combination is unchanged. PPO's multi-epoch
    reuse / clipping / adaptive-KL do NOT appear here: those mechanisms only
    exist because PPO moves theta while reusing data; a single gradient at
    theta has ratio 1 by construction. Advantages are NOT normalized, to keep
    V0_grad's magnitude commensurate with V1_grad's.

    critic_v0 is fit against the GAE returns (advantages + values) instead of
    Monte-Carlo rtg0; rtg0 is dropped entirely. The QCQP never used the V0
    scalar (only V1's), so the logged "V0" is now the mean GAE return at
    episode-start states.

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

        # RND extension
        self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg) if rnd_cfg else None

        # Symmetry extension
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # Components
        self.actor = actor.to(self.device)
        self.critic_v0 = critic_v0.to(self.device)
        self.critic_v1 = critic_v1.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critic``.
        self._raw_actor = self.actor
        self._raw_critic_v0 = self.critic_v0
        self._raw_critic_v1 = self.critic_v1

        self.critic_lr = 1e-3
        self.critic_optimizer = torch.optim.Adam(
            chain(self.critic_v0.parameters(), self.critic_v1.parameters()), lr=self.critic_lr
        )

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # RL-SGF extras. No rtg0: V0's weights come from GAE (storage.advantages),
        # and the QCQP never consumed the V0 scalar.
        self.r1   = torch.zeros(storage.num_transitions_per_env, storage.num_envs, 1, device=self.device)
        self.rtg1 = torch.zeros(storage.num_transitions_per_env, storage.num_envs, 1, device=self.device)
        # Timeout flags per step: at a timeout row `dones` masks the next value out
        # of the GAE recursion, so the bootstrap must be re-added explicitly
        # (mirrors PPO's reward-side timeout bootstrapping). Captured here because
        # `extras` is gone by compute_returns time.
        self.time_outs = torch.zeros(storage.num_transitions_per_env, storage.num_envs, 1, device=self.device)

        # RL-SGF parameters
        self.h = h
        self.alpha = alpha

        # ---- __init__ ----
        self.h_ref     = self.h          # the h you already tuned; valid at the starting sigma
        self.sigma_ref = None            # captured lazily on the first update
        self.h_exp     = 2.0
        self.h_cap     = 1 / self.alpha        # Lemma 1 needs h < 1/alpha
        self.h_min     = 1e-7

        self.weight_decay = 0

        # mask: decay weight matrices only -- not biases, NOT log_std_param
        mask = []
        for name, p in self.actor.named_parameters():
            decay = ("weight" in name) and ("std" not in name)
            mask.append(torch.full((p.numel(),), float(decay), device=self.device))
        self.wd_mask = torch.cat(mask)

        self.max_theta_step = max_theta_step
        self.memory_chunk_size = 50

        self.env = env
        self.critic_use_epochs = critic_use_epochs
        self.critic_epochs = critic_epochs
        self.critic_minibatch = critic_minibatch
        self.bootstrap_final_state = bootstrap_final_state

        # PPO parameters (only gamma/lam drive the GAE weights; the epoch/clip/KL
        # parameters are accepted for config compatibility but unused -- a single
        # gradient at theta has ratio 1, so clipping/reuse mechanics never engage)
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

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic_v0.get_hidden_state())
        # Compute the actions and V0 values (the values feed the GAE recursion)
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

        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
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
        """Compute V1's Monte-Carlo safety returns and V0's GAE(lambda) targets."""
        st = self.storage

        if self.bootstrap_final_state:
            # The observation passed by the runner is s_{T}. Preserve recurrent
            # state because evaluating the critics must not advance the rollout.
            critic_v0_hidden_state = self.critic_v0.get_hidden_state()
            critic_v1_hidden_state = self.critic_v1.get_hidden_state()
            last_values = self.critic_v0(obs).detach()
            prev_r1 = self.critic_v1(obs).detach()
            self.critic_v0.reset(hidden_state=critic_v0_hidden_state)
            self.critic_v1.reset(hidden_state=critic_v1_hidden_state)
        else:
            last_values = torch.zeros(st.num_envs, 1, device=self.device)
            prev_r1 = torch.zeros(st.num_envs, 1, device=self.device)

        # ---- V1: unchanged Monte-Carlo safety returns ----
        for step in reversed(range(st.num_transitions_per_env)):
            next_is_not_terminal = 1.0 - st.dones[step].float()
            self.rtg1[step] = self.r1[step] + self.gamma * prev_r1 * next_is_not_terminal
            prev_r1 = self.rtg1[step]

        self.rtg1 = self.rtg1.detach()

        # ---- V0: GAE(lambda) on the reward, exactly PPO's recursion, with the
        # timeout bootstrap applied on the reward side (V(s_t) stands in for
        # V(s_{t+1}) at timeout rows, as in PPO). Advantages are NOT normalized:
        # V0_grad's magnitude must stay commensurate with V1_grad's. ----
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            step_reward = st.rewards[step] + self.gamma * st.values[step] * self.time_outs[step]
            delta = step_reward + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        # In-place into the preallocated buffer: the runner calls this inside
        # torch.inference_mode(), and a rebinding assignment would make
        # st.advantages an inference tensor that autograd cannot save for
        # backward when the gradient loop later multiplies by it.
        st.advantages[:] = st.returns - st.values

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
            self.sigma_ref = sigma_min                    # sigma at iteration 0
        h = self.h_ref * (sigma_min / self.sigma_ref) ** self.h_exp
        h = min(h, self.h_ref, self.h_cap)                # never exceed the tuned value
        h = max(h, self.h_min)
        return float(h), sigma_min

    def update(self) -> dict[str, float]:
        st = self.storage
        T, N = st.num_transitions_per_env, st.num_envs

        dones = st.dones.squeeze(-1).bool()                       # (T,N): episode ended AT this row
        rows = torch.arange(T, device=self.device).unsqueeze(1).expand(T, N)

        # Row 0 starts a rollout segment (and, in legacy reset mode, a genuine
        # episode); every other row starts a new episode after a done.
        starts = torch.zeros_like(dones)
        starts[0] = True
        starts[1:] = dones[:-1]

        # --- episode-local time tau: rows since this env's most recent start ---
        start_row = torch.where(starts, rows, torch.full_like(rows, -1))
        start_row = torch.cummax(start_row, dim=0).values
        tau = rows - start_row                                    # (T,N), 0 at each start

        if self.bootstrap_final_state:
            # Every trajectory segment has a valid target: terminal segments end
            # at done, while the final unfinished segment ends at a critic value.
            usable = torch.ones_like(dones)
            n_ep = starts.sum().clamp(min=1).float()
        else:
            # Legacy behavior only uses episodes that terminate in-buffer.
            ended = torch.flip(torch.cummax(torch.flip(dones, [0]).long(), 0).values, [0]).bool()
            usable = ended
            n_ep = dones.sum().clamp(min=1).float()

        gamma_tau = (self.gamma ** tau.float()).unsqueeze(-1)     # (T,N,1)
        keep = usable.unsqueeze(-1).float()                       # (T,N,1)

        rtg1 = self.rtg1.squeeze(-1)
        returns0 = st.returns.squeeze(-1)

        # Values at the start of each usable episode/rollout segment. V0 is
        # logging-only (the QCQP consumes only V1's scalar); it now comes from
        # the GAE returns instead of Monte-Carlo rtg0.
        good_start = starts & usable
        self.V0 = -returns0[good_start].mean()
        self.V1 =  rtg1[good_start].mean()

        obs_all = st.observations
        act_all = st.actions

        with torch.no_grad():
            v1_b = self.critic_v1(obs_all.flatten(0, 1)).reshape(T, N, 1)

            # explained variance -- v0 against its GAE targets (st.values holds the
            # rollout-time critic_v0 predictions), v1 against the MC safety returns
            g0_flat = st.returns.reshape(-1)
            g1_flat = self.rtg1.reshape(-1)
            p0 = st.values.reshape(-1)
            p1 = v1_b.reshape(-1)
            ev0 = (1.0 - (g0_flat - p0).var() / (g0_flat.var() + 1e-8)).item()
            ev1 = (1.0 - (g1_flat - p1).var() / (g1_flat.var() + 1e-8)).item()

        w0 = st.advantages           # the only change to V0's estimator: GAE weights
        w1 = self.rtg1 - v1_b

        d = sum(p.numel() for p in self.actor.parameters())
        g0 = torch.zeros(d, device=self.device)
        g1 = torch.zeros(d, device=self.device)

        chunk = self.memory_chunk_size
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            obs_c = obs_all[start:end].flatten(0, 1)
            act_c = act_all[start:end].flatten(0, 1)

            self.actor(obs_c, stochastic_output=True)
            logp_c = self.actor.get_output_log_prob(act_c).reshape(end - start, N, 1)

            # Episode/segment-local discount, normalized by the number used.
            gt_c = gamma_tau[start:end] * keep[start:end]
            J0_c = -(gt_c * logp_c * w0[start:end]).sum() / n_ep
            J1_c =  (gt_c * logp_c * w1[start:end]).sum() / n_ep

            g0 += torch.nn.utils.parameters_to_vector(
                torch.autograd.grad(J0_c, self.actor.parameters(), retain_graph=True))
            g1 += torch.nn.utils.parameters_to_vector(
                torch.autograd.grad(J1_c, self.actor.parameters()))

        self.theta = torch.nn.utils.parameters_to_vector(self.actor.parameters()).detach()
        self.V0_grad = g0 + 2.0 * self.weight_decay * self.wd_mask * self.theta
        self.V1_grad = g1

        # ---- QCQP (unchanged) ----
        self.A = self.V1_grad.norm() ** 2 - 2 * self.alpha * self.V1
        self.B = 2 * self.A
        self.C = 2 * self.V1_grad.dot(self.V0_grad) - self.V0_grad.norm() ** 2 - 2 * self.alpha * self.V1
        self.delta = 4 * ((self.V1_grad - self.V0_grad).norm() ** 2) * self.A

        self.h, sigma_min = self._current_h()

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

        # ---- critic update: identical structure to the original, with critic_v0's
        # targets switched from Monte-Carlo rtg0 to the GAE returns. ----
        if not self.critic_use_epochs:
            # ORIGINAL: one full-batch step, chunked for memory
            self.critic_optimizer.zero_grad()
            v0_loss_total = 0.0
            v1_loss_total = 0.0
            for start in range(0, T, chunk):
                end = min(start + chunk, T)
                obs_c = obs_all[start:end].flatten(0, 1)
                v0_c = self.critic_v0(obs_c).reshape(end - start, N, 1)
                v1_c = self.critic_v1(obs_c).reshape(end - start, N, 1)
                frac = (end - start) / T
                v0_loss_c = ((v0_c - st.returns[start:end]) ** 2).mean() * frac
                v1_loss_c = ((v1_c - self.rtg1[start:end]) ** 2).mean() * frac
                (v0_loss_c + v1_loss_c).backward()
                v0_loss_total += v0_loss_c.item()
                v1_loss_total += v1_loss_c.item()
            self.critic_optimizer.step()
        else:
            # minibatched epochs
            obs_flat = obs_all.reshape(T * N, *obs_all.shape[2:])
            returns0_flat = st.returns.reshape(T * N, 1)
            rtg1_flat = self.rtg1.reshape(T * N, 1)
            n_samples = T * N
            v0_loss_total = v1_loss_total = 0.0
            n_batches = 0
            for _ in range(self.critic_epochs):
                perm = torch.randperm(n_samples, device=self.device)
                for mb in range(0, n_samples, self.critic_minibatch):
                    idx = perm[mb: mb + self.critic_minibatch]
                    self.critic_optimizer.zero_grad()
                    v0_c = self.critic_v0(obs_flat[idx])
                    v1_c = self.critic_v1(obs_flat[idx])
                    v0_loss_c = ((v0_c - returns0_flat[idx]) ** 2).mean()
                    v1_loss_c = ((v1_c - rtg1_flat[idx]) ** 2).mean()
                    (v0_loss_c + v1_loss_c).backward()
                    self.critic_optimizer.step()
                    v0_loss_total += v0_loss_c.item()
                    v1_loss_total += v1_loss_c.item()
                    n_batches += 1
            v0_loss_total /= max(n_batches, 1)
            v1_loss_total /= max(n_batches, 1)

        # ---- reset storage AND per-iteration buffers so next rollout starts fresh ----
        self.storage.clear()
        self.r1.zero_()
        self.time_outs.zero_()

        return {
            "V0": self.V0.item(), "V1": self.V1.item(),
            "A": self.A.item(), "B": self.B.item(), "C": self.C.item(),
            "delta": self.delta.item(),
            "u": self.u.item() if torch.is_tensor(self.u) else float(self.u),
            "V0_grad_norm": self.V0_grad.norm().item(),
            "V1_grad_norm": self.V1_grad.norm().item(),
            "critic_v0_loss": v0_loss_total, "critic_v1_loss": v1_loss_total,
            "clipped": float(clipped),
            "raw_step_norm": raw_step_norm.item(),
            "step_norm": (self.p_theta - self.theta).norm().item(),
            "h": self.h,
            "wd_frac": (2.0 * self.weight_decay * (self.wd_mask * self.theta).norm() / g0.norm()).item(),
            "theta_norm": self.theta.norm().item(),
            "N_eff": n_ep.item(),        # episodes/segments used this update -- WATCH THIS
            "critic_v0_ev": ev0,
            "critic_v1_ev": ev1,
        }

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic_v0.train()
        self.critic_v1.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic_v0.eval()
        self.critic_v1.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_v0_state_dict": self._raw_critic_v0.state_dict(),
            "critic_v1_state_dict": self._raw_critic_v1.state_dict(),
            "optimizer_state_dict": self.critic_optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd.optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic_v0.load_state_dict(loaded_dict["critic_v0_state_dict"], strict=strict)
            self._raw_critic_v1.load_state_dict(loaded_dict["critic_v1_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.critic_optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd.optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        """Compile actor and critic with ``torch.compile``.

        See :func:`~rsl_rl.utils.compile_model` for the set of accepted modes.

        Args:
            mode: ``torch.compile`` mode. Defaults to ``None``, in which case compilation is disabled.
        """
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic_v0 = compile_model(self._raw_critic_v0, mode)
        self.critic_v1 = compile_model(self._raw_critic_v1, mode)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> RLSGF_PPOGrad:
        """Construct the algorithm."""
        # Resolve class callables
        alg_class: type[RLSGF_PPOGrad] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic_v0: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        critic_v1: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: RLSGF_PPOGrad = alg_class(
            actor, critic_v0, critic_v1, storage, device=device, env=env, **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        # Compile the algorithm's models if requested
        alg.compile(cfg.get("torch_compile_mode"))

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self._raw_actor.state_dict(), self._raw_critic_v0.state_dict(), self._raw_critic_v1.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self._raw_actor.load_state_dict(model_params[0])
        self._raw_critic_v0.load_state_dict(model_params[1])
        self._raw_critic_v1.load_state_dict(model_params[2])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic_v0.parameters(), self.critic_v1.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
