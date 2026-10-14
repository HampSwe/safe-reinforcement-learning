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


class IMP_RLSGF:
    """RL-SGF algorithm.

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
        alpha: float = 1.0, #5.0
        max_theta_step: float = 1.0,    # superseded by the desired_kl trust region; kept for config compatibility
        fisher_damping: float = 1e-3,
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
        if symmetry_cfg is not None and (actor.is_recurrent or self.critic_v0.is_recurrent or self.critic_v1.is_recurrent):
            raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
        self.symmetry = Symmetry(**symmetry_cfg) if symmetry_cfg else None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic_v0 = critic_v0.to(self.device)
        self.critic_v1 = critic_v1.to(self.device)

        # Handles to the uncompiled modules for state_dict operations and export. If compilation is disabled, these
        # simply alias ``self.actor`` / ``self.critic``.
        self._raw_actor = self.actor
        self._raw_critic_v0 = self.critic_v0
        self._raw_critic_v1 = self.critic_v1

        # Create the optimizer
        # self.optimizer = resolve_optimizer(optimizer)(
        #     chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
        # )  # type: ignore

        self.critic_lr = 1e-3
        self.critic_optimizer = torch.optim.Adam(
            chain(self.critic_v0.parameters(), self.critic_v1.parameters()), lr=1e-3
        )

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # RL-SGF extras
        self.r1   = torch.zeros(storage.num_transitions_per_env, storage.num_envs, 1, device=self.device)
        self.rtg0 = torch.zeros(storage.num_transitions_per_env, storage.num_envs, 1, device=self.device)
        self.rtg1 = torch.zeros(storage.num_transitions_per_env, storage.num_envs, 1, device=self.device)
        
        # RL-SGF parameters
        self.h = h
        self.alpha = alpha
        self.fisher_damping = fisher_damping
        # desired_kl (set below with the PPO parameters) doubles as the
        # KL trust-region budget of the projection.

        self.max_theta_step = max_theta_step
        self.memory_chunk_size = 50 #int(50*0.5)

        # PPO parameters
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
        self.transition.hidden_states = (self.actor.get_hidden_state(), None)
        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = torch.zeros(obs.batch_size[0], 1, device=self.device)   # unused; storage requires it
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
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        self.r1[self.storage.step] = extras["r1"].to(self.device).unsqueeze(-1)

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        # if "time_outs" in extras:
        #     self.transition.rewards += self.gamma * torch.squeeze(
        #         self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
        #         1,
        #     )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic_v0.reset(dones)
        self.critic_v1.reset(dones)


    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage

        # Compute values for the last step
        # critic_hidden_state = self.critic.get_hidden_state()
        # last_values = self.critic(obs).detach()
        # # Restore the critic's hidden state so the next rollout is not affected by the forward pass
        # self.critic.reset(hidden_state=critic_hidden_state)
        # Compute returns and advantages
        
        prev_r0 = torch.zeros(st.num_envs, 1, device=self.device)
        prev_r1 = torch.zeros(st.num_envs, 1, device=self.device)

        for step in reversed(range(st.num_transitions_per_env)):
            next_is_not_terminal = 1.0 - st.dones[step].float()

            self.rtg0[step] = st.rewards[step] + self.gamma * prev_r0 * next_is_not_terminal
            self.rtg1[step] = self.r1[step] + self.gamma * prev_r1 * next_is_not_terminal
            prev_r0 = self.rtg0[step]
            prev_r1 = self.rtg1[step]

        self.rtg0 = self.rtg0.detach()
        self.rtg1 = self.rtg1.detach()


    def update(self) -> dict[str, float]:
        st = self.storage

        dones = self.storage.dones.squeeze(-1).bool()
        starts = torch.zeros_like(dones)
        starts[0] = True
        starts[1:] = dones[:-1]

        # rtg at every start row is that episode's full return G_0
        # rtg0/rtg1 are (T, N, 1) -> squeeze to (T, N)
        rtg0 = self.rtg0.squeeze(-1)
        rtg1 = self.rtg1.squeeze(-1)

        # Compute V0 and V1
        self.V0 = -rtg0[starts].mean()
        self.V1 =  rtg1[starts].mean()

        T, N = st.num_transitions_per_env, st.num_envs
        obs_all = st.observations                    # (T, N, ...) — don't flatten yet
        act_all = st.actions                         # (T, N, A)

        # baseline, no graph
        with torch.no_grad():
            v0_b = self.critic_v0(obs_all.flatten(0,1)).reshape(T, N, 1)
            v1_b = self.critic_v1(obs_all.flatten(0,1)).reshape(T, N, 1)
        w0 = self.rtg0 - v0_b                          # (T, N, 1) detached
        w1 = self.rtg1 - v1_b

        gamma_t = (self.gamma ** torch.arange(T, device=self.device)).view(T, 1, 1)

        # snapshot the policy on a probe batch for exact realized-KL logging
        probe_obs = obs_all[0]
        with torch.no_grad():
            self.actor(probe_obs, stochastic_output=True)
            mu_old, std_old = (p.clone() for p in self.actor.output_distribution_params)

        # accumulate gradients over timestep chunks
        d = sum(p.numel() for p in self.actor.parameters())
        g0 = torch.zeros(d, device=self.device)
        g1 = torch.zeros(d, device=self.device)
        f  = torch.zeros(d, device=self.device)    # diagonal Fisher accumulator

        chunk = self.memory_chunk_size                                     # tune to fit memory
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            obs_c = obs_all[start:end].flatten(0, 1)   # (chunk*N, ...)
            act_c = act_all[start:end].flatten(0, 1)

            self.actor(obs_c, stochastic_output=True)
            logp_c = self.actor.get_output_log_prob(act_c).reshape(end - start, N, 1)

            gt_c = gamma_t[start:end]
            J0_c = -(gt_c * logp_c * w0[start:end]).sum() / N
            J1_c =  (gt_c * logp_c * w1[start:end]).sum() / N

            # entropy bonus (diagonal Gaussian: gradient only touches the std parameters)
            std_c = self.actor.output_distribution_params[1]
            J0_c = J0_c - self.entropy_coef * torch.log(std_c).sum() / N

            # accumulate — retain graph only within the chunk, free after
            g0 += torch.nn.utils.parameters_to_vector(
                torch.autograd.grad(J0_c, self.actor.parameters(), retain_graph=True))
            g1 += torch.nn.utils.parameters_to_vector(
                torch.autograd.grad(J1_c, self.actor.parameters(), retain_graph=True))
            # one Rademacher probe per chunk: E[v**2] = sum of per-sample squared scores
            eps_c = torch.randint(0, 2, logp_c.shape, device=self.device, dtype=logp_c.dtype) * 2.0 - 1.0
            v = torch.nn.utils.parameters_to_vector(
                torch.autograd.grad((eps_c * logp_c).sum(), self.actor.parameters()))
            f += v.pow(2)
            # chunk graph freed here

        self.V0_grad = g0
        self.V1_grad = g1

        # damped diagonal Fisher (Rademacher estimate of E[score**2] per parameter)
        F = f / (T * N)
        F = F + self.fisher_damping * F.mean().clamp_min(1e-12)
        Fi = F.rsqrt()                                 # F^{-1/2}, elementwise

        # Compute next theta using closed form formula.
        # Whitened trust-region projection (x = F^{1/2} * step):
        #     min ||x - z||^2   s.t.   at.x <= c   (hard: linearized CBF condition)
        #                              ||x|| <= r  (KL budget desired_kl, r = sqrt(2*desired_kl))
        # z = -h F^{-1/2} grad(V0) is the natural-gradient step, at = F^{-1/2} grad(V1),
        # c = -h alpha V1. With F = I and desired_kl = inf this is the plain
        # halfspace projection z - lambda*at, i.e. vanilla RL-SGF.
        z  = -self.h * Fi * self.V0_grad
        at = Fi * self.V1_grad
        r  = (2.0 * self.desired_kl) ** 0.5
        c  = -self.h * self.alpha * self.V1

        a2 = at.dot(at).clamp_min(1e-12)
        az = at.dot(z)

        if c < -r * a2.sqrt():
            x = (c / a2) * at                          # (5) conflict: minimal-KL safety-restoring step
            case = 5
        elif az <= c and z.norm() <= r:
            x = z                                      # (1) interior: pure natural-gradient step
            case = 1
        elif az <= c:
            x = z * (r / z.norm().clamp_min(1e-12))    # (2) KL-limited
            case = 2
            if at.dot(x) > c:                          # shrinking left the safe halfspace
                x, case = self._corner(at, a2, z, az, c, r), 4
        else:
            x = z - ((az - c) / a2) * at               # (3) safety-limited (vanilla RL-SGF in the F-metric)
            case = 3
            if x.norm() > r:
                x, case = self._corner(at, a2, z, az, c, r), 4

        self.theta = torch.nn.utils.parameters_to_vector(self.actor.parameters())
        step = Fi * x                                  # un-whiten
        self.p_theta = self.theta + step

        with torch.no_grad():
            torch.nn.utils.vector_to_parameters(self.p_theta.detach(), self.actor.parameters())

        # exact realized KL on the probe batch (gap to quad_kl = linearization drift)
        with torch.no_grad():
            self.actor(probe_obs, stochastic_output=True)
            mu_new, std_new = self.actor.output_distribution_params
            realized_kl = (torch.log(std_new / std_old)
                           + (std_old.pow(2) + (mu_old - mu_new).pow(2)) / (2.0 * std_new.pow(2))
                           - 0.5).sum(-1).mean()


        # Critic update maybe wrong?
        # chunked critic update (single optimizer over both critics)
        self.critic_optimizer.zero_grad()
        v0_loss_total = 0.0
        v1_loss_total = 0.0
        for start in range(0, T, chunk):
            end = min(start + chunk, T)
            obs_c = obs_all[start:end].flatten(0, 1)
            v0_c = self.critic_v0(obs_c).reshape(end - start, N, 1)
            v1_c = self.critic_v1(obs_c).reshape(end - start, N, 1)

            frac = (end - start) / T           # so chunk-means sum to the full mean
            v0_loss_c = ((v0_c - self.rtg0[start:end]) ** 2).mean() * frac
            v1_loss_c = ((v1_c - self.rtg1[start:end]) ** 2).mean() * frac
            (v0_loss_c + v1_loss_c).backward()  # accumulates into .grad across chunks

            v0_loss_total += v0_loss_c.item()   # for logging
            v1_loss_total += v1_loss_c.item()

        self.critic_optimizer.step()


        # Reset storage
        self.storage.clear()
        self.r1.zero_()

        return {
            "V0": self.V0.item(),
            "V1": self.V1.item(),
            "case": float(case),                          # 1 interior | 2 KL-limited | 3 safety-limited | 4 corner | 5 conflict
            "quad_kl": (0.5 * x.dot(x)).item(),           # KL spent per the quadratic form; <= desired_kl in cases 1-4
            "realized_kl": realized_kl.item(),            # exact KL on the probe batch
            "safety_slack": (c - at.dot(x)).item(),       # >= 0 in cases 1-4; < 0 only in case 5
            "V0_grad_norm": self.V0_grad.norm().item(),
            "V1_grad_norm": self.V1_grad.norm().item(),

            "critic_v0_loss": v0_loss_total,
            "critic_v1_loss": v1_loss_total,
            
            "nat_step_norm": z.norm().item(),             # what the step WANTED to be (whitened)
            "step_norm": step.norm().item(),              # what it actually was (parameter space)
        }


    @staticmethod
    def _corner(at: torch.Tensor, a2: torch.Tensor, z: torch.Tensor,
                az: torch.Tensor, c: torch.Tensor, r: float) -> torch.Tensor:
        """Projection onto the circle {at.x = c} intersected with {||x|| = r} (case 4)."""
        x_par = (c / a2) * at
        x_perp = (z - ((az - c) / a2) * at) - x_par     # halfspace projection of z, component perp. to at
        pn = x_perp.norm()
        rad2 = (r * r - x_par.dot(x_par)).clamp_min(0.0)
        if pn <= 1e-12:
            return x_par
        return x_par + rad2.sqrt() * (x_perp / pn)

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
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> RLSGF:
        """Construct the PPO algorithm."""
        # Resolve class callables
        alg_class: type[RLSGF] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
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
        #critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 2, **cfg["critic"]).to(device)
        critic_v0: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        critic_v1: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        #print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize the algorithm
        alg: RLSGF = alg_class(actor, critic_v0, critic_v1, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

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