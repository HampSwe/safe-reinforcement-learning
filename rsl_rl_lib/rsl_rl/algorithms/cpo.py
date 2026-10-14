"""CPO (Constrained Policy Optimization) for the rsl_rl fork (v5.4.x API).

Port (2026-09-14) of the Isaac Drone Racer implementation ``isaac_drone_racer/algorithms/cpo.py``
(written for stock rsl-rl-lib 3.0.1, see CPO_DESIGN.md in the same hand-off zip) to this fork's
``MLPModel`` actor / critic, ``RolloutStorage`` and ``construct_algorithm`` conventions, so it can be
run on the planar drone with the same rollout / GAE / critic setup as the fork's ``PPO``.
The update itself (dual solve, sign convention, line search) is unchanged from the Isaac version.

Reference:
    - Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel, "Constrained Policy Optimization",
      ICML 2017, https://arxiv.org/abs/1705.10528
    - Update/line-search structure follows OpenAI safety-starter-agents (``cpo``) and OmniSafe.

Each iteration takes ONE trust-region step that solves the local problem

    max_x  g^T x    s.t.   b^T x + c <= 0,   0.5 x^T H x <= target_kl

    g = grad of the reward surrogate  E[ratio * A_r]
    b = grad of the cost surrogate    E[ratio * A_c]
    H = Fisher information (KL Hessian), applied via conjugate gradient
    c = J_C(pi) - cost_limit          (current constraint violation)

in closed form through its dual, falling back to a pure cost-descent recovery step when the
linearized problem is infeasible, and backtracking until the KL / improvement conditions hold.

The cost is the same r1 channel RL-SGF constrains (``extras["r1"]``, environment.py:r1_reward).
The constraint is on the AVERAGE PER-STEP cost over the on-policy state distribution,
``c = mean_batch(r1) - cost_limit`` -- the safety-starter-agents ``(EpCost - limit) / EpLen`` form,
which puts c in the same per-step units as the cost surrogate. With cost_limit = 0 this is
RL-SGF's constraint V1 <= 0 under gamma1 = 1: E[#unsafe steps] <= beta * T.

Like the fork's PPO this is NOT episodic: it runs on the stock rollout loop (short buffers, no
forced resets) with GAE on both channels, so it is directly comparable to the PPO baseline --
CPO vs PPO differs only in the policy update.
"""

from __future__ import annotations

import math
from collections import deque
from itertools import chain

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import compile_model, resolve_callable, resolve_obs_groups

_EPS = 1e-8


class CPO:
    """Constrained Policy Optimization with a single cost channel (r1)."""

    actor: MLPModel
    critic_r: MLPModel
    critic_c: MLPModel

    def __init__(
        self,
        actor: MLPModel,
        critic_r: MLPModel,
        critic_c: MLPModel,
        storage: RolloutStorage,
        gamma: float = 0.99,
        lam: float = 0.95,
        cost_gamma: float = 0.99,
        cost_lam: float = 0.95,
        # --- trust region ---
        target_kl: float = 0.01,
        cg_iters: int = 15,
        cg_damping: float = 0.1,
        line_search_steps: int = 15,
        line_search_decay: float = 0.8,
        fvp_subsample: int = 5,
        # --- constraint ---
        cost_limit: float = 0.0,
        # --- critic fitting ---
        critic_lr: float = 1.0e-3,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        max_grad_norm: float = 1.0,
        # episodic logging window (completed episodes)
        episode_window: int = 1024,
        device: str = "cpu",
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        if multi_gpu_cfg is not None:
            raise NotImplementedError("CPO is single-GPU only.")
        if rnd_cfg is not None or symmetry_cfg is not None:
            raise NotImplementedError("CPO does not support RND or symmetry augmentation.")
        self.device = device
        self.is_multi_gpu = False
        self.gpu_global_rank = 0
        self.gpu_world_size = 1

        # The runner checks these; CPO has no RND path.
        self.rnd = None
        self.symmetry = None
        self.intrinsic_rewards = None

        self.actor = actor.to(device)
        self.critic_r = critic_r.to(device)
        self.critic_c = critic_c.to(device)
        # Uncompiled handles for state_dict / export (aliases unless compile() is called).
        self._raw_actor = self.actor
        self._raw_critic_r = self.critic_r
        self._raw_critic_c = self.critic_c

        self.critic_optimizer = torch.optim.Adam(
            chain(self.critic_r.parameters(), self.critic_c.parameters()), lr=critic_lr
        )
        self.optimizer = self.critic_optimizer
        self.learning_rate = critic_lr  # logged by the runner; the policy step has no learning rate

        self.storage = storage
        self.transition = RolloutStorage.Transition()
        T, N = storage.num_transitions_per_env, storage.num_envs
        self.costs = torch.zeros(T, N, 1, device=device)       # r1, time-out bootstrapped (GAE target)
        self.raw_costs = torch.zeros(T, N, 1, device=device)   # r1 as exported (the constrained quantity)
        self.cost_values = torch.zeros(T, N, 1, device=device)
        self.cost_returns = torch.zeros(T, N, 1, device=device)
        self.cost_advantages = torch.zeros(T, N, 1, device=device)

        self.gamma, self.lam = gamma, lam
        self.cost_gamma, self.cost_lam = cost_gamma, cost_lam
        self.target_kl = target_kl
        self.cg_iters = cg_iters
        self.cg_damping = cg_damping
        self.line_search_steps = line_search_steps
        self.line_search_decay = line_search_decay
        # Fisher-vector products use every k-th sample of the batch (OpenAI baselines TRPO uses 5,
        # OmniSafe 4). Only the trust-region metric H is subsampled; g, b, c and the line-search
        # checks use the full batch. Measured (2026-09-14, 2048 envs x 100 steps, RTX 3070): the 30
        # full-batch FVPs of one update take 2.6 s of a 3.4 s update; at 1/10 they take 0.27 s.
        self.fvp_subsample = max(int(fvp_subsample), 1)
        self.cost_limit = cost_limit
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm

        # Per-env running episode sums, for the logged episodic cost (not used by the update).
        self._ep_cost = torch.zeros(N, device=device)
        self._ep_len = torch.zeros(N, device=device)
        self._ep_cost_buf = deque(maxlen=episode_window)
        self._ep_len_buf = deque(maxlen=episode_window)

    # ------------------------------------------------------------------ #
    # policy parameter helpers
    # ------------------------------------------------------------------ #
    def _policy_parameters(self) -> list[nn.Parameter]:
        # MLP weights + the distribution's std parameter (the obs normalizer keeps buffers, not params).
        return [p for p in self.actor.parameters() if p.requires_grad]

    def _clamp_std(self) -> None:
        """Keep the std PARAMETER inside the distribution's std_range (the distribution only clamps
        on the forward pass; a natural-gradient step can push the raw parameter far outside, which
        makes the Fisher (~1/sigma^2) ill-conditioned). Same reasoning as RLSGF.std_min."""
        dist = self.actor.distribution
        with torch.no_grad():
            if dist.std_type == "log":
                dist.log_std_param.clamp_(dist.log_std_range[0], dist.log_std_range[1])
            else:
                dist.std_param.clamp_(dist.std_range[0], dist.std_range[1])

    # ------------------------------------------------------------------ #
    # rollout hooks (called by the stock OnPolicyRunner.learn)
    # ------------------------------------------------------------------ #
    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic_r.get_hidden_state())
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic_r(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs
        self._cost_value = self.critic_c(obs).detach()
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self.actor.update_normalization(obs)
        self.critic_r.update_normalization(obs)
        self.critic_c.update_normalization(obs)

        step = self.storage.step
        cost = extras["r1"].to(self.device)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        self.cost_values[step] = self._cost_value
        self.costs[step] = cost.unsqueeze(-1)
        self.raw_costs[step] = cost.unsqueeze(-1)

        # Bootstrap on time-outs (truncation, not termination) -- as rsl-rl PPO does for rewards.
        if "time_outs" in extras:
            to = extras["time_outs"].unsqueeze(1).to(self.device).float()
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * to, 1)
            self.costs[step] += self.cost_gamma * self._cost_value * to

        # Episodic cost bookkeeping (logging only).
        self._ep_cost += cost
        self._ep_len += 1
        done_ids = (dones > 0).nonzero(as_tuple=False).squeeze(-1)
        if done_ids.numel() > 0:
            self._ep_cost_buf.extend(self._ep_cost[done_ids].tolist())
            self._ep_len_buf.extend(self._ep_len[done_ids].tolist())
            self._ep_cost[done_ids] = 0.0
            self._ep_len[done_ids] = 0.0

        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic_r.reset(dones)
        self.critic_c.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """GAE on both channels, bootstrapped from the critics at the final observation."""
        st = self.storage
        last_v = self.critic_r(obs).detach()
        last_vc = self.critic_c(obs).detach()
        adv = torch.zeros_like(last_v)
        adv_c = torch.zeros_like(last_vc)
        for t in reversed(range(st.num_transitions_per_env)):
            next_v = last_v if t == st.num_transitions_per_env - 1 else st.values[t + 1]
            next_vc = last_vc if t == st.num_transitions_per_env - 1 else self.cost_values[t + 1]
            not_done = 1.0 - st.dones[t].float()

            delta = st.rewards[t] + not_done * self.gamma * next_v - st.values[t]
            adv = delta + not_done * self.gamma * self.lam * adv
            st.returns[t] = adv + st.values[t]

            delta_c = self.costs[t] + not_done * self.cost_gamma * next_vc - self.cost_values[t]
            adv_c = delta_c + not_done * self.cost_gamma * self.cost_lam * adv_c
            self.cost_returns[t] = adv_c + self.cost_values[t]

        # In place: this runs under the runner's inference_mode, and a freshly allocated tensor here
        # would be an inference tensor that update() cannot use in autograd.
        st.advantages.copy_(st.returns - st.values)
        self.cost_advantages.copy_(self.cost_returns - self.cost_values)

    # ------------------------------------------------------------------ #
    # trust-region machinery
    # ------------------------------------------------------------------ #
    def _dist(self, obs_flat: TensorDict):
        """Set and return the current policy distribution over a flat batch of observations."""
        self.actor(obs_flat, stochastic_output=True)
        return self.actor.distribution

    @staticmethod
    def _kl(old_mean, old_std, dist) -> torch.Tensor:
        """Mean over the batch of KL(pi_old || pi_theta), summed over action dimensions."""
        new_mean, new_std = dist.mean, dist.std
        kl = (
            torch.log(new_std / old_std)
            + (old_std.pow(2) + (old_mean - new_mean).pow(2)) / (2.0 * new_std.pow(2))
            - 0.5
        )
        return kl.sum(-1).mean()

    def _conjugate_gradient(self, fvp, b: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = b.clone()
        rr = r.dot(r)
        for _ in range(self.cg_iters):
            Ap = fvp(p)
            alpha = rr / (p.dot(Ap) + _EPS)
            x += alpha * p
            r -= alpha * Ap
            rr_new = r.dot(r)
            if rr_new < 1e-10:
                break
            p = r + (rr_new / rr) * p
            rr = rr_new
        return x

    # ------------------------------------------------------------------ #
    # the CPO update
    # ------------------------------------------------------------------ #
    def update(self) -> dict[str, float]:  # noqa: C901
        st = self.storage
        T, N = st.num_transitions_per_env, st.num_envs
        obs = st.observations.flatten(0, 1)
        actions = st.actions.flatten(0, 1)

        adv_r = st.advantages.flatten(0, 1).squeeze(-1)
        adv_c = self.cost_advantages.flatten(0, 1).squeeze(-1)
        # Standardize the reward advantage; only CENTER the cost advantage so the cost surrogate
        # stays in the same per-step units as c (as safety-starter-agents / OmniSafe do).
        adv_r = (adv_r - adv_r.mean()) / (adv_r.std() + _EPS)
        adv_c = adv_c - adv_c.mean()

        # Constraint violation in per-step units.
        cost_mean = self.raw_costs.mean().item()
        c = cost_mean - self.cost_limit

        params = self._policy_parameters()

        # pi_old, re-evaluated with the CURRENT observation normalizer so that theta_old is exactly
        # the point we linearize around (the normalizer may have drifted during the rollout).
        with torch.no_grad():
            d_old = self._dist(obs)
            old_mean, old_std = d_old.mean.clone(), d_old.std.clone()
            old_logp = d_old.log_prob(actions)  # summed over action dims by the distribution

        # --- surrogate gradients g (reward) and b (cost) ---
        dist = self._dist(obs)
        ratio = torch.exp(dist.log_prob(actions) - old_logp)
        surr_r = (ratio * adv_r).mean()
        surr_c = (ratio * adv_c).mean()
        g = torch.nn.utils.parameters_to_vector(torch.autograd.grad(surr_r, params, retain_graph=True))
        b = torch.nn.utils.parameters_to_vector(torch.autograd.grad(surr_c, params))

        # --- Fisher-vector product via double backprop of the KL (on a subsample of the batch) ---
        k = self.fvp_subsample
        dist = self._dist(obs[::k])
        kl = self._kl(old_mean[::k], old_std[::k], dist)
        grad_kl = torch.nn.utils.parameters_to_vector(torch.autograd.grad(kl, params, create_graph=True))

        def fvp(v: torch.Tensor) -> torch.Tensor:
            hv = torch.autograd.grad(grad_kl.dot(v), params, retain_graph=True)
            return torch.nn.utils.parameters_to_vector(hv).detach() + self.cg_damping * v

        x = self._conjugate_gradient(fvp, g)          # H^-1 g
        q = g.dot(x).item()
        delta = self.target_kl

        # --- solve the dual (Achiam et al. 2017, Theorem 1 / Appendix) ---
        if b.dot(b).item() <= 1e-8 and c < 0:
            # Cost gradient is ~0 and we are feasible: the constraint is inactive -> TRPO step.
            p = torch.zeros_like(g)
            r = s = A = B = 0.0
            optim_case = 4
        else:
            p = self._conjugate_gradient(fvp, b)      # H^-1 b
            r = g.dot(p).item()
            s = b.dot(p).item()
            A = q - r ** 2 / (s + _EPS)
            B = 2.0 * delta - c ** 2 / (s + _EPS)
            if c < 0 and B < 0:
                optim_case = 3                        # whole trust region is feasible
            elif c < 0 <= B:
                optim_case = 2                        # feasible, constraint may bind
            elif c >= 0 and B >= 0:
                optim_case = 1                        # infeasible, but recoverable in one step
            else:
                optim_case = 0                        # infeasible -> pure recovery step

        if optim_case in (3, 4):
            lam = math.sqrt(q / (2.0 * delta))
            nu = 0.0
        elif optim_case in (1, 2):
            # NOTE the signs: g here is the ASCENT gradient of the reward surrogate. The usual
            # statement (safety-starter-agents, OmniSafe) uses the gradient of the policy LOSS,
            # r_loss = -r, which is where their "lam*c - r" / "- r*c/s" come from. Stationarity of
            # the dual in nu gives b^T x + c = 0 at nu = (lam*c + r)/s.
            # lambda ranges where nu is positive (LA) or clipped to 0 (LB).
            LA, LB = [0.0, -r / c], [-r / c, math.inf]
            LA, LB = (LA, LB) if c < 0 else (LB, LA)

            def proj(v, L):
                return max(L[0], min(L[1], v))

            lam_a = proj(math.sqrt(A / (B + _EPS)), LA)
            lam_b = proj(math.sqrt(q / (2.0 * delta)), LB)

            def f_a(lm):
                return -0.5 * (A / (lm + _EPS) + B * lm) + r * c / (s + _EPS)

            def f_b(lm):
                return -0.5 * (q / (lm + _EPS) + 2.0 * delta * lm)

            lam = lam_a if f_a(lam_a) >= f_b(lam_b) else lam_b
            nu = max(0.0, lam * c + r) / (s + _EPS)
        else:
            lam = 0.0
            nu = math.sqrt(2.0 * delta / (s + _EPS))

        if optim_case > 0:
            step_dir = (x - nu * p) / (lam + _EPS)
        else:
            step_dir = -nu * p

        # --- backtracking line search ---
        theta_old = torch.nn.utils.parameters_to_vector(params).detach()
        with torch.no_grad():
            surr_r_old = surr_r.item()
            surr_c_old = surr_c.item()
            accepted = False
            frac = 1.0
            kl_new = surr_r_new = surr_c_new = float("nan")
            ls_steps = self.line_search_steps
            for j in range(self.line_search_steps):
                torch.nn.utils.vector_to_parameters(theta_old + frac * step_dir, params)
                d_new = self._dist(obs)
                ratio_new = torch.exp(d_new.log_prob(actions) - old_logp)
                surr_r_new = (ratio_new * adv_r).mean().item()
                surr_c_new = (ratio_new * adv_c).mean().item()
                kl_new = self._kl(old_mean, old_std, d_new).item()
                finite = math.isfinite(surr_r_new) and math.isfinite(surr_c_new) and math.isfinite(kl_new)
                if (
                    finite
                    and kl_new <= delta
                    and (surr_r_new >= surr_r_old if optim_case > 1 else True)
                    and (surr_c_new - surr_c_old) <= max(-c, 0.0)
                ):
                    accepted = True
                    ls_steps = j
                    break
                frac *= self.line_search_decay
            if not accepted:
                torch.nn.utils.vector_to_parameters(theta_old, params)
                frac = 0.0
        self._clamp_std()
        final_step_norm = (torch.nn.utils.parameters_to_vector(params).detach() - theta_old).norm().item()

        # --- critic fitting (minibatched epochs, both channels) ---
        returns_r = st.returns.flatten(0, 1)
        returns_c = self.cost_returns.flatten(0, 1)
        n = T * N
        mb = max(n // self.num_mini_batches, 1)
        loss_r_tot = loss_c_tot = 0.0
        n_batches = 0
        for _ in range(self.num_learning_epochs):
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, mb):
                idx = perm[i:i + mb]
                obs_mb = obs[idx]
                loss_r = (self.critic_r(obs_mb) - returns_r[idx]).pow(2).mean()
                loss_c = (self.critic_c(obs_mb) - returns_c[idx]).pow(2).mean()
                self.critic_optimizer.zero_grad()
                (loss_r + loss_c).backward()
                nn.utils.clip_grad_norm_(
                    chain(self.critic_r.parameters(), self.critic_c.parameters()), self.max_grad_norm
                )
                self.critic_optimizer.step()
                loss_r_tot += loss_r.item()
                loss_c_tot += loss_c.item()
                n_batches += 1

        # Explained variance (logging only).
        with torch.no_grad():
            v_r = st.values.flatten(0, 1)
            ev_r = (1.0 - (returns_r - v_r).var() / (returns_r.var() + _EPS)).item()
            ev_c = (1.0 - (returns_c - self.cost_values.flatten(0, 1)).var() / (returns_c.var() + _EPS)).item()

        # Leave the actor's distribution set at theta_new (the runner logs get_policy().output_std).
        with torch.no_grad():
            self._dist(obs[:1])

        st.clear()

        ep_cost = sum(self._ep_cost_buf) / len(self._ep_cost_buf) if self._ep_cost_buf else 0.0
        ep_len = sum(self._ep_len_buf) / len(self._ep_len_buf) if self._ep_len_buf else 0.0
        return {
            "value_function": loss_r_tot / max(n_batches, 1),
            "cost_value_function": loss_c_tot / max(n_batches, 1),
            "cost_mean": cost_mean,                   # batch mean r1 (the constrained quantity)
            "cost_violation": c,                      # cost_mean - cost_limit; <= 0 is feasible
            "cost_unsafe_frac": (self.raw_costs > 0.0).float().mean().item(),
            "ep_cost": ep_cost,                       # episodic sum of r1 (window of completed episodes)
            "ep_len": ep_len,
            "optim_case": float(optim_case),          # 0 recovery, 1 infeasible, 2 binding, 3/4 free
            "lambda": lam,
            "nu": nu,
            "q": q, "r": r, "s": s, "A": A, "B": B,
            "g_norm": g.norm().item(),
            "b_norm": b.norm().item(),
            "kl": kl_new if accepted else 0.0,
            "line_search_step": float(ls_steps),
            "line_search_accepted": float(accepted),
            "step_frac": frac,
            "step_norm": final_step_norm,
            "surr_r_improve": (surr_r_new - surr_r_old) if accepted else 0.0,
            "surr_c_change": (surr_c_new - surr_c_old) if accepted else 0.0,
            "critic_r_ev": ev_r,
            "critic_c_ev": ev_c,
        }

    # ------------------------------------------------------------------ #
    # runner contract (mirrors PPO / RLSGF)
    # ------------------------------------------------------------------ #
    def train_mode(self) -> None:
        self.actor.train()
        self.critic_r.train()
        self.critic_c.train()

    def eval_mode(self) -> None:
        self.actor.eval()
        self.critic_r.eval()
        self.critic_c.eval()

    def save(self) -> dict:
        return {
            "actor_state_dict": self._raw_actor.state_dict(),
            "critic_r_state_dict": self._raw_critic_r.state_dict(),
            "critic_c_state_dict": self._raw_critic_c.state_dict(),
            "optimizer_state_dict": self.critic_optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor"):
            self._raw_actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self._raw_critic_r.load_state_dict(loaded_dict["critic_r_state_dict"], strict=strict)
            self._raw_critic_c.load_state_dict(loaded_dict["critic_c_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.critic_optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        return self._raw_actor

    def compile(self, mode: str | None = None) -> None:
        self.actor = compile_model(self._raw_actor, mode)  # type: ignore
        self.critic_r = compile_model(self._raw_critic_r, mode)  # type: ignore
        self.critic_c = compile_model(self._raw_critic_c, mode)  # type: ignore

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> CPO:
        """Construct the CPO algorithm (same shape as PPO.construct_algorithm, two critics)."""
        alg_class: type[CPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
        # The runner reads cfg["algorithm"]["rnd_cfg"] for logging; CPO has no RND.
        cfg["algorithm"].setdefault("rnd_cfg", None)

        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        critic_r: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        critic_c: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Models (reward, cost): {critic_r}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        alg: CPO = alg_class(
            actor, critic_r, critic_c, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"]
        )
        alg.compile(cfg.get("torch_compile_mode"))
        return alg

    def broadcast_parameters(self) -> None:
        raise NotImplementedError("CPO is single-GPU only.")

    def reduce_parameters(self) -> None:
        raise NotImplementedError("CPO is single-GPU only.")
