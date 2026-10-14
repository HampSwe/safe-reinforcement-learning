"""Batched 2D planar quadrotor environment, compatible with rsl_rl's VecEnv.

All state is held as torch tensors of shape (num_envs, ...) on `device`, so the
whole batch steps in parallel -- this is what lets rsl_rl's PPO collect from
thousands of envs at once. This is the tensor port of your single-env Drone2D.

VecEnv contract (https://leggedrobotics.github.io/rsl_rl/api/env.html):
  required attributes : num_envs, num_actions, max_episode_length,
                        episode_length_buf, device, cfg
  required methods     : get_observations(), step(actions)
  step() returns       : (obs TensorDict, rewards (N,), dones (N,), extras dict)
"""

import math
import torch
from tensordict import TensorDict

WORLD_X = 10 # [-WORLD_X, WORLD_X]
WORLD_Y = 10 # [0, WORLD_Y]

FAILURE_PENALTY = 10000
TARGET_BONUS = 100*1
TARGET_DIS_SCALE = 1
TARGET_RADIUS = 0.05


TARGET_X_WIDTH = 6
TARGET_Y_OFFSET, TARGET_Y_WIDTH = 1, 8

OBSTACLE_RADIUS = 0.4
UNSAFE_REGION_OBSTACLE_RADIUS = OBSTACLE_RADIUS * 1.25
OBSTACLE_MID_STD = 0.1 #1.0
OBSTACLE_SPAWN_CLEARANCE = 0.5
OBSTACLE_TARGET_CLEARANCE = OBSTACLE_SPAWN_CLEARANCE
R0_SCALER = 1e-3

class VecDrone2D:
    def __init__(self, num_envs, device, cfg=None, subtract_r1_from_r0=False, r1_multiplier=1, perturb=0, beta=5e-2, k=0.5):
        # --- required VecEnv attributes ---
        self.num_envs = num_envs
        self.num_actions = 2                       # [thrust, moment]
        self.device = torch.device(device)
        self.cfg = cfg or {}
        self.max_episode_length = 300             # control steps per episode
        self.episode_length_buf = torch.zeros(
            num_envs, dtype=torch.long, device=self.device)

        # --- physics params ---
        self.m, self.J, self.g = 1.0, 0.5, 9.81
        self.dt = 1e-3                             # physics step
        self.decimation = 20                      # physics steps per control step
        self.hover = self.m * self.g

        self.TARGET_RADIUS_SQ = TARGET_RADIUS**2
        self.OBSTACLE_RADIUS_SQ = OBSTACLE_RADIUS**2
        self.UNSAFE_REGION_OBSTACLE_RADIUS_SQUARED = UNSAFE_REGION_OBSTACLE_RADIUS**2
        self.beta = beta
        self.k = k
        self.subtract_r1_from_r0 = subtract_r1_from_r0
        self.r1_multiplier = r1_multiplier # this is only applicable when r1 is subtracted from r0
        self.perturb = perturb

        self.prev_action = torch.zeros(num_envs, self.num_actions, device=self.device)

        # --- action scaling: raw policy output (~[-1,1]) -> F, M ---
        self.thrust_scale = self.hover
        self.moment_scale = 4.0

        # --- hover setpoint (could be randomized per env for "move" tasks) ---
        self.target = torch.zeros(num_envs, 2, device=self.device)

        self.obstacle = torch.zeros(num_envs, 2, device=self.device)

        # --- state: [x, y, theta, vx, vy, omega], shape (num_envs, 6) ---
        self.state = torch.zeros(num_envs, 6, device=self.device)
        self.reset_idx(torch.arange(num_envs, device=self.device))

        # compile the physics hot path for kernel fusion
        self._integrate = torch.compile(self._integrate)
        #self._integrate = torch.compile(self._integrate, mode="reduce-overhead")
        self._reward   = torch.compile(self._reward)
        self.r1_reward = torch.compile(self.r1_reward)

    # ------------------------------------------------------------------ #
    # required: get_observations  ->  TensorDict with the groups your
    # config1.yaml references ("policy" for actor, "+privileged" for critic)
    # ------------------------------------------------------------------ #
    def get_observations(self):
        x, y, th, vx, vy, w = self.state.unbind(dim=1)
        ex, ey = x - self.target[:, 0], y - self.target[:, 1]

        # Normalization
        ex = ex / (WORLD_X*2)
        ey = ey / WORLD_Y
        # th = th / 5
        vx = vx / 5
        vy = vy / 5
        w = w / 5

        pa0 = self.prev_action[:, 0]
        pa1 = self.prev_action[:, 1]

        ox = (self.obstacle[:, 0] - x) / (WORLD_X*2)
        oy = (self.obstacle[:, 1] - y) / WORLD_Y

        # actor sees this. sin/cos instead of raw theta (no wrap discontinuity)
        policy = torch.stack([ex, ey, torch.sin(th), torch.cos(th), vx, vy, w,
        pa0, pa1, ox, oy], dim=1)
        # <-- inject observation noise here to emulate a real state estimator

        # critic-only privileged info (clean, plus any hidden params)
        privileged = torch.stack([th, self.m * torch.ones_like(th)], dim=1)

        return TensorDict(
            {"policy": policy, "privileged": privileged},
            batch_size=[self.num_envs],
        )

    # ------------------------------------------------------------------ #
    # required: step
    # ------------------------------------------------------------------ #
    def step(self, actions):
        actions = actions.to(self.device).clamp(-1.0, 1.0)
        F = (self.hover + actions[:, 0] * self.thrust_scale).clamp(min=0.0)
        M = actions[:, 1] * self.moment_scale

        # integrate physics with zero-order hold on (F, M) across substeps
        for _ in range(self.decimation):
            self._integrate(F, M)
            #self.state = self.state.clone()   # break the CUDA-graph buffer aliasing

        self.episode_length_buf += 1

        terminated = self._terminated()
        timed_out = self.episode_length_buf >= self.max_episode_length
        dones = terminated | timed_out

        # should make r1 penalties continuous probably
        r0 = self._reward(actions)
        r1 = self.r1_reward(actions)

        #r1 = FAILURE_PENALTY * terminated.float()
        if self.subtract_r1_from_r0:
            r0 -= r1 * self.r1_multiplier
        rewards = r0
        # r1 is included in extras

        self.prev_action = actions.clone()

        # auto-reset finished envs -- rsl_rl expects the POST-reset observation
        reset_ids = dones.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            self.reset_idx(reset_ids)

        obs = self.get_observations()
        extras = {"time_outs": timed_out,
                  "r1": r1}         # timeout != terminal, for bootstrap
        return obs, rewards, dones, extras

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _integrate(self, F, M):
        x, y, th, vx, vy, w = self.state.unbind(dim=1)

        ax = (1.0 / self.m) * torch.sin(th) * F
        ay = (1.0 / self.m) * torch.cos(th) * F - self.g
        alpha = -M / self.J

        vx, vy, w = vx + self.dt * ax, vy + self.dt * ay, w + self.dt * alpha   # velocities first
        x, y, th = x + self.dt * vx, y + self.dt * vy, th + self.dt * w          # then positions

        below = y < 0.0                            # ground clamp
        y = torch.where(below, torch.zeros_like(y), y)
        vy = torch.where(below, torch.zeros_like(vy), vy)

        self.state = torch.stack([x, y, th, vx, vy, w], dim=1)


    def _reward(self, actions):
        x, y, th, vx, vy, w = self.state.unbind(dim=1)
        pos_err = (x - self.target[:, 0]) ** 2 + (y - self.target[:, 1]) ** 2
        close = pos_err < self.TARGET_RADIUS_SQ
        rate = ((actions - self.prev_action) ** 2).sum(dim=1)
        linear_pos_err = torch.sqrt(pos_err)

        thrust = actions[:, 0]
        moment = actions[:, 1]

        return (
            3 * (20-pos_err)
            + close.float() * TARGET_BONUS
            #- 3 * thrust ** 2 # PPO wants
            #- 3 * moment ** 2 # PPO wants
            - 2.0 * rate
            - 0.5 * w ** 2 # was 0.5 before #PPO wants
            - 1 * (vx ** 2 + vy ** 2) # was 0.5 before
            #- 1.0 * (1.0 - torch.cos(th))
            #+ 3 * (20-linear_pos_err)
            #+ TARGET_BONUS*torch.exp(-linear_pos_err / TARGET_DIS_SCALE)
        ) * R0_SCALER
    
    def r1_reward(self, actions):
        x, y, th, vx, vy, w = self.state.unbind(dim=1)
        obst_d = torch.sqrt((x - self.obstacle[:, 0]) ** 2 + (y - self.obstacle[:, 1]) ** 2) - UNSAFE_REGION_OBSTACLE_RADIUS
        in_safe_set = obst_d >= 0

        # How should we handle the walls and floor? Should we just terminate?

        safe_cost   = self.beta * (torch.exp(-obst_d * self.k) - 1.0)
        unsafe_cost = torch.full_like(obst_d, 1.0 - self.beta)
        return torch.where(in_safe_set, safe_cost, unsafe_cost)

    def _terminated(self):
        x, y, th, vx, vy, w = self.state.unbind(dim=1)
        crashed = y <= 0.0 
        out_of_bounds = (x.abs() > WORLD_X) | (y > WORLD_Y)
        obst_d2 = (x - self.obstacle[:, 0]) ** 2 + (y - self.obstacle[:, 1]) ** 2
        hit_obstacle = obst_d2 < self.OBSTACLE_RADIUS_SQ
        #tipped = th.abs() > (math.pi / 2)
        return crashed | out_of_bounds | hit_obstacle #| tipped

    def _spawn_obstacle(self, spawn_xy, target_xy):
        n = spawn_xy.shape[0]
        random_mode = torch.rand(n, device=self.device) < 0.5

        # 50% uniform-random in the spawn box, 50% midpoint + Gaussian
        cx = TARGET_X_WIDTH * 2 * (torch.rand(n, device=self.device) - 0.5)
        cy = TARGET_Y_OFFSET + TARGET_Y_WIDTH * torch.rand(n, device=self.device)
        rand_pos = torch.stack([cx, cy], dim=1)

        mid_pos = (
            0.5 * (spawn_xy + target_xy)
            + OBSTACLE_MID_STD * torch.randn(n, 2, device=self.device)
        )

        obst = torch.where(random_mode.unsqueeze(1), rand_pos, mid_pos)

        spawn_clear_sq = (
            UNSAFE_REGION_OBSTACLE_RADIUS + OBSTACLE_SPAWN_CLEARANCE
        ) ** 2

        target_clear_sq = (
            UNSAFE_REGION_OBSTACLE_RADIUS + OBSTACLE_TARGET_CLEARANCE
        ) ** 2

        for _ in range(1000):
            too_close_to_spawn = ((obst - spawn_xy) ** 2).sum(dim=1) < spawn_clear_sq
            too_close_to_target = ((obst - target_xy) ** 2).sum(dim=1) < target_clear_sq

            invalid = too_close_to_spawn | too_close_to_target

            if not invalid.any():
                break

            # Only generate as many candidates as needed
            num_invalid = invalid.sum()
            cx = TARGET_X_WIDTH * 2 * (
                torch.rand(num_invalid, device=self.device) - 0.5
            )
            cy = TARGET_Y_OFFSET + TARGET_Y_WIDTH * torch.rand(
                num_invalid, device=self.device
            )

            obst[invalid] = torch.stack([cx, cy], dim=1)

        return obst

    def old_spawn_obstacle(self, spawn_xy, target_xy):
        n = spawn_xy.shape[0]
        random_mode = torch.rand(n, device=self.device) < 0.5

        # 50% uniform-random in the spawn box, 50% midpoint + gaussian
        cx = TARGET_X_WIDTH * 2 * (torch.rand(n, device=self.device) - 0.5)
        cy = TARGET_Y_OFFSET + TARGET_Y_WIDTH * torch.rand(n, device=self.device)
        rand_pos = torch.stack([cx, cy], dim=1)
        mid_pos = 0.5 * (spawn_xy + target_xy) + OBSTACLE_MID_STD * torch.randn(n, 2, device=self.device)
        obst = torch.where(random_mode.unsqueeze(1), rand_pos, mid_pos)

        # too close to spawn (either mode) -> resample randomly until clear
        clear_sq = (UNSAFE_REGION_OBSTACLE_RADIUS + OBSTACLE_SPAWN_CLEARANCE) ** 2
        for _ in range(200):
            too_close = ((obst - spawn_xy) ** 2).sum(dim=1) < clear_sq
            if not too_close.any():
                break
            cx = TARGET_X_WIDTH * 2 * (torch.rand(n, device=self.device) - 0.5)
            cy = TARGET_Y_OFFSET + TARGET_Y_WIDTH * torch.rand(n, device=self.device)
            cand = torch.stack([cx, cy], dim=1)
            obst = torch.where(too_close.unsqueeze(1), cand, obst)
        return obst

    def reset_idx(self, ids):
        n = ids.numel()
        new = torch.zeros(n, 6, device=self.device)
        new[:, 0] = TARGET_X_WIDTH * 2 * (torch.rand(n, device=self.device) - 0.5)   # x in [-1, 1]
        new[:, 1] = TARGET_Y_OFFSET + TARGET_Y_WIDTH * torch.rand(n, device=self.device)     # y in [1, 3], always airborne
        new[:, 2] = self.perturb * 2 * 0.3 * torch.randn(n, device=self.device)          # small tilt
        new[:, 3:6] = self.perturb * 2 * 0.5 * torch.randn(n, 3, device=self.device)     # small initial velocities
        self.state[ids] = new

        self.target[ids, 0] = TARGET_X_WIDTH * 2 * (torch.rand(n, device=self.device) - 0.5)   # x_target in [-2, 2]
        self.target[ids, 1] = TARGET_Y_OFFSET + TARGET_Y_WIDTH * torch.rand(n, device=self.device)     # y_target in [1, 3]

        self.obstacle[ids] = self._spawn_obstacle(new[:, :2], self.target[ids])

        self.episode_length_buf[ids] = 0
        self.prev_action[ids] = 0.0

    def reset(self):
        """Not in the abstract interface, but handy. Returns (obs, extras)."""
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_observations(), {}