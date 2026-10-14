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

# --- square world:  x in [-WORLD_X, WORLD_X],  y in [0, WORLD_Y] ---
WORLD_X = 10   # half-width, so full x-range is [-10, 10]  (width 20)
WORLD_Y = 20   # y in [0, 20]  -> square 20 x 20

FAILURE_PENALTY = 10000
TARGET_BONUS = int(100 * 4)
TARGET_DIS_SCALE = 1
TARGET_RADIUS = 0.05

OBSTACLE_RADIUS = 0 #0.5
UNSAFE_REGION_OBSTACLE_RADIUS = OBSTACLE_RADIUS * 1.25

# ------------------------------------------------------------------ #
# new spawn scheme
#   - obstacle fixed at CENTER (+ small jitter)
#   - drone placed randomly on the screen (+ small jitter)
#   - target on the OPPOSITE side of the center from the drone, at a
#     random distance from the center within [TARGET_DIST_MIN, MAX]
#   - drone and target must clear the obstacle by their margins; if not,
#     resample everything (up to 100 tries)
# ------------------------------------------------------------------ #
CENTER = (0.0, WORLD_Y / 2.0)          # obstacle nominal center (x, y) = (0, 10)

DRONE_SPAWN_STD    = 0.05              # jitter on drone position (start tiny)
OBSTACLE_SPAWN_STD = 1             # jitter on obstacle position
TARGET_SPAWN_STD   = 0.05              # jitter on target position

TARGET_DIST_MIN = 0.5                  # target distance from center, min
TARGET_DIST_MAX = 8.0                  # target distance from center, max

# min clearance between the obstacle and the drone / target
DRONE_MARGIN  = UNSAFE_REGION_OBSTACLE_RADIUS + 0.2
TARGET_MARGIN = UNSAFE_REGION_OBSTACLE_RADIUS + 0.2

# drone spawn box (kept inside the walls / off the floor & ceiling)
SPAWN_HALF_X = WORLD_X - 1.0           # x in [-9, 9]
SPAWN_Y_MIN, SPAWN_Y_MAX = 1.0, WORLD_Y - 1.0   # y in [1, 19]

R0_SCALER = 1e-3


class VecDrone2D:
    def __init__(self, num_envs, device, cfg=None, subtract_r1_from_r0=False, r1_multiplier=1, perturb=0, beta=5e-2, k=0.5):
        # --- required VecEnv attributes ---
        self.num_envs = num_envs
        self.num_actions = 2                       # [thrust, moment]
        self.device = torch.device(device)
        self.cfg = cfg or {}
        self.max_episode_length = 400             # control steps per episode
        self.episode_length_buf = torch.zeros(
            num_envs, dtype=torch.long, device=self.device)

        # --- physics params ---
        self.m, self.J, self.g = 1.0, 0.5, 9.81
        self.dt = 1e-3                             # physics step
        self.decimation = 20                      # physics steps per control step
        self.hover = self.m * self.g

        self.TARGET_RADIUS_SQ = TARGET_RADIUS**2
        self.OBSTACLE_RADIUS_SQ = OBSTACLE_RADIUS**2
        self.UNSAFE_REGION_OBSTACLE_RADIUS = UNSAFE_REGION_OBSTACLE_RADIUS
        self.UNSAFE_REGION_OBSTACLE_RADIUS_SQUARED = UNSAFE_REGION_OBSTACLE_RADIUS**2
        self.beta = beta
        self.k = k
        self.subtract_r1_from_r0 = subtract_r1_from_r0
        self.r1_multiplier = r1_multiplier # this is only applicable when r1 is subtracted from r0
        self.perturb = perturb

        self.prev_action = torch.zeros(num_envs, self.num_actions, device=self.device)

        self.ACTION_HIST = 10
        self.action_hist = torch.zeros(num_envs, self.ACTION_HIST, self.num_actions, device=self.device)
        self.VAR_COEF = 5.0    # tune

        # --- action scaling: raw policy output (~[-1,1]) -> F, M ---
        self.thrust_scale = self.hover
        self.moment_scale = 2.0

        # --- target / obstacle buffers ---
        self.target = torch.zeros(num_envs, 2, device=self.device)
        self.obstacle = torch.zeros(num_envs, 2, device=self.device)

        # --- state: [x, y, theta, vx, vy, omega], shape (num_envs, 6) ---
        self.state = torch.zeros(num_envs, 6, device=self.device)
        self.reset_idx(torch.arange(num_envs, device=self.device))

        # compile the physics hot path for kernel fusion — GPU only
        if self.device.type != "cpu":
            self._integrate = torch.compile(self._integrate)
            self._reward   = torch.compile(self._reward)
            self.r1_reward = torch.compile(self.r1_reward)

        # # compile the physics hot path for kernel fusion
        # self._integrate = torch.compile(self._integrate)
        # #self._integrate = torch.compile(self._integrate, mode="reduce-overhead")
        # self._reward   = torch.compile(self._reward)
        # self.r1_reward = torch.compile(self.r1_reward)

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

        self.episode_length_buf += 1

        terminated = self._terminated()
        timed_out = self.episode_length_buf >= self.max_episode_length
        dones = terminated | timed_out

        # --- update action history + variance (uncompiled, stateful) ---
        self.action_hist = torch.roll(self.action_hist, shifts=-1, dims=1)
        self.action_hist[:, -1] = actions
        action_var = self.action_hist.var(dim=1).sum(dim=1)          # (N,)

        r0 = self._reward(actions)
        r0 = r0 - self.VAR_COEF * action_var * R0_SCALER            # <-- add penalty here
        r1 = self.r1_reward(actions)

        # should make r1 penalties continuous probably
        #r0 = self._reward(actions)
        #r1 = self.r1_reward(actions)

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

    def try_reward(self, actions):
        x, y, th, vx, vy, w = self.state.unbind(dim=1)

        pos_err_sq = (
            (x - self.target[:, 0]) ** 2
            + (y - self.target[:, 1]) ** 2
        )
        dist = torch.sqrt(pos_err_sq + 1e-8)

        speed_sq = vx**2 + vy**2
        action_rate = ((actions - self.prev_action) ** 2).sum(dim=1)

        near_target = torch.exp(-dist / 0.25)

        reward = (
            - 3.0 * dist
            + 20.0 * near_target
            - 1.0 * speed_sq
            - 0.5 * w**2
            - 0.25 * action_rate
            - 2.0 * near_target * speed_sq
        )

        return reward * R0_SCALER


    def _reward(self, actions):
        x, y, th, vx, vy, w = self.state.unbind(dim=1)
        pos_err = (x - self.target[:, 0]) ** 2 + (y - self.target[:, 1]) ** 2
        close = pos_err < self.TARGET_RADIUS_SQ
        rate = ((actions - self.prev_action) ** 2).sum(dim=1)
        linear_pos_err = torch.sqrt(pos_err)

        thrust = actions[:, 0]
        moment = actions[:, 1]

        return (
            + 0.1 * (28**2-pos_err)
            + 3 * (28-linear_pos_err)
            #+ close.float() * TARGET_BONUS #could add back
            - 4 * thrust ** 2 # PPO wants, was 3 before
            - 4 * moment ** 2 # PPO wants, was 3 before
            - 6 * rate
            - 3 * w ** 2 # was 0.5 before #PPO wants
            - 1.0 * (vx ** 2 + vy ** 2) # was 0.5 before
            #- 1.0 * (1.0 - torch.cos(th))
            #+ 3 * (20-linear_pos_err)
            + TARGET_BONUS*torch.exp(-linear_pos_err / TARGET_DIS_SCALE)
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

    # ------------------------------------------------------------------ #
    # spawn / reset
    # ------------------------------------------------------------------ #
    def reset_idx(self, ids):
        n = ids.numel()

        # obstacle nominal center (same for every env), as an (n, 2) tensor
        cx = torch.full((n,), CENTER[0], device=self.device)
        cy = torch.full((n,), CENTER[1], device=self.device)
        center = torch.stack([cx, cy], dim=1)                      # (n, 2)

        drone_xy  = torch.zeros(n, 2, device=self.device)
        target_xy = torch.zeros(n, 2, device=self.device)
        obst_xy   = torch.zeros(n, 2, device=self.device)

        # retry loop: for envs that violate a margin, resample EVERYTHING
        remaining = torch.ones(n, dtype=torch.bool, device=self.device)
        for _ in range(100):
            m = int(remaining.sum().item())
            if m == 0:
                break
            idx = remaining.nonzero(as_tuple=False).flatten()      # (m,)

            # obstacle: center + small jitter
            o = center[idx] + OBSTACLE_SPAWN_STD * torch.randn(m, 2, device=self.device)

            # drone: random on screen + small jitter
            dx = SPAWN_HALF_X * 2 * (torch.rand(m, device=self.device) - 0.5)
            dy = SPAWN_Y_MIN + (SPAWN_Y_MAX - SPAWN_Y_MIN) * torch.rand(m, device=self.device)
            d = torch.stack([dx, dy], dim=1) + DRONE_SPAWN_STD * torch.randn(m, 2, device=self.device)

            # target: opposite side of center from the drone, random distance from center
            dir_vec = center[idx] - d                              # drone -> center
            dir_norm = dir_vec / (dir_vec.norm(dim=1, keepdim=True) + 1e-8)
            dist = TARGET_DIST_MIN + (TARGET_DIST_MAX - TARGET_DIST_MIN) * torch.rand(m, 1, device=self.device)
            t = center[idx] + dir_norm * dist + TARGET_SPAWN_STD * torch.randn(m, 2, device=self.device)

            # margins: drone and target must clear the obstacle
            drone_ok  = ((d - o) ** 2).sum(dim=1) >= DRONE_MARGIN ** 2
            target_ok = ((t - o) ** 2).sum(dim=1) >= TARGET_MARGIN ** 2
            ok = drone_ok & target_ok

            good = idx[ok]
            drone_xy[good]  = d[ok]
            target_xy[good] = t[ok]
            obst_xy[good]   = o[ok]
            remaining[good] = False

        # build the state for these envs
        new = torch.zeros(n, 6, device=self.device)
        new[:, 0:2] = drone_xy
        new[:, 2]   = self.perturb * 2 * 0.3 * torch.randn(n, device=self.device)   # small tilt
        new[:, 3:6] = self.perturb * 2 * 0.5 * torch.randn(n, 3, device=self.device) # small initial velocities

        self.state[ids]    = new
        self.target[ids]   = target_xy
        self.obstacle[ids] = obst_xy

        self.episode_length_buf[ids] = 0     # <-- ADD: reset the timeout counter
        self.prev_action[ids] = 0.0          # <-- ADD: reset prev action
        self.action_hist[ids] = 0.0        # <-- add here

    def reset(self):
        self.episode_length_buf[:] = 0
        self.reset_idx(torch.arange(self.num_envs, device=self.device))
        return self.get_observations()