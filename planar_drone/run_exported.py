"""Live dashboard driven by the EXPORTED policy (TorchScript), not a checkpoint.

Loads logs/<exp>/exported/policy.pt directly with torch.jit -- no rsl_rl runner,
no config file. Reuses the drawing code from eval_dashboard.py so the visuals
stay identical; only the policy *loading* and the policy *call* differ.

Usage:
  standalone   ->  python run_exported.py        (uses the newest experiment)
  from train   ->  run_exported.run(path, device, num_envs)
"""

import math
from collections import deque
from pathlib import Path

import torch
import pygame

import environment
import log_handler
from eval_dashboard import draw_plot, draw_scene, GRAPH_H, HIST, W, SCENE_H, WIN_H


def run(policy_path, device="cpu", num_envs=64):
    """Open the dashboard and fly the exported policy. Blocks until the window
    is closed (Esc / Q / close button)."""

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    policy_path = Path(policy_path)
    env = environment.VecDrone2D(num_envs=num_envs, device=device, use_curriculum=False, difficulty_if_no_curr=1.0)

    print(f"Loading exported policy {policy_path}")
    policy = torch.jit.load(str(policy_path), map_location=device)
    policy.eval()

    pygame.init()
    pygame.display.set_caption(f"Exported policy  -  {policy_path.parent.parent.name}")
    screen = pygame.display.set_mode((W, WIN_H))
    scene = screen.subsurface((0, GRAPH_H, W, SCENE_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    fps = round(1.0 / (env.dt * env.decimation))            # ~50 -> real time

    gap = 70
    pw = (W - 3 * gap) // 2
    ph = GRAPH_H - 46
    py = 26
    rect_act = (gap, py, pw, ph)
    rect_th  = (2 * gap + pw, py, pw, ph)

    buf_F  = deque(maxlen=HIST)
    buf_M  = deque(maxlen=HIST)
    buf_th = deque(maxlen=HIST)

    view_i = 0
    ep_return, ep_count = 0.0, 0

    obs = env.get_observations()
    running = True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.KEYDOWN:
                if e.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif e.key == pygame.K_n:
                    view_i = (view_i + 1) % env.num_envs
                    buf_F.clear(); buf_M.clear(); buf_th.clear()
                    ep_return, ep_count = 0.0, 0

        with torch.no_grad():
            actions = policy(obs["policy"])        # exported actor takes the actor obs group

        a0 = float(actions[view_i, 0].clamp(-1.0, 1.0))
        a1 = float(actions[view_i, 1].clamp(-1.0, 1.0))
        F_view = max(0.0, env.hover + a0 * env.thrust_scale)
        M_view = a1 * env.moment_scale

        buf_F.append(a0)
        buf_M.append(a1)
        buf_th.append(float(env.state[view_i, 2]))

        obs, rewards, dones, extras = env.step(actions)

        ep_return += float(rewards[view_i])
        if bool(dones[view_i]):
            ep_count += 1
            ep_return = 0.0
            buf_F.clear(); buf_M.clear(); buf_th.clear()

        screen.fill((237, 235, 230))
        draw_plot(screen, rect_act, [buf_F, buf_M],
                  [(60, 120, 220), (200, 60, 60)],
                  ["norm force", "norm moment"], -1.1, 1.1,
                  "normalized action", font)
        draw_plot(screen, rect_th, [buf_th],
                  [(40, 160, 40)], ["theta (rad)"],
                  -math.pi / 2, math.pi / 2, "angle theta", font)
        draw_scene(scene, font, env, view_i, F_view, M_view, a0, a1, obs, ep_return, ep_count)

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    exp = log_handler.latest_exp_dir("logs")               # newest experiment
    run(exp / "exported" / "policy.pt", device=device, num_envs=64)


if __name__ == "__main__":
    main()
