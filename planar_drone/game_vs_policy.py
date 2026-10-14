"""Play the drone toward a target -- and hand control to the trained policy.

Built on VecDrone2D with num_envs=1 so the policy sees exactly the observation
(and action mapping) it was trained on. You fly with the keyboard; press X to
let the exported policy take over, X again to take back.

Run:  python game_vs_policy.py

Keys:
  W / UP      thrust up            (manual)
  S / DOWN    thrust down          (manual)
  A / LEFT    moment one way       (manual)
  D / RIGHT   moment other way     (manual)
  X           toggle MANUAL <-> POLICY
  R           reset (new start + target)
  Esc / Q     quit
"""

import math
import argparse

import torch
import pygame

import environment
import render
import log_handler
from eval_dashboard import draw_scene
from pathlib import Path
import train

#DEVICE      = train.device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

MANUAL_M    = 0.5          # manual moment magnitude (normalized, in [-1, 1])
# thrust: held W -> +1 (climb), held S -> -1 (fall), neither -> 0 (hover)


def main(exp_dir=None):
    env = environment.VecDrone2D(num_envs=1, device=DEVICE, use_curriculum=False, difficulty_if_no_curr=1.0)

    env.max_episode_length = 10**9        # effectively no timeout in free play

    project_dir = Path(__file__).resolve().parent
    logs_dir = project_dir / "logs"

    # Resolve which experiment directory to load the policy from.
    # If a path is given on the command line, use it (relative paths are
    # resolved against the project directory); otherwise use the latest.
    if exp_dir is not None:
        exp_path = Path(exp_dir)
        if not exp_path.is_absolute():
            exp_path = project_dir / exp_path
    else:
        exp_path = log_handler.latest_exp_dir(logs_dir)

    policy_path = exp_path / "exported" / "policy.pt"

    print(f"Loading exported policy {policy_path}")
    policy = torch.jit.load(str(policy_path), map_location=DEVICE)
    policy.eval()

    pygame.init()
    pygame.display.set_caption("Drone -- manual / policy")
    screen = pygame.display.set_mode((render.SCREEN_W, render.SCREEN_H))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)
    big = pygame.font.SysFont("monospace", 20, bold=True)

    fps = round(1.0 / (env.dt * env.decimation))     # real time

    policy_mode = True
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
                elif e.key == pygame.K_x:
                    policy_mode = not policy_mode          # hand over / take back
                elif e.key == pygame.K_r:
                    env.reset_idx(torch.zeros(1, dtype=torch.long, device=DEVICE))
                    ep_return = 0.0

        # --- decide the action ---
        if policy_mode:
            with torch.no_grad():
                action = policy(obs["policy"])             # policy flies
        else:
            keys = pygame.key.get_pressed()                # you fly
            a0 = 0.0
            if keys[pygame.K_w] or keys[pygame.K_UP]:
                a0 = 1.0
            elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
                a0 = -1.0
            a1 = 0.0
            if keys[pygame.K_a] or keys[pygame.K_LEFT]:
                a1 = MANUAL_M
            elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
                a1 = -MANUAL_M
            action = torch.tensor([[a0, a1]], device=DEVICE)

        # values for the HUD / arrow (same mapping the env uses)
        a0 = float(action[0, 0].clamp(-1.0, 1.0))
        a1 = float(action[0, 1].clamp(-1.0, 1.0))
        F_view = max(0.0, env.hover + a0 * env.thrust_scale)
        M_view = a1 * env.moment_scale

        obs, rewards, dones, extras = env.step(action)

        ep_return += float(rewards[0])
        if bool(dones[0]):
            ep_count += 1
            ep_return = 0.0

        # --- draw ---
        draw_scene(screen, font, env, 0, F_view, M_view, a0, a1, obs, ep_return, ep_count)

        # mode indicator (top-right)
        label = "MODE: POLICY  (X = take back)" if policy_mode else "MODE: MANUAL  (X = policy)"
        color = (40, 160, 40) if policy_mode else (200, 60, 60)
        surf = big.render(label, True, color)
        screen.blit(surf, (render.SCREEN_W - surf.get_width() - 16, 12))

        pygame.display.flip()
        clock.tick(fps)

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fly the drone manually or hand control to a trained policy.")
    parser.add_argument(
        "exp_dir",
        nargs="?",
        default=None,
        help="Relative (or absolute) path to the logs/expN directory to load the policy from. "
             "If omitted, the latest experiment is used.",
    )
    args = parser.parse_args()
    main(args.exp_dir)