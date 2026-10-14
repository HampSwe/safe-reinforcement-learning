"""Live policy dashboard: rolling graphs on top, drone view below.

Top strip:  normalized force + moment (left panel),  theta (right panel),
            both scrolling in real time.
Below:      the drone flying toward its target, with a physical/normalized HUD.

Graphs are drawn natively in pygame (rolling buffers -> polylines) so the whole
thing stays smooth at real-time speed -- no matplotlib per frame.

Run:  python eval_dashboard.py
Keys: N = next env   |   Esc / Q = quit
"""

import math
from collections import deque
from pathlib import Path

import torch
import pygame
import yaml

import environment
import render
import log_handler
from rsl_rl.runners import OnPolicyRunner


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
DEVICE   = "cuda:0" if torch.cuda.is_available() else "cpu"   # auto: GPU if present, else CPU
NUM_ENVS = 64
LOG_DIR  = None                    # None -> use the latest experiment in logs/
CONFIG   = Path(__file__).resolve().parent / "config" / "config_rlsgf.yaml"
ARM      = 0.3                     # drone half-span (metres), matches render.py

GRAPH_H  = 150                     # height of the top graph strip (px) -- smaller
HIST     = 250                     # points kept in the rolling window (~5 s)

W        = render.SCREEN_W
SCENE_H  = render.SCREEN_H
WIN_H    = GRAPH_H + SCENE_H


def latest_checkpoint(log_dir):
    ckpts = list(Path(log_dir).glob("model_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No model_*.pt checkpoints in {log_dir}")
    return str(max(ckpts, key=lambda p: int(p.stem.split("_")[1])))


# --------------------------------------------------------------------------- #
# a single rolling line-plot panel
# --------------------------------------------------------------------------- #
def draw_plot(surf, rect, series, colors, labels, ymin, ymax, title, font):
    px, py, pw, ph = rect
    pygame.draw.rect(surf, (255, 255, 255), rect)
    pygame.draw.rect(surf, (180, 180, 180), rect, 1)

    # zero reference line
    if ymin < 0 < ymax:
        zy = py + ph - int((0 - ymin) / (ymax - ymin) * ph)
        pygame.draw.line(surf, (215, 215, 215), (px, zy), (px + pw, zy), 1)

    # each data series as a polyline
    for s, col in zip(series, colors):
        n = len(s)
        if n < 2:
            continue
        pts = []
        for i, v in enumerate(s):
            x = px + int(i / (n - 1) * pw)
            vv = max(ymin, min(ymax, v))                 # clamp for drawing
            y = py + ph - int((vv - ymin) / (ymax - ymin) * ph)
            pts.append((x, y))
        pygame.draw.lines(surf, col, False, pts, 2)

    # title + inline legend
    surf.blit(font.render(title, True, (40, 40, 40)), (px, py - 17))
    lx = px + 6
    for lab, col in zip(labels, colors):
        surf.blit(font.render(lab, True, col), (lx, py + 3))
        lx += font.size(lab)[0] + 16


# --------------------------------------------------------------------------- #
# the drone scene (drawn into a subsurface; no flip here)
# --------------------------------------------------------------------------- #
def draw_scene(surf, font, env, i, F_view, M_view, a0, a1, obs, ep_return, ep_count):
    BG, INK, ROTOR, THRUST, TGT, OBST = (
        (247, 245, 240), (30, 30, 30), (200, 60, 60), (60, 120, 220), (40, 160, 40), (150, 60, 60)
    )
    surf.fill(BG)

    gy = render.to_screen(0, 0)[1]
    pygame.draw.line(surf, INK, (0, gy), (render.SCREEN_W, gy), 2)

    # obstacle (filled circle, drawn at its true collision radius)
    ox_w, oy_w = env.obstacle[i].tolist()
    ocx, ocy = render.to_screen(ox_w, oy_w)
    orad_px = max(2, int((env.OBSTACLE_RADIUS_SQ ** 0.5) * render.SCALE))

    # unsafe margin ring (orange outline), drawn first so the obstacle sits on top
    ORANGE = (255, 140, 0)
    unsafe_rad_px = max(2, int(environment.UNSAFE_REGION_OBSTACLE_RADIUS * render.SCALE))
    pygame.draw.circle(surf, ORANGE, (int(ocx), int(ocy)), unsafe_rad_px, 2)   # width=2 → outline only

    pygame.draw.circle(surf, OBST, (int(ocx), int(ocy)), orad_px)

    # target cross (sizes scale with render.ANIM_SCALE)
    tx_w, ty_w = env.target[i].tolist()
    tcx, tcy = render.to_screen(tx_w, ty_w)
    tr = max(2, int(9 * render.ANIM_SCALE))
    tl = max(3, int(13 * render.ANIM_SCALE))
    pygame.draw.circle(surf, TGT, (int(tcx), int(tcy)), tr, 2)
    pygame.draw.line(surf, TGT, (tcx - tl, tcy), (tcx + tl, tcy), 1)
    pygame.draw.line(surf, TGT, (tcx, tcy - tl), (tcx, tcy + tl), 1)

    # drone (line width + rotor radius scale with render.ANIM_SCALE)
    x, y, th, vx, vy, w = env.state[i].tolist()
    bx, by = math.cos(th), -math.sin(th)
    tx, ty = math.sin(th),  math.cos(th)

    left  = render.to_screen(x - ARM * bx, y - ARM * by)
    right = render.to_screen(x + ARM * bx, y + ARM * by)
    pygame.draw.line(surf, INK, left, right, render.ARM_W)
    pygame.draw.circle(surf, ROTOR, (int(left[0]),  int(left[1])),  render.ROTOR_R)
    pygame.draw.circle(surf, ROTOR, (int(right[0]), int(right[1])), render.ROTOR_R)

    # thrust arrow (defaults to render.ARROW_W / ARROW_HEAD, which scale)
    disp = F_view * 0.05
    tip = (x + tx * disp, y + ty * disp)
    render.draw_arrow(surf, THRUST, render.to_screen(x, y), render.to_screen(*tip))

    p = obs["policy"][i].tolist()    # [ex, ey, sin, cos, vx/5, vy/5, w/5]
    lines = [
        f"env {i}   episode {ep_count}   return {ep_return:8.1f}",
        "",
        "PHYSICAL",
        f"  x {x:7.2f}   y {y:7.2f}   th {th:7.3f}",
        f" vx {vx:7.2f}  vy {vy:7.2f}    w {w:7.3f}",
        f"  F {F_view:7.2f}    M {M_view:7.2f}",
        f"target ({tx_w:5.2f}, {ty_w:5.2f})",
        "",
        "NORMALIZED (policy input)",
        f" ex {p[0]:7.3f}  ey {p[1]:7.3f}",
        f"sin {p[2]:7.3f} cos {p[3]:7.3f}",
        f"vx~ {p[4]:7.3f} vy~ {p[5]:7.3f}  w~ {p[6]:7.3f}",
        "",
        "ACTION (normalized output)",
        f" F~ {a0:7.3f}   M~ {a1:7.3f}",
    ]
    for k, t in enumerate(lines):
        surf.blit(font.render(t, True, INK), (12, 10 + 18 * k))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():

    with open(CONFIG, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f)["runner"]
    
    env = environment.VecDrone2D(num_envs=NUM_ENVS, device=DEVICE, use_curriculum=False, difficulty_if_no_curr=1.0)

    runner = OnPolicyRunner(env=env, train_cfg=train_cfg, device=DEVICE)

    # resolve which experiment to load: explicit LOG_DIR, else the latest one
    if LOG_DIR is not None:
        log_dir = Path(LOG_DIR)
    else:
        log_dir = log_handler.latest_exp_dir(Path(__file__).resolve().parent / "logs")
    ckpt = latest_checkpoint(log_dir)
    print(f"Loading {ckpt}")
    runner.load(ckpt)
    policy = runner.get_inference_policy()

    pygame.init()
    pygame.display.set_caption("Drone policy dashboard")
    screen = pygame.display.set_mode((W, WIN_H))
    scene = screen.subsurface((0, GRAPH_H, W, SCENE_H))     # drone region
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    fps = round(1.0 / (env.dt * env.decimation))            # ~50 -> real time

    # graph panel rectangles (left = actions, right = theta)
    gap = 70
    pw = (W - 3 * gap) // 2
    ph = GRAPH_H - 46
    py = 26
    rect_act = (gap, py, pw, ph)
    rect_th  = (2 * gap + pw, py, pw, ph)

    # rolling buffers
    buf_F = deque(maxlen=HIST)
    buf_M = deque(maxlen=HIST)
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
            actions = policy(obs)
        a0 = float(actions[view_i, 0].clamp(-1.0, 1.0))
        a1 = float(actions[view_i, 1].clamp(-1.0, 1.0))
        F_view = max(0.0, env.hover + a0 * env.thrust_scale)
        M_view = a1 * env.moment_scale

        # record (normalized action + current angle), then step
        buf_F.append(a0)
        buf_M.append(a1)
        buf_th.append(float(env.state[view_i, 2]))

        obs, rewards, dones, extras = env.step(actions)

        ep_return += float(rewards[view_i])
        if bool(dones[view_i]):
            ep_count += 1
            ep_return = 0.0
            buf_F.clear(); buf_M.clear(); buf_th.clear()   # clean per-episode trace

        # ---- draw everything, one flip ----
        screen.fill((237, 235, 230))                       # graph-strip background
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


if __name__ == "__main__":
    main()