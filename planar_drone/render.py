import pygame
import math
import environment

SCALE  = 38 #60             # px per metre (your zoom knob)
MARGIN = 40             # px of breathing room below y=0

# Single multiplier for every pixel-sized drawing element (rotor dots, line
# widths, arrowheads, target cross). Anchored so that SCALE=60 reproduces the
# original hand-tuned sizes; smaller/larger SCALE scales them proportionally.
ANIM_SCALE = 1.25 * SCALE / 60

SCREEN_W = int(2 * environment.WORLD_X * SCALE)            # width fits the x-range
SCREEN_H = int(environment.WORLD_Y * SCALE) + MARGIN       # height fits y-range + margin
ORIGIN   = (SCREEN_W // 2, SCREEN_H - MARGIN)  # world (0,0): centred in x, MARGIN above bottom

# --- pixel sizes, all derived from ANIM_SCALE (edit the base numbers to taste) ---
ROTOR_R    = max(1, int(6  * ANIM_SCALE))   # rotor dot radius
ARM_W      = max(1, int(6  * ANIM_SCALE))   # drone-arm line width
ARROW_W    = max(1, int(3  * ANIM_SCALE))   # thrust-arrow line width
ARROW_HEAD = max(3, int(10 * ANIM_SCALE))   # thrust-arrow head length


def to_screen(wx, wy):
    """World metres -> screen pixels, flipping y so +y points up."""
    return (ORIGIN[0] + wx * SCALE, ORIGIN[1] - wy * SCALE)


def draw_arrow(surf, color, start, end, width=ARROW_W, head=ARROW_HEAD):
    pygame.draw.line(surf, color, start, end, width)
    ang = math.atan2(end[1] - start[1], end[0] - start[0])
    for da in (math.radians(150), math.radians(-150)):
        hx = end[0] + head * math.cos(ang + da)
        hy = end[1] + head * math.sin(ang + da)
        pygame.draw.line(surf, color, end, (hx, hy), width)


def render(surf, drone, arm=0.3):
    surf.fill((247, 245, 240))
    gy = to_screen(0, 0)[1]
    pygame.draw.line(surf, (40, 40, 40), (0, gy), (SCREEN_W, gy), 2)

    th = drone.theta
    cx, cy = drone.x, drone.y

    # paper convention: thrust along (sin θ, cos θ); body arm ⊥ to it = (cos θ, -sin θ)
    bx, by = math.cos(th), -math.sin(th)        # body-arm direction
    tx, ty = math.sin(th),  math.cos(th)        # thrust direction

    left  = to_screen(cx - arm * bx, cy - arm * by)
    right = to_screen(cx + arm * bx, cy + arm * by)
    pygame.draw.line(surf, (30, 30, 30), left, right, ARM_W)
    pygame.draw.circle(surf, (200, 60, 60), (int(left[0]),  int(left[1])),  ROTOR_R)
    pygame.draw.circle(surf, (200, 60, 60), (int(right[0]), int(right[1])), ROTOR_R)

    disp = drone.F * 0.05
    tip  = (cx + tx * disp, cy + ty * disp)
    draw_arrow(surf, (60, 120, 220), to_screen(cx, cy), to_screen(*tip))

    pygame.display.flip()