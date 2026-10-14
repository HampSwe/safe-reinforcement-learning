"""Manual keyboard test for the drone dynamics.

Not part of the real pipeline -- a sandbox to fly the drone by hand and
sanity-check the physics/rendering. Run it directly:

    python game.py
"""

import pygame
import dynamics_old as dynamics
import render

"""Keyboard control for the 2D drone -- the interactive 'play' layer.

Kept separate from the physics (dynamics.py) and the drawing (render.py):
this module's only job is to translate keyboard state into the drone's
inputs, F (thrust) and M (moment).

Two kinds of input, read two different ways:
  - held keys (thrust / moment)  -> pygame.key.get_pressed(), polled per frame
  - discrete keys (reset / quit) -> the event queue, fires once per press
"""

# --- tuning knobs ---
F_UP_FACTOR = 1.9       # held W / UP   -> climb   (multiple of hover thrust)
F_DOWN_FACTOR = 0.3     # held S / DOWN -> descend
M_CTRL = 2.0            # moment magnitude while turning


def handle_events(drone):
    """Process discrete events (quit, reset). Return False to stop the app."""
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            return False
        if e.type == pygame.KEYDOWN:
            if e.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if e.key == pygame.K_r:
                reset(drone)
    return True


def apply_keys(drone):
    """Read held keys once and set the drone's inputs (zero-order hold)."""
    keys = pygame.key.get_pressed()
    hover = drone.m * drone.g

    if keys[pygame.K_w] or keys[pygame.K_UP]:
        drone.F = F_UP_FACTOR * hover
    elif keys[pygame.K_s] or keys[pygame.K_DOWN]:
        drone.F = F_DOWN_FACTOR * hover
    else:
        drone.F = hover                 # neutral = hover
        drone.F = 0

    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        drone.M = M_CTRL
    elif keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        drone.M = -M_CTRL
    else:
        drone.M = 0.0


def reset(drone):
    """Return the drone to a hovering start state."""
    drone.x, drone.y, drone.theta = 0.0, 2.0, 0.0
    drone.x_prime = drone.y_prime = drone.theta_prime = 0.0


# Game start

def play():
    pygame.init()
    pygame.display.set_caption("Drone Game")
    screen = pygame.display.set_mode((render.SCREEN_W, render.SCREEN_H))
    clock = pygame.time.Clock()
    drone = dynamics.Drone2D(y=0, theta=0, dt=1e-3)

    accumulator, running = 0.0, True
    while running:
        running = handle_events(drone)   # discrete keys (quit, reset)
        apply_keys(drone)                # held keys -> F, M

        accumulator += clock.tick(120) / 1000.0
        while accumulator >= drone.dt:
            drone.iterate()
            accumulator -= drone.dt

        render.render(screen, drone)

    pygame.quit()


if __name__ == "__main__":
    play()