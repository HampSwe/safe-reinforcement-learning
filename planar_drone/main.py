import dynamics_old as dynamics
import time
import pygame
import render

def simulate():

    pygame.init()
    pygame.display.set_caption("Drone2D")
    screen = pygame.display.set_mode((render.SCREEN_W, render.SCREEN_H))
    clock = pygame.time.Clock()
    drone = dynamics.Drone2D(y=2, theta=0.2, F=9.81+0.1, dt=1e-3)   # F≈mg to roughly hover

    accumulator, running = 0.0, True
    while running:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        accumulator += clock.tick(120) / 1000.0
        while accumulator >= drone.dt:
            drone.iterate()
            accumulator -= drone.dt

        render.render(screen, drone)

    pygame.quit()


def main():
    simulate()


if __name__ == "__main__":
    main()