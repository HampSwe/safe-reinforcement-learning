import math

class Drone2D:
    def __init__(self, x=0, y=0, theta=0, x_prime=0, y_prime=0, theta_prime=0,
                 M=0, F=0,
                 m=1, J=1, g=9.81, dt=1e-3,
                 ):
        self.x = x
        self.y = y
        self.theta = theta
        self.x_prime = x_prime
        self.y_prime = y_prime
        self.theta_prime = theta_prime
        self.M = M
        self.F = F
        self.m = m
        self.J = J
        self.g = g
        self.dt = dt

    def iterate(self):
        self.x_prime = self.x_prime + self.dt * (1/self.m) * math.sin(self.theta) * self.F
        self.y_prime = self.y_prime + self.dt * ((1/self.m) * math.cos(self.theta) * self.F - self.g)
        self.theta_prime = self.theta_prime + self.dt * (- self.M / self.J)

        self.theta = self.theta + self.dt * self.theta_prime
        self.x = self.x + self.dt * self.x_prime
        self.y = self.y + self.dt * self.y_prime

        if self.y < 0:
            self.y = 0
            self.y_prime = 0
            self.x_prime = 0
            self.theta_prime = 0
            self.theta = 0
