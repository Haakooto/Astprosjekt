import numpy as np
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem


class SolarSystem(SolarSystem):
    def plot_orb(self):
        p = 1000
        N = self.number_of_planets

        f = np.transpose(np.array([np.linspace(0, 2 * np.pi, p)] * N))
        a = np.array([self.semi_major_axes] * p)
        e = np.array([self.eccentricities] * p)
        omega = np.array([self.semi_major_axis_angles] * p) + np.pi

        R = a * (1 - e ** 2) / (1 + e * np.cos(f - omega))
        x = R * np.cos(f)
        y = R * np.sin(f)

        plt.plot(x, y)

    def diff_orb(self):
        from ivp import ExponentialDecay as ED

        e = self.eccentricities
        a = self.semi_major_axes
        omega = self.semi_major_axis_angles + np.pi
        r = self.initial_positions
        v = self.initial_velocities
        h = r[0] * v[1] - r[1] * v[0]

        start_angle = np.arctan(self.initial_positions[1] / self.initial_positions[0])
        start_angle = np.where(self.initial_positions[0] >= 0, start_angle, start_angle + np.pi)

        T = 2
        dt = 1e-6

        orbits = ED(a, e, h, omega)
        t, u = orbits.solve(start_angle, T, dt)

        R = a * (1 - e ** 2) / (1 + e * np.cos(u - omega))

        self.X = R * np.cos(u)
        self.Y = R * np.sin(u)

        plt.plot(self.X, self.Y)


    def accelerate(self, r):
        return self.m * r * (np.linalg.norm(r, axis=0)) ** (-3)

    def simulate(self, T, dt):
        self.T = T
        self.dt = dt
        self.nt = int(T / dt)

        self.time = np.linspace(0, T, self.nt)

        self.pos = np.zeros((2, self.number_of_planets, self.nt))
        self.pos[:, :, 0] = self.initial_positions

        self.m = -const.G_sol * self.star_mass

        self.vel = self.initial_velocities

        self.acc_0 = self.accelerate(self.pos[:, :, 0])

        for t in range(self.nt - 1):
            self.pos[:, :, t + 1] = (
                self.pos[:, :, t]
                + self.vel * self.dt
                + 0.5 * self.acc_0 * self.dt ** 2
            )

            self.acc_1 = self.accelerate(self.pos[:, :, t + 1])
            self.vel = self.vel + 0.5 * (self.acc_0 + self.acc_1) * self.dt

            self.acc_0 = self.acc_1

    def load_pos(self, filename):
        self.pos = np.load(filename)


if __name__ == "__main__":
    seed = util.get_seed("haakooto")

    mission = SpaceMission(seed)
    system = SolarSystem(seed)

    system.diff_orb()

    system.plot_orb()
    year_conv = system.rotational_periods[0]
    years = 20
    dt = year_conv / 1e5

    # system.simulate(years * year_conv, dt)
    system.load_pos("backup_20yr.npy")

    X = system.pos
    for i in range(system.number_of_planets):
        plt.plot(X[0, i, :], X[1, i, :])
    plt.scatter(*system.initial_positions)
    plt.scatter(X[0, :, -1], X[1, :, -1])
    plt.scatter(system.X[-1], system.Y[-1])
    plt.axis("equal")
    plt.grid()
    plt.show()

    #np.save("planets_pos_20yr", system.pos)

    #system.verify_planet_positions(years * year_conv, system.pos)
