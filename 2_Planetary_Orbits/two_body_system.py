"""
Program for å simulere 1 2-legemesystem

All kode er egenskrevet

"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

font = {"family": "DejaVu Sans", "weight": "normal", "size": 22}

plt.rc("font", **font)


class SolarSys(SolarSystem):
    """
    Mye av det samme som SolarSys i orbits.py
    Her går planetene i bane rundt massesenter, ikke stjerna.
    bruker Leapfrog istdenfor spinnbevaring
    Helt lik klassen i n_body_system, men kun for 1 planet.
    """

    def __init__(self, seed, data_path=None, has_moons=True, verbose=True):
        SolarSystem.__init__(self, seed, data_path=None, has_moons=True, verbose=True)
        self.one_year = np.sqrt(self.semi_major_axes[0] ** 3 / self.star_mass)
        self.one_years = np.sqrt(self.semi_major_axes ** 3 / self.star_mass)

    def find_largest_attractor(self):
        F = (
            -const.G_sol
            * self.star_mass
            * self.masses
            * np.linalg.norm(self.initial_positions, axis=0) ** -2
        )
        return np.argmin(F)

    def two_body_system(self, yrs, dt_pr_yr):
        def A(r):
            R = r[0] - r[1]
            F = -const.G_sol * R * np.linalg.norm(R) ** -3
            return np.array([F * p_mass, -F * self.star_mass])

        T = self.one_year * yrs
        dt = self.one_year * dt_pr_yr
        n = int(T / dt)

        self.time = np.linspace(0, T, n)

        la_idx = self.find_largest_attractor()
        p_mass = self.masses[la_idx]

        # Position and velocity of centre of mass
        Rcm = p_mass * self.initial_positions[:, la_idx] / (p_mass + self.star_mass)
        Vcm = p_mass * self.initial_velocities[:, la_idx] / (p_mass + self.star_mass)

        R = np.zeros((n, 2, 2))  # [time][object][coord]
        R[0, 0] = -Rcm  # moving sun
        R[0, 1] = self.initial_positions[:, la_idx] - Rcm  # setting planet pos

        V = np.zeros((n, 2, 2))  # [time][object][coord]
        V[0, 0] = -Vcm  #
        V[0, 1] = self.initial_velocities[:, la_idx] - Vcm  # setting planet vel

        for t in range(n - 1):
            # Leapfrogintegration
            a0 = A(R[t])
            R[t + 1] = R[t] + V[t] * dt + 0.5 * a0 * dt ** 2
            a1 = A(R[t + 1])
            V[t + 1] = V[t] + 0.5 * (a0 + a1) * dt

        self.solar_orb = R[:, 0]
        self.planet_orb = R[:, 1]
        self.vel = V
        self.la_idx = la_idx
        self.p_mass = p_mass

    def plot_two_pos(self):
        S = np.transpose(self.solar_orb)
        P = np.transpose(self.planet_orb)

        plt.plot(*S, color=np.array(self.star_color) / 255, label="Sun")
        plt.plot(*P, "c", label="Planet")
        plt.scatter([0], [0], label="Centre of mass")

        plt.grid()
        plt.axis("equal")
        plt.title("Two-body system around centre of mass")
        plt.legend(loc=1)
        plt.show()

    def radial_vel(self, i=np.pi / 2):
        xvel = np.sin(i) * self.vel[:, 0, 0]
        noise = np.random.normal(0, 0.2 * xvel.max(), len(xvel))
        V = 1444  # Peculiar velocity
        self.vnoise = xvel + noise + V

    def energy_conserve(self):
        relvel = np.linalg.norm(self.vel[:, 0] - self.vel[:, 1], axis=1)
        relpos = np.linalg.norm(self.solar_orb - self.planet_orb, axis=1)

        mu_hat = self.star_mass * self.p_mass / (self.star_mass + self.p_mass)

        self.E = (
            0.5 * mu_hat * relvel ** 2
            - const.G_sol * (self.star_mass + self.p_mass) * mu_hat / relpos
        )
        plt.plot(self.time, self.E)
        plt.title("Energy in two-body system")
        plt.xlabel("time")
        plt.ylabel("Total energy")
        plt.show()

    def light_curve(self):
        def section_area(h):
            return r ** 2 * np.arccos((r - h) / r) - (r - h) * np.sqrt(
                2 * r * h - h ** 2
            )  # equation 13 in Simulating planetary orbits

        try:
            la_idx = self.la_idx
        except:
            la_idx = self.find_largest_attractor()

        R = util.km_to_AU(self.star_radius)
        star_area = np.pi * R ** 2
        r = util.km_to_AU(self.radii[la_idx])

        velo = np.linalg.norm(self.initial_velocities[:, la_idx])

        partial_time = 2 * r / velo
        crossing_time = 2 * R / velo - partial_time

        Ps = 10000

        time_before = np.linspace(0, 5 * partial_time, Ps)
        time_enter = np.linspace(5 * partial_time, 6 * partial_time, Ps / 5)
        time_cross = np.linspace(6 * partial_time, 6 * partial_time + crossing_time, Ps)
        time_exit = np.linspace(time_cross[-1], time_cross[-1] + partial_time, Ps / 5)
        time_after = np.linspace(time_exit[-1], time_exit[-1] + time_before[-1], Ps)

        before = np.ones(Ps)
        after = np.ones_like(before)
        enter = 1 - section_area(2 * r * np.linspace(0, 1, Ps / 5)) / star_area
        exit = enter[::-1]
        cross = np.array([enter[-1]] * Ps)

        self.full_time = np.concatenate(
            (time_before, time_enter, time_cross, time_exit, time_after)
        )
        light_curve = np.concatenate((before, enter, cross, exit, after))

        self.light_curve = light_curve + np.random.normal(0, 0.2, len(light_curve))

        plt.plot(self.full_time, self.light_curve)
        plt.show()

    def assemble_data(self):
        try:
            rvs = np.concatenate(([self.time], [self.vnoise]))
            np.save("npys/radial_velocity_curve_single.npy", rvs)
        except:
            pass

        try:
            lc = np.concatenate(([self.full_time], [self.light_curve]))
            np.save("npys/light_curve.npy", lc)
        except:
            pass

        try:
            info = np.array(
                [self.p_mass, self.radii[self.la_idx], self.star_radius, self.star_mass]
            )
            np.save("npys/info.npy", info)
        except:
            pass


def planet_mass(vs, P, ms, i=np.pi / 2):
    return (
        ms ** (2 / 3)
        * vs
        * (2 * np.pi * const.G_sol) ** (-1 / 3)
        * P ** (1 / 3)
        / np.sin(i)
    )


if __name__ == "__main__":
    seed = 76117

    system = SolarSys(seed)

    yrs = 80
    dt = 1e-4

    system.two_body_system(yrs, dt)
    system.plot_two_pos()
    system.energy_conserve()
    system.radial_vel(i=2 * np.pi / 3)
    system.light_curve()
    system.assemble_data()
