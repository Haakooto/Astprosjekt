"""
Klasse for å finne og plotte planetposisjoner numerisk
Brukes som instans av SolarSystem fra ast2000tool i nesten alle filer

All kode er egenskrevet
"""

import numpy as np
import sys, os, time
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

font = {"family": "DejaVu Sans", "weight": "normal", "size": 22}

plt.rc("font", **font)


class SolarSys(SolarSystem):
    # SolarSys brukes av de fleste av våre programmer senere, og også for del 1
    def __init__(self, seed, data_path=None, has_moons=True, verbose=True):
        SolarSystem.__init__(self, seed, data_path=None, has_moons=True, verbose=True)
        self.one_year = np.sqrt(
            self.semi_major_axes[0] ** 3 / self.star_mass
        )  # period of homeplanet
        self.one_years = np.sqrt(
            self.semi_major_axes ** 3 / self.star_mass
        )  # period of all planests

        self.spin = (
            self.initial_positions[0] * self.initial_velocities[1]
            - self.initial_positions[1] * self.initial_velocities[0]
        )  # initial angular mumentum for all planets

        self.ordered_planets = np.argsort(
            np.argsort(np.linalg.norm(self.initial_positions, axis=0))
        )  # list of indices of planets by distance from star
        self.names = np.array(
            [
                "Vulcan",
                "Laconia",
                "Vogsphere",
                "Ilus",
                "Alderaan",
                "Apetos",
                "Auberon",
                "Zarkon",
                "Tellusia",
                "X",
            ]
        )  # Possible names of planets when plotting.
        # Best name-convention in all of astophysics, all innermost planets
        # in ALL planetary systems are called Vulcan, second innermost called Laconia, und so weiter.

    def year_convert_to(self, T, planet="L"):
        # We kinda screwed the pooch by always using years relative to "our" (Laconias) solar system
        # This is a bodge of a method but has become very useful
        # converts to the specified planet, if L: from earth to Laconia, if E other way
        T = np.asarray(T)
        if planet == "L":
            return T / self.one_year
        elif planet == "E":
            return T * self.one_year
        else:  # could be generalized to all planets in system, but nah
            raise AttributeError("unknown planet used in year conversion")

    def analytical_orbits(self):
        p = 10000
        N = self.number_of_planets

        f = np.transpose(np.array([np.linspace(0, 2 * np.pi, p)] * N))
        # f has one axis for each planet
        a = np.array([self.semi_major_axes] * p)
        e = np.array([self.eccentricities] * p)
        omega = np.array([self.aphelion_angles] * p) + np.pi

        # equation 5 in Simulating Planetary orbits
        R = a * (1 - e ** 2) / (1 + e * np.cos(f - omega))
        x = R * np.cos(f)
        y = R * np.sin(f)

        self.a_pos = np.array([x, y])

    def differential_orbits(self, yrs, dt_pr_yr, newpos=False):
        from ivp import Diff_eq as Deq

        if newpos:
            r0 = self.r0
        else:
            r0 = self.initial_positions

        T = self.year_convert_to(yrs, "E")  # yrs is in Laconia years, T is earth years
        dt = self.one_year * dt_pr_yr
        self.time = np.linspace(0, T, int(T / dt) + 1)

        if T != 0:
            e = self.eccentricities
            a = self.semi_major_axes
            omega = self.aphelion_angles + np.pi
            h = self.spin

            start_angle = np.arctan(r0[1] / r0[0])
            start_angle = np.where(r0[0] >= 0, start_angle, start_angle + np.pi)

            # ODE-solver in ivp.py
            orbits = Deq(a, e, h, omega)
            t, u = orbits.solve(start_angle, T, dt)

            R = a * (1 - e ** 2) / (1 + e * np.cos(u - omega))
            # equation 5 from our article on part 2

            x = R * np.cos(u)
            y = R * np.sin(u)

            self.d_pos = np.transpose([x, y], (0, 2, 1))
            self.angles_of_all_times = u

        else:
            self.time = np.zeros((1), dtype=int)
            self.d_pos = np.array([self.initial_positions.T])

    def iterated_orbits(self, yrs, dt_pr_yr):
        def accelerate(r):
            return self.constant * r * (np.linalg.norm(r, axis=0)) ** (-3)

        T = self.one_year * yrs
        dt = self.one_year * dt_pr_yr
        nt = int(T / dt)

        self.time = np.linspace(0, T, nt)

        pos = np.zeros((2, self.number_of_planets, nt))
        pos[:, :, 0] = self.initial_positions

        self.constant = -const.G_sol * self.star_mass

        vel = self.initial_velocities

        acc_0 = accelerate(pos[:, :, 0])

        # Leapfrog integration
        for t in range(nt - 1):
            pos[:, :, t + 1] = pos[:, :, t] + vel * dt + 0.5 * acc_0 * dt ** 2

            acc_1 = accelerate(pos[:, :, t + 1])
            vel = vel + 0.5 * (acc_0 + acc_1) * dt

            acc_0 = acc_1

        self.i_pos = pos

    def plot_orbits(self, array=1, a=False, init=True, final=True, title=""):
        if not type(array) is np.ndarray:
            if array != 1:
                print("Type of array for plotting not recognized, defaults to d_pos.")
            array = self.d_pos

        shap = array.shape
        if shap[0] != 2 or len(shap) != 3:
            print("You've done wrong!")
            return 0

        planet_names = self.names

        if a:  # analytic orbits
            try:
                self.a_pos
            except:
                self.analytical_orbits()
            plt.plot(*self.a_pos, "y")
            plt.plot(*self.a_pos[:, 0, 0], "y", label="Analytical orbits")

        for p in range(array.shape[1]):
            lab = f"{planet_names[self.ordered_planets[p]]}"
            plt.plot(*array[:, p, :], label=lab)

        if init:
            plt.scatter(*array[:, :, 0], label="start")
        if final:
            plt.scatter(*array[:, :, -1], label="final")

        plt.scatter([0], [0], s=80, color=np.array(self.star_color) / 255, label="Sun")

        plt.grid()
        plt.axis("equal")
        plt.xlabel("x, [AU]", fontsize=25)
        plt.ylabel("y, [AU]", fontsize=25)
        plt.title(title)
        plt.legend(loc=1)

    def animate_orbits(self, inn=0, ut=7):
        from matplotlib.animation import FuncAnimation

        fig = plt.figure()
        self.ani_pos = self.d_pos[:, inn: ut + 1, :]

        # Configure figure
        plt.axis("equal")
        plt.axis("off")
        xmax = np.max(abs(self.ani_pos))
        plt.axis((-xmax, xmax, -xmax, xmax))

        # Make an "empty" plot object to be updated throughout the animation
        (self.positions,) = plt.plot([], [], "o", lw=1)
        # Call FuncAnimation
        self.animation = FuncAnimation(
            fig,
            self._next_frame,
            frames=range(len(self.time)),
            repeat=None,
            interval=1,
            blit=True,
            save_count=100,
        )

    def _next_frame(self, i):

        self.positions.set_data(
            (0, *self.ani_pos[0, :, i]), (0, *self.ani_pos[1, :, i])
        )
        self.positions.set_label(("p1", "p2", "p3"))
        return (self.positions,)


def verify(yrs=20):
    print("Verifying orbits")
    years = yrs
    dt = 1e-4

    system.analytical_orbits()
    system.differential_orbits(years, dt)

    system.verify_planet_positions(
        years * system.one_year,
        system.d_pos,
        f"{path}/planet_trajectories_{years}yr.npy",
    )
    system.generate_orbit_video(system.time, system.d_pos)


def show_orbits(yrs=500):
    print("\nPlotting orbits")
    years = yrs
    dt = 1e-4

    system.analytical_orbits()
    system.differential_orbits(years, dt)

    system.plot_orbits(a=True, title="Differential method")
    plt.show()


def do_iterated(yrs=30):
    print("\nPlotting orbits calculated different way")
    years = yrs
    dt = 1e-4

    system.iterated_orbits(years, dt)
    system.plot_orbits(system.i_pos)

    plt.show()


def animate(yrs=20):
    print("\nAnimating orbits")
    years = yrs
    dt = 1e-3

    system.differential_orbits(years, dt)
    system.animate_orbits()
    plt.show()


if __name__ == "__main__":
    seed = 76117
    path = "./../verification_data"
    system = SolarSys(seed, path, False, True)

    verify()
    show_orbits()
    do_iterated()
    animate()
