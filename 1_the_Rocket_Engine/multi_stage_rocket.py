"""
Kode ikke i bruk, ignorer
"""

import numpy as np
from Engine import Engine
from Rocket import Rocket
import sys, os
import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem
import matplotlib.pyplot as plt


class Rocket:
    def __init__(self, r0, M):
        self.stages = []
        self.mass = 0
        self.r = r0
        self.R = r0
        self.v = 0
        self.M = M
        self.time = 0
        self.dt = 0.01

    def add_stage(self, drymass, fuel_mass, engine, name):
        self.stages.append([drymass, fuel_mass, engine, name])
        self.mass += drymass + fuel_mass

    def next_stage(self, n):
        return self.stages[n]

    def launch(self):

        stage = self.stages[0]
        print(f"Stage {stages[3]}")
        engine = stages[2]
        engine.build()
        engine.ignite()
        T = engine.thrust

        mF = stages[1]
        mass_use = 0

        while not self.escaped(self.r, self.v):

            A0 = self.acceleration(self.r, self.mass, T)
            self.r += self.v * self.dt + 0.5 * A0 * self.dt ** 2
            A1 = self.acceleration(self.r, self.mass, T)
            self.v += +0.5 * (A0 + A1) * self.dt
            mass_use += engine.consume * self.dt
            self.mass -= engine.consume * self.dt

            if mass_use >= mF:
                print("Stage over, starting next stage")
                self.statusrapport()

                self.mass -= stage[0]
                break

            if self.r < self.R:
                self.statusrapport()
                print("RUD in LAO")
                sys.exit()

            self.time += self.dt

    self.statusrapport()
    print("We escaped")

    def escaped(self, r, v):
        ke = 0.5 * v ** 2
        pe = const.G * self.M / r
        return ke > pe

    def acceleration(self, r, m, T):
        return (T - const.G * m * self.M / r ** 2) / m

    def statusrapport(self):
        print(f"Altitude: {self.r - self.R}")
        print(f"Speed: {self.v}")
        print(f"Time: {self.time}")
        print(f"mass: {self.mass}")


np.random.seed(14441111)
seed = util.get_seed("haakooto")


def escape_veolocity(r, v, m):
    ke = 0.5 * v ** 2
    pe = -G * M / r
    return ke > pe


mission = SpaceMission(seed)
system = SolarSystem(seed)

G = const.G
M = system.masses[0] * const.m_sun
R = system.radii[0] * 1000

craft_mass = mission.spacecraft_mass
craft_area = mission.spacecraft_area

# Variables for rocketengine
T = 10000  # temperature in K
L = 1e-6  # lengt of box in m
N = int(1e5)  # number of particles

area = np.array([16, 38, 40])
Ne = area / L ** 2  # numer of engineboxes
nozzle = 1

dt = 1e-12
ts = 1000

inp = lambda n: [N, Ne[n], nozzle, T, L, dt, ts]

F1 = 3000
F2 = 40000
F3 = 150000

rocket = Rocket(R, M)
engine1 = Engine(*inp(0))
engine2 = Engine(*inp(1))
engine3 = Engine(*inp(2))
rocket.add_stage(11000, F3, engine3, "terrum")
rocket.add_stage(4000, F2, engine2, "venum")
rocket.add_stage(1100, F1, engine1, "mercum")

rocket.launch()
print(rocket.statusrapport())

