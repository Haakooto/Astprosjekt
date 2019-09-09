"""
Program for å gjennomføre rakettoppskytning

All kode er egenskrevet
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

from engine import Engine
from rocket import Rocket

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

util.check_for_newer_version()

np.random.seed(14441111)
seed = util.get_seed("haakooto")

mission = SpaceMission(seed)
system = SolarSystem(seed)


# Variables for engine
N = int(1e5)  # number of particles
nozzle = 0.25  # nozzle size, as percent of surface
L = 1e-7  # lengt of box in m
T = 3000  # temperature in K
dt_e = 1e-12
ts = 1000

# Variables for rocket
mass = mission.spacecraft_mass  # rocket drymass in kg
R = system.radii[0] * 1000  # planet radius in m
M = system.masses[0] * const.m_sun  # planet mass in kg
dt_r = 0.01

throttle = 14  # how much to trottle engine, must be larger than 1
rocket_area = mission.spacecraft_area
Ne = rocket_area / L ** 2 / throttle  # numer of engineboxes
fuel_load = 40000

# haakooto: throttle = 14, fuel = 40000
# tobiasob: throttle = 4, fuel = 136000

# Packages to build rocket and engine
engine_build = [N, nozzle, T, L, dt_e, ts]
rocket_build = [mass, R, M, dt_r]


Volcano = Rocket(*rocket_build)
Epstein = Engine()

Epstein.build(*engine_build)
Volcano.assemble(Epstein, fuel_load, Ne)

Volcano.launch()


def verify():

	thrust = Epstein.thrust  # thrust pr box
	dm = Epstein.consume  # mass loss rate
	fuel = fuel_load  # loaded fuel
	T1 = Volcano.time  # launch duration
	planet_pos = system.initial_positions[:, 0]
	pos_x = planet_pos[0] + R / const.AU  # x-position relative to star
	pos_y = planet_pos[1]
	T0 = 0  # start of launch

	verify = [thrust, dm, Ne, fuel, T1, (pos_x, pos_y), T0]

	mission.set_launch_parameters(*verify)
	mission.launch_rocket()

	orb_speed = system.initial_velocities[:, 0] / const.yr
	rot_speed = 2 * np.pi * R / (system.rotational_periods[0] * const.day * const.AU)
	final_position = (
		np.asarray([planet_pos[0] + Volcano.r / const.AU, pos_y])
		+ orb_speed * T1
		+ rot_speed * T1
	)

	mission.verify_launch_result(final_position)


if __name__ == "__main__":
	verify()
