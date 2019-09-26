"""
Program for å gjennomføre rakettoppskytning

All kode er egenskrevet
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os

from engine import Engine
from rocket import Rocket

sys.path.append(os.path.abspath("../2_Planetary_Orbits"))

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from orbits import SolarSys

seed = 76117

mission = SpaceMission(seed)
system = SolarSys(seed)
dummy_system = SolarSys(seed)

# Variables for engine
N = int(1e5)  # number of particles
nozzle = 0.25  # nozzle size, as percent of surface
L = 1e-7  # lengt of box in m
T = 2800  # temperature in K
dt_e = 1e-12
ts = 1000

# Variables for rocket
mass = mission.spacecraft_mass  # rocket drymass in kg
R = system.radii[0] * 1000  # planet radius in m
M = system.masses[0] * const.m_sun  # planet mass in kg
dt_r = 0.01

throttle = 1 / 12  # how much to trottle engine, must be larger than 1
rocket_area = mission.spacecraft_area
Ne = rocket_area / L ** 2 * throttle  # numer of engineboxes
fuel_load = 50000

# haakooto: throttle = 12, fuel = 50000
# tobiasob: throttle = 4, fuel = 136000

# Packages to build rocket and engine
engine_build = lambda : [N, nozzle, T, L, dt_e, ts]
rocket_build = lambda : [mass, R, M, dt_r]
asseble = lambda engine: [engine, fuel_load, Ne]

def do_launch():
	Volcano = Rocket(*rocket_build())
	Epstein = Engine()

	Epstein.build(*engine_build())
	Volcano.assemble(*asseble(Epstein))

	Volcano.launch()

	return Volcano, Epstein


def verify(mission, system, rocket, engine, site=np.pi, T0=0):
	# T0 is given in laconia years

	thrust = engine.thrust  # thrust pr box
	dm = engine.consume  # mass loss rate
	fuel = fuel_load  # loaded fuel
	T1 = rocket.time  # launch duration

	planet_pos = system.d_pos[system.time[-1], 0, :]

	launch_site = site		#angle of launch site on the equator [0,2pi]
	pos_x = planet_pos[0] + R*np.cos(launch_site) / const.AU  # x-position relative to star
	pos_y = planet_pos[1] + R*np.sin(launch_site)/const.AU	# y-position relative to star
	T0 = system.year_convert_to(T0, "E")  # start of launch in earth years

	verify = [thrust, dm, Ne, fuel, T1, (pos_x, pos_y), T0]

	mission.set_launch_parameters(*verify)
	mission.launch_rocket()

	orb_speed = system.initial_velocities[:, 0] / const.yr
	abs_rot_speed = 2 * np.pi * R / (system.rotational_periods[0] * const.day * const.AU)
	rot_speed = np.asarray(
		[-np.sin(launch_site)*abs_rot_speed, np.cos(launch_site) * abs_rot_speed]
	)
	final_position = (
		np.asarray([planet_pos[0] + np.cos(launch_site) * rocket.r / const.AU, planet_pos[1] + np.sin(launch_site) * rocket.r / const.AU])
		+ orb_speed * T1
		+ rot_speed * T1
	)
	print(final_position)

	mission.verify_launch_result(final_position)


if __name__ == "__main__":
	years = 0
	dt = 1e-3

	dummy_system.differential_orbits(years, dt)

	rocket, engine = do_launch()
	verify(mission, dummy_system, rocket, engine, T0=years)
