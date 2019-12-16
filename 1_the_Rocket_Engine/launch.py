"""
Program for å gjennomføre rakettoppskytning

All kode er egenskrevet
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os, time

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission

sys.path.append(os.path.abspath("../2_Planetary_Orbits"))
from orbits import SolarSys
from engine import Engine
from rocket import Rocket


seed = 76117
path = "./../verification_data/"

launch_mission = SpaceMission(seed, path, False, True)
launch_system = SolarSys(seed, path, False, True)

# Variables for engine
N = int(1e5)  # number of particles
nozzle = 0.25  # nozzle size, as percent of surface
L = 1e-7  # lengt of box in m
T = 2800  # temperature in K
dt_e = 1e-12  # engine
ts = 1000

# Variables for rocket
mass = launch_mission.spacecraft_mass  # rocket drymass in kg
R = launch_system.radii[0] * 1000  # planet radius in m
M = launch_system.masses[0] * const.m_sun  # planet mass in kg
dt_r = 0.01  # rocket

throttle = 1 / 12  # how much to trottle engine, must be larger than 1
rocket_area = launch_mission.spacecraft_area
Ne = rocket_area / L ** 2 * throttle  # numer of engineboxes
fuel_load = 50000

# 76117: throttle = 12, fuel = 50000
# 65212: throttle = 4, fuel = 136000

# Packages to build rocket and engine
engine_build = lambda: [N, nozzle, T, L, dt_e, ts]
rocket_build = lambda: [mass, R, M, dt_r]
parts = lambda engine: [engine, fuel_load, Ne]


def do_launch(Rocket=Rocket, verb=True):
    Volcano = Rocket(*rocket_build(), verbose=verb)
    Epstein = Engine(*engine_build())
    # Create instance of rocket and engine

    Volcano.assemble(*parts(Epstein))
    # Insert engine into rocket

    Volcano.launch()
    # Perform the launch, where the engine is run to use the performance values in the launch

    return Volcano, Epstein


def change_reference(mission, system, rocket, engine, site=0, T0=0):
    """
    Changes reference from planet-centred to sol-centred, and verifies result
    Is generalized for launch-site and launch time, as Part 3 stipulates
    """
    # site is angle in star reference, 0 is along x-axis
    # T0 is given in laconia years

    thrust = engine.thrust  # thrust pr box
    dm = engine.consume  # mass loss rate
    fuel = fuel_load  # loaded fuel
    T1 = rocket.time  # launch duration

    if T0 == 0:
        planet_pos = system.initial_positions[:, 0]
        planet_vel = system.initial_velocities[:, 0]

    else:
        time_idx = np.argmin(abs(system.time - system.year_convert_to(T0, "E"))) - 1
        T0 = system.time[time_idx]

        planet_pos = system.d_pos[:, 0, time_idx]
        planet_vel = (
            system.d_pos[:, 0, time_idx] - system.d_pos[:, 0, time_idx - 1]
        ) / (system.time[1])
    # Find position and vel of planet after launch

    launch_site = site  # angle of launch site on the equator [0,2pi]
    pos_x = (
        planet_pos[0] + R * np.cos(launch_site) / const.AU
    )  # x-position relative to star as if T0 = 0
    pos_y = (
        planet_pos[1] + R * np.sin(launch_site) / const.AU
    )  # y-position relative to star as if T0 = 0

    params = [thrust, dm, Ne, fuel, T1, (pos_x, pos_y), T0]

    orb_speed = planet_vel / const.yr  # planet velocity around star
    abs_rot_speed = (
        2 * np.pi * R / (system.rotational_periods[0] * const.day * const.AU)
    )
    rot_speed = np.asarray(
        [-np.sin(launch_site) * abs_rot_speed, np.cos(launch_site) * abs_rot_speed]
    )  # surface velocity around planet

    final_position = (
        np.asarray(
            [
                planet_pos[0] + np.cos(launch_site) * rocket.r / const.AU,
                planet_pos[1] + np.sin(launch_site) * rocket.r / const.AU,
            ]
        )
        + orb_speed * T1
        + rot_speed * T1
    )  # Do launch as if T0 = 0, then shift position with planet as T0 = T

    mission.set_launch_parameters(*params)
    mission.launch_rocket()
    mission.verify_launch_result(final_position)


if __name__ == "__main__":
    years = 25.245
    dt = 1e-4

    launch_system.differential_orbits(years, dt)

    Volcano, Epstein = do_launch()
    change_reference(launch_mission, launch_system, Volcano, Epstein, 0, years)
