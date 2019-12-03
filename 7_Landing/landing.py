"""
Program for simulering av landing

All kode er egenskrevet
"""

import numpy as np
import sys, os
import time as tim
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../1_the_Rocket_Engine"))
sys.path.append(os.path.abspath("../2_Planetary_Orbits"))
sys.path.append(os.path.abspath("../3_Habitable_zone"))
sys.path.append(os.path.abspath("../4_Onboard_Orientation_Software"))
sys.path.append(os.path.abspath("../6_Preparing_for_Landing"))



import launch
from orbits import SolarSys
from navigate import navigate
from rocket import Rocket
from atmosphere import P, T, rho

import ast2000tools.utils as util
import ast2000tools.constants as const
# sys.path.append(os.path.abspath("../decompiled"))
# from space_mission import SpaceMission

from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

if __name__ == "__main__":
    seed = 76117
    path = "./../verification_data/"
    system = SolarSys(seed, path)
    mission = SpaceMission(seed, path)

    R = system.radii[1]*1000
    def F_d(A, r, v):
        omega = 2*np.pi/(system.rotational_periods[1]*const.day)
        w = omega*np.asarray([-r[1], r[0], 0])
        v_d = v - w
        dens = rho(np.linalg.norm(r) - R)
        return 1/2*dens*A*np.linalg.norm(v_d)*(-v_d)

    r0 = np.asarray([500000 + R, 0, 0])
    v0 = np.asarray([2000, 0, 0])

    A = 2*const.G*system.masses[1]*const.m_sun*mission.lander_mass/(system.atmospheric_densities[1]*(3*R)**2)

    launch_time = 0.75
    site = 0
    destination = 1
    Volcano, Epstein = launch.do_launch(Rocket=Rocket, verb=False)
    launch.change_reference(mission, system, Volcano, Epstein, site, launch_time)
    mission.verify_manual_orientation(*navigate(system, mission, path))
    Volcano.begin_interplanetary_journey(system, mission, destination=destination, k=1)
    mission.begin_landing_sequence()
