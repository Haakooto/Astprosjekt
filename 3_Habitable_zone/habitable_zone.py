"""
Program for å finne den beboelige sonen i solsystemet

All kode er egenskrevet

"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem


def set_globals(system):
    global T
    T = system.star_temperature
    global R
    R = system.star_radius * 1000
    global L
    L = const.sigma * T ** 4 * A(R)  # Star's luminosity


def A(r):
    return 4 * np.pi * r ** 2


def F(r):
    return L / A(r)  # flux from the star in a distance r


def distance(T):
    return np.sqrt(
        L / (8 * np.pi * const.sigma * T ** 4)
    )  # returns distance from star in m where surf_temp = T


def planet_temp(system):
    set_globals(system)

    R_p = (
        np.linalg.norm(system.initial_positions, axis=0) * const.AU
    )  # initial distance of planets in m
    F_p = F(R_p)  # flux from the star at planet orbits
    E_p = F_p * A(system.radii * 1000) / 2  # total energy hitting planets

    surf_temp = (F_p / (2 * const.sigma)) ** (1 / 4)  # temp of planets in K

    panel_area = 1
    E_lander = F_p * panel_area * 0.12  # energy hitting the solar panels on each planet

    min_T = 260
    max_T = 390
    max_R = distance(min_T)
    min_R = distance(max_T)
    index = []
    for i, j in enumerate(R_p):
        if min_R < j < max_R:
            index.append(i)
    return surf_temp, R_p, min_R, max_R, E_lander


if __name__ == "__main__":
    seed = util.get_seed("haakooto")
    system = SolarSystem(seed)

    surf_temp, R_p, min_R, max_R, E_lander = planet_temp(system)
    print(E_lander)
