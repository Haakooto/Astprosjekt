"""
Program for å finne den beboelige sonen i solsystemet

All kode er egenskrevet

"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("./../2_Planetary_Orbits"))

from orbits import SolarSys

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem


def set_globals(system):
	global T, R, L
	T = system.star_temperature
	R = system.star_radius * 1000
	L = const.sigma * T ** 4 * A(R)  # Star's luminosity


def A(r):
	return 4 * np.pi * r ** 2

def F(r):
	return L / A(r)  # flux from the star in a distance r

def distance(T):
	return np.sqrt(
		L / (8 * np.pi * const.sigma * T ** 4)
	)  # returns distance from star in m where surf_temp = T


def habitable_planets(system):
	set_globals(system)

	R_p = (
		np.linalg.norm(system.initial_positions, axis=0) * const.AU
	)  # initial distance of planets in m
	F_p = F(R_p)  # flux from the star at planet orbits
	E_p = F_p * A(system.radii * 1000) / 2  # total energy hitting planets

	surf_temp = (F_p / (2 * const.sigma)) ** (1 / 4)  # temp of planets in K

	panel_area = 1
	E_lander = F_p * panel_area * 0.12  # energy hitting the solar panels on each planet

	min_T = 260  # -13C
	max_T = 390  # 117C
	max_R = distance(min_T)
	min_R = distance(max_T)
	index = []

	for i, j in enumerate(R_p):
		if min_R < j < max_R:
			index.append(i)
	return index, util.m_to_AU(max_R), util.m_to_AU(min_R)

def plot_habzone(system, mx, mn):
	year = 40
	dt = 1e-4
	system.differential_orbits(year, dt)
	N = 100

	phi = np.linspace(0, 2 * np.pi, 1000)
	X = np.cos(phi)
	Y = np.sin(phi)
	R = mn
	dr = (mx - mn) / N

	for _ in range(N):
		x = R * X
		y = R * Y
		plt.plot(x, y, "xkcd:bright green", lw=1)
		R += dr

	system.plot_orbits(d=True, init=False, final=False)
	plt.plot([0], [0], "xkcd:bright green", label="Habitable Zone")

	plt.title("Habitable zone in solar system", fontsize=25)
	plt.show()

if __name__ == "__main__":
	seed = 76117
	system = SolarSys(seed)

	idx, mxR, mnR = habitable_planets(system)
	print(idx)
	print(mnR, mxR)
	plot_habzone(system, mxR, mnR)
