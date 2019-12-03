"""
Program for å implementere modellen av Vogspheres atmosfære

All kode er egenskrevet
"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem


seed = util.get_seed(76117)

system = SolarSystem(seed)

G = const.G
M = system.masses[1] * const.m_sun
m = 26.5 * const.m_p
k = const.k_B
R = system.radii[1] * 1000
rho0 = system.atmospheric_densities[1]
T0 = 280
P0 = rho0 * k * T0 / m
c = T0 ** (7 / 5) * (P0) ** (-2 / 5)
c1 = P0 ** (2 / 7) - 2 * G * m * M / (7 * k * c ** (5 / 7) * R)
r_maks = 4 * G * M * m * R / (2 * G * M * m - c1 * 7 * k * c ** (5 / 7) * R) - R
T_maks = 140


def P(r):
	if 0 <= r <= r_maks:
		return (2 * G * m * M / (7 * k * c ** (5 / 7) * (r + R)) + c1) ** (7 / 2)
	elif r > r_maks:
		c2 = np.log(P(r_maks)) - G * M * m / (k * T_maks * (r_maks + R))
		return np.exp(c2 + G * M * m / (k * T_maks * (r + R)))


def T(r):
	if 0 <= r <= r_maks:
		return c ** (5 / 7) * P(r) ** (2 / 7)
	elif r > r_maks:
		return T_maks

Pv = np.vectorize(P)
Tv = np.vectorize(T)

def density(h):
	if h < 200000:
		return Pv(h) * m / k / Tv(h)
	else: return 0

densityv = np.vectorize(density)



if __name__ == "__main__":
	rs = np.linspace(0, r_maks + 10000, 10000)
	plt.plot(rs, Pv(rs))
	plt.show()
	plt.plot(rs, Tv(rs))
	plt.show()
	plt.plot(rs, density(rs))
	plt.show()
