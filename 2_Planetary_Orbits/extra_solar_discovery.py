"""
Program for å _lage hastighetsplot, og løse det

All kode er egenskrevet

"""

import numpy as np
import sys, os
import matplotlib.pyplot as plt

import ast2000tools.utils as util
import ast2000tools.constants as const
from ast2000tools.space_mission import SpaceMission
from ast2000tools.solar_system import SolarSystem

class SolarSys(SolarSystem):
	def __init__(self, seed, data_path=None, has_moons=True, verbose=True):
		SolarSystem.__init__(self, seed, data_path=None, has_moons=True, verbose=True)
		self.one_year = np.sqrt(self.semi_major_axes[0] ** 3 / self.star_mass)

	def find_largest_attractor(self):
		F = - const.G_sol * self.star_mass * self.masses * np.linalg.norm(self.initial_positions, axis = 0) ** -2
		return np.argmin(F)

	def two_body_system(self, yrs, dt_pr_yr):

		def A(r):
			R = r[0] - r[1]
			F = const.G_sol * R * np.linalg.norm(R) ** -3
			return np.array([-F * p_mass, F * self.star_mass])

		T = self.one_year * yrs
		dt = self.one_year * dt_pr_yr
		n = int(T / dt)

		la_idx = self.find_largest_attractor()
		p_mass = self.masses[la_idx]

		Rcm = p_mass * self.initial_positions[:, la_idx] / (p_mass + self.star_mass)
		Vcm = p_mass * self.initial_velocities[:, la_idx] / (p_mass + self.star_mass)

		R = np.zeros((n, 2, 2)) # [time][object][coord]
		R[0, 0] = -Rcm # moving sun
		R[0, 1] = self.initial_positions[:, la_idx] - Rcm # setting planet pos

		V = np.zeros((2,2)) #[object][coord]
		V[0] = -Vcm #
		V[1] = self.initial_velocities[:, la_idx] - Vcm # setting planet vel

		for t in range(n - 1):

			a0 = A(R[t])
			R[t + 1] = R[t] + V * dt + 0.5 * a0 * dt ** 2
			a1 = A(R[t + 1])
			V = V + 0.5 * (a0 + a1) * dt


		self.solar_orb = R[:, 0]
		self.planet_orb = R[:, 1]

	def plot_two_pos(self):
		S = np.transpose(self.solar_orb)
		P = np.transpose(self.planet_orb)

		plt.plot(*S, color=np.array(self.star_color)/255, label="Sun")
		plt.plot(*P,  "c", label="Planet")
		plt.scatter([0], [0], label="Centre of mass")

		plt.grid()
		plt.axis("equal")
		plt.legend(loc=1)
		plt.show()



if __name__ == "__main__":
	seed = util.get_seed("haakooto")

	system = SolarSys(seed)

	yrs = 10
	dt = 1e-4

	system.two_body_system(yrs, dt)
	system.plot_two_pos()


