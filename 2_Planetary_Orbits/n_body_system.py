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
		self.one_years = np.sqrt(self.semi_major_axes ** 3 / self.star_mass)

	def two_body_system(self, yrs, dt_pr_yr):
		def A(r):
			R_p = r[0] - r[1:]
			F_p = -const.G_sol * R_p * self.star_mass * np.transpose([self.masses]*2) * np.transpose([np.linalg.norm(R_p, axis = 1) ** -3]*2)
			A_s = np.sum(F_p, axis=0) / self.star_mass
			A_p = -F_p / np.transpose([self.masses]*2)
			return np.concatenate(([A_s], A_p))

		T = self.one_year * yrs
		dt = self.one_year * dt_pr_yr
		n = int(T / dt)

		self.time = np.linspace(0, T, n)

		# self.all_masses = np.concatenate((self.star_mass, self.masses))

		Rcm = np.sum(self.masses * self.initial_positions, axis=1) / (sum(self.masses) + self.star_mass)
		Vcm = np.sum(self.masses * self.initial_velocities, axis=1) / (sum(self.masses) + self.star_mass)

		R = np.zeros((n, self.number_of_planets + 1, 2))  # [time][object][coord]
		R[0, 0] = -Rcm  # moving sun
		R[0, 1:, 0] = self.initial_positions[0] - Rcm[0]  # setting planet pos
		R[0, 1:, 1] = self.initial_positions[1] - Rcm[1]  # setting planet pos

		V = np.zeros_like(R)  # [time][object][coord]
		V[0, 0] = -Vcm  #
		V[0, 1:, 0] = self.initial_velocities[0] - Vcm[0]  # setting planet vel
		V[0, 1:, 1] = self.initial_velocities[1] - Vcm[1]  # setting planet vel

		for t in range(n - 1):

			a0 = A(R[t])
			R[t + 1] = R[t] + V[t] * dt + 0.5 * a0 * dt ** 2
			a1 = A(R[t + 1])
			V[t + 1] = V[t] + 0.5 * (a0 + a1) * dt

		self.solar_orb = R[:, 0]
		self.planet_orbs = R[:, 1:]
		self.vels = V

	# def energy_conserve(self):
	# 	relvel = np.linalg.norm(self.vel[:, 0] - self.vel[:, 1:], axis=1)
	# 	relpos = np.linalg.norm(self.solar_orb - self.planet_orb, axis=1)

	# 	mu_hat = self.star_mass * self.p_mass / (self.star_mass + self.p_mass)

	# 	self.E = (
	# 		0.5 * mu_hat * relvel ** 2
	# 		- const.G_sol * (self.star_mass + self.p_mass) * mu_hat / relpos
	# 	)

	# 	plt.plot(self.time, self.E)
	# 	plt.show()

	def radial_vel(self, i=np.pi):
		xvel = np.sin(i) * self.vel[:, 0, 0]
		noise = np.random.normal(0, 0.2 * xvel.max(), len(xvel))
		V = 1444
		self.vnoise = xvel + noise + V

		plt.plot(self.time, self.vnoise)
		plt.show()

	def plot_n_pos(self):
		S = np.transpose(self.solar_orb)
		P = np.transpose(self.planet_orbs, (1, 2, 0))

		planet_names = ["Vulcan", "Laconia", "Vogsphere", "Ilus", "Alderaan", "Apetos", "Auberon", "Zarkon", "Tellusia", "X"]

		plt.plot(*S, color=np.array(self.star_color) / 255, label="Sun")
		for p in range(self.number_of_planets):
			lab = planet_names[p]
			plt.plot(*P[p], "c", label=lab)
		plt.scatter([0], [0], label="Centre of mass")

		plt.grid()
		plt.axis("equal")
		plt.legend(loc=1)
		plt.show()

	def assemble_data(self):
		first = np.concatenate(([self.star_mass], self.vnoise))
		sec = np.concatenate(([self.p_mass], self.time))
		data = np.concatenate(([first], [sec]))
		np.save("radial_velocity_curve.npy", data)



if __name__ == "__main__":
	seed = util.get_seed("mikkelme")

	system = SolarSys(seed)

	yrs = 2220
	dt = 1e-4

	system.two_body_system(yrs, dt)
	system.plot_n_pos()
	# system.energy_conserve()
	# system.radial_vel(i=2*np.pi/3)
	# system.assemble_data()

	# np.save("star_data.npy", system.vnoise)
	# np.save("times.npy", system.time)
	# times = system.time
	# data = system.vnoise

	# data = np.load("star_data.nyp.npy")
	# times = np.load("times.npy")
