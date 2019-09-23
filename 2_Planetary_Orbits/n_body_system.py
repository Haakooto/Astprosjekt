"""
Program for å simulere N 2-legemesystem sammtidig

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
	def __init__(self, seed, N, data_path=None, has_moons=True, verbose=True):
		SolarSystem.__init__(self, seed, data_path=None, has_moons=True, verbose=True)
		self.one_year = np.sqrt(self.semi_major_axes[0] ** 3 / self.star_mass)
		self.one_years = np.sqrt(self.semi_major_axes ** 3 / self.star_mass)
		N = min(N, self.number_of_planets)
		self.ordered_planets = np.argsort(
			np.argsort(np.linalg.norm(self.initial_positions, axis=0))
		)

		las = self.find_largest_attractor()
		self.planets = las[:N]
		self.my_number_of_planets = N
		self.my_masses = self.masses[self.planets]
		self.my_initial_positions = self.initial_positions[:, self.planets]
		self.my_initial_velocities = self.initial_velocities[:, self.planets]

	def find_largest_attractor(self):
		F = (
			-const.G_sol
			* self.star_mass
			* self.masses
			* np.linalg.norm(self.initial_positions, axis=0) ** -2
		)
		return np.argsort(F)

	def N_body_system(self, yrs, dt_pr_yr):
		def A(r):
			R_p = r[0] - r[1:]
			F_p = (
				-const.G_sol
				* R_p
				* self.star_mass
				* np.transpose([self.my_masses] * 2)
				* np.transpose([np.linalg.norm(R_p, axis=1) ** -3] * 2)
			)
			A_s = np.sum(F_p, axis=0) / self.star_mass
			A_p = -F_p / np.transpose([self.my_masses] * 2)
			return np.concatenate(([A_s], A_p))

		T = self.one_year * yrs
		dt = self.one_year * dt_pr_yr
		n = int(T / dt)

		self.time = np.linspace(0, T, n)

		Rcm = np.sum(self.my_masses * self.my_initial_positions, axis=1) / (
			sum(self.my_masses) + self.star_mass
		)
		Vcm = np.sum(self.my_masses * self.my_initial_velocities, axis=1) / (
			sum(self.my_masses) + self.star_mass
		)

		R = np.zeros((n, self.my_number_of_planets + 1, 2))  # [time][object][coord]
		R[0, 0] = -Rcm  # moving sun
		R[0, 1:, 0] = self.my_initial_positions[0] - Rcm[0]  # setting planet pos
		R[0, 1:, 1] = self.my_initial_positions[1] - Rcm[1]  # setting planet pos

		V = np.zeros_like(R)  # [time][object][coord]
		V[0, 0] = -Vcm  #
		V[0, 1:, 0] = self.my_initial_velocities[0] - Vcm[0]  # setting planet vel
		V[0, 1:, 1] = self.my_initial_velocities[1] - Vcm[1]  # setting planet vel

		# print(A(R[0]))
		for t in range(n - 1):

			a0 = A(R[t])
			R[t + 1] = R[t] + V[t] * dt + 0.5 * a0 * dt ** 2
			a1 = A(R[t + 1])
			V[t + 1] = V[t] + 0.5 * (a0 + a1) * dt

		self.solar_orb = R[:, 0]
		self.planet_orbs = R[:, 1:]
		self.vels = V
		self.R = R

	def radial_vel(self, i=np.pi):
		xvel = np.sin(i) * self.vels[:, 0, 0]
		noise = np.random.normal(0, 0.2 * xvel.max(), len(xvel))
		V = 1444
		self.vnoise = xvel + noise + V

		plt.plot(self.time, self.vnoise)
		plt.show()

	def plot_n_pos(self):
		S = np.transpose(self.solar_orb)
		P = np.transpose(self.planet_orbs, (1, 2, 0))

		planet_names = [
			"Vulcan",
			"Laconia",
			"Vogsphere",
			"Ilus",
			"Alderaan",
			"Apetos",
			"Auberon",
			"Zarkon",
			"Tellusia",
			"X",
		]

		plt.plot(*S, color=np.array(self.star_color) / 255, label="Sun")
		for p in range(self.my_number_of_planets):
			lab = planet_names[self.ordered_planets[p]]
			plt.plot(*P[p], label=lab)
		plt.scatter([0], [0], label="Centre of mass")

		plt.grid()
		plt.axis("equal")
		plt.legend(loc=1)
		plt.show()

	def assemble_data(self):
		data = np.concatenate(([self.vnoise], [self.time]))
		np.save("radial_velocity_curve_multiple.npy", data)

	def animate_orbits(self):
		from matplotlib.animation import FuncAnimation

		fig = plt.figure()

		self.RR = np.transpose(self.R, (2, 1, 0))

		# Configure figure
		plt.axis("equal")
		plt.axis("off")
		xmax = np.max(abs(self.RR))
		plt.axis((-xmax, xmax, -xmax, xmax))

		# Make an "empty" plot object to be updated throughout the animation
		self.positions, = plt.plot([], [], "o", lw=1)
		# print(self.d_pos[0, :, 100-0:100+1])
		# print(*np.vsplit(self.d_pos[0, :, 100-10:100+1], 1)[0])
		# Call FuncAnimation
		self.animation = FuncAnimation(
			fig,
			self._next_frame,
			frames=range(len(self.time)),
			repeat=None,
			interval=1,  # 000 * self.dt,
			blit=True,
			save_count=100,
		)
		plt.show()

	def _next_frame(self, i):
		self.positions.set_data((0, *self.RR[0, :, i]), (0, *self.RR[1, :, i]))

		return (self.positions,)


if __name__ == "__main__":
	# import time

	# timer = time.time()
	seed = util.get_seed("haakooto")

	system = SolarSys(seed, 4)

	yrs = 40
	dt = 1e-3

	system.N_body_system(yrs, dt)
	print(system.time)
	# print(time.time() - timer)
	# system.plot_n_pos()
	# system.energy_conserve()
	system.animate_orbits()
	system.radial_vel(i=2 * np.pi / 3)

	# system.assemble_data()

	# np.save("star_data.npy", system.vnoise)
	# np.save("times.npy", system.time)
	# times = system.time
	# data = system.vnoise

	# data = np.load("npys/radial_velocity_curve_multiple.npy")
	# plt.plot(data[1], data[0])
	# plt.show()
